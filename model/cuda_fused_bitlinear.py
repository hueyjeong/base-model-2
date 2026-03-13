"""Fused BitLinear CUDA 커널 래퍼

activation quant + ternary matmul + dequant를 단일 커널로 실행.
Forward만 fused, backward는 기존 autograd로 처리.

Weight는 캐시된 INT8 ternary + scale을 사용.
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Function

from model.bitlinear import BitLinear, quantize_weights_158, quantize_activations_8bit

# JIT 빌드
_FUSED_EXT = None
try:
    from torch.utils.cpp_extension import load as _cpp_load
    _kdir = os.path.dirname(__file__)
    _FUSED_EXT = _cpp_load(
        name="cuda_fused_bitlinear_ext",
        sources=[
            os.path.join(_kdir, "cuda_fused_bitlinear_ext.cpp"),
            os.path.join(_kdir, "cuda_fused_bitlinear_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
except Exception as e:
    import warnings
    warnings.warn(f"Fused BitLinear CUDA 빌드 실패: {e}")


class FusedBitLinearFn(Function):
    """Fused forward (CUDA) + standard backward"""

    @staticmethod
    def forward(ctx, x, weight, bias, norm_weight, norm_bias, norm_eps):
        # LayerNorm (elementwise_affine=False이므로 weight/bias 무시)
        x_norm = torch.nn.functional.layer_norm(x, (x.shape[-1],), None, None, norm_eps)

        # Weight quantization (캐시는 상위 모듈에서 관리)
        w_quant, w_scale = quantize_weights_158(weight)
        w_int8 = w_quant.to(torch.int8)

        # Fused CUDA forward
        out = _FUSED_EXT.forward(x_norm, w_int8, w_scale.reshape(1), bias)

        # backward용 저장
        ctx.save_for_backward(x, weight, bias)
        ctx.norm_eps = norm_eps

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors

        # Standard BitLinear backward (STE 호환)
        x_norm = torch.nn.functional.layer_norm(x, (x.shape[-1],), None, None, ctx.norm_eps)
        x_quant, x_scale = quantize_activations_8bit(x_norm)
        w_quant, w_scale = quantize_weights_158(weight)

        # grad_x: dL/dx = dL/dout @ w_quant * scales * layernorm_grad
        # 단순화: standard autograd로 위임
        # 여기서는 forward를 재계산하여 autograd 그래프를 활용
        x_det = x.detach().requires_grad_(True)
        w_det = weight.detach().requires_grad_(True)

        with torch.enable_grad():
            x_n = torch.nn.functional.layer_norm(x_det, (x_det.shape[-1],), None, None, ctx.norm_eps)
            xq, xs = quantize_activations_8bit(x_n)
            wq, ws = quantize_weights_158(w_det)
            out = torch.nn.functional.linear(xq, wq, bias)
            out = out * (ws * xs)

        out.backward(grad_output)

        grad_x = x_det.grad
        grad_w = w_det.grad
        grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1))) if bias is not None else None

        return grad_x, grad_w, grad_bias, None, None, None


class FusedBitLinear(nn.Module):
    """Fused BitLinear — forward는 CUDA fused 커널, backward는 standard autograd"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        self._weight_version = -1
        self._w_quant_cache = None
        self._w_scale_cache = None

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        if _FUSED_EXT is not None and x.is_cuda:
            return FusedBitLinearFn.apply(
                x, self.weight, self.bias,
                None, None, self.norm.eps,
            )
        # Fallback: standard BitLinear
        x_norm = self.norm(x)
        x_quant, x_scale = quantize_activations_8bit(x_norm)

        v = self.weight._version
        if v != self._weight_version:
            w_quant, w_scale = quantize_weights_158(self.weight)
            self._w_quant_cache = w_quant
            self._w_scale_cache = w_scale
            self._weight_version = v

        out = torch.nn.functional.linear(x_quant, self._w_quant_cache, self.bias)
        out = out * (self._w_scale_cache * x_scale)
        return out


def replace_bitlinear_with_fused(model: nn.Module) -> nn.Module:
    """모델 내 모든 BitLinear를 FusedBitLinear로 교체"""
    for name, child in model.named_children():
        if isinstance(child, BitLinear):
            fused = FusedBitLinear(child.in_features, child.out_features,
                                    bias=child.bias is not None)
            fused.weight = child.weight
            if child.bias is not None:
                fused.bias = child.bias
            setattr(model, name, fused)
        else:
            replace_bitlinear_with_fused(child)
    return model


def is_available():
    return _FUSED_EXT is not None
