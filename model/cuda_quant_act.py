"""Fused CUDA quantize_activations_8bit — 단일 커널로 STE 양자화

Forward: 5+ PyTorch 커널 (abs, amax, clamp, mul, div, round, clamp) → 1 CUDA 커널.
Backward: STE identity (d_x = d_x_quant / x_scale) — 1 PyTorch 연산.

커널 감소: ~110 calls × 5+ kernels = 550+ → ~110 (forward), 550+ → ~110 (backward).
"""
import os
import torch
from torch.autograd import Function

_EXT = None
try:
    from torch.utils.cpp_extension import load as _cpp_load
    _kdir = os.path.dirname(__file__)
    _EXT = _cpp_load(
        name="cuda_quant_act_ext",
        sources=[
            os.path.join(_kdir, "cuda_quant_act_ext.cpp"),
            os.path.join(_kdir, "cuda_quant_act_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
except Exception as e:
    import warnings
    warnings.warn(f"CUDA fused quant_act 빌드 실패: {e}")


class _CUDAQuantActFn(Function):
    """Fused forward (CUDA) + STE backward

    Forward: per-row absmax + scale → round → clamp (단일 커널)
    Backward: d_x = d_x_quant / x_scale (STE, amax gradient 무시)
    """

    @staticmethod
    def forward(ctx, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K)

        # bf16/fp16 → fp32 변환 (CUDA 커널은 fp32 전용)
        orig_dtype = x_2d.dtype
        if orig_dtype != torch.float32:
            x_2d = x_2d.float()
        x_2d = x_2d.contiguous()

        x_quant_2d, x_scale_2d = _EXT.forward(x_2d)

        # 원래 dtype 복원
        if orig_dtype != torch.float32:
            x_quant_2d = x_quant_2d.to(orig_dtype)

        # 원래 shape 복원
        x_quant = x_quant_2d.reshape(orig_shape)
        scale_shape = list(orig_shape)
        scale_shape[-1] = 1
        x_scale = x_scale_2d.reshape(scale_shape)

        ctx.save_for_backward(x_scale)
        return x_quant, x_scale

    @staticmethod
    def backward(ctx, d_x_quant, d_x_scale):
        x_scale, = ctx.saved_tensors
        # STE: d_x = d_x_quant / x_scale
        # amax gradient 무시 (standard QAT, ~0.006%, torch.compile NaN 원인)
        d_x = d_x_quant / x_scale.clamp(min=1e-8)
        return d_x


def cuda_quantize_activations_8bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA fused quantize_activations_8bit

    quantize_activations_8bit()과 동일한 입출력.
    CUDA 커널로 5+ op fusion → 1 커널 launch.
    """
    return _CUDAQuantActFn.apply(x)


def is_available() -> bool:
    """CUDA fused quant_act 커널 사용 가능 여부"""
    return _EXT is not None
