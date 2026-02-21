"""INT8 BitLinear — PyTorch 양자화 + INT8 tensor core matmul

Triton 커널 없이 순수 PyTorch 연산으로 양자화 수행.
F.linear만 INT8 _int_mm으로 교체하여 tensor core 활용.
torch.compile과 완전히 호환.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _quantize_activations(x: torch.Tensor):
    """Per-row 8-bit absmax quantization (PyTorch ops)."""
    eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * 127.0 / eta).round().clamp(-128, 127)
    scale = eta / 127.0
    return x_q, scale


def _quantize_weights(w: torch.Tensor):
    """Absmean ternary quantization (PyTorch ops)."""
    gamma = w.abs().mean().clamp(min=1e-5)
    w_q = (w / gamma).clamp(-1, 1).round()
    return w_q, gamma


def _int_mm_safe(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    """INT8 matmul — 차원이 8의 배수가 아니면 FP16 fallback.

    a: (M, K) int8, b: (K, N) int8 → (M, N) int32
    """
    M, K = a_int8.shape
    _, N = b_int8.shape

    if M % 8 == 0 and K % 8 == 0 and N % 8 == 0:
        return torch._int_mm(a_int8, b_int8)
    # FP16 fallback (tensor core 활용, INT8보단 느리지만 FP32보다 빠름)
    return (a_int8.half() @ b_int8.half()).int()


# ---------------------------------------------------------------------------
# Custom autograd: INT8 forward + backward
# ---------------------------------------------------------------------------
class _BitLinearInt8Fn(torch.autograd.Function):
    """Forward: PyTorch 양자화 + INT8 matmul
    Backward: INT8 for grad_x, FP16 for grad_w, pure STE
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x_norm, weight, bias):
        x_q, x_scale = _quantize_activations(x_norm)
        w_q, w_scale = _quantize_weights(weight)

        batch_shape = x_norm.shape[:-1]
        M = x_q.reshape(-1, x_q.shape[-1]).shape[0]
        x_int8 = x_q.reshape(M, -1).to(torch.int8)
        w_int8 = w_q.to(torch.int8)

        out_int32 = _int_mm_safe(x_int8, w_int8.t())
        out = out_int32.float().reshape(*batch_shape, -1)
        out = out * (w_scale * x_scale)

        if bias is not None:
            out = out + bias

        # backward 캐싱
        ctx.save_for_backward(x_norm.reshape(M, -1).contiguous(), w_int8, w_scale)
        ctx.batch_shape = batch_shape
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x_norm_2d, w_int8, w_scale = ctx.saved_tensors
        batch_shape = ctx.batch_shape
        M = x_norm_2d.shape[0]
        out_f = w_int8.shape[0]

        go_2d = grad_output.reshape(M, -1).float()

        # grad_x: INT8 (weight=ternary, lossless)
        go_abs_max = go_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        go_int8 = (go_2d * 127.0 / go_abs_max).round().clamp(-128, 127).to(torch.int8)
        go_scale = go_abs_max / 127.0

        grad_x_int32 = _int_mm_safe(go_int8, w_int8)
        # Original BitLinear chain rule: dL/dx = dL/dx_q * d(x_q)/dx = dL/dy * w_scale * w_q
        grad_x = (grad_x_int32.float() * go_scale * w_scale).reshape(*batch_shape, -1)

        # grad_w: float matmul (uses TF32 natively on Ampere/Ada/Blackwell, prevents FP16 overflow)
        grad_w = go_2d.t() @ x_norm_2d

        grad_bias = go_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, grad_bias


# ---------------------------------------------------------------------------
# BitLinearTriton module
# ---------------------------------------------------------------------------
class BitLinearTriton(nn.Module):
    """INT8 tensor core BitLinear — drop-in replacement for BitLinear"""

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        # weight/bias를 x와 같은 device로 보정 (grad checkpointing 호환)
        weight = self.weight
        bias = self.bias
        if weight.device != x_norm.device:
            weight = weight.to(x_norm.device)
        if bias is not None and bias.device != x_norm.device:
            bias = bias.to(x_norm.device)
        return _BitLinearInt8Fn.apply(x_norm, weight, bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, quant=1.58bit+int8"


def replace_bitlinear_with_triton(model: nn.Module) -> nn.Module:
    """모델 내 모든 BitLinear를 BitLinearTriton으로 교체."""
    from model.bitlinear import BitLinear
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            replacements.append(name)
    for name in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old = getattr(parent, parts[-1])
        new = BitLinearTriton(old.in_features, old.out_features, bias=old.bias is not None)
        new.weight.data.copy_(old.weight.data)
        if old.bias is not None:
            new.bias.data.copy_(old.bias.data)
        new.norm.load_state_dict(old.norm.state_dict())
        setattr(parent, parts[-1], new)
    print(f"[BitLinearTriton] Replaced {len(replacements)} BitLinear layers")
    return model
