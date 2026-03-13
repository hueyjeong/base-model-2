"""공통 빌딩 블록 — RMSNorm, BitNetFFN, BatchedBitNetFFN

BitEditor의 여러 모듈에서 재사용되는 기본 구성 요소.
Triton fused 커널 자동 감지 → 폴백: 순수 PyTorch.
"""
import torch
import torch.nn as nn

from model.bitlinear import BitLinear, BatchedBitLinear

# Triton fused 커널 감지
_TRITON_KERNELS = False
_fused_rms_norm = None
_fused_sigmoid_mul = None

try:
    from model.triton_kernels import fused_rms_norm as _fused_rms_norm_fn
    from model.triton_kernels import fused_sigmoid_mul as _fused_sigmoid_mul_fn
    _TRITON_KERNELS = True
    _fused_rms_norm = _fused_rms_norm_fn
    _fused_sigmoid_mul = _fused_sigmoid_mul_fn
except ImportError:
    pass


@torch.compiler.disable
def _triton_rms_norm_wrapper(x, weight, eps):
    return _fused_rms_norm(x, weight, eps)


@torch.compiler.disable
def _triton_sigmoid_mul_wrapper(gate, up):
    return _fused_sigmoid_mul(gate, up)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    CUDA 사용 시 Triton fused 커널 자동 적용.
    bf16 안전성:
      - bf16 exponent = 8bit (f32과 동일) → overflow/underflow 없음
      - rsqrt(eps) = rsqrt(1e-6) = 1000 → bf16 범위(~65504) 이내
      - PAD 0-vector: rsqrt(0+eps) = 1000, weight*0*1000=0 → 안전
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _TRITON_KERNELS and x.is_cuda:
            return _triton_rms_norm_wrapper(x, self.weight, self.eps)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight.to(x.dtype)


class BitNetFFN(nn.Module):
    """BitNet Feed-Forward Network (SwiGLU)

    x → [BitLinear(gate), BitLinear(up)] → sigmoid(gate)*up → BitLinear(down)
    CUDA 사용 시 sigmoid*mul을 Triton fused 커널로 처리.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)
        self.up_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.gate_proj(x)
        up = self.up_proj(x)
        if _TRITON_KERNELS and x.is_cuda:
            x = _triton_sigmoid_mul_wrapper(gate_out, up)
        else:
            x = torch.sigmoid(gate_out) * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class BatchedBitNetFFN(nn.Module):
    """Batched BitNet FFN: E개 expert를 단일 batched 연산으로 처리

    16회 sequential expert forward → 1회 batched bmm.
    MoE 커널 호출 수: ~288/layer → ~18/layer.
    수학적으로 E개 독립 BitNetFFN과 동일.
    """

    def __init__(self, n_experts: int, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = BatchedBitLinear(n_experts, d_model, d_ff)
        self.up_proj = BatchedBitLinear(n_experts, d_model, d_ff)
        self.down_proj = BatchedBitLinear(n_experts, d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (E, capacity, d_model)
        Returns:
            (E, capacity, d_model)
        """
        gate = self.gate_proj(x)   # (E, C, d_ff)
        up = self.up_proj(x)       # (E, C, d_ff)
        # Triton fused sigmoid_mul은 2D 전용이므로 reshape 후 적용
        if _TRITON_KERNELS and x.is_cuda:
            E, C, F = gate.shape
            x = _triton_sigmoid_mul_wrapper(
                gate.reshape(E * C, F), up.reshape(E * C, F),
            ).reshape(E, C, F)
        else:
            x = torch.sigmoid(gate) * up
        x = self.dropout(x)
        x = self.down_proj(x)      # (E, C, d_model)
        return x
