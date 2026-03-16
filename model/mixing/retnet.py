"""BiRetention Mixing Layer — 양방향 Retention

RetNet (Sun et al., 2023) 의 multi-scale retention을 양방향으로 확장.
학습: parallel 모드 (retention matrix with exponential decay mask)
추론(CPU): recurrent 모드 (state = γ·state + k⊗v, out = state @ q)

State: n_heads × headdim × headdim = 8×32×32 = 8KB → L1 캐시 적중
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear


class RetentionScan(nn.Module):
    """단방향 multi-scale retention

    per-head 고정 decay gamma로 exponential decay 적용.
    학습 시 parallel 모드 (O(T²) but GPU-friendly matmul),
    추론 시 recurrent 모드 (O(1) per token).
    """

    def __init__(self, d_model: int, n_heads: int, headdim: int,
                 gamma_min: float = 0.8, gamma_max: float = 0.999):
        super().__init__()
        self.n_heads = n_heads
        self.headdim = headdim

        # 프로젝션 (ternary weight)
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

        # Gating (SiLU, non-ternary for precision)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head decay gamma (고정, 학습 불가)
        gammas = torch.linspace(gamma_min, gamma_max, n_heads)
        self.register_buffer("gammas", gammas)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.headdim

        q = self.q_proj(x).view(B, T, H, D)  # (B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)
        g = F.silu(self.g_proj(x)).view(B, T, H, D)

        # Parallel retention (학습 모드)
        # decay[i,j] = gamma^(i-j) for i >= j, 0 otherwise
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        # (T, T) lower-triangular decay matrix per head
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        # gammas: (H,) → (1, H, 1, 1)
        decay = self.gammas.view(1, H, 1, 1) ** diff.clamp(min=0).unsqueeze(0).unsqueeze(0)
        # 상삼각 마스킹 (causal)
        causal_mask = (diff >= 0).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        decay = decay * causal_mask

        # Q @ K^T with decay → attention-like
        # (B, H, T, D) @ (B, H, D, T) → (B, H, T, T)
        qk = torch.einsum("bthd,bshd->bhts", q, k)
        qk = qk * decay

        # 정규화
        qk = qk / (qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0))

        # (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        out = torch.einsum("bhts,bshd->bthd", qk, v)

        # Gating
        out = out * g

        # Reshape & output projection
        out = out.reshape(B, T, H * D)
        out = self.o_proj(out)

        return out


class BiRetentionMixing(MixingLayer):
    """양방향 Retention — forward + backward addition

    두 독립적인 RetentionScan을 사용하여 양방향 문맥 포착.
    """

    def __init__(self, cfg):
        super().__init__()
        self.fwd = RetentionScan(
            cfg.d_model, cfg.n_heads, cfg.headdim,
            cfg.retnet_gamma_min, cfg.retnet_gamma_max,
        )
        self.bwd = RetentionScan(
            cfg.d_model, cfg.n_heads, cfg.headdim,
            cfg.retnet_gamma_min, cfg.retnet_gamma_max,
        )

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        out = fwd_out + bwd_out

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
