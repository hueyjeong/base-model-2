"""BiRetention Mixing Layer — 양방향 Retention

GPU 학습: fla fused_recurrent_retention Triton 커널
CPU 추론: Python sequential scan fallback

State: n_heads × headdim × headdim (L1 적중)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# fla Triton 커널 감지
_FLA_RETENTION = False
_fused_recurrent_retention = None
try:
    from fla.ops.retention import fused_recurrent_retention
    _FLA_RETENTION = True
    _fused_recurrent_retention = fused_recurrent_retention
except ImportError:
    pass


@torch.compiler.disable
def _fla_retention_wrapper(q, k, v, reverse=False):
    """fla fused_recurrent_retention 래퍼 — torch.compile 충돌 방지"""
    return _fused_recurrent_retention(q, k, v, reverse=reverse)


class RetentionScan(nn.Module):
    """단방향 multi-scale retention

    GPU: fla fused_recurrent_retention (Triton 커널)
    CPU: parallel mode (einsum) 또는 sequential scan
    """

    def __init__(self, d_model: int, n_heads: int, headdim: int,
                 gamma_min: float = 0.8, gamma_max: float = 0.999):
        super().__init__()
        self.n_heads = n_heads
        self.headdim = headdim

        # Fused q,k,v projection (양자화 1회)
        self.qkv_proj = BitLinear(d_model, 3 * d_model)
        self.o_proj = BitLinear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)

        gammas = torch.linspace(gamma_min, gamma_max, n_heads)
        self.register_buffer("gammas", gammas)

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.headdim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(H * D, dim=-1)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        g = F.silu(self.g_proj(x)).view(B, T, H, D)

        if _FLA_RETENTION and x.is_cuda:
            # fla 레이아웃: (B, H, T, D)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            # retention decay를 k에 포지셔널 스케일링으로 적용
            positions = torch.arange(T, device=x.device, dtype=q.dtype)
            log_gammas = torch.log(self.gammas.to(q.dtype))
            decay = (log_gammas.unsqueeze(1) * positions.unsqueeze(0)).exp()
            k = k * decay.unsqueeze(0).unsqueeze(-1)

            result = _fla_retention_wrapper(q, k, v, reverse=reverse)
            out = result[0] if isinstance(result, tuple) else result
            out = out.transpose(1, 2).contiguous()
        else:
            # CPU: parallel mode (einsum)
            if reverse:
                q, k, v = q.flip(1), k.flip(1), v.flip(1)

            positions = torch.arange(T, device=x.device, dtype=torch.float32)
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            decay = self.gammas.view(1, H, 1, 1) ** diff.clamp(min=0).unsqueeze(0).unsqueeze(0)
            causal_mask = (diff >= 0).unsqueeze(0).unsqueeze(0)
            decay = decay * causal_mask

            qk = torch.einsum("bthd,bshd->bhts", q, k) * decay
            qk = qk / qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
            out = torch.einsum("bhts,bshd->bthd", qk, v)

            if reverse:
                out = out.flip(1)

        out = out * g
        out = out.reshape(B, T, H * D)
        return self.o_proj(out)


class BiRetentionMixing(MixingLayer):
    """양방향 Retention — forward + backward addition"""

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
        fwd_out = self.fwd(x, reverse=False)
        bwd_out = self.bwd(x, reverse=True)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
