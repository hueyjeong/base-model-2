"""BiMLSTM Mixing Layer — 양방향 mLSTM (Matrix LSTM)

xLSTM (Beck et al., NeurIPS 2024) 의 mLSTM 변형.
Matrix memory cell: C = f*C + i*outer(k,v), h = C@q / max(n@q, 1)

GPU 학습: fla fused_recurrent_delta_rule 또는 parallel einsum
CPU 추론: sequential scan (C 커널)

State: n_heads × headdim × headdim
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# fla delta_rule 커널 감지 (mLSTM ≈ delta rule with gating)
_FLA_DELTA = False
_fused_recurrent_delta_rule = None
try:
    from fla.ops.delta_rule import fused_recurrent_delta_rule
    _FLA_DELTA = True
    _fused_recurrent_delta_rule = fused_recurrent_delta_rule
except ImportError:
    pass


@torch.compiler.disable
def _fla_delta_wrapper(q, k, v, beta):
    return _fused_recurrent_delta_rule(q, k, v, beta)


class MLSTMScan(nn.Module):
    """단방향 mLSTM

    GPU: parallel einsum (학습 시 O(T²) but GPU-friendly)
    CPU: sequential scan (C 커널)
    """

    def __init__(self, d_model: int, n_heads: int, headdim: int):
        super().__init__()
        self.n_heads = n_heads
        self.headdim = headdim

        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)
        # Gating: i (input), f (forget)
        self.i_proj = nn.Linear(d_model, n_heads, bias=False)  # per-head scalar gate
        self.f_proj = nn.Linear(d_model, n_heads, bias=False)  # per-head scalar gate

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.headdim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        # Per-head scalar gates
        i_gate = self.i_proj(x)  # (B, T, H)
        f_gate = self.f_proj(x)  # (B, T, H)

        if reverse:
            q, k, v = q.flip(1), k.flip(1), v.flip(1)
            i_gate, f_gate = i_gate.flip(1), f_gate.flip(1)

        # Gate activations
        f_act = torch.sigmoid(f_gate)  # (B, T, H)
        i_act = torch.exp(i_gate.clamp(-10, 10))  # (B, T, H)

        # Parallel mode: causal attention with gating
        # 간소화: f를 cumulative product로 → decay mask
        # log_f_cumsum[t] = sum(log(f[1..t]))
        log_f = torch.log(f_act.clamp(min=1e-6))  # (B, T, H)
        log_f_cumsum = log_f.cumsum(dim=1)  # (B, T, H)

        # decay[t, s] = exp(log_f_cumsum[t] - log_f_cumsum[s]) * i[s]
        # 어텐션 스코어: q[t] @ k[s] * decay[t,s]
        # (B, T, H, D) @ (B, T, H, D)^T → (B, H, T, T)
        qk = torch.einsum("bthd,bshd->bhts", q, k)  # (B, H, T, T)

        # Decay mask: exp(cumsum_f[t] - cumsum_f[s]) for t >= s
        log_decay = log_f_cumsum.transpose(1, 2).unsqueeze(-1) - log_f_cumsum.transpose(1, 2).unsqueeze(-2)
        # (B, H, T, 1) - (B, H, 1, T) = (B, H, T, T)

        # Causal mask
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(0).T  # lower triangular
        log_decay = log_decay.masked_fill(~causal, -1e9)

        # i_act scaling: decay * i[s]
        i_scale = i_act.transpose(1, 2).unsqueeze(-2)  # (B, H, 1, T)
        attn = qk * torch.exp(log_decay) * i_scale
        attn = attn.masked_fill(~causal, 0.0)

        # Normalize
        denom = attn.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        attn = attn / denom

        out = torch.einsum("bhts,bshd->bthd", attn, v)

        if reverse:
            out = out.flip(1)

        out = out.reshape(B, T, H * D)
        return self.o_proj(out)


class BiMLSTMMixing(MixingLayer):
    """양방향 mLSTM — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        self.fwd = MLSTMScan(cfg.d_model, cfg.n_heads, cfg.headdim)
        self.bwd = MLSTMScan(cfg.d_model, cfg.n_heads, cfg.headdim)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None, reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reverse=False)
        bwd_out = self.bwd(x, reverse=True)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
