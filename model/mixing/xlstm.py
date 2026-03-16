"""BiSLSTM Mixing Layer — 양방향 sLSTM (Scalar LSTM)

xLSTM (Beck et al., NeurIPS 2024) 의 sLSTM 변형.
exponential gating으로 전통 LSTM의 gate saturation 문제 해결.

State per head: scalar c, n = 수 바이트 → 레지스터 적중
O(T × d_model) 복잡도, 가장 가벼운 recurrence.

CPU 최적: 4개 gate projection은 i8_sgemv, scan은 element-wise.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear


class SLSTMScan(nn.Module):
    """단방향 sLSTM

    Exponential input gate + sigmoid forget gate:
        i[t] = exp(i_gate[t] - m[t])  (stabilized)
        f[t] = sigmoid(f_gate[t])
        z[t] = tanh(z_gate[t])
        o[t] = sigmoid(o_gate[t])
        c[t] = f[t]*c[t-1] + i[t]*z[t]
        n[t] = f[t]*n[t-1] + i[t]
        h[t] = o[t] * c[t] / max(|n[t]|, 1)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 4개 gate projection (ternary)
        self.i_proj = BitLinear(d_model, d_model)
        self.f_proj = BitLinear(d_model, d_model)
        self.z_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        # Gate projections (한번에)
        i_gate = self.i_proj(x)  # (B, T, D)
        f_gate = self.f_proj(x)
        z_gate = self.z_proj(x)
        o_gate = self.o_proj(x)

        # Sequential scan
        c = x.new_zeros(B, D)  # cell state
        n = x.new_zeros(B, D)  # normalizer

        hs = []
        for t in range(T):
            f_t = torch.sigmoid(f_gate[:, t])
            # Exponential input gate with stabilization
            i_raw = i_gate[:, t]
            # log-space stabilization: m = max(f*prev_m, i_raw)
            # 간단한 구현: exp(i - max) 사용
            i_t = torch.exp(i_raw - i_raw.detach().clamp(min=-10, max=10))
            z_t = torch.tanh(z_gate[:, t])
            o_t = torch.sigmoid(o_gate[:, t])

            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            h_t = o_t * c / n.abs().clamp(min=1.0)
            hs.append(h_t)

        return torch.stack(hs, dim=1)


class BiSLSTMMixing(MixingLayer):
    """양방향 sLSTM — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        self.fwd = SLSTMScan(cfg.d_model)
        self.bwd = SLSTMScan(cfg.d_model)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        out = fwd_out + bwd_out

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
