"""BiMamba Mixing Layer — 양방향 Mamba-1 Selective Scan

Mamba (Gu & Dao, 2023) — selective state space model.
GPU 학습: mamba_ssm CUDA 커널 자동 감지 → Python fallback
CPU 추론: recurrent scan (O(T × d_inner × d_state))

State: d_inner × d_state = 512×16 = 32KB → L1/L2 캐시 경계
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear


class MambaBlock(nn.Module):
    """Mamba-1 단방향 블록

    구조: in_proj → conv1d → SSM scan → out_proj
    in_proj: d → 2*d_inner (x_branch + z_branch)
    SSM: h = exp(A·Δ)·h + Δ·B·x, y = C·h + D·x
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(d_model // 16, 1)

        # in_proj: d → 2*d_inner (x + z branch)
        self.in_proj = BitLinear(d_model, 2 * self.d_inner)

        # conv1d (작은 커널, float)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM 파라미터 프로젝션
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: (d_inner, d_state) — log-space, negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # out_proj
        self.out_proj = BitLinear(self.d_inner, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        # in_proj
        xz = self.in_proj(x)
        x_branch, z = xz.split(self.d_inner, dim=-1)

        # conv1d
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        # SSM 파라미터
        x_ssm = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        B_ssm = B_ssm  # (B, T, d_state)
        C_ssm = C_ssm  # (B, T, d_state)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Sequential scan (CPU/학습 호환)
        y = self._scan(x_conv, dt, A, B_ssm, C_ssm)

        # skip + gate
        y = y + self.D * x_conv
        y = y * F.silu(z)

        # out_proj
        return self.out_proj(y)

    def _scan(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor) -> Tensor:
        """Sequential selective scan

        h[t] = exp(A * dt[t]) * h[t-1] + dt[t] * B[t] * x[t]
        y[t] = C[t] @ h[t]
        """
        batch, T, d_inner = x.shape
        d_state = self.d_state

        h = x.new_zeros(batch, d_inner, d_state)
        ys = []

        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            x_t = x[:, t, :].unsqueeze(-1)    # (B, d_inner, 1)
            B_t = B[:, t, :].unsqueeze(1)      # (B, 1, d_state)
            C_t = C[:, t, :].unsqueeze(1)      # (B, 1, d_state)

            # discretize
            dA = torch.exp(A.unsqueeze(0) * dt_t)  # (B, d_inner, d_state)
            dB = dt_t * B_t * x_t  # (B, d_inner, d_state)

            h = dA * h + dB
            y_t = (C_t * h).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, d_inner)


class BiMambaMixing(MixingLayer):
    """양방향 Mamba — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        self.fwd = MambaBlock(
            cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv, cfg.mamba_expand,
        )
        self.bwd = MambaBlock(
            cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv, cfg.mamba_expand,
        )

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        out = fwd_out + bwd_out

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
