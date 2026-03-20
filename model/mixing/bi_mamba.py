"""BiMamba Mixing Layer — 양방향 Mamba-1 Selective Scan

GPU 학습: mamba_ssm CUDA 커널 (selective_scan_fn) 자동 감지
CPU 추론: Python sequential scan fallback
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# mamba_ssm CUDA 커널 감지
_MAMBA_CUDA = False
_selective_scan_fn = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_CUDA = True
    _selective_scan_fn = selective_scan_fn
except ImportError:
    pass


@torch.compiler.disable
def _selective_scan_wrapper(u, delta, A, B, C, D, z, delta_bias, delta_softplus):
    """mamba_ssm selective_scan_fn 래퍼 — torch.compile 충돌 방지"""
    return _selective_scan_fn(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=False,
    )


class MambaBlock(nn.Module):
    """Mamba-1 단방향 블록

    GPU: mamba_ssm selective_scan_fn (fused CUDA kernel)
    CPU: Python sequential scan fallback
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

        # conv1d
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

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        # in_proj
        xz = self.in_proj(x)
        x_branch, z = xz.split(self.d_inner, dim=-1)

        # BOS 리셋: x_branch와 z를 0으로 → state 자연소멸
        # (selective_scan에서 x=0이면 dB=dt*B*0=0 → state에 새 정보 안 들어감)
        # (dt를 크게 하면 exp(A*dt) ≈ 0 → 기존 state도 flush)
        if reset_mask is not None:
            rst = reset_mask.unsqueeze(-1).to(x_branch.dtype)  # (B, T, 1)
            x_branch = x_branch * (1 - rst)
            z = z * (1 - rst)

        # conv1d
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = F.silu(x_conv)

        # SSM 파라미터
        x_for_proj = x_conv.transpose(1, 2)
        x_ssm = self.x_proj(x_for_proj)
        dt, B_ssm, C_ssm = x_ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)  # (B, T, d_inner) — softplus는 selective_scan 내부에서

        # BOS에서 dt를 크게 → exp(A*dt) ≈ 0 → 기존 state 완전 flush
        if reset_mask is not None:
            rst = reset_mask.unsqueeze(-1).to(dt.dtype)
            dt = dt + rst * 1e4  # softplus(1e4) ≈ 1e4 → exp(-1e4) ≈ 0

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        if _MAMBA_CUDA and x.is_cuda:
            # selective_scan_fn은 모든 입력이 동일 dtype (float32) 필요
            dt_t = dt.float().transpose(1, 2).contiguous()
            B_t = B_ssm.float().transpose(1, 2).contiguous()
            C_t = C_ssm.float().transpose(1, 2).contiguous()
            z_t = z.float().transpose(1, 2).contiguous()

            y = _selective_scan_wrapper(
                x_conv.float().contiguous(), dt_t, A, B_t, C_t,
                D=self.D.float(), z=z_t,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y = y.to(x.dtype).transpose(1, 2)
        else:
            # Python fallback (CPU)
            x_f = x_conv.transpose(1, 2)  # (B, T, d_inner)
            dt_sp = F.softplus(dt)
            y = self._scan_sequential(x_f, dt_sp, A, B_ssm, C_ssm)
            y = y + self.D * x_f
            y = y * F.silu(z)

        return self.out_proj(y)

    def _scan_sequential(self, x, dt, A, B, C):
        """Python sequential scan fallback (CPU용)"""
        batch, T, d_inner = x.shape
        d_state = self.d_state
        h = x.new_zeros(batch, d_inner, d_state)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)
            x_t = x[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)
            dB = dt_t * B_t * x_t
            h = dA * h + dB
            y_t = (C_t * h).sum(dim=-1)
            ys.append(y_t)
        return torch.stack(ys, dim=1)


class BiMambaMixing(MixingLayer):
    """양방향 Mamba — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        self.fwd = MambaBlock(cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv, cfg.mamba_expand)
        self.bwd = MambaBlock(cfg.d_model, cfg.mamba_d_state, cfg.mamba_d_conv, cfg.mamba_expand)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        bwd_reset = reset_mask.flip(1) if reset_mask is not None else None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
