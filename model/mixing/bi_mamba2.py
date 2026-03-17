"""BiMamba2 Mixing Layer — 양방향 Mamba-2 SSD (Structured State Space Duality)

GPU 학습: mamba_ssm.Mamba2 fused CUDA kernel (chunk-parallel SSD)
CPU 추론: Python sequential scan fallback

Mamba-1 대비:
  - GPU: chunk-parallel → Tensor Core 활용 → ~2x 빠름
  - CPU: 스칼라 decay (exp 불필요) + headdim 벡터화 → 멀티스레드 개선
  - 프로젝션: x_proj+dt_proj 제거 → in_proj 하나로 통합
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# mamba_ssm.Mamba2 CUDA 커널 감지 (CUDA 필수)
_MAMBA2_CUDA = False
try:
    import torch as _torch_check
    if _torch_check.cuda.is_available():
        from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2Module
        _MAMBA2_CUDA = True
except ImportError:
    pass


@torch.compiler.disable
def _mamba2_cuda_wrapper(mamba2_module, x, seq_idx):
    """mamba_ssm.Mamba2 래퍼 — torch.compile 충돌 방지"""
    return mamba2_module(x, seq_idx=seq_idx)


class Mamba2Block(nn.Module):
    """Mamba-2 SSD 단방향 블록

    GPU: mamba_ssm.Mamba2 fused CUDA kernel (chunk-parallel SSD)
    CPU: Python sequential scan fallback

    파라미터 구조 (d=640, expand=2, d_state=64, headdim=64, ngroups=1):
      in_proj:  (2708, 640) — x(1280) + z(1280) + B(64) + C(64) + dt(20)
      conv1d:   (1408, 1, 4) — depthwise on [x, B, C]
      norm:     (1280,) — RMSNorm
      out_proj: (640, 1280)
      A_log, D, dt_bias: (20,) each
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, ngroups: int = 1,
                 chunk_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % headdim == 0, \
            f"d_inner({self.d_inner})은 headdim({headdim})으로 나누어떨어져야 함"

        if _MAMBA2_CUDA:
            self.mamba2 = _Mamba2Module(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                chunk_size=chunk_size,
                rmsnorm=True,
                bias=False,
            )
        else:
            # Python fallback — 수동 프로젝션 + sequential scan
            d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
            d_conv_in = self.d_inner + 2 * ngroups * d_state

            self.in_proj = BitLinear(d_model, d_in_proj)
            self.conv1d = nn.Conv1d(
                d_conv_in, d_conv_in,
                kernel_size=d_conv, padding=d_conv - 1,
                groups=d_conv_in, bias=True,
            )
            self.out_proj = BitLinear(self.d_inner, d_model)
            self.norm = nn.RMSNorm(self.d_inner)

            # SSM 파라미터 (head당 스칼라)
            self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
            A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        if _MAMBA2_CUDA and x.is_cuda:
            seq_idx = None
            if reset_mask is not None:
                seq_idx = (reset_mask.int().cumsum(dim=1) - 1).to(torch.int32)
            return _mamba2_cuda_wrapper(self.mamba2, x, seq_idx)
        else:
            return self._forward_fallback(x, reset_mask)

    def _forward_fallback(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """Python sequential scan fallback (CPU / non-CUDA)"""
        B_batch, T, _ = x.shape
        di = self.d_inner
        ds = self.d_state
        nh = self.nheads
        hd = self.headdim
        ng = self.ngroups

        # in_proj: d → (2*di + 2*ng*ds + nh)
        proj = self.in_proj(x)

        # Split: x_branch, z, B, C, dt
        x_branch = proj[:, :, :di]
        z = proj[:, :, di:2*di]
        B_ssm = proj[:, :, 2*di:2*di + ng*ds]
        C_ssm = proj[:, :, 2*di + ng*ds:2*di + 2*ng*ds]
        dt_raw = proj[:, :, 2*di + 2*ng*ds:]  # (B, T, nh)

        # conv1d on [x_branch, B, C]
        xBC = torch.cat([x_branch, B_ssm, C_ssm], dim=-1)  # (B, T, di+2*ng*ds)
        xBC = self.conv1d(xBC.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Split back after conv
        x_conv = F.silu(xBC[:, :, :di])
        B_conv = xBC[:, :, di:di + ng*ds]
        C_conv = xBC[:, :, di + ng*ds:]

        # dt → decay
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, nh)
        A = -torch.exp(self.A_log.float())  # (nh,) — negative
        decay = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # (B, T, nh)

        # Sequential scan
        # x_conv: (B, T, di) → (B, T, nh, hd)
        x_heads = x_conv.view(B_batch, T, nh, hd)
        # B_conv: (B, T, ng*ds) → (B, T, ng, ds)
        B_heads = B_conv.view(B_batch, T, ng, ds)
        # C_conv: (B, T, ng*ds) → (B, T, ng, ds)
        C_heads = C_conv.view(B_batch, T, ng, ds)

        # 각 그룹의 헤드 수
        heads_per_group = nh // ng

        y = self._scan_sequential(x_heads, B_heads, C_heads, decay, reset_mask,
                                  heads_per_group)

        # Skip connection: D * x
        D_expanded = self.D.view(1, 1, nh, 1).expand_as(y)
        y = y + D_expanded * x_heads

        # Reshape back + SiLU gating
        y = y.reshape(B_batch, T, di)
        y = self.norm(y)
        y = y * F.silu(z)

        return self.out_proj(y)

    def _scan_sequential(self, x, B, C, decay, reset_mask, heads_per_group):
        """Mamba-2 SSD sequential scan (in-place 연산 회피 — autograd 호환)

        h[n,d] = decay[h] * h[n,d] + B[g,n] * x[h,d]
        y[h,d] = Σ_n C[g,n] * h[n,d]

        Args:
            x: (B, T, nheads, headdim)
            B: (B, T, ngroups, d_state)
            C: (B, T, ngroups, d_state)
            decay: (B, T, nheads)
            reset_mask: (B, T) bool or None
            heads_per_group: int
        Returns:
            y: (B, T, nheads, headdim)
        """
        batch, T, nh, hd = x.shape
        ds = self.d_state

        # state: (B, nheads, d_state, headdim)
        h = x.new_zeros(batch, nh, ds, hd)
        ys = []

        for t in range(T):
            # 리셋
            if reset_mask is not None:
                rst = reset_mask[:, t].view(batch, 1, 1, 1).float()
                h = h * (1 - rst)

            a_t = decay[:, t, :].view(batch, nh, 1, 1)  # (B, nh, 1, 1)
            b_t = B[:, t, :]  # (B, ngroups, ds)
            c_t = C[:, t, :]  # (B, ngroups, ds)
            x_t = x[:, t, :]  # (B, nh, hd)

            # 그룹별 B, C를 헤드별로 확장
            # (B, ngroups, ds) → (B, nh, ds)
            b_expanded = b_t.repeat_interleave(heads_per_group, dim=1)  # (B, nh, ds)
            c_expanded = c_t.repeat_interleave(heads_per_group, dim=1)  # (B, nh, ds)

            # state update: h_new = a * h + outer(b, x)
            # outer(b, x): (B, nh, ds, 1) * (B, nh, 1, hd) → (B, nh, ds, hd)
            h = a_t * h + b_expanded.unsqueeze(-1) * x_t.unsqueeze(2)

            # output: y = einsum(C @ h) → (B, nh, hd)
            # (B, nh, ds, 1) * (B, nh, ds, hd) → sum over ds → (B, nh, hd)
            y_t = (c_expanded.unsqueeze(-1) * h).sum(dim=2)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, nh, hd)


class BiMamba2Mixing(MixingLayer):
    """양방향 Mamba-2 — forward + backward addition"""

    def __init__(self, cfg):
        super().__init__()
        ds = getattr(cfg, 'mamba2_d_state', 64)
        hd = getattr(cfg, 'mamba2_headdim', 64)
        ng = getattr(cfg, 'mamba2_ngroups', 1)
        cs = getattr(cfg, 'mamba2_chunk_size', 256)
        expand = getattr(cfg, 'mamba_expand', 2)
        d_conv = getattr(cfg, 'mamba_d_conv', 4)

        self.fwd = Mamba2Block(cfg.d_model, d_state=ds, d_conv=d_conv,
                               expand=expand, headdim=hd, ngroups=ng,
                               chunk_size=cs)
        self.bwd = Mamba2Block(cfg.d_model, d_state=ds, d_conv=d_conv,
                               expand=expand, headdim=hd, ngroups=ng,
                               chunk_size=cs)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        bwd_reset = reset_mask.flip(1) if reset_mask is not None else None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
