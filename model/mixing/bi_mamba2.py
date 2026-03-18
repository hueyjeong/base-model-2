"""BiMamba2 Mixing Layer — 양방향 Mamba-2 SSD (Structured State Space Duality)

GPU 학습: nn.Linear proj + mamba_ssm fused CUDA kernel (mamba_split_conv1d_scan_combined)
         + proximity regularization으로 ternary 근접 가중치 학습
CPU 추론: TernaryLinear proj + Python sequential scan fallback

in_proj/out_proj는 nn.Linear로 유지 — fused kernel과 호환 + 최대 학습 속도.
export 시 ternary quantization 적용 (proximity loss로 양자화 오차 최소화).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer

# CUDA ops 감지
_MAMBA2_CUDA_OPS = False
_mamba_split_conv1d_scan_combined = None

try:
    import torch as _torch_check
    if _torch_check.cuda.is_available():
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined as _mscsc
        _mamba_split_conv1d_scan_combined = _mscsc
        _MAMBA2_CUDA_OPS = True
except ImportError:
    pass


class Mamba2Block(nn.Module):
    """Mamba-2 SSD 단방향 블록

    GPU: nn.Linear proj + mamba_ssm fused kernel (conv+scan+norm+gate+outproj)
    CPU: nn.Linear proj + Python sequential scan fallback

    파라미터 구조 (d=640, expand=2, d_state=64, headdim=64, ngroups=1):
      in_proj:  (2708, 640) — nn.Linear — z(1280) + xBC(1408) + dt(20)
      conv1d:   (1408, 1, 4) — depthwise on [x, B, C]
      norm:     (1280,) — RMSNorm weight (fused kernel 내부에서 사용)
      out_proj: (640, 1280) — nn.Linear
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

        # 프로젝션 차원 (mamba_ssm 순서: z, xBC, dt)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        d_conv_in = self.d_inner + 2 * ngroups * d_state

        # nn.Linear projection — fused kernel 호환, 최대 학습 속도
        # export 시 ternary quantization 적용 (proximity loss로 양자화 오차 최소화)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            d_conv_in, d_conv_in,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_conv_in, bias=True,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # RMSNorm weight — GPU에서는 fused kernel 내부에서 사용, CPU에서는 수동 적용
        self.norm = nn.RMSNorm(self.d_inner)

        # SSM 파라미터 (head당 스칼라)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        if _MAMBA2_CUDA_OPS and x.is_cuda:
            return self._forward_cuda(x, reset_mask)
        else:
            return self._forward_fallback(x, reset_mask)

    def _forward_cuda(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """nn.Linear in_proj → mamba_split_conv1d_scan_combined fused kernel"""
        # seq_idx for document isolation
        seq_idx = None
        if reset_mask is not None:
            seq_idx = (reset_mask.int().cumsum(dim=1) - 1).to(torch.int32)

        # 1) nn.Linear in_proj
        zxbcdt = self.in_proj(x)  # (B, T, d_in_proj)

        A = -torch.exp(self.A_log.float())

        # 2) Fused: conv1d + chunk_scan + RMSNorm + gate + out_proj
        y = _mamba_split_conv1d_scan_combined(
            zxbcdt,
            self.conv1d.weight.squeeze(1),  # (d_conv_in, d_conv)
            self.conv1d.bias,
            self.dt_bias,
            A,
            self.D,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=1e-5,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=False,
        )

        return y

    def _forward_fallback(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        """Python sequential scan fallback (CPU / non-CUDA)"""
        B_batch, T, _ = x.shape
        di = self.d_inner
        ds = self.d_state
        nh = self.nheads
        hd = self.headdim
        ng = self.ngroups
        d_conv_in = di + 2 * ng * ds

        # in_proj — mamba_ssm 순서: z, xBC, dt
        proj = self.in_proj(x)

        z = proj[:, :, :di]
        xBC_raw = proj[:, :, di:di + d_conv_in]
        dt_raw = proj[:, :, di + d_conv_in:]  # (B, T, nh)

        # conv1d on [x, B, C]
        xBC = self.conv1d(xBC_raw.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Split + SiLU on x (B, C는 raw)
        x_conv = F.silu(xBC[:, :, :di])
        B_conv = xBC[:, :, di:di + ng * ds]
        C_conv = xBC[:, :, di + ng * ds:]

        # dt → decay
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, nh)
        A = -torch.exp(self.A_log.float())  # (nh,) — negative
        decay = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # (B, T, nh)

        # Sequential scan
        x_heads = x_conv.view(B_batch, T, nh, hd)
        B_heads = B_conv.view(B_batch, T, ng, ds)
        C_heads = C_conv.view(B_batch, T, ng, ds)

        heads_per_group = nh // ng
        y = self._scan_sequential(x_heads, B_heads, C_heads, decay, reset_mask,
                                  heads_per_group)

        # Skip connection: D * x
        D_expanded = self.D.view(1, 1, nh, 1).expand_as(y)
        y = y + D_expanded * x_heads

        # Reshape + gate→norm (norm_before_gate=False)
        y = y.reshape(B_batch, T, di)
        y = y * F.silu(z)
        y = self.norm(y)

        return self.out_proj(y)

    def _scan_sequential(self, x, B, C, decay, reset_mask, heads_per_group):
        """Mamba-2 SSD sequential scan (in-place 연산 회피 — autograd 호환)"""
        batch, T, nh, hd = x.shape
        ds = self.d_state

        h = x.new_zeros(batch, nh, ds, hd)
        ys = []

        for t in range(T):
            if reset_mask is not None:
                rst = reset_mask[:, t].view(batch, 1, 1, 1).float()
                h = h * (1 - rst)

            a_t = decay[:, t, :].view(batch, nh, 1, 1)
            b_t = B[:, t, :]
            c_t = C[:, t, :]
            x_t = x[:, t, :]

            b_expanded = b_t.repeat_interleave(heads_per_group, dim=1)
            c_expanded = c_t.repeat_interleave(heads_per_group, dim=1)

            h = a_t * h + b_expanded.unsqueeze(-1) * x_t.unsqueeze(2)
            y_t = (c_expanded.unsqueeze(-1) * h).sum(dim=2)
            ys.append(y_t)

        return torch.stack(ys, dim=1)


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
        if reset_mask is not None:
            bwd_reset = reset_mask.flip(1).clone()
            bwd_reset[:, 0] = True  # flipped 시퀀스 시작에 BOS 보장 (seq_idx >= 0)
        else:
            bwd_reset = None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
