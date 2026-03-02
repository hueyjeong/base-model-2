"""Mamba SSM Block — BitLinear + CUDA 가속

Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Selective scan (S6) mechanism
- 1D causal convolution
- Input-dependent Δ, B, C parameters
- BitLinear 1.58-bit 양자화 적용 (in_proj, x_proj, out_proj)
- mamba_ssm CUDA 커널 사용 가능 시 자동 활용
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bitlinear import BitLinear

# CUDA 가속 selective scan (설치 시 자동 사용)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_SELECTIVE_SCAN_CUDA = True
except ImportError:
    HAS_SELECTIVE_SCAN_CUDA = False


class MambaBlock(nn.Module):
    """Mamba SSM Block

    구조:
        x → Linear(d_model → d_inner*2) → split → [z, x']
        x' → Conv1D → SiLU → SSM(Δ, B, C) → y
        y * SiLU(z) → Linear(d_inner → d_model) → output

    SSM은 selective scan으로 구현:
        Δ = softplus(Linear(x'))
        B = Linear(x')
        C = Linear(x')
        h[t] = Ā·h[t-1] + B̄·x'[t]
        y[t] = C[t]·h[t]
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 48,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        # Input projection: d_model → 2*d_inner (x', z 분리) — BitLinear
        self.in_proj = BitLinear(d_model, d_inner * 2)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal: 왼쪽 패딩
            groups=d_inner,      # depthwise
            bias=True,
        )

        # SSM 파라미터 projection — BitLinear
        # x' → Δ(dt_rank), B(d_state), C(d_state)
        self.x_proj = BitLinear(d_inner, dt_rank + d_state * 2)

        # Δ projection: dt_rank → d_inner (일반 Linear 유지 — softplus 안정성)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM 파라미터 A, D
        # A: (d_inner, d_state) — 로그 스케일로 초기화
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: d_inner → d_model — BitLinear
        self.out_proj = BitLinear(d_inner, d_model)

    # Document isolation: BOS 위치에서 SSM state를 완전히 리셋하기 위한 dt 값.
    # A_log가 학습 중 -5까지 drift하는 최악의 경우에도
    # exp(-0.0067 × 1e4) = exp(-67) ≈ 0 보장.
    _RESET_DT = 1e4

    def forward(self, x: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            reset_mask: (batch, seq_len) bool — True인 위치에서 SSM state 리셋
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 1. Input projection + split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # 각 (B, L, d_inner)

        # 2. 1D causal convolution
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # causal: 미래 제거
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        # ★ Document isolation: BOS 위치에서 SSM 입력 제로링
        #   x_branch=0 → x_proj 출력(dt,B,C)≈0 → dB×x=0 (입력 기여 제거)
        #   z=0 → gated output에서 BOS 위치의 SSM 출력도 0
        if reset_mask is not None:
            keep = (~reset_mask).unsqueeze(-1).float()  # (B, L, 1)
            x_branch = x_branch * keep
            z = z * keep

        # 3. SSM 파라미터 계산
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ: dt_rank → d_inner, softplus 활성화
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        # Compute softplus in FP32 then clamp to prevent BF16 exponential explosions
        dt_f32 = dt.float()
        dt = F.softplus(dt_f32).clamp(min=1e-5).to(dt.dtype)

        # ★ Document isolation: BOS 위치에서 dt 극대화
        #   dA = exp(A × dt) where A<0 → exp(negative_large) ≈ 0
        #   → h[t] = 0 × h[t-1] + 0 = 0 (완벽한 state 리셋)
        if reset_mask is not None:
            dt = dt.masked_fill(reset_mask.unsqueeze(-1), self._RESET_DT)

        # 4. Selective scan (CUDA 가능 시 자동 사용)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state) FP32

        if HAS_SELECTIVE_SCAN_CUDA and x_branch.is_cuda:
            # dtype 통일 (AMP 환경에서 BitLinear=FP32, nn.Linear=FP16 혼재 방지)
            scan_dtype = x_branch.dtype
            u = x_branch.transpose(1, 2).contiguous()       # (B, d_inner, L)
            delta = dt.to(scan_dtype).transpose(1, 2).contiguous()  # (B, d_inner, L)
            B_t = B.to(scan_dtype).transpose(1, 2).contiguous()     # (B, d_state, L)
            C_t = C.to(scan_dtype).transpose(1, 2).contiguous()     # (B, d_state, L)

            y = selective_scan_fn(
                u, delta, A, B_t, C_t,
                self.D.float(),
                z=None,
                delta_softplus=False,  # 이미 softplus 적용함
                return_last_state=False,
            )
            y = y.transpose(1, 2)  # (B, L, d_inner)
        else:
            # Python fallback
            y = self._selective_scan(x_branch, dt, A, B, C)

        # 5. Gated output
        y = y * F.silu(z)

        # 6. Output projection
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,   # (B, L, d_inner)
        dt: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,    # (d_inner, d_state)
        B: torch.Tensor,    # (B, L, d_state)
        C: torch.Tensor,    # (B, L, d_state)
    ) -> torch.Tensor:
        """Selective Scan (S6) — Python fallback

        CUDA 커널 미설치 시 사용. 느리지만 정확.
        """
        batch, seq_len, d_inner = x.shape

        # 이산화 (Compute in FP32 to prevent exp() underflow to exactly 0.0 or Inf)
        orig_dtype = dt.dtype
        dA = torch.einsum("bld,dn->bldn", dt.float(), A.float())
        dA = torch.exp(dA).to(orig_dtype)
        dB = torch.einsum("bld,bln->bldn", dt, B)

        x_expanded = x.unsqueeze(-1)

        # 재귀 스캔
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_expanded[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, d_conv={self.d_conv}, "
            f"dt_rank={self.dt_rank}"
        )
