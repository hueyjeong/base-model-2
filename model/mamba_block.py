"""Mamba SSM Block — 순수 PyTorch 구현

Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Selective scan (S6) mechanism
- 1D causal convolution
- Input-dependent Δ, B, C parameters
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Input projection: d_model → 2*d_inner (x', z 분리)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal: 왼쪽 패딩
            groups=d_inner,      # depthwise
            bias=True,
        )

        # SSM 파라미터 projection
        # x' → Δ(dt_rank), B(d_state), C(d_state)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        # Δ projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM 파라미터 A, D
        # A: (d_inner, d_state) — 로그 스케일로 초기화
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
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

        # 3. SSM 파라미터 계산
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ: dt_rank → d_inner, softplus 활성화
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)

        # 4. Selective scan
        y = self._selective_scan(x_branch, dt, B, C)

        # 5. Gated output
        y = y * F.silu(z)

        # 6. Output projection
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,   # (B, L, d_inner)
        dt: torch.Tensor,   # (B, L, d_inner)
        B: torch.Tensor,    # (B, L, d_state)
        C: torch.Tensor,    # (B, L, d_state)
    ) -> torch.Tensor:
        """Selective Scan (S6) — 입력 의존적 SSM 재귀

        이산화:
            Ā = exp(A · Δ)
            B̄ = Δ · B (simplified Euler)

        재귀:
            h[t] = Ā[t] · h[t-1] + B̄[t] · x[t]
            y[t] = C[t] · h[t] + D · x[t]
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        # A: (d_inner, d_state)
        A = -torch.exp(self.A_log)  # 음수로 안정성 확보

        # 이산화  
        # Ā = exp(A · Δ): (B, L, d_inner, d_state)
        dA = torch.einsum("bld,dn->bldn", dt, A)
        dA = torch.exp(dA)

        # B̄ = Δ · B: (B, L, d_inner, d_state)
        dB = torch.einsum("bld,bln->bldn", dt, B)

        # x 확장: (B, L, d_inner, 1)
        x_expanded = x.unsqueeze(-1)

        # 재귀 스캔
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_expanded[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip connection with D
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, d_conv={self.d_conv}, "
            f"dt_rank={self.dt_rank}"
        )
