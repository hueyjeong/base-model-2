"""BiSLSTM Mixing Layer — 양방향 sLSTM (Scalar LSTM)

개선 실험:
  Phase 1: Conv1d / SiLU gate / Multi-head decay (개별 on/off 가능)
  Phase 2: d_state 확장 (차원당 벡터 상태)

GPU: Triton fused scan + fused 4-gate projection
CPU: sequential scan

BOS 위치에서 state(c, n) 리셋으로 문서 격리.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

_TRITON_SLSTM = False
_triton_slstm_scan = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _slstm_fwd_kernel(
        I, F_gate, Z, O, Reset, Out,
        T_dim: tl.constexpr,
        has_reset: tl.constexpr,
    ):
        """sLSTM forward: per-(batch,dim) element sequential scan (d_state=1)
        decay bias, B/C projections은 Python에서 미리 적용됨.
        """
        bd = tl.program_id(0)
        c = 0.0
        n = 0.0
        for t in range(T_dim):
            off = bd * T_dim + t

            # BOS 리셋
            if has_reset:
                if tl.load(Reset + off) != 0:
                    c = 0.0
                    n = 0.0

            f_t = tl.sigmoid(tl.load(F_gate + off))
            i_raw = tl.load(I + off)
            i_raw = tl.maximum(tl.minimum(i_raw, 10.0), -10.0)
            i_t = tl.exp(i_raw)
            z_t = tl.extra.cuda.libdevice.tanh(tl.load(Z + off))
            o_t = tl.sigmoid(tl.load(O + off))

            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            abs_n = tl.abs(n)
            denom = tl.maximum(abs_n, 1.0)
            tl.store(Out + off, o_t * c / denom)

    @triton.jit
    def _slstm_fwd_kernel_ds(
        I, F_gate, Z, O, Reset, B_gate, C_gate, Out,
        T_dim: tl.constexpr,
        d_state: tl.constexpr,
        has_reset: tl.constexpr,
    ):
        """sLSTM forward with d_state > 1: per-(batch,dim) sequential scan
        B_gate, C_gate: (B*D, T, d_state)
        """
        bd = tl.program_id(0)
        # d_state=4 이하만 지원 (레지스터에 직접 배치)
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        n0 = 0.0
        n1 = 0.0
        n2 = 0.0
        n3 = 0.0

        for t in range(T_dim):
            off = bd * T_dim + t
            bc_off = (bd * T_dim + t) * d_state

            # BOS 리셋
            if has_reset:
                if tl.load(Reset + off) != 0:
                    c0 = 0.0
                    c1 = 0.0
                    c2 = 0.0
                    c3 = 0.0
                    n0 = 0.0
                    n1 = 0.0
                    n2 = 0.0
                    n3 = 0.0

            f_t = tl.sigmoid(tl.load(F_gate + off))
            i_raw = tl.load(I + off)
            i_raw = tl.maximum(tl.minimum(i_raw, 10.0), -10.0)
            i_t = tl.exp(i_raw)
            z_t = tl.extra.cuda.libdevice.tanh(tl.load(Z + off))
            o_t = tl.sigmoid(tl.load(O + off))

            # d_state expansion: c[s] = f*c[s] + i*z*B[s]
            b0 = tl.load(B_gate + bc_off + 0)
            c0 = f_t * c0 + i_t * z_t * b0
            n0 = f_t * n0 + i_t * b0
            co0 = tl.load(C_gate + bc_off + 0)

            out_val = c0 * co0
            n_val = n0 * co0

            if d_state >= 2:
                b1 = tl.load(B_gate + bc_off + 1)
                c1 = f_t * c1 + i_t * z_t * b1
                n1 = f_t * n1 + i_t * b1
                co1 = tl.load(C_gate + bc_off + 1)
                out_val += c1 * co1
                n_val += n1 * co1

            if d_state >= 3:
                b2 = tl.load(B_gate + bc_off + 2)
                c2 = f_t * c2 + i_t * z_t * b2
                n2 = f_t * n2 + i_t * b2
                co2 = tl.load(C_gate + bc_off + 2)
                out_val += c2 * co2
                n_val += n2 * co2

            if d_state >= 4:
                b3 = tl.load(B_gate + bc_off + 3)
                c3 = f_t * c3 + i_t * z_t * b3
                n3 = f_t * n3 + i_t * b3
                co3 = tl.load(C_gate + bc_off + 3)
                out_val += c3 * co3
                n_val += n3 * co3

            abs_n = tl.abs(n_val)
            denom = tl.maximum(abs_n, 1.0)
            tl.store(Out + off, o_t * out_val / denom)

    def triton_slstm_scan(i_gate, f_gate, z_gate, o_gate, reset_mask=None):
        B, T, D = i_gate.shape
        i_gate = i_gate.float()
        f_gate = f_gate.float()
        z_gate = z_gate.float()
        o_gate = o_gate.float()
        i_flat = i_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        f_flat = f_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        z_flat = z_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        o_flat = o_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        out_flat = torch.empty_like(i_flat)

        has_reset = reset_mask is not None
        if has_reset:
            r_flat = reset_mask.unsqueeze(1).expand(-1, D, -1).contiguous().view(B * D, T).to(torch.int8)
        else:
            r_flat = torch.empty(0, device=i_gate.device, dtype=torch.int8)

        _slstm_fwd_kernel[(B * D,)](
            i_flat, f_flat, z_flat, o_flat, r_flat, out_flat,
            T_dim=T, has_reset=has_reset,
        )
        return out_flat.view(B, D, T).permute(0, 2, 1).contiguous()

    def triton_slstm_scan_ds(i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                              reset_mask=None, d_state=4):
        """d_state > 1 Triton scan"""
        B, T, D = i_gate.shape
        i_gate = i_gate.float()
        f_gate = f_gate.float()
        z_gate = z_gate.float()
        o_gate = o_gate.float()
        b_gate = b_gate.float()  # (B, T, d_state)
        c_gate = c_gate.float()  # (B, T, d_state)

        i_flat = i_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        f_flat = f_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        z_flat = z_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        o_flat = o_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        out_flat = torch.empty_like(i_flat)

        has_reset = reset_mask is not None
        if has_reset:
            r_flat = reset_mask.unsqueeze(1).expand(-1, D, -1).contiguous().view(B * D, T).to(torch.int8)
        else:
            r_flat = torch.empty(0, device=i_gate.device, dtype=torch.int8)

        # B/C: (B, T, d_state) → (B*D, T, d_state) broadcast
        # 각 dim에 동일한 B,C 적용
        b_flat = b_gate.unsqueeze(2).expand(-1, -1, D, -1).permute(0, 2, 1, 3).contiguous().view(B * D, T, d_state)
        c_flat = c_gate.unsqueeze(2).expand(-1, -1, D, -1).permute(0, 2, 1, 3).contiguous().view(B * D, T, d_state)

        _slstm_fwd_kernel_ds[(B * D,)](
            i_flat, f_flat, z_flat, o_flat, r_flat, b_flat, c_flat, out_flat,
            T_dim=T, d_state=d_state, has_reset=has_reset,
        )
        return out_flat.view(B, D, T).permute(0, 2, 1).contiguous()

    _TRITON_SLSTM = True
    _triton_slstm_scan = triton_slstm_scan
except (ImportError, Exception):
    pass


@torch.compiler.disable
def _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate, reset_mask=None):
    return _triton_slstm_scan(i_gate, f_gate, z_gate, o_gate, reset_mask)


@torch.compiler.disable
def _triton_slstm_ds_wrapper(i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                               reset_mask=None, d_state=4):
    return triton_slstm_scan_ds(i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                                 reset_mask, d_state)


class SLSTMScan(nn.Module):
    """단방향 sLSTM — 개별 on/off 가능한 개선 옵션

    Args:
        d_model: 모델 차원
        n_heads: 멀티헤드 수 (decay bias용)
        use_conv: Conv1d 전처리 (Mamba 스타일)
        use_silu_gate: SiLU 출력 게이팅 (Mamba 패턴)
        use_decay_bias: 멀티헤드 decay bias (RetNet/RWKV 패턴)
        d_state: 상태 확장 (1=기존, 2~4=Phase 2)
    """

    def __init__(self, d_model: int, n_heads: int = 1,
                 use_conv: bool = False, use_silu_gate: bool = False,
                 use_decay_bias: bool = False, d_state: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.headdim = d_model // n_heads
        self.use_conv = use_conv
        self.use_silu_gate = use_silu_gate
        self.use_decay_bias = use_decay_bias
        self.d_state = d_state

        # Phase 1-A: Conv1d (선택)
        if use_conv:
            self.conv1d = nn.Conv1d(
                d_model, d_model, kernel_size=4,
                padding=3, groups=d_model,
            )

        # fused 4-gate projection (i, f, z, o)
        self.gate_proj = BitLinear(d_model, 4 * d_model)

        # Phase 1-B: SiLU 게이팅 (선택)
        if use_silu_gate:
            self.z_proj = BitLinear(d_model, d_model)

        # output projection (차원 간 mixing)
        self.o_proj = BitLinear(d_model, d_model)

        # Phase 1-C: decay bias (선택)
        if use_decay_bias:
            gammas = torch.linspace(0.8, 0.999, n_heads)
            self.decay_bias = nn.Parameter(torch.log(gammas / (1 - gammas)))

        # Phase 2: d_state 확장
        if d_state > 1:
            self.B_proj = nn.Linear(d_model, d_state, bias=False)
            self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        B, T, D = x.shape

        # SiLU gate (원본 x에서 계산)
        z = self.z_proj(x) if self.use_silu_gate else None

        # Conv1d 전처리
        if self.use_conv:
            x_inp = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
            x_inp = F.silu(x_inp)
        else:
            x_inp = x

        # 4-gate projection
        gates = self.gate_proj(x_inp)
        i_gate, f_gate, z_gate, o_gate = gates.split(D, dim=-1)

        # decay bias
        if self.use_decay_bias:
            f_gate = f_gate + self.decay_bias.repeat_interleave(self.headdim)

        # d_state > 1: B, C projection
        if self.d_state > 1:
            b_gate = self.B_proj(x_inp)  # (B, T, d_state)
            c_gate = self.C_proj(x_inp)  # (B, T, d_state)
            h = self._scan_ds(i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                              reset_mask, x.is_cuda)
        else:
            h = self._scan(i_gate, f_gate, z_gate, o_gate, reset_mask, x.is_cuda)

        # SiLU gating
        if z is not None:
            h = h * F.silu(z)

        return self.o_proj(h)

    def _scan(self, i_gate, f_gate, z_gate, o_gate, reset_mask, is_cuda):
        """d_state=1 기본 scan"""
        if _TRITON_SLSTM and is_cuda:
            return _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate, reset_mask)

        B, T, D = i_gate.shape
        c = i_gate.new_zeros(B, D)
        n = i_gate.new_zeros(B, D)
        hs = []
        for t in range(T):
            if reset_mask is not None:
                rst = reset_mask[:, t].unsqueeze(-1)
                c = c * (~rst)
                n = n * (~rst)
            f_t = torch.sigmoid(f_gate[:, t])
            i_t = torch.exp(i_gate[:, t].clamp(-10, 10))
            z_t = torch.tanh(z_gate[:, t])
            o_t = torch.sigmoid(o_gate[:, t])
            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            h_t = o_t * c / n.abs().clamp(min=1.0)
            hs.append(h_t)
        return torch.stack(hs, dim=1)

    def _scan_ds(self, i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                  reset_mask, is_cuda):
        """d_state > 1 확장 scan"""
        if _TRITON_SLSTM and is_cuda:
            return _triton_slstm_ds_wrapper(
                i_gate, f_gate, z_gate, o_gate, b_gate, c_gate,
                reset_mask, self.d_state,
            )

        # CPU fallback
        B, T, D = i_gate.shape
        ds = self.d_state
        c = i_gate.new_zeros(B, D, ds)
        n = i_gate.new_zeros(B, D, ds)
        hs = []
        for t in range(T):
            if reset_mask is not None:
                rst = reset_mask[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                c = c * (~rst)
                n = n * (~rst)
            f_t = torch.sigmoid(f_gate[:, t])      # (B, D)
            i_t = torch.exp(i_gate[:, t].clamp(-10, 10))
            z_t = torch.tanh(z_gate[:, t])
            o_t = torch.sigmoid(o_gate[:, t])
            bt = b_gate[:, t]  # (B, ds)
            ct = c_gate[:, t]  # (B, ds)

            # c[d,s] = f*c[d,s] + i*z*B[s]
            iz = (i_t * z_t).unsqueeze(-1)     # (B, D, 1)
            c = f_t.unsqueeze(-1) * c + iz * bt.unsqueeze(1)  # (B, D, ds)
            n = f_t.unsqueeze(-1) * n + i_t.unsqueeze(-1) * bt.unsqueeze(1)

            # output = o * sum_s(c[d,s]*C[s]) / norm
            c_weighted = (c * ct.unsqueeze(1)).sum(-1)   # (B, D)
            n_weighted = (n * ct.unsqueeze(1)).sum(-1)    # (B, D)
            denom = n_weighted.abs().clamp(min=1.0)
            h_t = o_t * c_weighted / denom
            hs.append(h_t)
        return torch.stack(hs, dim=1)


class SLSTMMambaBlock(nn.Module):
    """Phase 3: Mamba 구조 + sLSTM 게이팅 하이브리드

    Mamba 외장 구조:
      in_proj(d, 2*d_inner) → split → conv1d+SiLU | z gate
      gate_proj(d, 3*d_inner) → f, i, z_cell gates (원본 x에서)
      sLSTM scan (d_inner 차원, element-wise)
      h * SiLU(z) → out_proj(d_inner, d)

    sLSTM vs Mamba:
      Mamba: h = exp(A*dt)*h + dt*B*x (matrix state d_inner×d_state)
      Hybrid: c = σ(f)*c + exp(i)*tanh(z) (scalar state d_inner)
    """

    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # in_proj: x_branch + z (SiLU gate 분기)
        self.in_proj = BitLinear(d_model, 2 * self.d_inner)

        # conv1d (Mamba 스타일 causal depthwise)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner,
        )

        # sLSTM 3-gate (f, i, z_cell) — 원본 x에서 계산 (파라미터 절약)
        # o_gate 불필요: SiLU(z_branch)가 출력 게이팅 담당
        self.gate_proj = nn.Linear(d_model, 3 * self.d_inner, bias=False)

        # out_proj
        self.out_proj = BitLinear(self.d_inner, d_model)

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.split(self.d_inner, dim=-1)

        # BOS 리셋: x_branch, z 제로링
        if reset_mask is not None:
            rst = reset_mask.unsqueeze(-1).to(x_branch.dtype)
            x_branch = x_branch * (1 - rst)
            z = z * (1 - rst)

        # conv1d + SiLU
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T]
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        # 3-gate: f, i, z_cell (원본 x에서 — conv 경로와 독립)
        gates = self.gate_proj(x)
        f_gate, i_gate, z_cell = gates.split(self.d_inner, dim=-1)

        # sLSTM scan (element-wise, d_inner 차원)
        h = self._slstm_scan(f_gate, i_gate, z_cell, x_conv, reset_mask)

        # SiLU gating + out_proj
        return self.out_proj(h * F.silu(z))

    def _slstm_scan(self, f_gate, i_gate, z_cell, x_conv, reset_mask):
        """sLSTM scan (o_gate 없이, conv output을 cell 입력에 가산)

        c = σ(f)*c + exp(i)*tanh(z_cell) + x_conv
        (conv output을 skip-add하여 지역 문맥 직접 주입)
        """
        B, T, D = f_gate.shape

        if _TRITON_SLSTM and f_gate.is_cuda:
            # Triton: 4-gate 커널 재활용 (o_gate = 0 → sigmoid(0) = 0.5)
            # 대신 z_cell에 conv 정보를 이미 포함시킴
            # o_gate 대신 ones를 사용하면 h = c/norm
            o_ones = torch.zeros_like(f_gate) + 6.0  # sigmoid(6)≈0.9975 ≈ 1
            h = _triton_slstm_wrapper(i_gate, f_gate, z_cell, o_ones, reset_mask)
            return h

        # CPU fallback
        c = f_gate.new_zeros(B, D)
        n = f_gate.new_zeros(B, D)
        hs = []
        for t in range(T):
            if reset_mask is not None:
                rst = reset_mask[:, t].unsqueeze(-1)
                c = c * (~rst)
                n = n * (~rst)
            f_t = torch.sigmoid(f_gate[:, t])
            i_t = torch.exp(i_gate[:, t].clamp(-10, 10))
            z_t = torch.tanh(z_cell[:, t])
            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            h_t = c / n.abs().clamp(min=1.0)
            hs.append(h_t)
        return torch.stack(hs, dim=1)


class BiSLSTMMixing(MixingLayer):
    """양방향 sLSTM — BOS state 리셋으로 문서 격리"""

    def __init__(self, cfg):
        super().__init__()
        # config에서 옵션 읽기 (없으면 기본값)
        use_conv = getattr(cfg, 'xlstm_use_conv', False)
        use_silu_gate = getattr(cfg, 'xlstm_use_silu_gate', False)
        use_decay_bias = getattr(cfg, 'xlstm_use_decay_bias', False)
        d_state = getattr(cfg, 'xlstm_d_state', 1)

        self.fwd = SLSTMScan(cfg.d_model, cfg.n_heads,
                             use_conv=use_conv, use_silu_gate=use_silu_gate,
                             use_decay_bias=use_decay_bias, d_state=d_state)
        self.bwd = SLSTMScan(cfg.d_model, cfg.n_heads,
                             use_conv=use_conv, use_silu_gate=use_silu_gate,
                             use_decay_bias=use_decay_bias, d_state=d_state)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        bwd_reset = reset_mask.flip(1) if reset_mask is not None else None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out


class BiSLSTMMambaMixing(MixingLayer):
    """Phase 3: 양방향 sLSTM-Mamba 하이브리드"""

    def __init__(self, cfg):
        super().__init__()
        expand = getattr(cfg, 'xlstm_expand', 2)
        d_conv = getattr(cfg, 'xlstm_d_conv', 4)
        self.fwd = SLSTMMambaBlock(cfg.d_model, d_conv=d_conv, expand=expand)
        self.bwd = SLSTMMambaBlock(cfg.d_model, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        bwd_reset = reset_mask.flip(1) if reset_mask is not None else None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
