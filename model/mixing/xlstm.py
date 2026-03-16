"""BiSLSTM Mixing Layer — 양방향 sLSTM (Scalar LSTM)

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
        """sLSTM forward: per-(batch,dim) element sequential scan
        Reset: (B*D, T) bool — True면 state 리셋 (BOS 위치)
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

        # reset_mask: (B, T) → (B*D, T) (각 dim에 동일 mask 복제)
        has_reset = reset_mask is not None
        if has_reset:
            # (B, T) → (B, 1, T) → (B, D, T) → (B*D, T)
            r_flat = reset_mask.unsqueeze(1).expand(-1, D, -1).contiguous().view(B * D, T).to(torch.int8)
        else:
            r_flat = torch.empty(0, device=i_gate.device, dtype=torch.int8)

        _slstm_fwd_kernel[(B * D,)](
            i_flat, f_flat, z_flat, o_flat, r_flat, out_flat,
            T_dim=T, has_reset=has_reset,
        )
        return out_flat.view(B, D, T).permute(0, 2, 1).contiguous()

    _TRITON_SLSTM = True
    _triton_slstm_scan = triton_slstm_scan
except (ImportError, Exception):
    pass


@torch.compiler.disable
def _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate, reset_mask=None):
    return _triton_slstm_scan(i_gate, f_gate, z_gate, o_gate, reset_mask)


class SLSTMScan(nn.Module):
    """단방향 sLSTM — fused 4-gate, BOS state 리셋"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gate_proj = BitLinear(d_model, 4 * d_model)

    def forward(self, x: Tensor, reset_mask: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        gates = self.gate_proj(x)
        i_gate, f_gate, z_gate, o_gate = gates.split(D, dim=-1)

        if _TRITON_SLSTM and x.is_cuda:
            return _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate, reset_mask)

        # CPU fallback
        c = x.new_zeros(B, D)
        n = x.new_zeros(B, D)
        hs = []
        for t in range(T):
            # BOS 리셋
            if reset_mask is not None:
                rst = reset_mask[:, t].unsqueeze(-1)  # (B, 1)
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


class BiSLSTMMixing(MixingLayer):
    """양방향 sLSTM — BOS state 리셋으로 문서 격리"""

    def __init__(self, cfg):
        super().__init__()
        self.fwd = SLSTMScan(cfg.d_model)
        self.bwd = SLSTMScan(cfg.d_model)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        fwd_out = self.fwd(x, reset_mask=reset_mask)
        # backward: reset_mask도 뒤집기
        bwd_reset = reset_mask.flip(1) if reset_mask is not None else None
        bwd_out = self.bwd(x.flip(1), reset_mask=bwd_reset).flip(1)
        out = fwd_out + bwd_out
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)
        return out
