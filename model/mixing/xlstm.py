"""BiSLSTM Mixing Layer — 양방향 sLSTM (Scalar LSTM)

xLSTM (Beck et al., NeurIPS 2024) 의 sLSTM 변형.

GPU: Triton fused scan + fused 4-gate projection (양자화 1회)
CPU: sequential scan (C 커널)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# Triton fused sLSTM scan
_TRITON_SLSTM = False
_triton_slstm_scan = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _slstm_fwd_kernel(
        I, F_gate, Z, O, Out,
        T_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """sLSTM forward: per-(batch,dim) element sequential scan on GPU"""
        bd = tl.program_id(0)

        c = 0.0
        n = 0.0
        for t in range(T_dim):
            off = bd * T_dim + t
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

    def triton_slstm_scan(i_gate, f_gate, z_gate, o_gate):
        B, T, D = i_gate.shape
        i_flat = i_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        f_flat = f_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        z_flat = z_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        o_flat = o_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        out_flat = torch.empty_like(i_flat)
        BLOCK_T = triton.next_power_of_2(T)
        _slstm_fwd_kernel[(B * D,)](
            i_flat, f_flat, z_flat, o_flat, out_flat,
            T_dim=T, BLOCK_T=BLOCK_T,
        )
        return out_flat.view(B, D, T).permute(0, 2, 1).contiguous()

    _TRITON_SLSTM = True
    _triton_slstm_scan = triton_slstm_scan
except (ImportError, Exception):
    pass


@torch.compiler.disable
def _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate):
    return _triton_slstm_scan(i_gate, f_gate, z_gate, o_gate)


class SLSTMScan(nn.Module):
    """단방향 sLSTM — fused 4-gate projection

    4개 gate를 1개 BitLinear(d, 4d)로 처리 → 양자화+RMSNorm 1회.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Fused: 1개 BitLinear(d, 4d) — i, f, z, o 순서로 concat
        self.gate_proj = BitLinear(d_model, 4 * d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        # Fused gate projection: 양자화+RMSNorm 1회만
        gates = self.gate_proj(x)  # (B, T, 4D)
        i_gate, f_gate, z_gate, o_gate = gates.split(D, dim=-1)

        if _TRITON_SLSTM and x.is_cuda:
            return _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate)

        # CPU fallback
        c = x.new_zeros(B, D)
        n = x.new_zeros(B, D)
        hs = []
        for t in range(T):
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
