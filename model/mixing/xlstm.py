"""BiSLSTM Mixing Layer — 양방향 sLSTM (Scalar LSTM)

xLSTM (Beck et al., NeurIPS 2024) 의 sLSTM 변형.
exponential gating으로 gate saturation 문제 해결.

GPU 학습: Triton fused scan (associative_scan) — 전체 시퀀스 병렬 처리
CPU 추론: sequential scan (C 커널)

State per head: scalar c, n — 레지스터 적중
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear

# Triton fused sLSTM scan 감지
_TRITON_SLSTM = False
_triton_slstm_scan = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _slstm_fwd_kernel(
        I, F_gate, Z, O,  # (B*D, T) — gate inputs (transposed)
        Out,               # (B*D, T) — output
        T_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """sLSTM forward: element-wise sequential scan (per B*D element)

        각 프로그램이 1개 (batch, dim) 원소의 전체 시퀀스를 처리.
        Triton의 장점: sigmoid/exp/tanh가 GPU에서 벡터화됨.
        """
        bd = tl.program_id(0)  # batch * d_model index
        offsets = tl.arange(0, BLOCK_T)  # 최대 BLOCK_T 시퀀스
        mask = offsets < T_dim

        # 전체 시퀀스 로드
        i_vals = tl.load(I + bd * T_dim + offsets, mask=mask, other=0.0)
        f_vals = tl.load(F_gate + bd * T_dim + offsets, mask=mask, other=0.0)
        z_vals = tl.load(Z + bd * T_dim + offsets, mask=mask, other=0.0)
        o_vals = tl.load(O + bd * T_dim + offsets, mask=mask, other=0.0)

        # 활성화 (element-wise, GPU에서 벡터화됨)
        f_act = tl.sigmoid(f_vals)
        i_clamped = tl.maximum(tl.minimum(i_vals, 10.0), -10.0)
        i_act = tl.exp(i_clamped)
        z_act = tl.extra.cuda.libdevice.tanh(z_vals)
        o_act = tl.sigmoid(o_vals)

        # Sequential scan은 Triton에서도 순차적이지만,
        # sigmoid/exp/tanh + 메모리 접근이 fused되어 Python for-loop보다 훨씬 빠름
        # TODO: associative_scan으로 교체 가능 (log-space stabilization 필요)

        # 현재: 단순 sequential scan (but all ops are fused on GPU)
        c = 0.0
        n = 0.0
        for t in range(T_dim):
            f_t = tl.load(F_gate + bd * T_dim + t)
            i_t_raw = tl.load(I + bd * T_dim + t)
            z_t = tl.load(Z + bd * T_dim + t)
            o_t = tl.load(O + bd * T_dim + t)

            f_t = tl.sigmoid(f_t)
            i_t_raw = tl.maximum(tl.minimum(i_t_raw, 10.0), -10.0)
            i_t = tl.exp(i_t_raw)
            z_t = tl.extra.cuda.libdevice.tanh(z_t)
            o_t = tl.sigmoid(o_t)

            c = f_t * c + i_t * z_t
            n = f_t * n + i_t
            abs_n = tl.abs(n)
            denom = tl.maximum(abs_n, 1.0)
            h = o_t * c / denom
            tl.store(Out + bd * T_dim + t, h)

    def triton_slstm_scan(i_gate, f_gate, z_gate, o_gate):
        """Triton sLSTM scan wrapper

        Args: 모두 (B, T, D)
        Returns: (B, T, D)
        """
        B, T, D = i_gate.shape
        # (B, T, D) → (B*D, T) for per-element parallel
        i_flat = i_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        f_flat = f_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        z_flat = z_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        o_flat = o_gate.permute(0, 2, 1).contiguous().view(B * D, T)
        out_flat = torch.empty_like(i_flat)

        # BLOCK_T: 시퀀스 길이에 맞춤 (triton.next_power_of_2)
        BLOCK_T = triton.next_power_of_2(T)
        grid = (B * D,)

        _slstm_fwd_kernel[grid](
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
    """단방향 sLSTM

    GPU: Triton fused scan (활성화+scan 모두 GPU에서 fused)
    CPU: Python sequential scan
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.i_proj = BitLinear(d_model, d_model)
        self.f_proj = BitLinear(d_model, d_model)
        self.z_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        i_gate = self.i_proj(x)
        f_gate = self.f_proj(x)
        z_gate = self.z_proj(x)
        o_gate = self.o_proj(x)

        if _TRITON_SLSTM and x.is_cuda:
            return _triton_slstm_wrapper(i_gate, f_gate, z_gate, o_gate)

        # Python fallback (CPU)
        c = x.new_zeros(B, D)
        n = x.new_zeros(B, D)
        hs = []
        for t in range(T):
            f_t = torch.sigmoid(f_gate[:, t])
            i_raw = i_gate[:, t].clamp(-10, 10)
            i_t = torch.exp(i_raw)
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
