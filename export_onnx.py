"""DenseEditor (BiMamba-2 + BitNet) → ONNX 변환

Mamba-2 SSD를 matmul 기반으로 재구현하고 BitLinear 가중치를 dequantize하여
ONNX Runtime CPU/GPU 모두에서 추론 가능한 모델 생성.

사용법:
    python export_onnx.py checkpoint.pt -o exported_onnx/
    python export_onnx.py checkpoint.pt -o exported_onnx/ --max-seq-len 512

출력:
    exported_onnx/
    ├── model.onnx               # ONNX 모델
    ├── config.json              # 모델 설정
    ├── tokenizer_config.json    # 토크나이저 메타
    └── keyboard_tokenizer.json  # 토크나이저 vocab
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────
#  ONNX용 RMSNorm (Triton 의존성 없음)
# ──────────────────────────────────────────────

class RMSNormOnnx(nn.Module):
    """ONNX-exportable RMSNorm — pure PyTorch 연산만 사용"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight

    @staticmethod
    def from_original(norm) -> "RMSNormOnnx":
        """원본 RMSNorm에서 가중치 복사"""
        m = RMSNormOnnx(norm.weight.shape[0], eps=norm.eps)
        m.weight.data.copy_(norm.weight.data)
        return m


# ──────────────────────────────────────────────
#  BitLinear → FP32 dequantized Linear
# ──────────────────────────────────────────────

class BitLinearOnnx(nn.Module):
    """BitLinear을 FP32 matmul로 변환 — ONNX 호환

    원본: LayerNorm(x) → per-token INT8 quant → ternary matmul → scale 복원
    ONNX: LayerNorm(x) → FP32 matmul(x_norm, w_dequant)

    Activation INT8 양자화 노이즈 제거 → 정확도 동일 or 미세 개선
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(self.norm(x), self.weight, self.bias)

    @staticmethod
    def from_original(bl) -> "BitLinearOnnx":
        """원본 BitLinear에서 가중치를 dequantize하여 복사"""
        w = bl.weight.data.float()
        gamma = w.abs().mean().clamp(min=1e-5)
        w_dequant = (w / gamma).clamp(-1, 1).round() * gamma

        m = BitLinearOnnx(bl.in_features, bl.out_features, bias=bl.bias is not None)
        m.weight.data.copy_(w_dequant)
        if bl.bias is not None:
            m.bias.data.copy_(bl.bias.data)
        return m


# ──────────────────────────────────────────────
#  Mamba-2 SSD — matmul 기반 ONNX-exportable
# ──────────────────────────────────────────────

def _ssd_chunk_parallel(x: Tensor, B_proj: Tensor, C_proj: Tensor,
                        dt: Tensor, A_neg: Tensor, D: Tensor,
                        nheads: int, headdim: int, d_state: int, ngroups: int,
                        chunk_size: int = 256) -> Tensor:
    """Mamba-2 SSD chunk-parallel forward — ONNX 표준 op만 사용

    C 커널(mamba2_ssd.c)과 동일한 chunk-parallel 알고리즘을 matmul로 구현.
    청크 내부: (cs,cs) attention matrix — 메모리 O(cs²)로 제한.
    청크 간: state passing (sequential, 청크 수만큼만 loop).

    Args:
        x: (B, T, d_inner), B_proj: (B, T, ng*ds), C_proj: (B, T, ng*ds)
        dt: (B, T, nh), A_neg: (nh,), D: (nh,)
        chunk_size: 청크 크기 (기본 256, 학습과 동일)

    Returns:
        y: (B, T, d_inner)
    """
    B_batch, T, d_inner = x.shape
    cs = chunk_size
    heads_per_group = nheads // ngroups
    nchunks = (T + cs - 1) // cs

    # T를 chunk_size 배수로 패딩
    pad_len = nchunks * cs - T
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
        B_proj = F.pad(B_proj, (0, 0, 0, pad_len))
        C_proj = F.pad(C_proj, (0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len))
    T_padded = nchunks * cs

    # reshape to chunks: (B, nchunks, cs, ...)
    x_c = x.view(B_batch, nchunks, cs, nheads, headdim)
    B_c = B_proj.view(B_batch, nchunks, cs, ngroups, d_state)
    C_c = C_proj.view(B_batch, nchunks, cs, ngroups, d_state)
    dt_c = dt.view(B_batch, nchunks, cs, nheads)

    # ── 1. dA cumsum per chunk ──
    dA_c = A_neg.view(1, 1, 1, nheads) * dt_c  # (B, nc, cs, nh)
    dA_cum_c = torch.cumsum(dA_c, dim=2)        # (B, nc, cs, nh) — 청크 내 cumsum

    # ── 2. chunk_states: per-chunk 누적 상태 ──
    # state[b,c,h,hd,ds] = Σ_l exp(dA_last - dA_l) * dt_l * outer(B_l, x_l)
    dA_last = dA_cum_c[:, :, -1:, :]  # (B, nc, 1, nh) — 청크 마지막 위치
    decay_to_end = torch.exp(dA_last - dA_cum_c)  # (B, nc, cs, nh) — 각 위치 → 청크 끝까지 decay

    # scale = decay_to_end * dt
    scale = decay_to_end * dt_c  # (B, nc, cs, nh)

    # B 확장: (B, nc, cs, ng, ds) → (B, nc, cs, nh, ds)
    B_exp = B_c.repeat_interleave(heads_per_group, dim=3) if heads_per_group > 1 else B_c

    # chunk_states[b,c,h,p,n] = Σ_l scaled_x[b,c,l,h,p] * B[b,c,l,h,n]
    # einsum "bclhp,bclhn->bchpn" → bmm: (B*nc*nh, hd, cs) @ (B*nc*nh, cs, ds)
    scaled_x = scale.unsqueeze(-1) * x_c  # (B, nc, cs, nh, hd)
    BNH = B_batch * nchunks * nheads
    sx = scaled_x.permute(0, 1, 3, 4, 2).reshape(BNH, headdim, cs)  # (B*nc*nh, hd, cs)
    be = B_exp.permute(0, 1, 3, 2, 4).reshape(BNH, cs, d_state)     # (B*nc*nh, cs, ds)
    chunk_states = torch.bmm(sx, be).view(B_batch, nchunks, nheads, headdim, d_state)

    # ── 3. state_passing: 청크 간 상태 전파 (loop-free, ONNX 호환) ──
    # loop: prev[0]=0, prev[c] = decay[c-1]*prev[c-1] + cs[c-1]
    # decay[c] = exp(log_decay[c]) where log_decay[c] = dA_cum[c,-1]

    # vectorized: (nc, nc) causal decay matrix
    # decay_mat[c,c'] = chunk c'의 state가 chunk c에 도달하는 decay factor
    # = exp(log_decay[c'+1] + ... + log_decay[c-1])  for c > c'+1
    # = 1  for c = c'+1 (인접: decay 없이 바로 전달)
    # = 0  for c <= c'

    log_decay = dA_last.squeeze(2)  # (B, nc, nh) — 각 청크의 총 log-decay

    # inclusive prefix sum: cum[c] = ld[0]+...+ld[c]
    cum_log = torch.cumsum(log_decay, dim=1)  # (B, nc, nh)

    # exclusive prefix sum: excl[0]=0, excl[c]=ld[0]+...+ld[c-1]
    zeros_pad = torch.zeros(B_batch, 1, nheads, device=x.device, dtype=x.dtype)
    excl_cum = torch.cat([zeros_pad, cum_log[:, :-1, :]], dim=1)  # (B, nc, nh)

    # decay_mat[c,c'] = exp(excl_cum[c] - cum_log[c'])  for c > c'
    # 검증: c=c'+1 → exp(excl_cum[c'+1] - cum_log[c']) = exp(cum_log[c'] - cum_log[c']) = 1 ✓
    # c=c'+2 → exp(excl_cum[c'+2] - cum_log[c']) = exp(ld[0]+..+ld[c'] - (ld[0]+..+ld[c']))
    #         = exp(ld[c'+1] + ... + ld[c'+1])... 잠깐, excl_cum[c'+2] = ld[0]+...+ld[c'+1]
    #         = exp(ld[0]+...+ld[c'+1] - ld[0]-...-ld[c']) = exp(ld[c'+1]) ✓ (1 chunk decay)

    diff_mat = excl_cum.unsqueeze(2) - cum_log.unsqueeze(1)  # (B, nc, nc, nh)
    causal_chunks = torch.tril(torch.ones(nchunks, nchunks, device=x.device, dtype=torch.bool), diagonal=-1)
    diff_mat = diff_mat.masked_fill(~causal_chunks.unsqueeze(0).unsqueeze(-1), float("-inf"))
    decay_mat = torch.exp(diff_mat)  # (B, nc, nc, nh)

    # prev_states[c,h,p,n] = Σ_{c'<c} decay_mat[c,c',h] * chunk_states[c',h,p,n]
    # einsum "bcjh,bjhpn->bchpn" → bmm per head: (B*nh, nc, nc) @ (B*nh, nc, hd*ds)
    BH = B_batch * nheads
    dm = decay_mat.permute(0, 3, 1, 2).reshape(BH, nchunks, nchunks)  # (B*nh, nc, nc)
    cs_flat = chunk_states.permute(0, 2, 1, 3, 4).reshape(BH, nchunks, headdim * d_state)  # (B*nh, nc, hd*ds)
    prev_states = torch.bmm(dm, cs_flat).view(B_batch, nheads, nchunks, headdim, d_state)
    prev_states = prev_states.permute(0, 2, 1, 3, 4)  # (B, nc, nh, hd, ds)

    # ── 4. chunk_scan: 청크 내 출력 계산 ──
    # 4a. intra-chunk: (cs, cs) attention per chunk
    # CB[b,c,t,s,g] = C[b,c,t,g,:] @ B[b,c,s,g,:].T
    # CB[b,c,i,j,g] = Σ_n C[b,c,i,g,n] * B[b,c,j,g,n] → bmm
    BNG = B_batch * nchunks * ngroups
    C_flat = C_c.permute(0, 1, 3, 2, 4).reshape(BNG, cs, d_state)    # (B*nc*ng, cs, ds)
    B_flat = B_c.permute(0, 1, 3, 4, 2).reshape(BNG, d_state, cs)    # (B*nc*ng, ds, cs)
    CB_c = torch.bmm(C_flat, B_flat).view(B_batch, nchunks, ngroups, cs, cs)
    CB_c = CB_c.permute(0, 1, 3, 4, 2)  # (B, nc, cs, cs, ng)
    if heads_per_group > 1:
        CB_c = CB_c.repeat_interleave(heads_per_group, dim=-1)  # → (B, nc, cs, cs, nh)

    # decay within chunk: exp(dA_cum[t] - dA_cum[s]) for t >= s
    diff_c = dA_cum_c.unsqueeze(3) - dA_cum_c.unsqueeze(2)  # (B, nc, cs, cs, nh)
    causal = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool))
    diff_c = diff_c.masked_fill(~causal.unsqueeze(0).unsqueeze(0).unsqueeze(-1), float("-inf"))
    decay_intra = torch.exp(diff_c)  # (B, nc, cs, cs, nh)

    # attn = CB * decay * dt[source]
    attn_c = CB_c * decay_intra * dt_c.unsqueeze(2)  # (B, nc, cs, cs, nh)

    # intra_out[b,c,i,h,p] = Σ_s attn[b,c,i,s,h] * x[b,c,s,h,p] → bmm
    attn_flat = attn_c.permute(0, 1, 4, 2, 3).reshape(BNH, cs, cs)   # (B*nc*nh, cs, cs)
    x_flat = x_c.permute(0, 1, 3, 2, 4).reshape(BNH, cs, headdim)    # (B*nc*nh, cs, hd)
    intra_out = torch.bmm(attn_flat, x_flat).view(B_batch, nchunks, nheads, cs, headdim)
    intra_out = intra_out.permute(0, 1, 3, 2, 4)  # (B, nc, cs, nh, hd)

    # 4b. inter-chunk: 이전 청크들의 누적 상태 기여
    # state_decay[b,c,t,h] = exp(dA_cum[b,c,t,h]) — 청크 시작 → t까지 decay
    state_decay = torch.exp(dA_cum_c)  # (B, nc, cs, nh)

    # C 확장
    C_exp = C_c.repeat_interleave(heads_per_group, dim=3) if heads_per_group > 1 else C_c

    # inter_out[b,c,i,h,p] = Σ_n C[b,c,i,h,n] * prev[b,c,h,p,n] → bmm
    # (B*nc*nh, cs, ds) @ (B*nc*nh, ds, hd) → (B*nc*nh, cs, hd)
    C_flat2 = C_exp.permute(0, 1, 3, 2, 4).reshape(BNH, cs, d_state)       # (B*nc*nh, cs, ds)
    ps_flat = prev_states.permute(0, 1, 2, 4, 3).reshape(BNH, d_state, headdim)  # (B*nc*nh, ds, hd)
    inter_out = torch.bmm(C_flat2, ps_flat).view(B_batch, nchunks, nheads, cs, headdim)
    inter_out = inter_out.permute(0, 1, 3, 2, 4)  # (B, nc, cs, nh, hd)
    inter_out = inter_out * state_decay.unsqueeze(-1)  # (B, nc, cs, nh, hd)

    # 합산 + skip
    y = intra_out + inter_out + D.view(1, 1, 1, nheads, 1) * x_c
    y = y.reshape(B_batch, T_padded, d_inner)

    # 패딩 제거
    if pad_len > 0:
        y = y[:, :T_padded - pad_len, :]

    return y


class Mamba2BlockOnnx(nn.Module):
    """Mamba-2 SSD 단방향 블록 — ONNX exportable

    원본 Mamba2Block._forward_fallback와 동일한 연산 흐름:
    in_proj → split(z, xBC, dt) → conv1d → silu → SSD → gate(silu(z)) → norm → out_proj
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64, ngroups: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        d_conv_in = self.d_inner + 2 * ngroups * d_state

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        # Conv1d: padding=0, 수동 left-pad로 ONNX 호환성 보장
        self.conv1d = nn.Conv1d(d_conv_in, d_conv_in, kernel_size=d_conv,
                                padding=0, groups=d_conv_in, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner)

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        di = self.d_inner
        ds = self.d_state
        ng = self.ngroups
        d_conv_in = di + 2 * ng * ds

        # 1. in_proj
        proj = self.in_proj(x)  # (B, T, d_in_proj)
        z = proj[:, :, :di]
        xBC_raw = proj[:, :, di:di + d_conv_in]
        dt_raw = proj[:, :, di + d_conv_in:]

        # 2. conv1d (수동 left-padding으로 ONNX 호환)
        xBC_t = xBC_raw.transpose(1, 2)  # (B, d_conv_in, T)
        xBC_padded = F.pad(xBC_t, (self.d_conv - 1, 0))  # left-pad
        xBC = self.conv1d(xBC_padded)  # (B, d_conv_in, T)
        xBC = xBC.transpose(1, 2)     # (B, T, d_conv_in)

        # 3. SiLU activation (GPU fused kernel activation="silu"와 동일)
        x_conv = F.silu(xBC[:, :, :di])
        B_conv = F.silu(xBC[:, :, di:di + ng * ds])
        C_conv = F.silu(xBC[:, :, di + ng * ds:])

        # 4. dt (FP32 강제 — 수치 안정성)
        dt = F.softplus(dt_raw + self.dt_bias)
        A_neg = -torch.exp(self.A_log.float())

        # 5. SSD forward (chunk-parallel matmul)
        y = _ssd_chunk_parallel(x_conv, B_conv, C_conv, dt, A_neg, self.D,
                                self.nheads, self.headdim, self.d_state, self.ngroups)

        # 6. gate(silu(z)) → norm → out_proj
        y = y * F.silu(z)
        y = self.norm(y)
        return self.out_proj(y)

    @staticmethod
    def from_original(block) -> "Mamba2BlockOnnx":
        """원본 Mamba2Block 또는 Mamba2BitLinearBlock에서 가중치 복사

        원본 conv1d는 padding=d_conv-1, 여기서는 padding=0 + 수동 left-pad.
        Mamba2BitLinearBlock: in_proj_down(Linear) + in_proj_up(BitLinear) → 단일 Linear로 합침
        """
        from model.mixing.bi_mamba2 import Mamba2BitLinearBlock

        m = Mamba2BlockOnnx(
            d_model=block.d_model, d_state=block.d_state,
            d_conv=block.d_conv, expand=block.d_inner // block.d_model,
            headdim=block.headdim, ngroups=block.ngroups,
        )

        if isinstance(block, Mamba2BitLinearBlock):
            # BitLinear 저랭크 in_proj: down(d→rank) → up(rank→d_in_proj)
            # → 단일 FP32 Linear(d→d_in_proj)로 합침
            # up은 BitLinear: LayerNorm → quant → matmul → scale
            # dequant weight: round(clip(W/gamma,-1,1)) * gamma
            w_up = block.in_proj_up.weight.data.float()
            gamma = w_up.abs().mean().clamp(min=1e-5)
            w_up_dequant = (w_up / gamma).clamp(-1, 1).round() * gamma

            # 합친 가중치: W_combined = W_up @ LayerNorm(W_down @ x)
            # LayerNorm이 중간에 있으므로 단순 합성 불가 → 별도 모듈로 처리
            # in_proj를 두 단계로 분리
            w_down = block.in_proj_down.weight.data.float()
            d_in_proj = w_up_dequant.shape[0]
            rank = w_down.shape[0]

            # in_proj를 nn.Sequential(Linear, LayerNorm, Linear)로 대체
            m.in_proj = nn.Sequential(
                nn.Linear(block.d_model, rank, bias=False),
                nn.LayerNorm(rank, elementwise_affine=False),
                nn.Linear(rank, d_in_proj, bias=block.in_proj_up.bias is not None),
            )
            m.in_proj[0].weight.data.copy_(w_down)
            m.in_proj[2].weight.data.copy_(w_up_dequant)
            if block.in_proj_up.bias is not None:
                m.in_proj[2].bias.data.copy_(block.in_proj_up.bias.data)

            # out_proj도 BitLinear → dequantized Linear
            w_out = block.out_proj.weight.data.float()
            gamma_out = w_out.abs().mean().clamp(min=1e-5)
            w_out_dequant = (w_out / gamma_out).clamp(-1, 1).round() * gamma_out

            # out_proj를 LayerNorm + Linear로
            m.out_proj = nn.Sequential(
                nn.LayerNorm(block.d_inner, elementwise_affine=False),
                nn.Linear(block.d_inner, block.d_model, bias=block.out_proj.bias is not None),
            )
            m.out_proj[1].weight.data.copy_(w_out_dequant)
            if block.out_proj.bias is not None:
                m.out_proj[1].bias.data.copy_(block.out_proj.bias.data)
        else:
            # 일반 Mamba2Block: nn.Linear 직접 복사
            m.in_proj.weight.data.copy_(block.in_proj.weight.data)
            m.out_proj.weight.data.copy_(block.out_proj.weight.data)

        m.conv1d.weight.data.copy_(block.conv1d.weight.data)
        m.conv1d.bias.data.copy_(block.conv1d.bias.data)
        m.norm.weight.data.copy_(block.norm.weight.data)
        m.dt_bias.data.copy_(block.dt_bias.data)
        m.A_log.data.copy_(block.A_log.data)
        m.D.data.copy_(block.D.data)
        return m


class BiMamba2OnnxMixing(nn.Module):
    """양방향 Mamba-2 — ONNX exportable

    fwd scan + bwd scan(flip) → element-wise add.
    배포 시 단일 문서이므로 reset_mask/seq_idx 문서 격리 생략.
    """

    def __init__(self):
        super().__init__()
        self.fwd = None  # from_original에서 설정
        self.bwd = None

    def forward(self, x: Tensor) -> Tensor:
        fwd_out = self.fwd(x)
        bwd_out = self.bwd(x.flip(1)).flip(1)
        return fwd_out + bwd_out

    @staticmethod
    def from_original(bi_mixing) -> "BiMamba2OnnxMixing":
        """원본 BiMamba2Mixing에서 가중치 복사"""
        m = BiMamba2OnnxMixing()
        m.fwd = Mamba2BlockOnnx.from_original(bi_mixing.fwd)
        m.bwd = Mamba2BlockOnnx.from_original(bi_mixing.bwd)
        return m


# ──────────────────────────────────────────────
#  BitNetFFN → ONNX
# ──────────────────────────────────────────────

class BitNetFFNOnnx(nn.Module):
    """BitNetFFN ONNX 변환 — BitLinear을 dequantized FP32로"""

    def __init__(self, d_ff: int, fused: bool):
        super().__init__()
        self.d_ff = d_ff
        self.fused = fused

    def forward(self, x: Tensor) -> Tensor:
        if self.fused:
            gu = self.gate_up_proj(x)
            gate_out, up = gu.split(self.d_ff, dim=-1)
        else:
            gate_out = self.gate_proj(x)
            up = self.up_proj(x)
        x = F.relu(gate_out) * up
        return self.down_proj(x)

    @staticmethod
    def from_original(ffn) -> "BitNetFFNOnnx":
        """원본 BitNetFFN에서 변환"""
        m = BitNetFFNOnnx(ffn.d_ff, ffn.fused)
        if ffn.fused:
            m.gate_up_proj = BitLinearOnnx.from_original(ffn.gate_up_proj)
        else:
            m.gate_proj = BitLinearOnnx.from_original(ffn.gate_proj)
            m.up_proj = BitLinearOnnx.from_original(ffn.up_proj)
        m.down_proj = BitLinearOnnx.from_original(ffn.down_proj)
        return m


# ──────────────────────────────────────────────
#  DenseEditorLayer + DenseEditor → ONNX
# ──────────────────────────────────────────────

class DenseEditorLayerOnnx(nn.Module):
    """DenseEditorLayer ONNX 변환"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixing(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    @staticmethod
    def from_original(layer) -> "DenseEditorLayerOnnx":
        m = DenseEditorLayerOnnx()
        m.norm1 = RMSNormOnnx.from_original(layer.norm1)
        m.mixing = BiMamba2OnnxMixing.from_original(layer.mixing)
        m.norm2 = RMSNormOnnx.from_original(layer.norm2)
        m.ffn = BitNetFFNOnnx.from_original(layer.ffn)
        return m


class DenseEditorOnnx(nn.Module):
    """DenseEditor 전체 모델 — ONNX exportable

    입력: input_ids (B, T) int64
    출력: tag_logits (B, T, n_tags) float32
    """

    def __init__(self, cfg):
        super().__init__()
        self.embed_scale = math.sqrt(cfg.d_model)
        self.bos_id = cfg.bos_id

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embedding(input_ids) * self.embed_scale

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return self.tag_head(x)

    @staticmethod
    def from_original(model) -> "DenseEditorOnnx":
        """학습된 DenseEditor에서 ONNX 모델 생성"""
        cfg = model.cfg
        m = DenseEditorOnnx(cfg)

        # Embedding
        m.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        m.embedding.weight.data.copy_(model.embedding.weight.data)

        # Layers
        m.layers = nn.ModuleList([
            DenseEditorLayerOnnx.from_original(layer) for layer in model.layers
        ])

        # Final norm
        m.final_norm = RMSNormOnnx.from_original(model.final_norm)

        # Tag head
        m.tag_head = BitLinearOnnx.from_original(model.tag_head)

        return m


# ──────────────────────────────────────────────
#  수치 검증
# ──────────────────────────────────────────────

def verify_ssd(original_block, onnx_block, T: int = 128, B: int = 1, atol: float = 1e-3):
    """SSD 단위 수치 검증: 원본 sequential scan vs ONNX matmul"""
    d = original_block.d_model
    x = torch.randn(B, T, d)

    original_block.eval()
    onnx_block.eval()

    with torch.no_grad():
        # 원본 (CPU fallback)
        y_orig = original_block._forward_fallback(x)
        # ONNX
        y_onnx = onnx_block(x)

    diff = (y_orig - y_onnx).abs().max().item()
    rel_diff = diff / (y_orig.abs().max().item() + 1e-8)
    print(f"  SSD 검증 (T={T}): max_diff={diff:.6f}, rel_diff={rel_diff:.6f}")

    if diff > atol:
        print(f"  ⚠ 차이가 atol={atol}을 초과 — 상대 오차 확인 필요")
    else:
        print(f"  ✓ OK (atol={atol} 이내)")
    return diff


def verify_model(original_model, onnx_model, seq_lens=(32, 64, 128)):
    """전체 모델 수치 검증: argmax 태그 일치 확인"""
    original_model.eval()
    onnx_model.eval()

    for T in seq_lens:
        ids = torch.randint(3, 300, (1, T))  # special token 회피
        ids[0, 0] = original_model.cfg.bos_id

        with torch.no_grad():
            logits_orig = original_model(ids)
            logits_onnx = onnx_model(ids)

        tags_orig = logits_orig.argmax(dim=-1)
        tags_onnx = logits_onnx.argmax(dim=-1)
        match = (tags_orig == tags_onnx).float().mean().item()

        logit_diff = (logits_orig - logits_onnx).abs().max().item()
        print(f"  T={T}: tag 일치율={match*100:.1f}%, logit max_diff={logit_diff:.4f}")


# ──────────────────────────────────────────────
#  Export
# ──────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """체크포인트에서 DenseEditor 로드"""
    from model.dense_editor_config import DenseEditorConfig
    from model.dense_editor import DenseEditor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # config 로드
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        if isinstance(cfg_dict, dict):
            cfg = DenseEditorConfig(**{k: v for k, v in cfg_dict.items()
                                       if k in DenseEditorConfig.__dataclass_fields__})
        else:
            cfg = cfg_dict
    else:
        raise ValueError("체크포인트에 config가 없습니다")

    model = DenseEditor(cfg)

    # state_dict 로드
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # DDP prefix 제거
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, cfg


def export_onnx(checkpoint_path: str, output_dir: str, opset: int = 18,
                max_seq_len: int = 256, verify: bool = True, fp16: bool = True):
    """DenseEditor → ONNX 변환 + 검증 + 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"체크포인트 로딩: {checkpoint_path}")
    model, cfg = load_model(checkpoint_path)
    print(f"모델 로드 완료: d={cfg.d_model}, {cfg.n_layers}L, mixing={cfg.mixing_type}")

    if cfg.mixing_type != "mamba2":
        raise ValueError(f"이 스크립트는 mamba2 mixing만 지원 (현재: {cfg.mixing_type})")

    # ONNX 모델 생성
    print("\nONNX 모델 변환 중...")
    onnx_model = DenseEditorOnnx.from_original(model)
    onnx_model.eval()

    # 파라미터 수 비교
    orig_params = sum(p.numel() for p in model.parameters())
    onnx_params = sum(p.numel() for p in onnx_model.parameters())
    print(f"원본 파라미터: {orig_params:,}")
    print(f"ONNX 파라미터: {onnx_params:,}")

    # 수치 검증
    if verify:
        print("\n=== SSD 단위 검증 ===")
        for i, layer in enumerate(model.layers):
            if i == 0:
                verify_ssd(layer.mixing.fwd, onnx_model.layers[0].mixing.fwd, T=64)
                break

        print("\n=== 전체 모델 검증 ===")
        verify_model(model, onnx_model, seq_lens=[32, 64, 128])

    # ONNX export
    print(f"\nONNX export (opset={opset})...")
    dummy_ids = torch.randint(3, 300, (1, max_seq_len))
    dummy_ids[0, 0] = cfg.bos_id

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        onnx_model,
        (dummy_ids,),
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["tag_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "tag_logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"ONNX 저장 완료: {onnx_path} ({size_mb:.1f}MB)")

    # ONNX simplifier (선택)
    try:
        import onnxsim
        import onnx
        print("onnxsim 적용 중...")
        model_onnx = onnx.load(str(onnx_path))
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simp, str(onnx_path))
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            print(f"simplify 완료: {size_mb:.1f}MB")
        else:
            print("simplify 실패 — 원본 유지")
    except ImportError:
        print("onnxsim 미설치 — 스킵 (pip install onnxsim)")

    # config.json 저장
    from dataclasses import asdict
    cfg_path = output_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # 토크나이저 파일 복사
    project_root = Path(__file__).parent
    for fname in ["keyboard_tokenizer.json", "jamo_token_map.json"]:
        src = project_root / "keyboard_tokenizer" / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    tok_config = {
        "type": "keyboard",
        "vocab_size": cfg.vocab_size,
        "pad_id": cfg.pad_id,
        "bos_id": cfg.bos_id,
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)

    # FP16 변환 (CUDA Tensor Core 활용 → ~2x 가속)
    if fp16:
        try:
            import onnx as _onnx
            from onnxruntime.transformers.float16 import convert_float_to_float16
            print("\nFP16 변환...")
            fp16_path = output_dir / "model_fp16.onnx"
            model_loaded = _onnx.load(str(onnx_path))
            model_fp16 = convert_float_to_float16(model_loaded, keep_io_types=True)
            _onnx.save(model_fp16, str(fp16_path))
            fp16_mb = fp16_path.stat().st_size / 1024 / 1024
            print(f"FP16 저장: {fp16_path} ({fp16_mb:.1f}MB)")
        except ImportError:
            print("\nFP16 변환 스킵 (onnxruntime.transformers 필요)")

    print(f"\n완료! 출력 디렉토리: {output_dir}")

    # ONNX RT 검증 + 벤치마크
    _run_ort_validation(onnx_path, onnx_model, cfg)


def setup_cuda_libs():
    """pip nvidia 패키지의 CUDA 라이브러리 경로를 LD_LIBRARY_PATH에 추가"""
    import site, sys
    nvidia_dirs = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_base = Path(sp) / "nvidia"
        if nvidia_base.exists():
            for sub in nvidia_base.iterdir():
                lib_dir = sub / "lib"
                if lib_dir.exists():
                    nvidia_dirs.append(str(lib_dir))
    if nvidia_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + ":" + existing
        return True
    return False


def _run_ort_validation(onnx_path, onnx_model, cfg):
    """ONNX Runtime CPU + CUDA 검증 및 벤치마크"""
    try:
        import onnxruntime as ort
        import numpy as np
        import time
    except ImportError:
        print("\nonnxruntime 미설치 — 스킵")
        return

    # chunk_size(256)의 배수로 테스트 (SSD reshape 호환)
    test_T = 256
    test_ids = torch.randint(3, 300, (1, test_T)).numpy().astype("int64")
    test_ids[0, 0] = cfg.bos_id

    with torch.no_grad():
        pt_out = onnx_model(torch.from_numpy(test_ids)).numpy()

    # CUDA EP 시도
    setup_cuda_libs()
    providers_list = [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "CUDA"),
        (["CPUExecutionProvider"], "CPU"),
    ]

    for providers, label in providers_list:
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            active = sess.get_providers()
        except Exception:
            continue

        ep_name = active[0].replace("ExecutionProvider", "")
        ort_out = sess.run(None, {"input_ids": test_ids})[0]
        diff = abs(pt_out - ort_out).max()
        tags_match = (pt_out.argmax(axis=-1) == ort_out.argmax(axis=-1)).mean()
        print(f"\n=== ORT {ep_name} ===")
        print(f"  PT vs ORT: max_diff={diff:.6f}, tag 일치율={tags_match*100:.1f}%")

        # 벤치마크 (chunk_size=256 배수)
        for T in [256, 1024, 4096]:
            ids_np = np.random.randint(3, 300, (1, T)).astype(np.int64)
            ids_np[0, 0] = cfg.bos_id
            for _ in range(3):
                sess.run(None, {"input_ids": ids_np})
            N = max(5, 30 // max(1, T // 256))
            t0 = time.time()
            for _ in range(N):
                sess.run(None, {"input_ids": ids_np})
            avg_ms = (time.time() - t0) / N * 1000
            print(f"  T={T:>4d}: {avg_ms:.1f}ms")

        # FP16 벤치마크 (CUDA만)
        if "CUDA" in ep_name:
            fp16_path = onnx_path.parent / "model_fp16.onnx"
            if fp16_path.exists():
                sess16 = ort.InferenceSession(str(fp16_path), providers=providers)
                print(f"\n=== ORT {ep_name} FP16 ===")
                for T in [256, 1024, 4096]:
                    ids_np = np.random.randint(3, 300, (1, T)).astype(np.int64)
                    ids_np[0, 0] = cfg.bos_id
                    for _ in range(3):
                        sess16.run(None, {"input_ids": ids_np})
                    N = max(5, 30 // max(1, T // 256))
                    t0 = time.time()
                    for _ in range(N):
                        sess16.run(None, {"input_ids": ids_np})
                    avg_ms = (time.time() - t0) / N * 1000
                    print(f"  T={T:>4d}: {avg_ms:.1f}ms")
        del sess


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DenseEditor → ONNX 변환")
    parser.add_argument("checkpoint", help="학습 체크포인트 경로 (.pt)")
    parser.add_argument("-o", "--output", default="exported_onnx", help="출력 디렉토리")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset 버전")
    parser.add_argument("--max-seq-len", type=int, default=256, help="export용 더미 시퀀스 길이")
    parser.add_argument("--no-verify", action="store_true", help="수치 검증 건너뛰기")
    parser.add_argument("--no-fp16", action="store_true", help="FP16 변환 건너뛰기")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, opset=args.opset,
                max_seq_len=args.max_seq_len, verify=not args.no_verify,
                fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
