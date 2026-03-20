"""Hybrid Mixing Layer — Conv1d + Window Attention + per-document FFT

세 가지 mixing을 조합하여 다중 스케일 패턴 캡처:
- Conv1d (depthwise, k=4): 자모 인접 패턴 (로컬)
- Window Attention (w=64, GQA+RoPE): 주변 어절 직접 참조 (중거리)
- FFT (1D, per-document): 문장 전체 분위기/문체 (글로벌)

흐름: x → conv1d(+residual) → window_attn + FFT → output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.bitlinear import Int8Linear
from model.mixing.base import MixingLayer
from model.mixing.full_attention import RotaryEmbedding, _apply_rotary


class HybridMixing(MixingLayer):
    """Conv1d + Window Attention + per-document FFT

    - Conv1d: depthwise conv로 자모 패턴 전처리 (residual)
    - Window Attention: GQA + RoPE, band_mask ∩ doc_mask로 윈도우 제한
    - FFT: 문서별 독립 1D FFT, 전체 문장 주파수 특성
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.attn_n_kv_heads
        self.headdim = cfg.headdim
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.window_size = cfg.hybrid_window_size

        d_kv = self.n_kv_heads * self.headdim

        # Conv1d (depthwise, symmetric padding)
        k = cfg.hybrid_conv_kernel
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False)

        # Window Attention Q,K,V,O (Int8Linear)
        self.q_proj = Int8Linear(d, d, bias=False)
        self.k_proj = Int8Linear(d, d_kv, bias=False)
        self.v_proj = Int8Linear(d, d_kv, bias=False)
        self.o_proj = Int8Linear(d, d, bias=False)

        # RoPE
        self.rope = RotaryEmbedding(self.headdim, max_seq_len=cfg.max_seq_len)

    def _init_weights(self):
        """가중치 초기화: Q,K,V Xavier uniform, O zero-init"""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self, x: Tensor,
        pad_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, D = x.shape

        # ── 1. Conv1d: 자모 패턴 전처리 (residual) ──
        h = x.transpose(1, 2)              # (B, D, T)
        conv_out = self.conv(h)
        if conv_out.size(2) != T:           # padding으로 길이 달라질 수 있음
            conv_out = conv_out[:, :, :T]
        x = x + conv_out.transpose(1, 2)   # (B, T, D)

        # ── 2. Window Attention ──
        H = self.n_heads
        H_kv = self.n_kv_heads
        d = self.headdim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H_kv, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H_kv, d).transpose(1, 2)

        q, k = self.rope(q, k)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        attn_mask = self._make_window_mask(reset_mask, pad_mask, T, x.device)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)

        # ── 3. Per-document FFT ──
        fft_out = self._per_doc_fft(x, reset_mask)

        # ── 4. 합산 ──
        out = attn_out + fft_out

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out

    def _make_window_mask(
        self,
        reset_mask: Tensor | None,
        pad_mask: Tensor | None,
        T: int,
        device: torch.device,
    ) -> Tensor | None:
        """Window + 문서 격리 + PAD 마스킹 → attention mask

        Returns:
            (B, 1, T, T) bool mask 또는 None
        """
        # Band mask: |i - j| <= w // 2
        w = self.window_size
        pos = torch.arange(T, device=device)
        band = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs() <= (w // 2)  # (T, T)

        masks = [band]

        if reset_mask is not None:
            doc_id = (reset_mask.int().cumsum(dim=1) - 1)  # (B, T)
            doc_mask = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1))  # (B, T, T)
            masks.append(doc_mask)

        if pad_mask is not None:
            key_mask = pad_mask.unsqueeze(1)  # (B, 1, T)
            masks.append(key_mask)

        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m

        return combined.unsqueeze(1)  # (B, 1, T, T)

    @torch.compiler.disable
    def _per_doc_fft(self, x: Tensor, reset_mask: Tensor | None) -> Tensor:
        """문서별 독립 1D FFT

        reset_mask에서 문서 경계를 찾아 각 문서 segment에 독립적으로 FFT 적용.
        문서 간 정보 누출 없음.
        """
        if reset_mask is None:
            # 패킹 없음 — 전체 시퀀스가 하나의 문서
            return torch.fft.fft(x.float(), dim=1).real.to(x.dtype)

        B, T, D = x.shape
        out = torch.zeros_like(x)
        x_f32 = x.float()

        for b in range(B):
            # BOS 위치 = 문서 시작점
            bos_positions = reset_mask[b].nonzero(as_tuple=True)[0]
            n_docs = bos_positions.size(0)

            for i in range(n_docs):
                start = bos_positions[i].item()
                end = bos_positions[i + 1].item() if i + 1 < n_docs else T

                # PAD 영역 제외 (end가 PAD 시작일 수 있음)
                doc_slice = x_f32[b, start:end]  # (doc_len, D)
                doc_fft = torch.fft.fft(doc_slice, dim=0).real
                out[b, start:end] = doc_fft.to(x.dtype)

        return out
