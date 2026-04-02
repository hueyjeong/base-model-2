"""Full Attention Mixing Layer — GQA + RoPE + 문서 격리

Full bidirectional self-attention (causal mask 없음).
Attention은 자연 양방향이므로 BiMamba처럼 fwd+bwd 분리 불필요.

구조:
    Q_proj(d, d) → RoPE → ┐
    K_proj(d, d_kv) → RoPE → GQA expand → SDPA(is_causal=False)
    V_proj(d, d_kv) → GQA expand ────────→ ┘
    → O_proj(d, d) → output
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.bitlinear import Int8Linear
from model.mixing.base import MixingLayer


# ── RoPE (Rotary Position Embedding) ──

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding

    cos/sin 캐시를 미리 계산하여 Q, K에 적용.
    half-rotary: x를 반으로 나눠 (x1, x2) → (x1*cos - x2*sin, x2*cos + x1*sin)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim//2)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Q, K에 RoPE 적용

        Args:
            q: (B, H, T, d_head)
            k: (B, H_kv, T, d_head)
        Returns:
            (q_rot, k_rot) 동일 shape
        """
        T = q.size(2)
        if T > self.cos_cache.size(0):
            self._build_cache(T)
        cos = self.cos_cache[:T].to(q.dtype)  # (T, d_head//2)
        sin = self.sin_cache[:T].to(q.dtype)
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-rotation: (x1, x2) → (x1*cos - x2*sin, x2*cos + x1*sin)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Full Attention Mixing ──

class FullAttentionMixing(MixingLayer):
    """Full bidirectional self-attention with GQA + RoPE + 문서 격리

    - GQA: n_kv_heads < n_heads → K,V를 repeat_interleave로 확장
    - RoPE: Q, K에 rotary position embedding
    - 문서 격리: reset_mask → doc_id → attention mask (같은 문서만 attend)
    - F.scaled_dot_product_attention 사용 (Flash Attention 2 자동)
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.attn_n_kv_heads
        self.headdim = cfg.headdim
        self.kv_repeat = self.n_heads // self.n_kv_heads

        d_kv = self.n_kv_heads * self.headdim

        # Q,K,V,O 프로젝션 (INT8 QAT: per-token act + per-tensor weight)
        # use_norm=False: INT8 absmax가 자체 스케일링 → Sub-LayerNorm 불필요, BF16 유지
        self.q_proj = Int8Linear(d, d, bias=False, use_norm=False)
        self.k_proj = Int8Linear(d, d_kv, bias=False, use_norm=False)
        self.v_proj = Int8Linear(d, d_kv, bias=False, use_norm=False)
        self.o_proj = Int8Linear(d, d, bias=False, use_norm=False)

        # RoPE
        self.rope = RotaryEmbedding(self.headdim, max_seq_len=cfg.max_seq_len)

    def _init_weights(self):
        """가중치 초기화: Q,K,V Xavier uniform, O zero-init"""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # O를 zero-init → 초기에 residual stream 그대로 통과
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self, x: Tensor,
        pad_mask: Tensor | None = None,
        reset_mask: Tensor | None = None,
    ) -> Tensor:
        B, T, D = x.shape
        H = self.n_heads
        H_kv = self.n_kv_heads
        d = self.headdim

        # 프로젝션
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)       # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H_kv, d).transpose(1, 2)    # (B, H_kv, T, d)
        v = self.v_proj(x).view(B, T, H_kv, d).transpose(1, 2)    # (B, H_kv, T, d)

        # RoPE
        q, k = self.rope(q, k)

        # GQA: K,V 확장
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)  # (B, H, T, d)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        # Attention mask (문서 격리 + PAD 마스킹)
        attn_mask = self._make_attn_mask(reset_mask, pad_mask, B, T, x.device)

        # Scaled dot-product attention (Flash Attention 2 자동)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # 출력 프로젝션
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)

        # PAD 위치 제로화
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out

    def _make_attn_mask(
        self,
        reset_mask: Tensor | None,
        pad_mask: Tensor | None,
        B: int, T: int,
        device: torch.device,
    ) -> Tensor | None:
        """문서 격리 + PAD 마스킹 → attention mask 생성

        Returns:
            (B, 1, T, T) bool mask — True = attend, False = ignore
            또는 None (마스킹 불필요 시)
        """
        masks = []

        # 문서 격리: 같은 문서 내 토큰끼리만 attend
        if reset_mask is not None:
            doc_id = (reset_mask.int().cumsum(dim=1) - 1)  # (B, T)
            doc_mask = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1))  # (B, T, T)
            masks.append(doc_mask)

        # PAD 마스킹: PAD key에 attend 차단
        if pad_mask is not None:
            key_mask = pad_mask.unsqueeze(1)  # (B, 1, T) — broadcast over query dim
            masks.append(key_mask)

        if not masks:
            return None

        # 마스크 결합
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m

        return combined.unsqueeze(1)  # (B, 1, T, T)
