"""Cross-Attention — GQA + FlashAttention

Grouped-Query Attention (GQA) with FlashAttention (SDPA) 지원.
Q는 n_heads개, K/V는 n_kv_heads개 (n_kv_heads < n_heads).
PyTorch 2.0+ scaled_dot_product_attention으로 FlashAttention-2 자동 활용.

GQA 모드:
  - n_kv_heads == n_heads → 표준 MHA
  - n_kv_heads == 1       → MQA (Multi-Query Attention)
  - 1 < n_kv_heads < n_heads → GQA
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bitlinear import BitLinear


class CrossAttention(nn.Module):
    """Grouped-Query Cross-Attention with BitLinear + FlashAttention

    디코더 hidden state에서 Q를 생성하고 (n_heads개),
    인코더 output에서 K, V를 생성 (n_kv_heads개, repeat로 확장).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        # GQA: n_kv_heads가 None이면 MHA (n_kv_heads == n_heads)
        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert d_model % n_heads == 0, \
            f"d_model({d_model}) must be divisible by n_heads({n_heads})"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads({n_heads}) must be divisible by n_kv_heads({n_kv_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 각 KV 헤드가 반복되는 횟수
        self.d_head = d_model // n_heads

        # Q: 디코더 hidden → n_heads개 query head
        self.q_proj = BitLinear(d_model, n_heads * self.d_head)
        # K, V: 인코더 output → n_kv_heads개 key/value head
        self.k_proj = BitLinear(d_model, n_kv_heads * self.d_head)
        self.v_proj = BitLinear(d_model, n_kv_heads * self.d_head)
        # Output projection
        self.o_proj = BitLinear(n_heads * self.d_head, d_model)

        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """KV 헤드를 n_rep번 반복하여 n_heads개로 확장

        (B, n_kv_heads, seq_len, d_head) → (B, n_heads, seq_len, d_head)
        """
        if self.n_rep == 1:
            return x
        B, n_kv, seq_len, d_head = x.shape
        # (B, n_kv, 1, seq_len, d_head) → (B, n_kv, n_rep, seq_len, d_head)
        x = x.unsqueeze(2).expand(B, n_kv, self.n_rep, seq_len, d_head)
        return x.reshape(B, self.n_heads, seq_len, d_head)

    def forward(
        self,
        x: torch.Tensor,            # decoder hidden: (B, tgt_len, d_model)
        encoder_out: torch.Tensor,   # encoder output: (B, src_len, d_model)
        encoder_mask: torch.Tensor | None = None,  # (B, src_len) bool
    ) -> torch.Tensor:
        """
        Args:
            x: 디코더 hidden state (B, tgt_len, d_model)
            encoder_out: 인코더 출력 (B, src_len, d_model)
            encoder_mask: 인코더 패딩 마스크 (True = 유효, False = 패딩)

        Returns:
            (B, tgt_len, d_model)
        """
        B, tgt_len, _ = x.shape
        _, src_len, _ = encoder_out.shape

        # Q/K/V projection
        Q = self.q_proj(x)           # (B, tgt_len, n_heads * d_head)
        K = self.k_proj(encoder_out) # (B, src_len, n_kv_heads * d_head)
        V = self.v_proj(encoder_out) # (B, src_len, n_kv_heads * d_head)

        # Reshape to multi-head: (B, n_heads/n_kv_heads, seq_len, d_head)
        Q = Q.view(B, tgt_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, src_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = V.view(B, src_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        # GQA: KV 헤드 반복 → n_heads 크기로 확장
        K = self._repeat_kv(K)  # (B, n_heads, src_len, d_head)
        V = self._repeat_kv(V)  # (B, n_heads, src_len, d_head)

        # 마스크 변환: SDPA는 (B, 1, tgt_len, src_len) 또는 (B, 1, 1, src_len) 형태 필요
        attn_mask = None
        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                # (B, src_len) → (B, 1, 1, src_len) — 전체 헤드/쿼리에 브로드캐스트
                attn_mask = encoder_mask.unsqueeze(1).unsqueeze(1)

        # FlashAttention via scaled_dot_product_attention
        # - FlashAttention-2 커널: Ampere+ GPU에서 자동 활성화
        # - CPU/이전 GPU에서는 math 백엔드로 fallback
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # cross-attention은 causal이 아님
        )  # (B, n_heads, tgt_len, d_head)

        # Concat heads
        out = out.transpose(1, 2).contiguous().view(B, tgt_len, self.d_model)

        return self.o_proj(out)

    def extra_repr(self) -> str:
        mode = "MHA" if self.n_rep == 1 else f"GQA(rep={self.n_rep})"
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, d_head={self.d_head}, {mode}"
        )
