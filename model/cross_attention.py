"""Cross-Attention — 디코더 → 인코더 어텐션

Multi-head cross-attention으로 디코더가 인코더 출력을 참조한다.
Q/K/V/O projection에 BitLinear를 사용하여 1.58-bit 양자화 적용.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bitlinear import BitLinear


class CrossAttention(nn.Module):
    """Multi-Head Cross-Attention with BitLinear projections

    디코더 hidden state에서 Q를 생성하고,
    인코더 output에서 K, V를 생성하여 attention을 수행한다.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model({d_model}) must be divisible by n_heads({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Q: 디코더 hidden → query
        self.q_proj = BitLinear(d_model, d_model)
        # K, V: 인코더 output → key, value
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        # Output projection
        self.o_proj = BitLinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # decoder hidden: (B, tgt_len, d_model)
        encoder_out: torch.Tensor,   # encoder output: (B, src_len, d_model)
        encoder_mask: torch.Tensor | None = None,  # (B, 1, 1, src_len) or (B, src_len)
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
        Q = self.q_proj(x)                  # (B, tgt_len, d_model)
        K = self.k_proj(encoder_out)        # (B, src_len, d_model)
        V = self.v_proj(encoder_out)        # (B, src_len, d_model)

        # Multi-head reshape: (B, n_heads, seq_len, d_head)
        Q = Q.view(B, tgt_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, src_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, src_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, tgt, src)

        # 마스크 적용
        if encoder_mask is not None:
            if encoder_mask.dim() == 2:
                # (B, src_len) → (B, 1, 1, src_len)
                encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~encoder_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, V)  # (B, H, tgt, d_head)
        out = out.transpose(1, 2).contiguous().view(B, tgt_len, self.d_model)

        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}"
        )
