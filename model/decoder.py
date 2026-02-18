"""Decoder — Mamba SSM + Cross-Attention + BitNet FFN 디코더

디코더 구조 (per layer):
    x → Mamba → (+residual) → RMSNorm
      → Cross-Attention(encoder_out) → (+residual) → RMSNorm
      → BitNet FFN → (+residual) → RMSNorm
"""
import torch
import torch.nn as nn

from model.mamba_block import MambaBlock
from model.cross_attention import CrossAttention
from model.encoder import RMSNorm, BitNetFFN


class DecoderLayer(nn.Module):
    """디코더 레이어: Mamba → RMSNorm → CrossAttention → RMSNorm → BitNet FFN → RMSNorm"""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        dt_rank: int,
        d_ff: int,
        n_heads: int,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Mamba SSM
        self.mamba = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=dt_rank,
        )
        self.norm1 = RMSNorm(d_model, eps=rms_norm_eps)

        # Cross-Attention (디코더 → 인코더)
        self.cross_attn = CrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(d_model, eps=rms_norm_eps)

        # BitNet FFN
        self.ffn = BitNetFFN(d_model, d_ff, dropout=dropout)
        self.norm3 = RMSNorm(d_model, eps=rms_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model)
            encoder_out: (batch, src_len, d_model)
            encoder_mask: (batch, src_len) — True=유효, False=패딩

        Returns:
            (batch, tgt_len, d_model)
        """
        # 1. Mamba + residual
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        # 2. Cross-Attention + residual
        residual = x
        x = self.cross_attn(x, encoder_out, encoder_mask)
        x = self.dropout(x)
        x = self.norm2(residual + x)

        # 3. FFN + residual
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(residual + x)

        return x


class Decoder(nn.Module):
    """N-layer 디코더 스택"""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        dt_rank: int,
        d_ff: int,
        n_heads: int,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                dt_rank=dt_rank,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout=dropout,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, d_model)  — 임베딩 출력
            encoder_out: (batch, src_len, d_model)
            encoder_mask: (batch, src_len)

        Returns:
            (batch, tgt_len, d_model)  — 디코더 최종 출력
        """
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_mask)
        return x
