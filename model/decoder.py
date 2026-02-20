"""Decoder — Mamba SSM + BitNet FFN 디코더 (Cross-Attention 없음)

디코더 구조 (per layer):
    x → Mamba → (+residual) → RMSNorm
      → BitNet FFN → (+residual) → RMSNorm

Cross-Attention 대신 encoder 출력을 decoder 입력에 concat하여
Mamba의 recurrent state를 통해 encoder 정보를 전달.
seq2seq.py에서 concat/slice 처리.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.mamba_block import MambaBlock
from model.encoder import RMSNorm, BitNetFFN


class DecoderLayer(nn.Module):
    """디코더 레이어: Mamba → RMSNorm → BitNet FFN → RMSNorm"""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        dt_rank: int,
        d_ff: int,
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

        # BitNet FFN
        self.ffn = BitNetFFN(d_model, d_ff, dropout=dropout)
        self.norm2 = RMSNorm(d_model, eps=rms_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — encoder_out + tgt가 concat된 시퀀스

        Returns:
            (batch, seq_len, d_model)
        """
        # 1. Mamba + residual
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        # 2. FFN + residual
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)

        return x


class Decoder(nn.Module):
    """N-layer 디코더 스택 (Cross-Attention 없음, Mamba concat 방식)"""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        dt_rank: int,
        d_ff: int,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                d_inner=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                dt_rank=dt_rank,
                d_ff=d_ff,
                dropout=dropout,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — concat[encoder_out, tgt_emb]

        Returns:
            (batch, seq_len, d_model) — 전체 시퀀스, seq2seq.py에서 tgt 부분만 슬라이스
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
