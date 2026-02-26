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

        # Linear Cross-Attention
        # Lazy import to avoid circular dependency if needed, but safe here
        from model.linear_attention import LinearCrossAttention
        # MQA (n_kv_heads=1) saves memory but has a very small context matrix (d_head x d_head) 
        # which heavily bottlenecks long sequences (e.g. 4096).
        # We default to MHA (n_kv_heads=n_heads) to increase capacity linearly with heads.
        n_heads = d_model // 64
        self.cross_attn = LinearCrossAttention(
            d_model=d_model,
            n_heads=n_heads,         # e.g., 4 for 8M model
            n_kv_heads=n_heads,      # Default to MHA for larger Context Matrix capacity
            dropout=dropout,
        )
        self.norm_cross = RMSNorm(d_model, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — encoder_out + tgt가 concat된 시퀀스
            encoder_out: (batch, src_len, d_model) - 인코더 파생 Context
            encoder_mask: (batch, src_len) - 패딩 마스크

        Returns:
            (batch, seq_len, d_model)
        """
        # 1. Mamba + residual
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        # 2. Linear Cross Attention + residual
        if encoder_out is not None:
            residual = x
            x = self.cross_attn(x, encoder_out, encoder_mask)
            x = self.dropout(x)
            x = self.norm_cross(residual + x)

        # 3. FFN + residual
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

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor | None = None,
        encoder_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — concat[encoder_out, tgt_emb]
            encoder_out: 인코더 문맥 (B, src_len, d)
            encoder_mask: 인코더 패딩 (B, src_len)

        Returns:
            (batch, seq_len, d_model) — 전체 시퀀스, seq2seq.py에서 tgt 부분만 슬라이스
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False args must be passed explicitly via kwargs for checkpoint
                x = checkpoint(layer, x, encoder_out, encoder_mask, use_reentrant=False)
            else:
                x = layer(x, encoder_out=encoder_out, encoder_mask=encoder_mask)
        return x
