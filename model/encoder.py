"""Encoder — Mamba SSM + BitNet FFN 인코더

인코더 구조 (per layer):
    x → Mamba → (+residual) → RMSNorm → BitNet FFN → (+residual) → RMSNorm
"""
import torch
import torch.nn as nn

from model.mamba_block import MambaBlock
from model.bitlinear import BitLinear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class BitNetFFN(nn.Module):
    """BitNet Feed-Forward Network

    x → BitLinear(d_model → d_ff) → SiLU → BitLinear(d_ff → d_model)
    
    SwiGLU variant:
    x → [BitLinear(gate), BitLinear(up)] → gate*SiLU(up) → BitLinear(down)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)
        self.up_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class EncoderLayer(nn.Module):
    """인코더 레이어: Mamba → RMSNorm → BitNet FFN → RMSNorm"""

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
        self.mamba = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=dt_rank,
        )
        self.norm1 = RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn = BitNetFFN(d_model, d_ff, dropout=dropout)
        self.norm2 = RMSNorm(d_model, eps=rms_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Mamba + residual
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        # FFN + residual
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)

        return x


class Encoder(nn.Module):
    """N-layer 인코더 스택"""

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
        self.layers = nn.ModuleList([
            EncoderLayer(
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
            x: (batch, seq_len, d_model)  — 임베딩 출력
        Returns:
            (batch, seq_len, d_model)  — 인코더 최종 출력
        """
        for layer in self.layers:
            x = layer(x)
        return x
