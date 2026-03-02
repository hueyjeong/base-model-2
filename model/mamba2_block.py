"""Mamba-2 SSD Block — mamba_ssm.Mamba2 래핑

Mamba-2 (Structured State Space Duality) 기반 시퀀스 모델링.
Mamba-1의 sequential scan을 chunk-parallel scan으로 교체하여 
GPU Tensor Core를 직접 활용, 대폭적인 속도 향상.

기존 MambaBlock과 동일한 인터페이스:
    forward(x, reset_mask=None) → (B, L, d_model)

주요 차이점 (vs Mamba-1):
    - d_state: 16 → 128 (더 많은 정보 유지)
    - Head 구조: headdim=64 기반 multi-head SSM
    - Chunk-parallel scan: chunk_size=256 내 matmul 병렬 처리
    - Document isolation: seq_idx 네이티브 지원 (reset_mask → cumsum 변환)
    - Fused kernel: conv1d + scan + norm + out_proj 한번에 처리
"""
import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2Block(nn.Module):
    """Mamba-2 SSD Block

    mamba_ssm.Mamba2를 래핑하여 기존 MambaBlock과 동일한 인터페이스 제공.
    
    구조:
        x → Mamba2(in_proj → Conv1D → SSD scan → RMSNorm → out_proj) → output
        
    Document Isolation:
        reset_mask (B, L) bool → seq_idx (B, L) int 변환으로
        Mamba-2의 네이티브 document isolation 활용.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int = 128,
        d_conv: int = 4,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim

        expand = d_inner / d_model
        # expand must be an integer for Mamba2
        assert d_inner % d_model == 0 or abs(expand - round(expand)) < 1e-6, \
            f"d_inner({d_inner}) must be a multiple of d_model({d_model})"
        expand_int = round(expand)

        self.mamba2 = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_int,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            rmsnorm=True,
            bias=False,
        )

    def forward(self, x: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            reset_mask: (batch, seq_len) bool — True인 위치에서 SSM state 리셋
                        (BOS 위치를 표시, Mamba-2의 seq_idx로 변환)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_idx = None
        if reset_mask is not None:
            # reset_mask: True = 새 문서 시작 (BOS)
            # seq_idx: 같은 문서의 위치는 같은 정수값
            # cumsum으로 True가 나올 때마다 ID 증가 → 자동 문서별 고유 ID
            seq_idx = (reset_mask.int().cumsum(dim=1) - 1).to(torch.int32)

        return self.mamba2(x, seq_idx=seq_idx)

    def extra_repr(self) -> str:
        nheads = self.d_inner // self.headdim
        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, d_conv={self.d_conv}, "
            f"headdim={self.headdim}, nheads={nheads}"
        )
