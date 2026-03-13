"""양방향 RWKV (BiRWKV)

Forward + Backward RWKV-6를 결합한 양방향 SSM.
Vision Mamba (Vim, ICML 2024) 표준에 따라 element-wise addition으로 융합.

구조:
    forward_rwkv: 좌→우 스캔
    backward_rwkv: 입력 flip → 스캔 → 출력 flip
    융합: addition (concat 대비 파라미터 절감, 성능 동등)
"""
import torch
import torch.nn as nn

from model.rwkv_block import RWKV6TimeMix


class BiRWKV(nn.Module):
    """양방향 RWKV-6 블록

    독립적인 forward/backward RWKV-6 가중치를 사용하여
    양방향 문맥을 포착한다.

    Args:
        d_model: 히든 차원
        n_heads: RWKV head 수
        headdim: head 차원
    """

    def __init__(self, d_model: int, n_heads: int, headdim: int):
        super().__init__()
        self.forward_rwkv = RWKV6TimeMix(d_model, n_heads, headdim)
        self.backward_rwkv = RWKV6TimeMix(d_model, n_heads, headdim)

    def _init_weights(self):
        """가중치 초기화 전파"""
        self.forward_rwkv._init_weights()
        self.backward_rwkv._init_weights()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            pad_mask: (B, T) bool — True가 유효 데이터 (PAD 위치 처리용)
        Returns:
            (B, T, d_model)
        """
        # Forward 방향: 좌 → 우
        fwd_out = self.forward_rwkv(x)

        # Backward 방향: 우 → 좌 (시간 축 역순)
        x_rev = x.flip(1)
        bwd_out = self.backward_rwkv(x_rev)
        bwd_out = bwd_out.flip(1)

        # 융합: element-wise addition
        out = fwd_out + bwd_out

        # PAD 마스킹 (선택적)
        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
