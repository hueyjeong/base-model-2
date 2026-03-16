"""Mixing Layer 추상 베이스 클래스

모든 토큰 믹싱 레이어는 이 ABC를 상속한다.
인터페이스: (B, T, d_model) → (B, T, d_model)
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class MixingLayer(nn.Module, ABC):
    """교체 가능한 토큰 믹싱 레이어 추상 클래스

    Args:
        x: (B, T, d_model) 입력
        pad_mask: (B, T) bool — True가 유효 데이터
        reset_mask: (B, T) bool — True = BOS 위치 (패킹 시 문서 경계 state 리셋)
    """

    @abstractmethod
    def forward(self, x: Tensor, pad_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> Tensor:
        ...
