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

    모든 mixing layer는 동일한 입출력 형태를 가진다:
        입력: (B, T, d_model)
        출력: (B, T, d_model)

    pad_mask: (B, T) bool — True가 유효 데이터. PAD 위치의 출력을 0으로 마스킹.
    """

    @abstractmethod
    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        ...
