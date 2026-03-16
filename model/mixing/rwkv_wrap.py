"""BiRWKV Mixing Layer 래퍼

기존 BiRWKV 구현을 MixingLayer ABC로 래핑한다.
"""
import torch
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bi_rwkv import BiRWKV


class BiRWKVMixing(MixingLayer):
    """기존 BiRWKV를 MixingLayer 인터페이스로 래핑

    BiRWKV는 forward+backward RWKV-6를 element-wise addition으로 융합.
    """

    def __init__(self, cfg):
        super().__init__()
        self.bi_rwkv = BiRWKV(cfg.d_model, cfg.n_heads, cfg.headdim)

    def _init_weights(self):
        self.bi_rwkv._init_weights()

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        return self.bi_rwkv(x, pad_mask=pad_mask, reset_mask=None)
