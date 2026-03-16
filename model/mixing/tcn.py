"""TCN (Temporal Convolutional Network) Mixing Layer

Non-causal dilated depthwise-separable convolution으로 토큰 믹싱.
인코더용이므로 symmetric (non-causal) padding 사용.
O(T × kernel_size × n_dilations) 복잡도, 완전 병렬화.

CPU 최적: depthwise conv는 MKL-DNN이 AVX2/VNNI 최적화,
각 채널 독립으로 캐시 친화적.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.mixing.base import MixingLayer
from model.bitlinear import BitLinear


class TCNMixing(MixingLayer):
    """Non-causal dilated depthwise conv + pointwise BitLinear

    구조:
        [depthwise_conv(dilation=2^i) for i in range(n_dilations)]
        → 합산 → BitLinear pointwise projection

    Receptive field = kernel_size × sum(dilations)
    k=7, n_dilations=6: RF = 7 × 63 = 441 per layer
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        k = cfg.tcn_kernel_size
        n_dil = cfg.tcn_n_dilations

        # Depthwise dilated convolutions
        self.convs = nn.ModuleList()
        for i in range(n_dil):
            dilation = 2 ** i
            # symmetric padding for non-causal
            padding = (k - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(d, d, k, padding=padding, dilation=dilation, groups=d, bias=False)
            )

        # Pointwise projection (ternary)
        self.proj = BitLinear(d, d)

    def forward(self, x: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        # x: (B, T, d) → (B, d, T) for Conv1d
        h = x.transpose(1, 2)

        # 다중 dilation 합산
        out = self.convs[0](h)
        for conv in self.convs[1:]:
            c = conv(h)
            # 패딩으로 인해 길이가 다를 수 있으므로 맞춤
            if c.size(2) != out.size(2):
                c = c[:, :, :out.size(2)]
            out = out + c

        out = F.relu(out)

        # (B, d, T) → (B, T, d)
        out = out.transpose(1, 2)

        # Pointwise projection
        out = self.proj(out)

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
