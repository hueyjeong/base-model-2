"""FNet Mixing Layer — FFT 기반 토큰 믹싱

학습 가능한 파라미터 없음. 시퀀스 차원에 1D FFT를 적용하여 전역 토큰 믹싱 수행.
O(T log T) 복잡도, 완전 병렬화 가능, 양방향.

참고: FNet (Lee-Thorp et al., 2021) — "FNet: Mixing Tokens with Fourier Transforms"
원본은 2D FFT (seq + feature)이나, 시퀀스 차원만 적용하는 것이 안정적.
"""
import torch
from torch import Tensor

from model.mixing.base import MixingLayer


class FNetMixing(MixingLayer):
    """FFT 기반 토큰 믹싱 — 파라미터 없음

    시퀀스 차원에 대해 1D real FFT → 실수부 추출.
    d_ff를 2048로 키워서 FFN에서 파라미터 보충.
    """

    def __init__(self, cfg):
        super().__init__()

    def forward(self, x: Tensor, pad_mask: Tensor | None = None, reset_mask: Tensor | None = None) -> Tensor:
        # (B, T, d) — 시퀀스 차원(dim=1)에 1D FFT
        # FP32로 변환 후 FFT (bf16에서 FFT 불안정)
        orig_dtype = x.dtype
        x_f32 = x.float()
        # 2D FFT (seq + feature) → 실수부 추출 (원본 FNet)
        x_freq = torch.fft.fft2(x_f32, dim=(-2, -1))
        out = x_freq.real.to(orig_dtype)

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(out.dtype)

        return out
