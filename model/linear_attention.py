"""리니어 어텐션 feature map

φ(x) = relu(x) + 1 — 양수 보장 특징 매핑.
SharedLinearSelfAttention에서 사용.
"""
import torch
import torch.nn.functional as F


def gelu1p_feature_map(x: torch.Tensor) -> torch.Tensor:
    """리니어 어텐션을 위한 양수 보장 특징 매핑: phi(x) = relu(x) + 1

    relu+1: CPU 인퍼런스에서 exp 연산 없이 빠름.
    출력 하한 = 1.0 → 양수 보장. denominator 폭발 방지.
    """
    return F.relu(x) + 1.0
