"""리니어 어텐션 feature map

φ(x) = gelu(x) + 1 — 양수 보장 특징 매핑.
SharedLinearSelfAttention에서 사용.
"""
import torch
import torch.nn.functional as F


def gelu1p_feature_map(x: torch.Tensor) -> torch.Tensor:
    """리니어 어텐션을 위한 양수 보장 특징 매핑: phi(x) = gelu(x) + 1

    relu+1과 달리 x < 0 영역에서도 smooth gradient가 전달되어
    attention weight 학습이 더 원활합니다.
    출력 하한 ≈ 0.83 (GELU 최솟값 ≈ -0.17 at x ≈ -0.75) → 양수 보장.
    """
    return F.gelu(x) + 1.0
