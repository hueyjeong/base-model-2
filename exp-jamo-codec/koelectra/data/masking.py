"""ELECTRA MLM 마스킹: per-token 포맷.

BBPETokenDataset 이 반환하는 shape:
  jamo_ids [B, P, S], jamo_mask [B, P, S], token_pad_mask [B, P], special_token_mask [B, P], n_tokens [B]

본 모듈은 배치 단위로 토큰 P 개 중 mask_ratio 를 무작위 선택해 해당 토큰에 속한
모든 자모 위치를 JAMO_MASK(=4) 로 치환한다. segment_ids / gather 불필요.
"""
from __future__ import annotations

import torch


# JamoTokenizer specials
JAMO_MASK_ID = 4
JAMO_PAD_ID = 0


def make_patch_mask(
    n_tokens: torch.Tensor,          # [B]
    max_patches: int,
    mask_ratio: float = 0.20,
    min_masked: int = 1,
    special_patch_mask: torch.Tensor | None = None,  # [B, P]
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """배치별 유효 토큰 중 mask_ratio 만큼 균등 샘플링.

    Returns:
        masked_patch_mask: [B, P] bool, True=마스킹 대상
    """
    B = n_tokens.size(0)
    device = n_tokens.device
    pos = torch.arange(max_patches, device=device).unsqueeze(0).expand(B, -1)
    valid = pos < n_tokens.unsqueeze(-1)  # [B, P]
    if special_patch_mask is not None:
        valid = valid & ~special_patch_mask

    if generator is not None:
        scores = torch.rand(B, max_patches, generator=generator, device=device)
    else:
        scores = torch.rand(B, max_patches, device=device)
    scores = scores.masked_fill(~valid, -1.0)

    valid_count = valid.sum(dim=1).float()
    k = (valid_count * mask_ratio).round().clamp(min=min_masked).long()
    max_k = int(k.max().item()) if k.numel() > 0 else 0
    if max_k == 0:
        return torch.zeros(B, max_patches, dtype=torch.bool, device=device)

    _, topk_idx = scores.topk(max_k, dim=1)
    rank = torch.arange(max_k, device=device).unsqueeze(0).expand(B, -1)
    keep = rank < k.unsqueeze(-1)

    masked = torch.zeros(B, max_patches, dtype=torch.bool, device=device)
    masked.scatter_(1, topk_idx, keep)
    return masked


def apply_mask(
    jamo_ids: torch.Tensor,          # [B, P, S]
    jamo_mask: torch.Tensor,         # [B, P, S] bool
    masked_patch_mask: torch.Tensor,  # [B, P] bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """마스킹 토큰의 전 슬롯을 JAMO_MASK 로 채움 (BBPE-level special 토큰 규칙과 통일).

    - masked 토큰: 32슬롯 모두 JAMO_MASK_ID, jamo_mask 도 전 슬롯 True
      → codec 이 "MASK 로 saturate 된 토큰" 으로 일관된 latent 산출
    - 비마스킹 토큰: 원본 유지

    Returns:
        masked_jamo_ids: [B, P, S] — codec 입력용
        masked_jamo_mask: [B, P, S] — codec 입력용 (masked 토큰 전 슬롯 True)
        per_jamo_mask: [B, P, S] — Gen CE loss target 마스크 (masked 토큰의 실자모만)
    """
    patch2 = masked_patch_mask.unsqueeze(-1)  # [B, P, 1]
    masked_jamo_ids = torch.where(
        patch2,
        torch.full_like(jamo_ids, JAMO_MASK_ID),
        jamo_ids,
    )
    masked_jamo_mask = jamo_mask | patch2  # masked 토큰은 전 슬롯 valid
    per_jamo_mask = patch2 & jamo_mask     # loss target: 원본 실자모 위치만
    return masked_jamo_ids, masked_jamo_mask, per_jamo_mask


# ── smoke test ──
if __name__ == "__main__":
    print("=== masking smoke test ===")
    B, P, S = 2, 8, 5
    # 토큰 3개만 유효한 배치
    jamo_ids = torch.randint(10, 330, (B, P, S))
    jamo_mask = torch.zeros(B, P, S, dtype=torch.bool)
    jamo_mask[:, :3, :4] = True  # 첫 3 토큰 × 앞 4 자모 유효
    n_tokens = torch.tensor([3, 3])
    special = torch.zeros(B, P, dtype=torch.bool)
    special[:, 0] = True  # BOS
    special[:, 2] = True  # EOS

    masked_patch_mask = make_patch_mask(n_tokens, P, mask_ratio=0.5,
                                         special_patch_mask=special)
    print("masked_patch_mask:")
    print(masked_patch_mask)
    # special/ padding 에 True 없어야 함
    assert not (masked_patch_mask & special).any()
    pos = torch.arange(P).unsqueeze(0).expand(B, -1)
    padding = pos >= n_tokens.unsqueeze(-1)
    assert not (masked_patch_mask & padding).any()

    masked_ids, masked_mask, per_jamo = apply_mask(jamo_ids, jamo_mask, masked_patch_mask)
    # 마스킹된 토큰의 전 슬롯이 JAMO_MASK_ID 인지
    patch2 = masked_patch_mask.unsqueeze(-1).expand(-1, -1, S)
    assert (masked_ids[patch2] == JAMO_MASK_ID).all(), "masked 토큰 전 슬롯 MASK 가 아님"
    # 마스킹되지 않은 토큰은 원본 유지
    assert (masked_ids[~patch2] == jamo_ids[~patch2]).all()
    # masked_jamo_mask: masked 토큰 전슬롯 True
    assert masked_mask[patch2].all(), "masked 토큰 전슬롯 valid 가 아님"
    # per_jamo_mask: masked 토큰의 실자모 위치만 True
    expected = masked_patch_mask.unsqueeze(-1) & jamo_mask
    assert (per_jamo == expected).all()

    # token 단위 diff: diff.any(dim=-1) & masked_patch_mask
    sampled = jamo_ids.clone()
    sampled[0, 1, 0] = 999
    diff = ((sampled != jamo_ids) & per_jamo).any(dim=-1)
    replaced = diff & masked_patch_mask
    assert replaced.shape == (B, P)
    print("replaced[0]:", replaced[0])
    print("OK")
