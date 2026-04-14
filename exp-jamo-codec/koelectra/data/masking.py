"""ELECTRA MLM 마스킹: BBPE 패치 단위 20% 마스킹.

BBPEJamoDataset은 `jamo_ids[B,L]`, `segment_ids[B,L]`, `n_segments[B]`를 반환한다.
이 모듈은 배치 단위로 패치(BBPE 토큰) 20%를 무작위 선택해 해당 패치에 속하는
모든 자모 위치를 JAMO_MASK(=4)로 치환한다.
"""
from __future__ import annotations

import torch

# JamoTokenizer의 MASK special id (tok/jamo_tokenizer.py 참조)
JAMO_MASK_ID = 4
JAMO_PAD_ID = 0


def make_patch_mask(n_segments: torch.Tensor, max_patches: int,
                    mask_ratio: float = 0.20,
                    min_masked: int = 1,
                    special_patch_mask: torch.Tensor | None = None,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """배치별 유효 패치 중 mask_ratio 만큼 균등 샘플링.

    Args:
        n_segments: [B] 배치별 유효 패치 수
        max_patches: 시퀀스 패치 차원 크기 P
        mask_ratio: 마스킹 비율 (special 제외 후의 "유효" 패치에 대한 비율)
        min_masked: 최소 마스크 수
        special_patch_mask: [B, P] bool, True=special 토큰(BOS/EOS/SEP) → 마스킹 대상 제외
        generator: 재현성을 위한 torch.Generator (옵션)

    Returns:
        masked_patch_mask: [B, P] bool, True=마스킹 대상
    """
    B = n_segments.size(0)
    device = n_segments.device
    # 패딩 패치는 제외 → 유효 범위 내에서 상위-k 샘플링
    pos = torch.arange(max_patches, device=device).unsqueeze(0).expand(B, -1)  # [B,P]
    valid = pos < n_segments.unsqueeze(-1)  # [B,P]
    if special_patch_mask is not None:
        valid = valid & ~special_patch_mask  # special은 마스킹 대상 아님

    # 랜덤 점수 생성, padding/special 위치엔 -inf → 상위-k에서 배제
    if generator is not None:
        scores = torch.rand(B, max_patches, generator=generator, device=device)
    else:
        scores = torch.rand(B, max_patches, device=device)
    scores = scores.masked_fill(~valid, -1.0)

    # 배치별 마스크 개수 — 유효(non-special) 패치 수 기준
    valid_count = valid.sum(dim=1).float()  # [B]
    k = (valid_count * mask_ratio).round().clamp(min=min_masked).long()  # [B]
    # 모든 샘플에서 같은 수의 topk를 취하기 위해 max_k 로 뽑고, 뒤에서 자름
    max_k = int(k.max().item()) if k.numel() > 0 else 0
    if max_k == 0:
        return torch.zeros(B, max_patches, dtype=torch.bool, device=device)

    _, topk_idx = scores.topk(max_k, dim=1)  # [B, max_k]
    rank = torch.arange(max_k, device=device).unsqueeze(0).expand(B, -1)  # [B, max_k]
    keep = rank < k.unsqueeze(-1)  # [B, max_k]

    masked = torch.zeros(B, max_patches, dtype=torch.bool, device=device)
    masked.scatter_(1, topk_idx, keep)
    return masked


def apply_mask(jamo_ids: torch.Tensor, segment_ids: torch.Tensor,
               jamo_mask: torch.Tensor, masked_patch_mask: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """마스킹 패치에 속한 자모 위치를 JAMO_MASK로 치환.

    Args:
        jamo_ids: [B, L] 원본 자모 ID
        segment_ids: [B, L] 각 자모의 패치 ID
        jamo_mask: [B, L] 유효 자모 여부
        masked_patch_mask: [B, P] True=해당 패치 전체 마스킹

    Returns:
        masked_jamo_ids: [B, L] JAMO_MASK로 치환된 사본
        per_jamo_mask: [B, L] True=이 위치가 마스킹 패치에 속함 & 유효
    """
    # gather: segment_ids가 가리키는 패치의 mask 값을 자모 위치로 펼침
    per_jamo_mask = masked_patch_mask.gather(1, segment_ids)  # [B, L]
    per_jamo_mask = per_jamo_mask & jamo_mask  # 유효 위치만

    masked_jamo_ids = torch.where(
        per_jamo_mask,
        torch.full_like(jamo_ids, JAMO_MASK_ID),
        jamo_ids,
    )
    return masked_jamo_ids, per_jamo_mask


def scatter_any_per_patch(per_jamo_flag: torch.Tensor, segment_ids: torch.Tensor,
                          max_patches: int) -> torch.Tensor:
    """자모 단위 bool 플래그를 패치 단위 OR로 축약 ([B,L] → [B,P]).

    "이 패치에 속한 자모 중 하나라도 True면 패치 True."

    Args:
        per_jamo_flag: [B, L] bool
        segment_ids: [B, L] 각 자모의 패치 ID
        max_patches: 출력 차원 P

    Returns:
        per_patch_flag: [B, P] bool
    """
    B, _ = per_jamo_flag.shape
    out = torch.zeros(B, max_patches, dtype=torch.bool, device=per_jamo_flag.device)
    # int로 scatter_add_ 후 >0 판정 (bool scatter_add_ 미지원)
    out_int = torch.zeros(B, max_patches, dtype=torch.int32, device=per_jamo_flag.device)
    out_int.scatter_add_(1, segment_ids, per_jamo_flag.to(torch.int32))
    return out_int > 0


if __name__ == "__main__":
    print("=== masking smoke test ===")
    B, L, P = 2, 15, 5
    jamo_ids = torch.randint(10, 330, (B, L))
    jamo_mask = torch.ones(B, L, dtype=torch.bool)
    # segment_ids: [0]*5 + [1]*3 + [2]*7  → 3 패치만 유효
    segment_ids = torch.tensor([[0]*5 + [1]*3 + [2]*7] * B, dtype=torch.long)
    n_segments = torch.tensor([3, 3])

    masked_patch_mask = make_patch_mask(n_segments, max_patches=P, mask_ratio=0.5)
    print("masked_patch_mask:", masked_patch_mask)
    assert (masked_patch_mask & (torch.arange(P) >= n_segments.unsqueeze(-1))).sum() == 0, \
        "padding 패치가 마스킹됨"

    masked_ids, per_jamo_mask = apply_mask(jamo_ids, segment_ids, jamo_mask, masked_patch_mask)
    print("per_jamo_mask[0]:", per_jamo_mask[0])
    print("masked_ids[0][per_jamo]:", masked_ids[0][per_jamo_mask[0]])
    assert (masked_ids[per_jamo_mask] == JAMO_MASK_ID).all()

    # scatter_any 테스트
    diff = torch.zeros(B, L, dtype=torch.bool)
    diff[0, 6] = True  # 패치 1에 속함
    per_patch = scatter_any_per_patch(diff, segment_ids, P)
    print("per_patch[0]:", per_patch[0])
    assert per_patch[0, 1].item() is True and per_patch[0, 0].item() is False

    print("OK")
