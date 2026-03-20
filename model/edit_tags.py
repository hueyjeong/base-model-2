"""편집 태그 시스템

Levenshtein 정렬을 사용하여 소스 → 타겟 간 편집 태그를 생성하고 적용한다.

태그 ID 체계 (vocab_size = V):
    KEEP      = 0
    DELETE    = 1
    REPLACE_x = 2 .. V+1        (V개, x는 대체할 토큰 ID)
    INSERT_x  = V+2 .. 2V+1     (V개, x는 삽입할 토큰 ID)

총 태그 수: 2 + 2V
"""
from __future__ import annotations

import torch


# 태그 상수
TAG_KEEP = 0
TAG_DELETE = 1


def tag_replace(token_id: int, vocab_size: int) -> int:
    """REPLACE_x 태그 ID 반환"""
    return 2 + token_id


def tag_insert(token_id: int, vocab_size: int) -> int:
    """INSERT_x 태그 ID 반환"""
    return 2 + vocab_size + token_id


def tag_to_op(tag_id: int, vocab_size: int) -> tuple[str, int]:
    """태그 ID → (연산, 토큰 ID) 변환

    Returns:
        ("keep", -1) | ("delete", -1) | ("replace", token_id) | ("insert", token_id)
    """
    if tag_id == TAG_KEEP:
        return ("keep", -1)
    elif tag_id == TAG_DELETE:
        return ("delete", -1)
    elif tag_id < 2 + vocab_size:
        return ("replace", tag_id - 2)
    else:
        return ("insert", tag_id - 2 - vocab_size)


def compute_edit_tags(
    source_ids: list[int],
    target_ids: list[int],
    vocab_size: int,
) -> list[int]:
    """Levenshtein DP + backtrace → 소스 위치별 편집 태그 생성

    다중 삽입(한 위치에 여러 토큰 삽입)은 첫 번째만 INSERT로 기록.
    나머지는 iterative refinement에서 처리.

    Args:
        source_ids: 소스 토큰 ID 시퀀스
        target_ids: 타겟 토큰 ID 시퀀스
        vocab_size: 어휘 크기

    Returns:
        소스 길이만큼의 편집 태그 리스트
    """
    n = len(source_ids)
    m = len(target_ids)

    # Levenshtein DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source_ids[i - 1] == target_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1],  # replace
                )

    # Backtrace → 연산 목록
    ops: list[tuple[str, int, int]] = []  # (op, src_idx, tgt_token)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source_ids[i - 1] == target_ids[j - 1]:
            ops.append(("match", i - 1, -1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, target_ids[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", i, target_ids[j - 1]))  # i = 삽입 직전 소스 위치
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            # 도달 불가능하지만 안전 가드
            break

    ops.reverse()

    # 연산 → 태그 변환
    tags = [TAG_KEEP] * n
    insert_used = set()  # 이미 INSERT가 할당된 소스 위치

    for op, src_idx, tgt_token in ops:
        if op == "match":
            tags[src_idx] = TAG_KEEP
        elif op == "sub":
            tags[src_idx] = tag_replace(tgt_token, vocab_size)
        elif op == "del":
            tags[src_idx] = TAG_DELETE
        elif op == "ins":
            # 삽입 위치: src_idx 직전 위치(src_idx-1)에 INSERT 태그
            # 다중 삽입은 첫 번째만 기록
            ins_at = max(0, src_idx - 1)
            if ins_at not in insert_used and tags[ins_at] == TAG_KEEP:
                tags[ins_at] = tag_insert(tgt_token, vocab_size)
                insert_used.add(ins_at)

    return tags


def apply_edit_tags(
    source_ids: list[int],
    tags: list[int],
    vocab_size: int,
) -> list[int]:
    """편집 태그를 적용하여 수정된 토큰 시퀀스 생성

    Args:
        source_ids: 소스 토큰 ID 시퀀스
        tags: 소스 길이만큼의 편집 태그 리스트
        vocab_size: 어휘 크기

    Returns:
        수정된 토큰 ID 시퀀스
    """
    result = []
    for i, (src_tok, tag) in enumerate(zip(source_ids, tags)):
        op, tok_id = tag_to_op(tag, vocab_size)
        if op == "keep":
            result.append(src_tok)
        elif op == "delete":
            pass  # 토큰 삭제
        elif op == "replace":
            result.append(tok_id)
        elif op == "insert":
            result.append(src_tok)   # 원본 유지
            result.append(tok_id)    # 삽입 토큰 추가
    return result


def compute_edit_tags_batch(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    vocab_size: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """배치 단위 편집 태그 계산

    Args:
        source_ids: (B, src_len) — 패딩 포함
        target_ids: (B, tgt_len) — 패딩 포함
        vocab_size: 어휘 크기
        pad_id: 패딩 토큰 ID

    Returns:
        (B, src_len) — 편집 태그 (PAD 위치는 TAG_KEEP)
    """
    B, src_len = source_ids.shape
    tags_batch = torch.full((B, src_len), TAG_KEEP, dtype=torch.long,
                            device=source_ids.device)

    for b in range(B):
        # PAD 제거
        src = source_ids[b].tolist()
        tgt = target_ids[b].tolist()
        src_valid = [t for t in src if t != pad_id]
        tgt_valid = [t for t in tgt if t != pad_id]

        if not src_valid:
            continue

        tags = compute_edit_tags(src_valid, tgt_valid, vocab_size)

        # 유효 위치에 태그 할당
        valid_idx = 0
        for i in range(src_len):
            if src[i] != pad_id and valid_idx < len(tags):
                tags_batch[b, i] = tags[valid_idx]
                valid_idx += 1

    return tags_batch


if __name__ == "__main__":
    # 기본 테스트
    vocab_size = 303

    # 테스트 1: 동일한 시퀀스 → all KEEP
    src = [1, 5, 10, 20]
    tgt = [1, 5, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    assert all(t == TAG_KEEP for t in tags), f"동일 시퀀스 KEEP 실패: {tags}"
    result = apply_edit_tags(src, tags, vocab_size)
    assert result == tgt, f"동일 시퀀스 roundtrip 실패: {result} != {tgt}"
    print("[PASS] 동일 시퀀스 → all KEEP")

    # 테스트 2: 치환
    src = [1, 5, 10, 20]
    tgt = [1, 99, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    result = apply_edit_tags(src, tags, vocab_size)
    assert result == tgt, f"치환 roundtrip 실패: {result} != {tgt}"
    print(f"[PASS] 치환: tags={tags}")

    # 테스트 3: 삭제
    src = [1, 5, 10, 20]
    tgt = [1, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    result = apply_edit_tags(src, tags, vocab_size)
    assert result == tgt, f"삭제 roundtrip 실패: {result} != {tgt}"
    print(f"[PASS] 삭제: tags={tags}")

    # 테스트 4: 삽입 (단일)
    src = [1, 10, 20]
    tgt = [1, 5, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    result = apply_edit_tags(src, tags, vocab_size)
    assert result == tgt, f"삽입 roundtrip 실패: {result} != {tgt}"
    print(f"[PASS] 삽입: tags={tags}")

    # 테스트 5: 배치
    source = torch.tensor([[1, 5, 10, 0], [1, 10, 20, 0]])
    target = torch.tensor([[1, 99, 10, 0], [1, 10, 0, 0]])
    tags_b = compute_edit_tags_batch(source, target, vocab_size, pad_id=0)
    print(f"[PASS] 배치 태그: shape={tags_b.shape}, tags={tags_b}")

    print("\n모든 편집 태그 테스트 통과!")
