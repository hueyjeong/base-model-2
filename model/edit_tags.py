"""편집 태그 시스템

Levenshtein 정렬을 사용하여 소스 → 타겟 간 편집 태그를 생성하고 적용한다.

태그 ID 체계 (vocab_size = V):
    KEEP         = 0
    DELETE       = 1
    REPLACE_x    = 2 .. V+1        (V개, x는 대체할 토큰 ID)
    INSERT_x     = V+2 .. 2V+1     (V개, x는 삽입할 토큰 ID) — 레거시 호환
    INSERT_START = V+2             (하이브리드 모드: 디코더가 삽입 토큰 결정)

레거시 모드: n_tags = 2 + 2V (INSERT_x 개별 태그)
하이브리드 모드: n_tags = 2 + V + 1 = V + 3 (INSERT_START 단일 태그)
"""
from __future__ import annotations

import torch


# 태그 상수
TAG_KEEP = 0
TAG_DELETE = 1


def tag_insert_start(vocab_size: int) -> int:
    """INSERT_START 태그 ID (하이브리드 모드)"""
    return 2 + vocab_size


def n_tags_hybrid(vocab_size: int) -> int:
    """하이브리드 모드 태그 수: KEEP + DELETE + REPLACE_x(V) + INSERT_START"""
    return 3 + vocab_size


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


def compute_edit_tags_hybrid(
    source_ids: list[int],
    target_ids: list[int],
    vocab_size: int,
    eos_id: int = 3,
) -> tuple[list[int], dict[int, list[int]]]:
    """Levenshtein DP → 하이브리드 태그 (INSERT_START + 삽입 시퀀스)

    기존 compute_edit_tags와 동일한 DP/backtrace를 사용하되,
    INSERT_x 대신 INSERT_START 태그 + 위치별 삽입 시퀀스를 반환한다.
    한 위치에 여러 토큰 삽입이 필요하면 전부 해당 위치의 시퀀스에 포함.

    Args:
        source_ids: 소스 토큰 ID 시퀀스
        target_ids: 타겟 토큰 ID 시퀀스
        vocab_size: 어휘 크기
        eos_id: 삽입 시퀀스 종료 토큰

    Returns:
        tags: 소스 길이만큼의 태그 리스트 (KEEP/DELETE/REPLACE_x/INSERT_START)
        insert_seqs: {위치: [토큰1, 토큰2, ..., eos_id]} 삽입 시퀀스
    """
    INSERT_START = tag_insert_start(vocab_size)
    n = len(source_ids)
    m = len(target_ids)

    # Levenshtein DP (compute_edit_tags와 동일)
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
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    # Backtrace
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source_ids[i - 1] == target_ids[j - 1]:
            ops.append(("match", i - 1, -1))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, target_ids[j - 1]))
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", i, target_ids[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            break
    ops.reverse()

    # 연산 → 태그 + 삽입 시퀀스
    tags = [TAG_KEEP] * n
    insert_seqs: dict[int, list[int]] = {}

    for op, src_idx, tgt_token in ops:
        if op == "match":
            pass
        elif op == "sub":
            tags[src_idx] = tag_replace(tgt_token, vocab_size)
        elif op == "del":
            tags[src_idx] = TAG_DELETE
        elif op == "ins":
            # 삽입 위치: src_idx 직전 (src_idx-1)
            ins_at = max(0, src_idx - 1)
            if ins_at not in insert_seqs:
                # 첫 삽입: INSERT_START 태그 설정 (KEEP 위치만)
                if tags[ins_at] == TAG_KEEP:
                    tags[ins_at] = INSERT_START
                insert_seqs[ins_at] = []
            insert_seqs[ins_at].append(tgt_token)

    # 삽입 시퀀스에 EOS 추가
    for pos in insert_seqs:
        insert_seqs[pos].append(eos_id)

    return tags, insert_seqs


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


# ── GPU-native iterative refinement (O(N) tag composition, CPU 전송 없음) ──

def refinement_step_gpu(
    current_ids: torch.Tensor,
    pred_tags: torch.Tensor,
    edit_tags: torch.Tensor,
    pad_mask: torch.Tensor,
    vocab_size: int,
    pad_id: int = 0,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-native refinement: apply tags + O(N) tag composition (Levenshtein 불필요)

    pred_tags를 적용하여 modified를 생성하고, old edit_tags와 비교하여
    다음 iteration의 정답 태그를 O(N)으로 계산한다.

    Args:
        current_ids: (B, T) 현재 입력 토큰
        pred_tags:   (B, T) 모델 예측 태그
        edit_tags:   (B, T) 현재 정답 태그
        pad_mask:    (B, T) bool
        vocab_size:  어휘 크기
        pad_id:      패딩 토큰 ID
        max_seq_len: 출력 최대 길이 (None=무제한, 패킹 시 고정 길이 유지)

    Returns:
        modified:  (B, T') 수정된 토큰
        new_tags:  (B, T') 다음 iteration 정답 태그
        new_mask:  (B, T') bool
    """
    B, T = current_ids.shape
    device = current_ids.device

    is_del_pred = (pred_tags == TAG_DELETE)
    is_ins_pred = (pred_tags >= 2 + vocab_size)
    is_rep_pred = (pred_tags >= 2) & (pred_tags < 2 + vocab_size)

    is_del_old = (edit_tags == TAG_DELETE)
    is_ins_old = (edit_tags >= 2 + vocab_size)
    is_rep_old = (edit_tags >= 2) & (edit_tags < 2 + vocab_size)

    # ── 출력 위치 계산 (cumsum) ──
    out_count = torch.ones(B, T, dtype=torch.long, device=device)
    out_count[is_del_pred] = 0
    out_count[is_ins_pred] = 2
    out_count[~pad_mask] = 0

    out_pos = out_count.cumsum(dim=1) - out_count
    max_out = int(out_count.sum(dim=1).max().item())
    if max_seq_len is not None:
        max_out = min(max_out, max_seq_len)
    max_out = max(max_out, 1)

    modified = torch.full((B, max_out), pad_id, dtype=torch.long, device=device)
    new_tags = torch.full((B, max_out), TAG_KEEP, dtype=torch.long, device=device)
    new_mask = torch.zeros(B, max_out, dtype=torch.bool, device=device)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(current_ids)
    valid_A = pad_mask & ~is_del_pred

    # ── Position A: 메인 토큰 + 태그 ──

    # 실제 토큰 (pred 적용 결과)
    actual_A = current_ids.clone()
    actual_A[is_rep_pred] = pred_tags[is_rep_pred] - 2

    # 목표 토큰 (old_tag이 지시하는 정답): KEEP/INSERT→current, REPLACE_x→x
    target_A = current_ids.clone()
    target_A[is_rep_old] = edit_tags[is_rep_old] - 2

    # 태그 결정
    tag_A = torch.full_like(edit_tags, TAG_KEEP)
    tag_A[is_del_old] = TAG_DELETE
    mismatch = (actual_A != target_A) & ~is_del_old & pad_mask
    tag_A[mismatch] = 2 + target_A[mismatch]
    # old=INSERT_z이고 pred가 INSERT가 아님 → INSERT 보존
    ins_not_done = is_ins_old & ~is_ins_pred & (tag_A == TAG_KEEP) & pad_mask
    tag_A[ins_not_done] = edit_tags[ins_not_done]

    # scatter (max_out 범위 내만)
    pos_A = out_pos[valid_A]
    in_range_A = pos_A < max_out
    if in_range_A.any():
        modified[batch_idx[valid_A][in_range_A], pos_A[in_range_A]] = actual_A[valid_A][in_range_A]
        new_tags[batch_idx[valid_A][in_range_A], pos_A[in_range_A]] = tag_A[valid_A][in_range_A]
        new_mask[batch_idx[valid_A][in_range_A], pos_A[in_range_A]] = True

    # ── Position B: INSERT 추가 토큰 + 태그 ──
    ins_mask = is_ins_pred & pad_mask
    if ins_mask.any():
        ins_token = pred_tags[ins_mask] - 2 - vocab_size
        ins_pos = out_pos[ins_mask] + 1

        # 기본: 불필요한 삽입 → DELETE
        tag_B = torch.full_like(ins_token, TAG_DELETE)
        old_was_ins = is_ins_old[ins_mask]
        if old_was_ins.any():
            old_ins_token = edit_tags[ins_mask][old_was_ins] - 2 - vocab_size
            actual_ins = ins_token[old_was_ins]
            correct = (actual_ins == old_ins_token)
            tag_B[old_was_ins] = torch.where(
                correct,
                torch.zeros_like(old_ins_token),   # KEEP
                2 + old_ins_token,                  # REPLACE_z
            )

        in_range_B = ins_pos < max_out
        if in_range_B.any():
            modified[batch_idx[ins_mask][in_range_B], ins_pos[in_range_B]] = ins_token[in_range_B]
            new_tags[batch_idx[ins_mask][in_range_B], ins_pos[in_range_B]] = tag_B[in_range_B]
            new_mask[batch_idx[ins_mask][in_range_B], ins_pos[in_range_B]] = True

    return modified, new_tags, new_mask


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
