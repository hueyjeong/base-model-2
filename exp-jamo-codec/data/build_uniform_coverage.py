"""build_uniform_coverage.py — BBPE 전체 vocab 균일 커버리지 데이터셋 생성

K-Exaone 153K BBPE vocab의 모든 토큰 ID에 대해 1개 row 생성.
각 row는 해당 토큰의 디코딩 텍스트를 ~target_chars(기본 2000)글자 채울 때까지
separator(기본 "|")로 반복 연결. 한 row 안에서 해당 토큰이 N회 등장.

이 구조의 이점:
- 단일 토큰 row보다 훨씬 긴 row → 데이터 로더 처리 효율 ↑ (동일 커버리지 대비 row 수 1/N)
- 한 번의 BBPE encode 로 수백 번의 gradient signal
- SegmentMaskedConvBlock이 토큰 간 상호작용을 차단하므로, 반복된 같은 토큰이라도
  각 세그먼트가 독립적으로 gradient 에 기여 (실제 문서 속 등장과 동일)

Fallback 토큰(자모 > 32)도 같은 방식으로 row 생성. 데이터셋 로더의
_decompose_ids 가 공백/문자 분할을 자동 수행하므로 별도 처리 불필요.

사용:
    # 기본 (토큰당 1행 × 2000자 ≈ 153,600행, parquet 저장)
    python exp-jamo-codec/data/build_uniform_coverage.py

    # 행당 4000자 + 기존 코퍼스 병합
    python exp-jamo-codec/data/build_uniform_coverage.py \\
        --target_chars 4000 \\
        --merge corpus/k-exaone_coverage_100.parquet \\
        --output corpus/k-exaone_coverage_100_uniform.parquet

    # fallback 토큰만 집중
    python exp-jamo-codec/data/build_uniform_coverage.py --fallback_only
"""
import argparse
import os
import re
import sys
import time
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_token_texts(
    bbpe_tok,
    jamo_tok,
    max_jamo_per_token: int = 32,
    fallback_only: bool = False,
    verbose: bool = True,
) -> dict:
    """전체 BBPE vocab → 토큰별 decode 텍스트 수집.

    Returns:
        {
            "token_texts": list[str],       # 토큰 ID 순서대로 decode 텍스트 (빈 문자열 제외)
            "n_fallback_tokens": int,        # 자모 > max_jamo_per_token 인 토큰 수
        }
    """
    vocab_size = len(bbpe_tok)
    token_texts: List[str] = []
    n_fallback = 0

    for tid in range(vocab_size):
        tok_str = bbpe_tok.decode([tid])
        if not tok_str:
            continue

        # 자모 분해 길이 확인 (fallback 판별)
        jids = jamo_tok.encode(tok_str, add_special=False)
        is_fallback = len(jids) > max_jamo_per_token

        if fallback_only and not is_fallback:
            continue

        token_texts.append(tok_str)
        if is_fallback:
            n_fallback += 1

    if verbose:
        print(f"Vocab 크기: {vocab_size:,}")
        print(f"수집된 토큰 텍스트: {len(token_texts):,} (빈 문자열 제외)")
        print(f"Fallback 토큰 (자모 > {max_jamo_per_token}): {n_fallback:,}")

    return {
        "token_texts": token_texts,
        "n_fallback_tokens": n_fallback,
    }


def _make_row(text: str, target_chars: int, sep: str) -> str:
    """한 토큰의 텍스트를 target_chars 까지 sep 로 구분하여 반복."""
    if not text:
        return ""
    unit_len = len(text) + len(sep)
    if unit_len <= 0:
        return text
    # 최소 1회 보장, sep 제외한 실 길이 기준 채우기
    n = max(1, target_chars // unit_len)
    return sep.join([text] * n)


def generate_rows(
    token_texts: List[str],
    target_chars: int = 2000,
    separator: str = "|",
    seed: int = 42,
    verbose: bool = True,
) -> List[str]:
    """각 토큰 텍스트 → target_chars 까지 반복 연결한 row 생성 후 셔플.

    Args:
        token_texts: 토큰별 decode 텍스트 리스트
        target_chars: row 당 목표 문자 수 (기본 2000)
        separator: 반복 사이 구분자 (기본 "|")
        seed: 셔플 시드

    Returns:
        셔플된 row 리스트 (토큰당 1개)
    """
    rows: List[str] = []
    total_chars = 0
    total_repeats = 0

    for text in token_texts:
        row = _make_row(text, target_chars, separator)
        if not row:
            continue
        rows.append(row)
        total_chars += len(row)
        # 반복 횟수 = row 길이 대비 unit 길이로 역산
        unit_len = len(text) + len(separator)
        n = len(row) // unit_len if unit_len > 0 else 1
        total_repeats += n

    if verbose:
        avg_chars = total_chars / max(len(rows), 1)
        avg_reps = total_repeats / max(len(rows), 1)
        print(f"생성된 row: {len(rows):,}")
        print(f"평균 row 길이: {avg_chars:.0f}자")
        print(f"평균 토큰 반복 횟수: {avg_reps:.0f}")
        print(f"총 텍스트 크기: {total_chars/1024/1024:.1f} MB")

    # 셔플
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    rows = [rows[i] for i in indices]

    if verbose:
        print(f"셔플 완료 (seed={seed})")

    return rows


def save_parquet(rows: List[str], output_path: str, verbose: bool = True):
    """텍스트 row 리스트 → Parquet 저장."""
    table = pa.table({"text": pa.array(rows, type=pa.string())})
    pq.write_table(table, output_path, compression="zstd")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    if verbose:
        print(f"저장: {output_path} ({len(rows):,} rows, {size_mb:.1f} MB)")


def merge_parquets(paths: List[str], output_path: str, seed: int = 42,
                   verbose: bool = True):
    """여러 Parquet 파일의 text 컬럼을 병합 + 셔플 → 저장."""
    all_texts: List[str] = []
    for p in paths:
        if verbose:
            print(f"병합 로드: {p}")
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=65536, columns=["text"]):
            for text in batch["text"].to_pylist():
                if text:
                    all_texts.append(text)

    if verbose:
        print(f"병합 전 총 row: {len(all_texts):,}")

    # 셔플
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]

    save_parquet(all_texts, output_path, verbose)


def main():
    parser = argparse.ArgumentParser(
        description="BBPE 전체 vocab 균일 커버리지 데이터셋 생성")
    parser.add_argument("--output", default="corpus/k-exaone_uniform_coverage.parquet",
                        help="출력 Parquet 경로")
    parser.add_argument("--target_chars", type=int, default=2000,
                        help="row 당 목표 문자 수 (기본 2000)")
    parser.add_argument("--separator", default="|",
                        help="토큰 반복 사이 구분자 (기본 '|')")
    parser.add_argument("--max_jamo_per_token", type=int, default=32,
                        help="이 이상의 자모 길이 토큰은 fallback 경로로 처리")
    parser.add_argument("--fallback_only", action="store_true",
                        help="fallback 필요한 토큰만 생성 (일반 토큰 제외)")
    parser.add_argument("--merge", nargs="*", default=None,
                        help="기존 Parquet 파일과 병합 (예: corpus/k-exaone_coverage_100.parquet)")
    parser.add_argument("--seed", type=int, default=42,
                        help="셔플 시드")
    parser.add_argument("--model_id", default="LGAI-EXAONE/K-EXAONE-236B-A23B")
    args = parser.parse_args()

    t0 = time.time()

    # 토크나이저 로드
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    from tok.jamo_tokenizer import JamoTokenizer

    print(f"토크나이저 로드: {args.model_id}")
    bbpe_tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    jamo_tok = JamoTokenizer()
    print(f"BBPE vocab: {len(bbpe_tok):,}, Jamo vocab: {jamo_tok.vocab_size}")

    # 토큰별 텍스트 + fallback part 수집
    result = build_token_texts(
        bbpe_tok, jamo_tok,
        max_jamo_per_token=args.max_jamo_per_token,
        fallback_only=args.fallback_only,
    )

    # row 생성 + 셔플
    rows = generate_rows(
        result["token_texts"],
        target_chars=args.target_chars,
        separator=args.separator,
        seed=args.seed,
    )

    if not rows:
        print("생성할 row 없음.")
        return

    if args.merge:
        # 먼저 균일 커버리지 임시 저장 후 병합
        tmp_path = args.output + ".tmp"
        save_parquet(rows, tmp_path)
        merge_parquets(
            args.merge + [tmp_path],
            args.output,
            seed=args.seed,
        )
        os.remove(tmp_path)
    else:
        save_parquet(rows, args.output)

    elapsed = time.time() - t0
    print(f"\n완료: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
