"""build_uniform_coverage.py — BBPE 전체 vocab 균일 커버리지 데이터셋 생성

K-Exaone 153K BBPE vocab의 모든 토큰 ID에 대해 동일 횟수(N)의 row를 생성.
각 row는 해당 토큰의 디코딩 텍스트 1개. 자모 분해 > max_jamo_per_token인
토큰은 fallback 경로(공백 분할 → 문자 분할)의 sub-part도 별도 row로 생성.

SegmentMaskedConvBlock이 토큰 간 상호작용을 차단하므로,
단일 토큰 row가 실제 문서 속 토큰과 동일한 gradient signal을 준다.

사용:
    # 기본 (토큰당 100행, parquet 저장)
    python exp-jamo-codec/data/build_uniform_coverage.py

    # 토큰당 500행 + 기존 코퍼스 병합
    python exp-jamo-codec/data/build_uniform_coverage.py \\
        --repeats 500 \\
        --merge corpus/k-exaone_coverage_100.parquet \\
        --output corpus/k-exaone_coverage_100_uniform.parquet

    # fallback 토큰만 집중 (각 1000행)
    python exp-jamo-codec/data/build_uniform_coverage.py \\
        --repeats 1000 --fallback_only
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
    """전체 BBPE vocab → 토큰별 텍스트 + fallback sub-part 텍스트 수집.

    Returns:
        {
            "token_texts": list[str],       # 토큰 ID 순서대로 decode 텍스트
            "fallback_parts": list[str],     # fallback 경로에서 나오는 sub-part 텍스트
            "n_fallback_tokens": int,        # fallback 필요한 토큰 수
            "n_fallback_parts": int,         # 총 fallback sub-part 수
        }
    """
    vocab_size = len(bbpe_tok)
    token_texts: List[str] = []
    fallback_parts: List[str] = []
    fallback_parts_set: set = set()
    n_fallback = 0

    for tid in range(vocab_size):
        tok_str = bbpe_tok.decode([tid])
        if not tok_str:
            tok_str = ""

        # 자모 분해 길이 확인
        jids = jamo_tok.encode(tok_str, add_special=False) if tok_str else []
        is_fallback = len(jids) > max_jamo_per_token

        if fallback_only and not is_fallback:
            continue

        token_texts.append(tok_str)

        if is_fallback:
            n_fallback += 1
            # fallback 경로 재현: 공백 분할 → 문자 분할
            parts = re.split(r'( )', tok_str)
            for part in parts:
                if not part:
                    continue
                pj = jamo_tok.encode(part, add_special=False)
                if len(pj) <= max_jamo_per_token:
                    if part not in fallback_parts_set:
                        fallback_parts_set.add(part)
                        fallback_parts.append(part)
                else:
                    # 문자 단위 분할
                    for ch in part:
                        cj = jamo_tok.encode(ch, add_special=False)
                        if cj and ch not in fallback_parts_set:
                            fallback_parts_set.add(ch)
                            fallback_parts.append(ch)

    if verbose:
        print(f"Vocab 크기: {vocab_size:,}")
        print(f"토큰 텍스트: {len(token_texts):,}")
        print(f"Fallback 토큰: {n_fallback:,} ({n_fallback/vocab_size*100:.2f}%)")
        print(f"Fallback sub-part: {len(fallback_parts):,} (중복 제거)")

    return {
        "token_texts": token_texts,
        "fallback_parts": fallback_parts,
        "n_fallback_tokens": n_fallback,
        "n_fallback_parts": len(fallback_parts),
    }


def generate_rows(
    token_texts: List[str],
    fallback_parts: List[str],
    repeats: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> List[str]:
    """토큰 텍스트와 fallback part를 repeats 횟수만큼 반복 → 셔플.

    Returns:
        셔플된 텍스트 row 리스트
    """
    rows: List[str] = []

    # 각 토큰 텍스트 N번 반복
    for text in token_texts:
        if text:  # 빈 문자열 제외
            rows.extend([text] * repeats)

    # fallback sub-part도 동일 횟수 반복
    for part in fallback_parts:
        if part:
            rows.extend([part] * repeats)

    if verbose:
        print(f"총 row 수 (셔플 전): {len(rows):,}")

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
    parser.add_argument("--repeats", type=int, default=100,
                        help="토큰당 반복 횟수 (기본 100)")
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
        result["fallback_parts"],
        repeats=args.repeats,
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
