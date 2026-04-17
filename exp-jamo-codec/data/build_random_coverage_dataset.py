"""build_random_coverage_dataset.py — vocab 균등 샘플링 기반 합성 데이터셋

K-Exaone vocab에서 각 대상 토큰을 per_token_samples번씩 bag에 담고 shuffle,
~target_chars 크기의 row로 분할하여 decode한 합성 텍스트 데이터셋을 만든다.

특징:
- 자연 텍스트 대비 row당 토큰 분포가 훨씬 다양함 (같은 토큰 반복 최소)
- per_token_samples * len(all_target) 만큼의 토큰으로 전체 vocab을 정확히 N회 커버
- codec identity 학습처럼 분포 다양성이 중요한 val/test에 적합

실행 예시:
    python exp-jamo-codec/data/build_random_coverage_dataset.py \\
        --output corpus/k-exaone_random_coverage_5_len2000.parquet \\
        --per_token_samples 5 \\
        --target_chars 2000
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
for _noisy in ("httpx", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


PARQUET_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("n_tokens", pa.int32()),
])


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    log.info(f"토크나이저 로드: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    log.info(f"  vocab_size={tok.vocab_size}")
    return tok


def get_target_ids(tokenizer) -> np.ndarray:
    """특수 토큰 제외 — 학습 대상 토큰 ID 배열"""
    special = set(getattr(tokenizer, "all_special_ids", []))
    vocab = tokenizer.get_vocab()
    target = sorted(tid for tid in vocab.values() if tid not in special)
    return np.array(target, dtype=np.int32)


def estimate_chars_per_token(rust_tokenizer, target_arr: np.ndarray, sample_size: int = 2000) -> float:
    """토큰당 평균 decode 문자 수 추정"""
    rng = np.random.default_rng(0)
    sample = rng.choice(target_arr, size=sample_size, replace=True).tolist()
    text = rust_tokenizer.decode(sample)
    cpt = len(text) / sample_size
    log.info(f"평균 char/token ≈ {cpt:.2f} (sample {sample_size})")
    return cpt


def build_dataset(
    tokenizer,
    rust_tokenizer,
    target_arr: np.ndarray,
    per_token_samples: int,
    target_chars: int,
    output_path: str,
    chunk_size: int,
    seed: int,
):
    """bag shuffle → ~target_chars row 분할 → Parquet 저장"""
    # 1. token bag: 각 target 토큰을 per_token_samples회 반복
    bag = np.tile(target_arr, per_token_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(bag)
    log.info(f"token bag 크기: {len(bag):,} ({len(target_arr):,} × {per_token_samples})")

    # 2. char/token 추정 → row당 대략 몇 토큰 넣을지 결정
    cpt = estimate_chars_per_token(rust_tokenizer, target_arr)
    approx_tokens_per_row = max(1, int(target_chars / cpt))
    log.info(f"row당 약 {approx_tokens_per_row:,} tokens 목표 (target_chars={target_chars})")

    # 3. bag을 row 단위로 분할하여 decode
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, schema=PARQUET_SCHEMA, compression="snappy")

    texts_buf, sources_buf, idxs_buf, ntok_buf = [], [], [], []
    row_idx = 0
    total_tokens = 0
    t0 = time.time()

    def flush():
        if not texts_buf:
            return
        table = pa.table({
            "text": pa.array(texts_buf, type=pa.string()),
            "source": pa.array(sources_buf, type=pa.string()),
            "row_idx": pa.array(idxs_buf, type=pa.int64()),
            "n_tokens": pa.array(ntok_buf, type=pa.int32()),
        }, schema=PARQUET_SCHEMA)
        writer.write_table(table)
        texts_buf.clear(); sources_buf.clear(); idxs_buf.clear(); ntok_buf.clear()

    pos = 0
    n_bag = len(bag)
    while pos < n_bag:
        # 첫 시도: approx_tokens_per_row개로 decode
        end = min(pos + approx_tokens_per_row, n_bag)
        chunk_ids = bag[pos:end].tolist()
        text = rust_tokenizer.decode(chunk_ids)

        # 부족하면 더 추가, 초과해도 일단 row로 저장 (균일 분포 유지 우선)
        while len(text) < target_chars and end < n_bag:
            add = min(max(1, int((target_chars - len(text)) / cpt)), n_bag - end)
            end += add
            chunk_ids = bag[pos:end].tolist()
            text = rust_tokenizer.decode(chunk_ids)

        if not text:
            pos = end
            continue

        texts_buf.append(text)
        sources_buf.append("random_vocab")
        idxs_buf.append(row_idx)
        ntok_buf.append(len(chunk_ids))
        row_idx += 1
        total_tokens += len(chunk_ids)
        pos = end

        if len(texts_buf) >= chunk_size:
            flush()

        if row_idx % 1000 == 0:
            elapsed = time.time() - t0
            rate = row_idx / max(elapsed, 1e-6)
            progress = total_tokens / n_bag * 100
            log.info(
                f"row {row_idx:,} | tokens {total_tokens:,}/{n_bag:,} "
                f"({progress:.1f}%) | {rate:.0f} rows/s"
            )

    flush()
    writer.close()

    elapsed = time.time() - t0
    log.info(
        f"완료: {row_idx:,} rows, {total_tokens:,} tokens, "
        f"avg {total_tokens/max(row_idx,1):.0f} tok/row, {elapsed:.1f}s"
    )
    log.info(f"저장: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="vocab 균등 샘플링 기반 합성 커버리지 데이터셋",
    )
    p.add_argument(
        "--output", default="corpus/k-exaone_random_coverage.parquet",
        help="최종 출력 Parquet 파일 경로",
    )
    p.add_argument(
        "--per_token_samples", type=int, default=5,
        help="각 토큰이 bag에 들어갈 횟수 (기본: 5)",
    )
    p.add_argument(
        "--target_chars", type=int, default=2000,
        help="row당 목표 문자 수 (기본: 2000)",
    )
    p.add_argument(
        "--chunk_size", type=int, default=1_000,
        help="Parquet writer flush 단위 (기본: 1000 rows)",
    )
    p.add_argument(
        "--model_id", default="LGAI-EXAONE/K-EXAONE-236B-A23B",
        help="K-Exaone 모델 ID",
    )
    p.add_argument("--seed", type=int, default=42, help="셔플 시드 (기본: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_id)
    rust_tokenizer = tokenizer.backend_tokenizer if hasattr(tokenizer, "backend_tokenizer") else tokenizer
    target_arr = get_target_ids(tokenizer)
    log.info(f"target 토큰 수: {len(target_arr):,}")

    build_dataset(
        tokenizer=tokenizer,
        rust_tokenizer=rust_tokenizer,
        target_arr=target_arr,
        per_token_samples=args.per_token_samples,
        target_chars=args.target_chars,
        output_path=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
