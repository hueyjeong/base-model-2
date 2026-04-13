"""/tmp/jamo_v3/clean/*.parquet → 중복 제거 + 98/1/1 split → 최종 parquet.

1패스:
1. 모든 clean parquet을 read_row_group으로 스트리밍
2. MD5 앞 8바이트로 정확 중복 제거 (set)
3. 버퍼 누적 후 셔플 + train/val/test 랜덤 할당 (ratio 98/1/1)
4. 각 split을 row_group 단위로 zstd parquet에 기록

최종 출력:
- corpus/jamo-codec-v3/train.parquet
- corpus/jamo-codec-v3/val.parquet
- corpus/jamo-codec-v3/test.parquet

중간 작업: /tmp/jamo_v3/final/ (SSD에서 처리 후 corpus/로 move)
"""

import argparse
import hashlib
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CLEAN_DIR = Path("/tmp/jamo_v3/clean")
TMP_FINAL_DIR = Path("/tmp/jamo_v3/final")
FINAL_DIR = Path("/workspace/base-model-2/corpus/jamo-codec-v3")

ROWS_PER_GROUP = 100_000
SEED = 42

TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
# test = 1 - TRAIN - VAL = 0.01


def text_hash(text: str) -> int:
    """텍스트 → 8바이트 해시 (충돌 확률 무시 가능)."""
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "little")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=Path, default=CLEAN_DIR)
    ap.add_argument("--tmp_dir", type=Path, default=TMP_FINAL_DIR)
    ap.add_argument("--out_dir", type=Path, default=FINAL_DIR)
    ap.add_argument("--keep_source", action="store_true",
                    help="source 컬럼도 출력에 유지 (기본: text만)")
    args = ap.parse_args()

    clean_files = sorted(args.clean_dir.glob("*.parquet"))
    if not clean_files:
        log.error(f"clean parquet 없음: {args.clean_dir}")
        sys.exit(1)

    log.info(f"입력 파일 {len(clean_files)}개:")
    total_rows = 0
    total_size_gb = 0.0
    for f in clean_files:
        pf = pq.ParquetFile(f)
        size_gb = f.stat().st_size / 1e9
        total_rows += pf.metadata.num_rows
        total_size_gb += size_gb
        log.info(f"  {f.name}: {pf.metadata.num_rows:,} rows, {size_gb:.2f}GB")
    log.info(f"합계: {total_rows:,} rows, {total_size_gb:.2f}GB")

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 스키마
    if args.keep_source:
        schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    else:
        schema = pa.schema([("text", pa.string())])

    writers = {
        split: pq.ParquetWriter(str(args.tmp_dir / f"{split}.parquet"),
                                schema, compression="zstd")
        for split in ("train", "val", "test")
    }

    bufs: dict[str, dict[str, list]] = {
        split: {"text": [], "source": []} for split in ("train", "val", "test")
    }
    counts = {"train": 0, "val": 0, "test": 0}
    seen: set[int] = set()
    stats = {"read": 0, "dup": 0}
    rng = random.Random(SEED)

    def flush(split: str):
        buf = bufs[split]
        if not buf["text"]:
            return
        cols = {"text": buf["text"]}
        if args.keep_source:
            cols["source"] = buf["source"]
        writers[split].write_table(pa.table(cols))
        buf["text"].clear()
        buf["source"].clear()

    t0 = time.time()
    last_log = t0

    for fpath in clean_files:
        pf = pq.ParquetFile(fpath)
        file_kept = 0
        file_dup = 0
        cols_to_read = ["text", "source"] if "source" in pf.schema.names else ["text"]

        for rg_idx in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(rg_idx, columns=cols_to_read)
            texts = batch.column("text").to_pylist()
            sources = (batch.column("source").to_pylist()
                       if "source" in cols_to_read
                       else [fpath.stem] * len(texts))

            for text, src in zip(texts, sources):
                stats["read"] += 1
                h = text_hash(text)
                if h in seen:
                    stats["dup"] += 1
                    file_dup += 1
                    continue
                seen.add(h)

                # split 할당
                r = rng.random()
                if r < TRAIN_RATIO:
                    split = "train"
                elif r < TRAIN_RATIO + VAL_RATIO:
                    split = "val"
                else:
                    split = "test"

                bufs[split]["text"].append(text)
                bufs[split]["source"].append(src)
                counts[split] += 1
                file_kept += 1

                if len(bufs[split]["text"]) >= ROWS_PER_GROUP:
                    flush(split)

            now = time.time()
            if now - last_log >= 30:
                log.info(f"  {fpath.name} rg={rg_idx+1}/{pf.metadata.num_row_groups} "
                         f"read={stats['read']:,} dup={stats['dup']:,} "
                         f"train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}")
                last_log = now
                sys.stdout.flush()

        dup_rate = file_dup / (file_dup + file_kept) * 100 if (file_dup + file_kept) else 0
        log.info(f"[{fpath.name}] kept={file_kept:,} dup={file_dup:,} ({dup_rate:.1f}%)")

    # flush all
    for split in ("train", "val", "test"):
        flush(split)
        writers[split].close()

    del seen

    # 결과 요약
    log.info("")
    log.info("=== split 완료 ===")
    log.info(f"총 read: {stats['read']:,}")
    log.info(f"중복 제거: {stats['dup']:,} ({stats['dup']/stats['read']*100:.1f}%)")
    for split in ("train", "val", "test"):
        tmp_path = args.tmp_dir / f"{split}.parquet"
        size_gb = tmp_path.stat().st_size / 1e9
        log.info(f"  {split}: {counts[split]:,} rows, {size_gb:.2f}GB")

    # /tmp → corpus/jamo-codec-v3로 이동
    log.info(f"\n이동: {args.tmp_dir} → {args.out_dir}")
    for split in ("train", "val", "test"):
        src = args.tmp_dir / f"{split}.parquet"
        dst = args.out_dir / f"{split}.parquet"
        log.info(f"  {src} → {dst}")
        shutil.move(str(src), str(dst))

    elapsed = time.time() - t0
    log.info(f"=== 전체 완료 ({elapsed:.0f}s) ===")


if __name__ == "__main__":
    main()
