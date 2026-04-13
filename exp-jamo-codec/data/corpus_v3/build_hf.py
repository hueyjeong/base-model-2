"""HuggingFace 한국어 코퍼스 → 정제 parquet (스트리밍, 목표 크기 도달 시 중단).

대상 소스:
- 위키: wikimedia/wikipedia (20231101.ko) — 전체 (~1.3GB)
- 나무위키: heegyu/namuwiki-extracted — 전체 (~6GB)
- fineweb-2 ko: HuggingFaceFW/fineweb-2 (kor_Hang) — 스트리밍, --target_gb 제한

정제 규칙 (기존 corpus/convert_fineweb2.py 패턴):
- language_score >= 0.8 (해당 필드가 있을 때)
- 길이 >= 30자 필터
- 1000자 초과 → 문장 단위로 1k 버퍼에 패킹

출력:
- /tmp/jamo_v3/clean/<source>.parquet (zstd, schema=[text: string, source: string])
- 최종 merge_split.py가 읽어가는 위치

실행 예시:
    # 전체 (위키 + 나무위키 + fineweb-2 ko 5GB)
    python exp-jamo-codec/data/corpus_v3/build_hf.py --all

    # 소스별 선택
    python exp-jamo-codec/data/corpus_v3/build_hf.py --sources wiki_ko namuwiki
    python exp-jamo-codec/data/corpus_v3/build_hf.py --sources fineweb2_ko --target_gb 5
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
for _noisy in ("httpx", "urllib3", "fsspec", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ── 소스 정의 ──
SOURCES = {
    "wiki_ko": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.ko",
        "split": "train",
        "text_key": "text",
        "score_key": None,
        "target_gb": None,  # 전체
        "description": "Wikipedia Korean (공식 덤프)",
    },
    "namuwiki": {
        "path": "heegyu/namuwiki-extracted",
        "name": None,
        "split": "train",
        "text_key": "text",
        "score_key": None,
        "target_gb": None,  # 전체
        "description": "NamuWiki (heegyu extracted)",
    },
    "fineweb2_ko": {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "kor_Hang",
        "split": "train",
        "text_key": "text",
        "score_key": "language_score",
        "target_gb": 5.0,  # 기본 5GB 샘플링
        "description": "FineWeb-2 Korean (kor_Hang, 웹 보충용)",
    },
}

# 정제 파라미터
MIN_SCORE = 0.8
MIN_LEN = 30
MAX_CHUNK = 1000
ROWS_PER_GROUP = 100_000

# 문장 분할: 마침표·물음표·느낌표 뒤 공백 또는 줄바꿈
_SENT_SPLIT = re.compile(r'(?<=[.?!。])\s+|\n+')

OUTPUT_DIR = Path("/tmp/jamo_v3/clean")


def chunk_text(text: str) -> list[str]:
    """1k자 초과 텍스트를 문장 단위로 1k 버퍼에 채워서 분리."""
    if len(text) <= MAX_CHUNK:
        return [text]

    sents = _SENT_SPLIT.split(text)
    chunks = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= MAX_CHUNK:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def iter_hf_stream(cfg: dict) -> Iterator[dict]:
    """HF datasets streaming → row dict 생성자."""
    from datasets import load_dataset

    kwargs = dict(path=cfg["path"], split=cfg["split"], streaming=True)
    if cfg.get("name"):
        kwargs["name"] = cfg["name"]

    log.info(f"HF 스트리밍 로드: {cfg['path']} name={cfg.get('name')}")
    ds = load_dataset(**kwargs)
    for row in ds:
        yield row


def process_source(source_key: str, target_gb_override: Optional[float] = None) -> Path:
    """한 소스를 스트리밍으로 처리하여 /tmp/jamo_v3/clean/<source>.parquet 생성."""
    cfg = SOURCES[source_key]
    target_gb = target_gb_override if target_gb_override is not None else cfg["target_gb"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{source_key}.parquet"

    log.info(f"[{source_key}] {cfg['description']}")
    log.info(f"  target_gb={target_gb}, out={out_path}")

    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd")

    text_key = cfg["text_key"]
    score_key = cfg.get("score_key")

    buf_text: list[str] = []
    buf_src: list[str] = []
    bytes_written = 0
    target_bytes = int(target_gb * 1e9) if target_gb else None

    stats = {
        "rows": 0, "kept": 0, "chunks": 0,
        "filtered_score": 0, "filtered_short": 0, "filtered_short_chunk": 0,
    }

    def flush():
        nonlocal bytes_written
        if not buf_text:
            return
        table = pa.table({"text": buf_text, "source": buf_src})
        writer.write_table(table)
        bytes_written += sum(len(t.encode("utf-8")) for t in buf_text)
        buf_text.clear()
        buf_src.clear()

    t0 = time.time()
    last_log = t0

    try:
        for row in iter_hf_stream(cfg):
            stats["rows"] += 1

            # 스코어 필터 (해당 필드가 있을 때만)
            if score_key and score_key in row:
                score = row.get(score_key)
                if score is not None and score < MIN_SCORE:
                    stats["filtered_score"] += 1
                    continue

            text = row.get(text_key)
            if not text or len(text) < MIN_LEN:
                stats["filtered_short"] += 1
                continue

            chunks = chunk_text(text)
            if len(chunks) > 1:
                stats["chunks"] += 1

            for ch in chunks:
                if len(ch) < MIN_LEN:
                    stats["filtered_short_chunk"] += 1
                    continue
                buf_text.append(ch)
                buf_src.append(source_key)
                stats["kept"] += 1

                if len(buf_text) >= ROWS_PER_GROUP:
                    flush()
                    if target_bytes and bytes_written >= target_bytes:
                        log.info(f"  목표 크기 도달: {bytes_written/1e9:.2f} GB, 중단")
                        raise StopIteration

            now = time.time()
            if now - last_log >= 30:
                speed = stats["rows"] / (now - t0)
                log.info(
                    f"  [{source_key}] rows={stats['rows']:,} kept={stats['kept']:,} "
                    f"bytes={bytes_written/1e9:.2f}GB speed={speed:.0f} rows/s"
                )
                last_log = now
                sys.stdout.flush()
    except StopIteration:
        pass

    flush()
    writer.close()

    size_gb = out_path.stat().st_size / 1e9
    elapsed = time.time() - t0
    log.info(
        f"[{source_key}] 완료: rows={stats['rows']:,} kept={stats['kept']:,} "
        f"out={size_gb:.2f}GB uncompressed={bytes_written/1e9:.2f}GB "
        f"(filtered: score={stats['filtered_score']:,} short={stats['filtered_short']:,} "
        f"short_chunk={stats['filtered_short_chunk']:,}) "
        f"chunked={stats['chunks']:,} in {elapsed:.0f}s"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", choices=list(SOURCES.keys()) + ["all"], default=["all"])
    ap.add_argument("--target_gb", type=float, default=None,
                    help="fineweb2_ko 등 target_gb 오버라이드")
    args = ap.parse_args()

    sources = list(SOURCES.keys()) if "all" in args.sources else args.sources
    log.info(f"대상 소스: {sources}")

    for src in sources:
        process_source(src, target_gb_override=args.target_gb)

    log.info("=== 모든 HF 소스 처리 완료 ===")


if __name__ == "__main__":
    main()
