"""NIKL parsed raw_texts → 정제 parquet (문서 ID 기반 1000자 패킹).

대상 (corpus/NIKL/parsed/raw_texts/ 재활용):
- 신문 (newspaper): v2 + 2020~2024 샘플링 (--newspaper_target_gb)
- 문어 (written): 전체
- 구어 (spoken): 전체
- 메신저 (messenger): 전체

raw_texts는 이미 문장 단위 JSONL이며 "id" 필드가 "DOCID.para.sent" 형식.
같은 문서(id prefix) 내 문장들을 순서대로 모아 1000자 이하 청크로 패킹한다.

출력:
- /tmp/jamo_v3/clean/nikl_<category>.parquet (zstd)

실행 예시:
    python exp-jamo-codec/data/corpus_v3/build_nikl.py --all
    python exp-jamo-codec/data/corpus_v3/build_nikl.py \\
        --sources newspaper written --newspaper_target_gb 12
"""

import argparse
import html
import json
import logging
import os
import random
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

# ── 경로 ──
NIKL_RAW = Path("/workspace/base-model-2/corpus/NIKL/parsed/raw_texts")
OUTPUT_DIR = Path("/tmp/jamo_v3/clean")

# ── 소스 그룹 정의 ──
NIKL_SOURCES = {
    "newspaper": {
        "files": [
            "nikl_newspaper_v2.jsonl",
            "nikl_newspaper_2020.jsonl",
            "nikl_newspaper_2021.jsonl",
            "nikl_newspaper_2022.jsonl",
            "nikl_newspaper_2023.jsonl",
            "nikl_newspaper_2024.jsonl",
        ],
        "target_gb": 12.0,  # 원본 32GB → 12GB 샘플링
        "description": "NIKL 신문 (v2 + 2020~2024, 샘플링)",
    },
    "written": {
        "files": ["nikl_written.jsonl", "nikl_np.jsonl",
                  "nikl_raw_2023.jsonl", "nikl_raw_2024.jsonl"],
        "target_gb": None,  # 전체 (~4.2GB)
        "description": "NIKL 문어 (written + np + raw writing)",
    },
    "spoken": {
        "files": ["nikl_spoken.jsonl",
                  "nikl_dialogue_2020.jsonl", "nikl_dialogue_2021.jsonl",
                  "nikl_dialogue_2022.jsonl", "nikl_dialogue_2023.jsonl",
                  "nikl_dialogue_2024.jsonl",
                  "nikl_om.jsonl"],
        "target_gb": None,  # 전체 (~4.0GB)
        "description": "NIKL 구어 (spoken + dialogue + 온라인대화)",
    },
    "messenger": {
        "files": ["nikl_messenger.jsonl"],
        "target_gb": None,  # 전체 (~80MB)
        "description": "NIKL 메신저 (2인 메신저 대화)",
    },
}

# 건너뛸 파일 (GEC 병렬/평가용)
SKIP_FILES = {"nikl_cola.jsonl"}

MAX_CHUNK = 1000
MIN_LEN = 30
ROWS_PER_GROUP = 100_000
SEED = 42

HTML_TAG_RE = re.compile(r'<[^>]+>')


def strip_html(text: str) -> str:
    """HTML 태그 제거 + 엔티티 디코딩 (nikl_newspaper_2024 등)."""
    return html.unescape(HTML_TAG_RE.sub('', text))


def doc_id_of(entry_id: str) -> str:
    """'WARW1800000007.1.1' → 'WARW1800000007' (문서 ID prefix)."""
    return entry_id.split(".", 1)[0] if "." in entry_id else entry_id


def pack_sentences(sentences: list[str], max_chars: int = MAX_CHUNK) -> list[str]:
    """문장 리스트를 max_chars 이하 청크로 패킹."""
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = buf + " " + s
        else:
            chunks.append(buf)
            # s 자체가 max_chars 초과하면 강제 분할
            while len(s) > max_chars:
                chunks.append(s[:max_chars])
                s = s[max_chars:]
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def process_file_packed(path: Path, source_key: str) -> Iterator[str]:
    """JSONL 파일을 읽어 문서 ID별로 그루핑 → 1000자 패킹 청크 생성.

    raw_texts는 id 순서로 정렬되어 있다고 가정 (실제 그렇게 추출됨).
    같은 doc_id가 연속될 때 누적 → doc_id 바뀌면 flush.
    """
    current_doc = None
    current_sents: list[str] = []

    def flush_doc():
        if not current_sents:
            return []
        chunks = pack_sentences(current_sents, MAX_CHUNK)
        return [c for c in chunks if len(c) >= MIN_LEN]

    for row in iter_jsonl(path):
        text = row.get("text", "")
        if not text:
            continue
        # HTML 태그 제거 (nikl_newspaper_2024 등)
        if "<" in text and ">" in text:
            text = strip_html(text)
        text = text.strip()
        if not text:
            continue

        entry_id = row.get("id", "")
        doc = doc_id_of(entry_id) if entry_id else None

        if doc != current_doc:
            # 문서 바뀜 → 이전 문서 flush
            for ch in flush_doc():
                yield ch
            current_doc = doc
            current_sents = [text]
        else:
            current_sents.append(text)

    # 마지막 문서
    for ch in flush_doc():
        yield ch


def process_source(source_key: str, target_gb_override: Optional[float] = None) -> Path:
    """한 NIKL 소스 그룹을 처리 → /tmp/jamo_v3/clean/nikl_<key>.parquet.

    target_gb가 지정되면 전체 파일들에서 랜덤 샘플링(수락 확률 기반)으로 크기 제한.
    """
    cfg = NIKL_SOURCES[source_key]
    target_gb = target_gb_override if target_gb_override is not None else cfg["target_gb"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"nikl_{source_key}.parquet"

    # 유효 파일
    paths = []
    for fname in cfg["files"]:
        if fname in SKIP_FILES:
            continue
        p = NIKL_RAW / fname
        if not p.exists():
            log.warning(f"  파일 없음, 스킵: {fname}")
            continue
        paths.append(p)

    if not paths:
        raise RuntimeError(f"[{source_key}] 처리할 파일이 없습니다")

    total_size = sum(p.stat().st_size for p in paths)
    # 샘플링 비율 계산 (target 미지정이면 1.0)
    if target_gb is None:
        sample_rate = 1.0
    else:
        # raw JSONL은 과도한 메타(source/category/id) 포함. 청크 후 실크기 감안 0.7배 기준
        # 보수적으로 JSONL 크기 기준으로 설정 (출력은 문장 합쳐지므로 덜 나감)
        sample_rate = min(1.0, target_gb * 1e9 / total_size)

    log.info(f"[{source_key}] {cfg['description']}")
    log.info(f"  파일 {len(paths)}개, raw {total_size/1e9:.2f}GB, "
             f"target_gb={target_gb}, sample_rate={sample_rate:.3f}")

    schema = pa.schema([("text", pa.string()), ("source", pa.string())])
    writer = pq.ParquetWriter(str(out_path), schema, compression="zstd")

    rng = random.Random(SEED + hash(source_key) % 10000)

    buf_text: list[str] = []
    buf_src: list[str] = []
    bytes_written = 0
    stats = {"chunks_in": 0, "kept": 0}

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

    for p in paths:
        file_kept = 0
        for chunk in process_file_packed(p, source_key):
            stats["chunks_in"] += 1
            # 샘플링: 문서-청크 단위
            if sample_rate < 1.0 and rng.random() >= sample_rate:
                continue
            buf_text.append(chunk)
            buf_src.append(f"nikl_{source_key}")
            stats["kept"] += 1
            file_kept += 1

            if len(buf_text) >= ROWS_PER_GROUP:
                flush()

            now = time.time()
            if now - last_log >= 30:
                log.info(f"  [{source_key}] {p.name}: kept_total={stats['kept']:,} "
                         f"bytes={bytes_written/1e9:.2f}GB")
                last_log = now
                sys.stdout.flush()

        log.info(f"  {p.name}: +{file_kept:,} (누적 {stats['kept']:,})")

    flush()
    writer.close()

    size_gb = out_path.stat().st_size / 1e9
    elapsed = time.time() - t0
    log.info(f"[{source_key}] 완료: chunks_in={stats['chunks_in']:,} "
             f"kept={stats['kept']:,} out={size_gb:.2f}GB "
             f"uncompressed={bytes_written/1e9:.2f}GB in {elapsed:.0f}s")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    choices=list(NIKL_SOURCES.keys()) + ["all"], default=["all"])
    ap.add_argument("--newspaper_target_gb", type=float, default=None,
                    help="신문 카테고리 target_gb 오버라이드 (기본 12GB)")
    args = ap.parse_args()

    sources = list(NIKL_SOURCES.keys()) if "all" in args.sources else args.sources
    log.info(f"대상: {sources}")

    for src in sources:
        override = args.newspaper_target_gb if src == "newspaper" else None
        process_source(src, target_gb_override=override)

    log.info("=== 모든 NIKL 소스 처리 완료 ===")


if __name__ == "__main__":
    main()
