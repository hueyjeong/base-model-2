"""build_coverage_dataset.py — K-Exaone 전체 토큰 커버리지 데이터셋 빌더

K-Exaone 153K BBPE 토크나이저의 모든 토큰 ID가 최소 1번 이상 등장하는
텍스트 row를 스트리밍으로 수집하고 Parquet으로 저장한다.

Byte fallback 커버리지:
- UTF-8 1~4바이트 전체 (U+0000~U+10FFFF) 커버를 목표로 한다.
- byte 토큰 ID 자체보다 실제 유니코드 문자가 포함된 row를 수집한다.

알고리즘 (1-pass):
1. HF 스트리밍으로 row를 순서대로 읽는다.
2. 각 row를 K-Exaone으로 토크나이징 → 등장한 고유 token_id 집합 추출.
3. 아직 목표 샘플 수(per_token_samples)에 미달인 token_id가 있는 row만 수집.
4. 수집된 row를 buf_dir 아래 청크(chunk) Parquet 파일에 버퍼링.
5. 전체 목표 달성(또는 스트림 소진) 후 청크를 병합해 최종 Parquet 저장.

메모리 안전 전략:
- token_to_rows는 Dict[int, int] (수집 카운트만 유지, row 내용은 버퍼에만 저장).
- 한 청크(chunk_size rows)마다 pyarrow로 디스크에 flush.
- covered 집합(목표 달성 token_id)이 증가할수록 스캔 속도 향상.

실행 예시:
    # 드라이런 (처음 10만 row 스캔):
    python exp-jamo-codec/data/build_coverage_dataset.py --dry_run --max_rows 100000

    # 실제 빌드:
    python exp-jamo-codec/data/build_coverage_dataset.py \\
        --output corpus/coverage_dataset.parquet \\
        --per_token_samples 10000 \\
        --chunk_size 50000

    # 1차 소스만 (FineWeb 10BT 영어):
    python exp-jamo-codec/data/build_coverage_dataset.py \\
        --sources fineweb_10bt \\
        --output corpus/coverage_fineweb_en.parquet
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 데이터셋 소스 정의 ──

SOURCE_CONFIGS = {
    "fineweb_10bt": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "text_key": "text",
        "description": "FineWeb sample-10BT (영어)",
    },
    "fineweb_100bt": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-100BT",
        "split": "train",
        "text_key": "text",
        "description": "FineWeb sample-100BT (영어)",
    },
    "fineweb2_ko": {
        "path": "HuggingFaceFW/fineweb-2",
        "name": "kor_Hang",
        "split": "train",
        "text_key": "text",
        "description": "FineWeb-2 Korean (한국어, kor_Hang)",
    },
}

DEFAULT_SOURCES = ["fineweb_10bt", "fineweb2_ko"]

# PyArrow 스키마
PARQUET_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("row_idx", pa.int64()),
    pa.field("n_tokens", pa.int32()),
])


# ── 토크나이저 유틸 ──

def load_tokenizer(model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    """K-Exaone AutoTokenizer 로드"""
    from transformers import AutoTokenizer
    log.info(f"토크나이저 로드: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    log.info(f"  vocab_size={tok.vocab_size}")
    return tok


def _gpt2_bytes_to_unicode() -> dict[str, int]:
    """GPT-2 / BBPE 스타일 Unicode 문자 → byte value (0~255) 역매핑.

    GPT-2 BBPE는 256개 base byte를 모두 표현 가능한 Unicode 문자로 인코딩한다:
    - 출력 가능한 ASCII (0x21~0x7E): 그대로 사용
    - Latin-1 일부 (0xA1~0xAC, 0xAE~0xFF): 그대로 사용
    - 제어 문자 등 나머지 94바이트: U+0100 이상의 문자로 매핑

    Returns:
        char → byte_value (0~255) dict, 크기 256
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def classify_vocab(tokenizer) -> dict:
    """vocab을 normal / byte_fallback / special 버킷으로 분류

    SentencePiece 스타일 (<0xXX>)과 GPT-2/BBPE 스타일(단일 Unicode 문자)
    두 가지 byte token 형식을 모두 감지한다.

    Returns:
        {
          "normal": set[int],         # 일반 텍스트 토큰
          "byte_fallback": dict[int, int],  # token_id → byte_value (0~255)
          "special": set[int],        # 특수 토큰 (<s>, </s>, <pad> 등)
          "all_target": set[int],     # 커버리지 목표 (normal + byte_fallback)
        }
    """
    import re

    vocab = tokenizer.get_vocab()  # str → int

    # 특수 토큰 집합
    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids = set(tokenizer.all_special_ids)

    normal: set[int] = set()
    byte_fallback: dict[int, int] = {}  # token_id → byte value

    # SentencePiece 스타일: <0xXX>
    byte_pat = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
    # GPT-2/BBPE 스타일: 길이 1 Unicode 문자 → byte value 매핑
    gpt2_char_to_byte = _gpt2_bytes_to_unicode()

    for tok_str, tid in vocab.items():
        if tid in special_ids:
            continue
        # SentencePiece 스타일
        m = byte_pat.match(tok_str)
        if m:
            byte_fallback[tid] = int(m.group(1), 16)
            continue
        # GPT-2/BBPE 스타일: 단일 문자가 byte 매핑에 있으면 byte token
        if len(tok_str) == 1 and tok_str in gpt2_char_to_byte:
            byte_fallback[tid] = gpt2_char_to_byte[tok_str]
            continue
        normal.add(tid)

    all_target = normal | set(byte_fallback.keys())

    log.info(
        f"Vocab 분류: normal={len(normal)}, "
        f"byte_fallback={len(byte_fallback)}, "
        f"special={len(special_ids)}, "
        f"target 합계={len(all_target)}"
    )
    return {
        "normal": normal,
        "byte_fallback": byte_fallback,
        "special": special_ids,
        "all_target": all_target,
    }


def build_unicode_target_map(byte_fallback: dict[int, int]) -> dict[str, list[int]]:
    """유니코드 문자 → 해당 문자를 인코딩하는 byte token ID 목록

    UTF-8 인코딩 시 필요한 byte token들을 문자마다 나열한다.
    byte fallback 커버리지 평가에 사용.

    범위: U+0000~U+10FFFF 중 byte_fallback에 있는 바이트로 표현 가능한 것.
    """
    # byte value → token_id 역맵
    byte_to_tid = {v: k for k, v in byte_fallback.items()}

    # U+0000~U+10FFFF에서 각 문자의 UTF-8 바이트 시퀀스 확인
    # 실용적으로: 대표 유니코드 범위에서 샘플링
    char_to_byte_tids: dict[str, list[int]] = {}

    def try_char(cp: int):
        try:
            ch = chr(cp)
            utf8_bytes = ch.encode("utf-8")
        except (ValueError, UnicodeEncodeError):
            return
        byte_tids = []
        for b in utf8_bytes:
            if b in byte_to_tid:
                byte_tids.append(byte_to_tid[b])
            else:
                return  # 이 문자에 필요한 byte token이 없음
        char_to_byte_tids[ch] = byte_tids

    # 1바이트 ASCII (U+0000~U+007F)
    for cp in range(0x0000, 0x0080):
        try_char(cp)

    # 2바이트 (U+0080~U+07FF)
    for cp in range(0x0080, 0x0800):
        try_char(cp)

    # 3바이트: 주요 범위만 샘플 (CJK, 한글 등)
    ranges_3byte = [
        (0x0800, 0x1000),    # 기본 다국어 면 (Devanagari, Armenian 등)
        (0x1100, 0x11FF),    # 한글 자모
        (0x3000, 0x3100),    # CJK 기호
        (0x3131, 0x3164),    # 한글 호환 자모
        (0x4E00, 0x4F00),    # CJK Unified Ideographs 앞부분
        (0xAC00, 0xAD00),    # 한글 음절 (가~힣 앞 256자)
        (0xD000, 0xD800),    # 한글 음절 뒷부분
        (0xFF00, 0x10000),   # 반각/전각 문자
    ]
    for start, end in ranges_3byte:
        for cp in range(start, min(end, 0x10000)):
            try_char(cp)

    # 4바이트: 이모지 및 Supplementary Multilingual Plane 대표 샘플
    ranges_4byte = [
        (0x1F300, 0x1F400),  # 날씨/자연 이모지
        (0x1F400, 0x1F500),  # 동물 이모지
        (0x1F600, 0x1F650),  # 이모티콘 (😀 등)
        (0x1F680, 0x1F700),  # 교통/여행 이모지
        (0x2F800, 0x2F900),  # CJK Compatibility Supplement
    ]
    for start, end in ranges_4byte:
        for cp in range(start, end):
            try_char(cp)

    log.info(f"유니코드 커버리지 대상 문자 수: {len(char_to_byte_tids)}")
    return char_to_byte_tids


# ── HuggingFace 스트리밍 데이터 소스 ──

def iter_hf_source(
    source_cfg: dict,
    max_rows: Optional[int] = None,
) -> Iterator[tuple[int, str]]:
    """HuggingFace datasets 스트리밍 → (row_idx, text) 생성자

    Args:
        source_cfg: SOURCE_CONFIGS의 항목
        max_rows: None이면 전체 스트림

    Yields:
        (row_idx, text)
    """
    from datasets import load_dataset

    log.info(
        f"HF 데이터셋 스트리밍 시작: {source_cfg['path']} "
        f"({source_cfg.get('name', 'default')}) — {source_cfg['description']}"
    )

    ds = load_dataset(
        source_cfg["path"],
        name=source_cfg.get("name"),
        split=source_cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )

    text_key = source_cfg["text_key"]
    row_idx = 0
    for row in ds:
        text = row.get(text_key, "")
        if not text or len(text) < 10:
            row_idx += 1
            continue
        yield row_idx, text
        row_idx += 1
        if max_rows is not None and row_idx >= max_rows:
            break


# ── 청크 버퍼 (디스크 flush) ──

class ChunkBuffer:
    """수집된 row를 청크 Parquet 파일로 버퍼링

    메모리 안전: chunk_size row마다 디스크에 flush.
    """

    def __init__(self, buf_dir: str, chunk_size: int = 50_000):
        self.buf_dir = Path(buf_dir)
        self.buf_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._buf: list[dict] = []
        self._chunk_idx = 0
        self._total_written = 0

    def add(self, text: str, source: str, row_idx: int, n_tokens: int):
        self._buf.append({
            "text": text,
            "source": source,
            "row_idx": row_idx,
            "n_tokens": n_tokens,
        })
        if len(self._buf) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        chunk_path = self.buf_dir / f"chunk_{self._chunk_idx:06d}.parquet"
        table = pa.table(
            {
                "text": pa.array([r["text"] for r in self._buf], type=pa.string()),
                "source": pa.array([r["source"] for r in self._buf], type=pa.string()),
                "row_idx": pa.array([r["row_idx"] for r in self._buf], type=pa.int64()),
                "n_tokens": pa.array([r["n_tokens"] for r in self._buf], type=pa.int32()),
            },
            schema=PARQUET_SCHEMA,
        )
        pq.write_table(table, str(chunk_path), compression="snappy")
        self._total_written += len(self._buf)
        log.debug(f"  청크 flush: {chunk_path.name} ({len(self._buf)} rows)")
        self._buf.clear()
        self._chunk_idx += 1

    def finalize(self) -> int:
        """남은 버퍼 flush → 총 저장 row 수 반환"""
        self.flush()
        return self._total_written

    def chunk_paths(self) -> list[Path]:
        """저장된 청크 파일 목록 (정렬)"""
        return sorted(self.buf_dir.glob("chunk_*.parquet"))


# ── 핵심 스캔 루프 (병렬 배치 토크나이징) ──

# 프로세스별 토크나이저 캐시 (ProcessPoolExecutor initializer가 설정)
_WORKER_TOKENIZER = None
_WORKER_MODEL_ID: str = ""


def _worker_init(model_id: str):
    """ProcessPoolExecutor 워커 초기화 — 프로세스당 한 번만 실행"""
    global _WORKER_TOKENIZER, _WORKER_MODEL_ID
    _WORKER_MODEL_ID = model_id
    from transformers import AutoTokenizer
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _tokenize_batch(texts: list[str]) -> list[list[int]]:
    """배치 토크나이징 워커 함수 — ProcessPoolExecutor 워커 프로세스에서 실행.

    각 워커 프로세스는 _worker_init()으로 이미 토크나이저를 갖고 있다.
    텍스트 리스트만 pickle로 전달받아 처리 후 token_ids 리스트 반환.
    """
    result = _WORKER_TOKENIZER(texts, add_special_tokens=False)
    # __call__ 반환값은 BatchEncoding (dict-like) 또는 리스트일 수 있음
    ids = result["input_ids"] if hasattr(result, "__getitem__") else result
    # ids가 tensor일 경우 list로 변환
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return [list(x) if not isinstance(x, list) else x for x in ids]


def scan_source(
    source_name: str,
    model_id: str,
    vocab_info: dict,
    counter: dict[int, int],
    covered: set[int],
    buf: ChunkBuffer,
    per_token_samples: int,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
    log_interval: int = 10_000,
    n_workers: int = 8,
    batch_size: int = 512,
) -> int:
    """단일 소스를 멀티프로세스 배치 토크나이징으로 스캔해 커버리지를 채운다.

    구조:
    - 메인 프로세스: 스트리밍 읽기 + 배치 조립 + 커버리지 상태 업데이트
    - 워커 프로세스 (n_workers개): 각자 토크나이저를 갖고 배치 병렬 토크나이징
    - 슬라이딩 윈도우: 최대 n_workers*2 개의 future를 동시 in-flight 유지

    Args:
        source_name: SOURCE_CONFIGS 키
        model_id: K-Exaone HuggingFace 모델 ID (워커 초기화에 사용)
        vocab_info: classify_vocab() 반환값
        counter: token_id → 현재 수집 카운트 (메인 프로세스 단독 접근)
        covered: 목표 달성 token_id 집합 (메인 프로세스 단독 접근)
        buf: ChunkBuffer
        per_token_samples: 토큰당 목표 row 수
        dry_run: True면 buf에 저장하지 않음
        max_rows: 이 소스에서 최대 읽을 row 수
        log_interval: 진행 로그 출력 간격
        n_workers: 토크나이징 병렬 프로세스 수
        batch_size: 한 번에 토크나이징할 row 수

    Returns:
        이 소스에서 스캔한 총 row 수
    """
    source_cfg = SOURCE_CONFIGS[source_name]
    all_target = vocab_info["all_target"]
    total_target = len(all_target)
    max_pending = n_workers * 2  # 슬라이딩 윈도우 크기

    scanned = 0
    selected = 0
    t0 = time.time()

    # pending: deque[(future, row_idxs, texts)]
    pending: deque = deque()

    def _process_batch(
        batch_token_ids: list[list[int]],
        batch_row_idxs: list[int],
        batch_texts: list[str],
    ):
        """완료된 배치 결과를 메인 프로세스에서 순차 처리"""
        nonlocal selected
        for token_ids, row_idx, text in zip(batch_token_ids, batch_row_idxs, batch_texts):
            if not token_ids:
                continue
            unique_ids = set(token_ids) & all_target
            # 아직 목표 미달인 토큰 중 이 row가 기여하는 것
            contrib_ids = unique_ids - covered
            if not contrib_ids:
                continue
            # counter 업데이트: 아직 목표 미달인 토큰별 카운트 증가
            # per_token_samples에 달한 경우 covered에 추가
            actually_useful = False
            for tid in contrib_ids:
                old = counter.get(tid, 0)
                if old < per_token_samples:  # 아직 여유 있는 토큰에만 카운트
                    counter[tid] = old + 1
                    actually_useful = True
                    if counter[tid] >= per_token_samples:
                        covered.add(tid)
            if not actually_useful:
                continue
            if not dry_run:
                buf.add(
                    text=text,
                    source=source_name,
                    row_idx=row_idx,
                    n_tokens=len(token_ids),
                )
            selected += 1

    cur_texts: list[str] = []
    cur_idxs: list[int] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(model_id,),
    ) as executor:
        for row_idx, text in iter_hf_source(source_cfg, max_rows=max_rows):
            scanned += 1

            if len(covered) >= total_target:
                log.info("모든 토큰 커버 완료 — 조기 종료")
                break

            cur_texts.append(text)
            cur_idxs.append(row_idx)

            if len(cur_texts) >= batch_size:
                # 텍스트 리스트만 워커로 전달 (토크나이저는 프로세스 내부에 있음)
                fut = executor.submit(_tokenize_batch, cur_texts)
                pending.append((fut, cur_idxs, cur_texts))
                cur_texts = []
                cur_idxs = []

                # 백프레셔: 슬라이딩 윈도우가 꽉 차면 가장 오래된 future 완료 대기
                while len(pending) >= max_pending:
                    f, idxs, txts = pending.popleft()
                    _process_batch(f.result(), idxs, txts)

            if scanned % log_interval == 0:
                elapsed = time.time() - t0
                rate = scanned / elapsed
                pct = len(covered) / total_target * 100
                log.info(
                    f"[{source_name}] 스캔 {scanned:,} | 선택 {selected:,} | "
                    f"커버 {len(covered):,}/{total_target:,} ({pct:.1f}%) | "
                    f"{rate:.0f} rows/s"
                )

        # 남은 부분 배치 제출
        if cur_texts:
            fut = executor.submit(_tokenize_batch, cur_texts)
            pending.append((fut, cur_idxs, cur_texts))

        # 모든 in-flight future 완료 대기
        while pending:
            f, idxs, txts = pending.popleft()
            _process_batch(f.result(), idxs, txts)

    elapsed = time.time() - t0
    log.info(
        f"[{source_name}] 완료 — 스캔 {scanned:,}, 선택 {selected:,}, "
        f"커버 {len(covered):,}/{total_target:,}, {elapsed:.1f}s"
    )
    return scanned


# ── 청크 병합 ──

def merge_chunks(
    buf: ChunkBuffer,
    output_path: str,
    dedup: bool = True,
) -> int:
    """버퍼 청크를 병합해 최종 Parquet 저장

    중복 row_idx 제거 옵션(dedup=True):
    한 row가 여러 토큰에 기여할 수 있어 중복 저장됐을 경우 제거.

    Returns:
        최종 저장된 row 수
    """
    chunk_paths = buf.chunk_paths()
    if not chunk_paths:
        log.warning("병합할 청크 파일 없음")
        return 0

    log.info(f"청크 {len(chunk_paths)}개 병합 → {output_path}")

    seen_keys: set[tuple[str, int]] = set()  # (source, row_idx)
    writer = pq.ParquetWriter(output_path, schema=PARQUET_SCHEMA, compression="snappy")
    total = 0

    for cp in chunk_paths:
        table = pq.read_table(str(cp))
        if dedup:
            # 중복 제거
            keep_mask = []
            sources = table["source"].to_pylist()
            row_idxs = table["row_idx"].to_pylist()
            for src, ridx in zip(sources, row_idxs):
                key = (src, ridx)
                if key not in seen_keys:
                    seen_keys.add(key)
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            import pyarrow.compute as pc
            mask = pa.array(keep_mask, type=pa.bool_())
            table = table.filter(mask)

        writer.write_table(table)
        total += len(table)

    writer.close()
    log.info(f"최종 Parquet 저장: {output_path} ({total:,} rows)")
    return total


# ── 커버리지 리포트 ──

def write_coverage_report(
    vocab_info: dict,
    counter: dict[int, int],
    covered: set[int],
    per_token_samples: int,
    output_path: str,
):
    """토큰별 커버리지 리포트 파일 저장"""
    all_target = vocab_info["all_target"]
    byte_fallback = vocab_info["byte_fallback"]
    missing = all_target - covered

    lines = [
        "=== K-Exaone 토큰 커버리지 리포트 ===",
        f"목표 토큰 수: {len(all_target):,}",
        f"커버된 토큰: {len(covered):,} ({len(covered)/len(all_target)*100:.2f}%)",
        f"미커버 토큰: {len(missing):,}",
        f"토큰당 목표 샘플: {per_token_samples:,}",
        "",
        "--- byte fallback 커버리지 ---",
        f"byte_fallback 전체: {len(byte_fallback):,}",
        f"byte_fallback 커버: {sum(1 for tid in byte_fallback if tid in covered):,}",
        "",
    ]

    if missing:
        lines.append("--- 미커버 토큰 목록 ---")
        # byte fallback 미커버
        missing_byte = {tid: byte_fallback[tid] for tid in missing if tid in byte_fallback}
        if missing_byte:
            lines.append(f"byte fallback 미커버 ({len(missing_byte)}개):")
            for tid, bval in sorted(missing_byte.items(), key=lambda x: x[1]):
                lines.append(f"  token_id={tid}, byte=0x{bval:02X}")

        # 일반 토큰 미커버
        missing_normal = missing - set(byte_fallback.keys())
        if missing_normal:
            lines.append(f"일반 토큰 미커버 ({len(missing_normal)}개):")
            for tid in sorted(missing_normal)[:200]:
                lines.append(f"  token_id={tid}, count={counter.get(tid, 0)}")
            if len(missing_normal) > 200:
                lines.append(f"  ... (총 {len(missing_normal)}개, 처음 200개만 표시)")

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    log.info(f"커버리지 리포트 저장: {output_path}")
    print(report)


# ── 메인 ──

def parse_args():
    p = argparse.ArgumentParser(description="K-Exaone 전체 토큰 커버리지 데이터셋 빌더")
    p.add_argument(
        "--output", default="corpus/coverage_dataset.parquet",
        help="최종 출력 Parquet 파일 경로 (기본: corpus/coverage_dataset.parquet)",
    )
    p.add_argument(
        "--buf_dir", default="corpus/.coverage_buf",
        help="임시 청크 버퍼 디렉토리 (기본: corpus/.coverage_buf)",
    )
    p.add_argument(
        "--sources", nargs="+", default=DEFAULT_SOURCES,
        choices=list(SOURCE_CONFIGS.keys()),
        help=f"사용할 소스 (기본: {DEFAULT_SOURCES})",
    )
    p.add_argument(
        "--per_token_samples", type=int, default=10_000,
        help="토큰당 최대 수집 row 수 (기본: 10,000)",
    )
    p.add_argument(
        "--chunk_size", type=int, default=50_000,
        help="청크 버퍼 크기 (기본: 50,000 rows)",
    )
    p.add_argument(
        "--max_rows", type=int, default=None,
        help="소스당 최대 스캔 row 수 (기본: None = 전체)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="드라이런: 버퍼 저장 없이 커버리지 통계만 계산",
    )
    p.add_argument(
        "--model_id", default="LGAI-EXAONE/K-EXAONE-236B-A23B",
        help="K-Exaone 모델 ID (HuggingFace Hub)",
    )
    p.add_argument(
        "--report", default="corpus/coverage_report.txt",
        help="커버리지 리포트 저장 경로",
    )
    p.add_argument(
        "--no_dedup", action="store_true",
        help="최종 Parquet 병합 시 중복 row 제거 안 함",
    )
    p.add_argument(
        "--log_interval", type=int, default=10_000,
        help="진행 로그 출력 간격 (기본: 10,000 rows)",
    )
    p.add_argument(
        "--n_workers", type=int, default=8,
        help="토크나이징 병렬 프로세스 수 (기본: 8)",
    )
    p.add_argument(
        "--batch_size", type=int, default=512,
        help="배치 토크나이징 크기 (기본: 512 rows/배치)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. 토크나이저 로드 및 vocab 분류 (메인 프로세스) ──
    tokenizer = load_tokenizer(args.model_id)
    vocab_info = classify_vocab(tokenizer)
    # 워커 프로세스에 전달할 필요 없음 — 각자 _worker_init에서 로드
    del tokenizer  # 메모리 절약 (워커가 각자 로드)
    all_target = vocab_info["all_target"]

    log.info(f"커버리지 목표: {len(all_target):,} 토큰, 토큰당 {args.per_token_samples:,} rows")

    # ── 2. 커버리지 추적 상태 ──
    counter: dict[int, int] = {}   # token_id → 수집된 row 수
    covered: set[int] = set()      # 목표 달성한 token_id

    # ── 3. 청크 버퍼 초기화 ──
    buf = ChunkBuffer(args.buf_dir, chunk_size=args.chunk_size)

    # ── 4. 소스별 스캔 ──
    total_scanned = 0
    for source_name in args.sources:
        if len(covered) >= len(all_target):
            log.info("소스 스캔 전 이미 전체 커버 완료")
            break

        remaining = len(all_target) - len(covered)
        log.info(
            f"\n[소스: {source_name}] 잔여 미커버 토큰: {remaining:,}"
        )

        n = scan_source(
            source_name=source_name,
            model_id=args.model_id,
            vocab_info=vocab_info,
            counter=counter,
            covered=covered,
            buf=buf,
            per_token_samples=args.per_token_samples,
            dry_run=args.dry_run,
            max_rows=args.max_rows,
            log_interval=args.log_interval,
            n_workers=args.n_workers,
            batch_size=args.batch_size,
        )
        total_scanned += n

    log.info(f"\n총 스캔: {total_scanned:,} rows")

    # ── 5. 버퍼 마무리 ──
    if not args.dry_run:
        n_buffered = buf.finalize()
        log.info(f"버퍼 총 저장: {n_buffered:,} rows")

        # ── 6. 청크 병합 → 최종 Parquet ──
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        n_final = merge_chunks(buf, args.output, dedup=not args.no_dedup)
        log.info(f"최종 데이터셋: {args.output} ({n_final:,} rows)")
    else:
        log.info("[드라이런] 파일 저장 안 함")

    # ── 7. 커버리지 리포트 ──
    if not args.dry_run:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_coverage_report(
        vocab_info=vocab_info,
        counter=counter,
        covered=covered,
        per_token_samples=args.per_token_samples,
        output_path=args.report if not args.dry_run else "/dev/null",
    )


if __name__ == "__main__":
    main()
