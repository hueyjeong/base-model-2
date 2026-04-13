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
- counter는 numpy int32 배열 (vocab_size 크기, ~600KB).
- 한 청크(chunk_size rows)마다 pyarrow로 디스크에 flush.
- 커버리지가 증가할수록 배치 조기 스킵으로 스캔 속도 향상.
- Rust-native encode_batch_fast로 단일 프로세스 토크나이징 (IPC 오버헤드 없음).

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
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

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

# httpx/urllib3/fsspec HTTP 로그 억제
for _noisy in ("httpx", "urllib3", "fsspec", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


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


# ── HuggingFace 스트리밍 데이터 소스 ──

def _get_hf_parquet_urls(source_cfg: dict) -> list[str]:
    """HF IterableDataset에서 Parquet 파일 URL 목록 추출 (내부 API)"""
    from datasets import load_dataset
    ds = load_dataset(
        source_cfg["path"],
        name=source_cfg.get("name"),
        split=source_cfg["split"],
        streaming=True,
    )
    files = []
    if hasattr(ds, "_ex_iterable") and hasattr(ds._ex_iterable, "kwargs"):
        files = ds._ex_iterable.kwargs.get("files", [])
    del ds
    return files


def iter_hf_batches(
    source_cfg: dict,
    batch_size: int = 512,
    max_rows: Optional[int] = None,
    n_readers: int = 4,
) -> Iterator[tuple[list[int], list[str]]]:
    """HF Parquet 멀티스레드 직접 읽기 → 배치 생성자

    HF datasets 내부 API에서 Parquet URL을 추출한 뒤
    N개 리더 스레드가 서로 다른 파일을 동시에 fsspec + PyArrow로 읽는다.
    각 스레드는 row group 단위로 읽어 공유 큐에 넣고,
    메인 스레드가 batch_size 단위로 잘라 yield한다.

    PyArrow Parquet 디컴프레션은 GIL 해제 → 스레드 간 실제 병렬 실행.
    HF 내부 API 접근 실패 시 .iter(batch_size) fallback.

    Yields:
        (row_idxs, texts) — 유효한 텍스트만 필터링된 배치
    """
    import queue
    import threading

    log.info(
        f"HF 데이터셋 스트리밍 시작: {source_cfg['path']} "
        f"({source_cfg.get('name', 'default')}) — {source_cfg['description']}"
    )

    text_key = source_cfg["text_key"]
    parquet_urls = _get_hf_parquet_urls(source_cfg)

    if not parquet_urls:
        log.warning("Parquet URL 추출 실패 — .iter(batch_size) fallback")
        yield from _iter_hf_batches_fallback(source_cfg, batch_size, max_rows)
        return

    n_readers = min(n_readers, len(parquet_urls))
    log.info(f"Parquet 직접 읽기: {len(parquet_urls)}개 파일, {n_readers}개 리더 스레드")

    # 파일을 리더들에게 라운드로빈 할당
    file_queues: list[list[str]] = [[] for _ in range(n_readers)]
    for i, url in enumerate(parquet_urls):
        file_queues[i % n_readers].append(url)

    # 리더 스레드들이 row group 텍스트를 공유 큐에 넣음
    rg_queue: queue.Queue = queue.Queue(maxsize=n_readers * 4)

    RG_CHUNK = 32  # row group 묶어 읽기 (디컴프레션 효율 + I/O 지터 완화)

    def _reader(urls: list[str]):
        """리더 스레드: 할당된 Parquet 파일들의 row group 묶어 읽기"""
        import fsspec
        try:
            for url in urls:
                fs, path = fsspec.core.url_to_fs(url)
                pf = pq.ParquetFile(fs.open(path))
                n_rg = pf.metadata.num_row_groups
                for start in range(0, n_rg, RG_CHUNK):
                    end = min(start + RG_CHUNK, n_rg)
                    table = pf.read_row_groups(
                        list(range(start, end)), columns=[text_key],
                    )
                    rg_queue.put(table[text_key].to_pylist())
        except Exception as e:
            log.error(f"Parquet reader 오류: {e}")
        finally:
            rg_queue.put(None)  # sentinel

    threads = []
    for urls in file_queues:
        t = threading.Thread(target=_reader, args=(urls,), daemon=True)
        t.start()
        threads.append(t)

    row_idx = 0
    sentinels = 0
    while sentinels < n_readers:
        raw_texts = rg_queue.get()
        if raw_texts is None:
            sentinels += 1
            continue

        for i in range(0, len(raw_texts), batch_size):
            chunk = raw_texts[i:i + batch_size]
            idxs, texts = [], []
            for text in chunk:
                if text and len(text) >= 10:
                    idxs.append(row_idx)
                    texts.append(text)
                row_idx += 1
            if texts:
                yield idxs, texts
            if max_rows is not None and row_idx >= max_rows:
                for t in threads:
                    t.join(timeout=2)
                return

    for t in threads:
        t.join(timeout=5)


def _iter_hf_batches_fallback(
    source_cfg: dict,
    batch_size: int,
    max_rows: Optional[int],
) -> Iterator[tuple[list[int], list[str]]]:
    """HF .iter(batch_size) fallback"""
    from datasets import load_dataset
    ds = load_dataset(
        source_cfg["path"],
        name=source_cfg.get("name"),
        split=source_cfg["split"],
        streaming=True,
    )
    text_key = source_cfg["text_key"]
    row_idx = 0
    for batch in ds.iter(batch_size=batch_size):
        raw_texts = batch[text_key]
        idxs, texts = [], []
        for text in raw_texts:
            if text and len(text) >= 10:
                idxs.append(row_idx)
                texts.append(text)
            row_idx += 1
        if texts:
            yield idxs, texts
        if max_rows is not None and row_idx >= max_rows:
            break


def iter_local_parquet_batches(
    parquet_path: str,
    batch_size: int = 512,
    max_rows: Optional[int] = None,
    n_readers: int = 4,
) -> Iterator[tuple[list[str], list[int], list[str]]]:
    """로컬 Parquet 파일 → (sources, row_idxs, texts) 배치 생성자

    기존 커버리지 데이터셋을 소스로 재샘플링할 때 사용.
    원본 source/row_idx를 그대로 전파하여 추적성 유지.

    Yields:
        (sources, row_idxs, texts) — 유효한 텍스트만 필터링된 배치
    """
    import queue
    import threading

    log.info(f"로컬 Parquet 읽기: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    avg_rows_per_rg = total_rows / max(n_rg, 1)
    log.info(f"  row_groups={n_rg}, total_rows={total_rows:,}, avg={avg_rows_per_rg:,.0f} rows/rg")

    n_readers = min(n_readers, n_rg)

    # 한 번에 읽을 row 수 목표 ~50K → row group 크기에 맞춰 RG_CHUNK 계산
    TARGET_ROWS_PER_CHUNK = 50_000
    RG_CHUNK = max(1, int(TARGET_ROWS_PER_CHUNK / max(avg_rows_per_rg, 1)))
    log.info(f"  리더 {n_readers}개, RG_CHUNK={RG_CHUNK} (~{RG_CHUNK * avg_rows_per_rg:,.0f} rows/묶음)")

    # 큐 크기도 메모리 관리: 리더당 2개까지만 buffer
    rg_queue: queue.Queue = queue.Queue(maxsize=n_readers * 2)

    # row group을 리더들에게 라운드로빈 할당
    rg_assignments: list[list[int]] = [[] for _ in range(n_readers)]
    for i in range(n_rg):
        rg_assignments[i % n_readers].append(i)

    def _reader(rg_indices: list[int]):
        try:
            pf_local = pq.ParquetFile(parquet_path)
            for start in range(0, len(rg_indices), RG_CHUNK):
                rgs = rg_indices[start:start + RG_CHUNK]
                table = pf_local.read_row_groups(
                    rgs, columns=["text", "source", "row_idx"],
                )
                rg_queue.put((
                    table["source"].to_pylist(),
                    table["row_idx"].to_pylist(),
                    table["text"].to_pylist(),
                ))
        except Exception as e:
            log.error(f"로컬 Parquet 리더 오류: {e}")
        finally:
            rg_queue.put(None)

    threads = []
    for rgs in rg_assignments:
        t = threading.Thread(target=_reader, args=(rgs,), daemon=True)
        t.start()
        threads.append(t)

    total_read = 0
    sentinels = 0
    while sentinels < n_readers:
        item = rg_queue.get()
        if item is None:
            sentinels += 1
            continue
        sources_all, idxs_all, texts_all = item

        for i in range(0, len(texts_all), batch_size):
            chunk_src = sources_all[i:i + batch_size]
            chunk_idx = idxs_all[i:i + batch_size]
            chunk_txt = texts_all[i:i + batch_size]

            out_src, out_idx, out_txt = [], [], []
            for s, r, t in zip(chunk_src, chunk_idx, chunk_txt):
                if t and len(t) >= 10:
                    out_src.append(s)
                    out_idx.append(r)
                    out_txt.append(t)
            total_read += len(chunk_txt)
            if out_txt:
                yield out_src, out_idx, out_txt
            if max_rows is not None and total_read >= max_rows:
                for th in threads:
                    th.join(timeout=2)
                return

    for t in threads:
        t.join(timeout=5)


# ── 청크 버퍼 (디스크 flush) ──

class ChunkBuffer:
    """수집된 row를 청크 Parquet 파일로 버퍼링

    메모리 안전: chunk_size row마다 디스크에 flush.
    """

    def __init__(self, buf_dir: str, chunk_size: int = 50_000):
        self.buf_dir = Path(buf_dir)
        self.buf_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        # columnar 저장: dict 오버헤드 제거 (청크당 ~12MB 절약)
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._row_idxs: list[int] = []
        self._n_tokens: list[int] = []
        self._count = 0
        self._chunk_idx = 0
        self._total_written = 0

    def add(self, text: str, source: str, row_idx: int, n_tokens: int):
        self._texts.append(text)
        self._sources.append(source)
        self._row_idxs.append(row_idx)
        self._n_tokens.append(n_tokens)
        self._count += 1
        if self._count >= self.chunk_size:
            self.flush()

    def flush(self):
        if self._count == 0:
            return
        chunk_path = self.buf_dir / f"chunk_{self._chunk_idx:06d}.parquet"
        table = pa.table(
            {
                "text": pa.array(self._texts, type=pa.string()),
                "source": pa.array(self._sources, type=pa.string()),
                "row_idx": pa.array(self._row_idxs, type=pa.int64()),
                "n_tokens": pa.array(self._n_tokens, type=pa.int32()),
            },
            schema=PARQUET_SCHEMA,
        )
        pq.write_table(table, str(chunk_path), compression="snappy")
        self._total_written += self._count
        log.debug(f"  청크 flush: {chunk_path.name} ({self._count} rows)")
        self._texts.clear()
        self._sources.clear()
        self._row_idxs.clear()
        self._n_tokens.clear()
        self._count = 0
        self._chunk_idx += 1

    def finalize(self) -> int:
        """남은 버퍼 flush → 총 저장 row 수 반환"""
        self.flush()
        return self._total_written

    def chunk_paths(self) -> list[Path]:
        """저장된 청크 파일 목록 (정렬)"""
        return sorted(self.buf_dir.glob("chunk_*.parquet"))


# ── 핵심 스캔 루프 (Rust-native 토크나이징) ──


def scan_source(
    source_name: str,
    rust_tokenizer,
    tokenizer_fallback,
    vocab_size: int,
    target_arr: "np.ndarray",
    counter: "np.ndarray",
    buf: ChunkBuffer,
    per_token_samples: int,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
    log_interval: int = 10_000,
    batch_size: int = 512,
    n_readers: int = 4,
) -> int:
    """단일 소스를 스캔해 커버리지를 채운다.

    구조:
    - iter_hf_batches: .iter(batch_size)로 Arrow 배치 이터레이션 (row-by-row 대비 ~1.4x)
    - encode_batch_fast: Rust rayon 내부 멀티스레드 토크나이징
    - numpy 배열 기반 커버리지 추적 + 배치 조기 스킵

    Args:
        source_name: SOURCE_CONFIGS 키
        rust_tokenizer: tokenizers.Tokenizer (Rust 백엔드) 또는 None
        tokenizer_fallback: rust_tokenizer 없을 때 HuggingFace tokenizer
        vocab_size: 토크나이저 vocab 크기
        target_arr: 커버리지 대상 token_id numpy 배열 (sorted)
        counter: np.ndarray[int32] — token_id별 수집 카운트
        buf: ChunkBuffer
        per_token_samples: 토큰당 목표 row 수
        dry_run: True면 buf에 저장하지 않음
        max_rows: 이 소스에서 최대 읽을 row 수
        log_interval: 진행 로그 출력 간격
        batch_size: 한 번에 토크나이징할 row 수

    Returns:
        이 소스에서 스캔한 총 row 수
    """
    source_cfg = SOURCE_CONFIGS[source_name]
    total_target = len(target_arr)

    scanned = 0
    selected = 0
    t0 = time.time()

    def _tokenize(texts: list[str]) -> list[list[int]]:
        """Rust-native encode_batch_fast, 없으면 HF fallback"""
        if rust_tokenizer is not None:
            encs = rust_tokenizer.encode_batch_fast(texts, add_special_tokens=False)
            return [enc.ids for enc in encs]
        result = tokenizer_fallback(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )
        return result["input_ids"]

    def _process_batch(
        batch_token_ids: list[list[int]],
        batch_row_idxs: list[int],
        batch_texts: list[str],
    ):
        """배치 결과 → numpy 벡터화 커버리지 업데이트"""
        nonlocal selected

        # 배치 레벨 조기 스킵: 배치 전체의 unique ID 중 미달인 것이 없으면 스킵
        all_ids_in_batch: set[int] = set()
        for ids in batch_token_ids:
            all_ids_in_batch.update(ids)
        if all_ids_in_batch:
            batch_arr = np.fromiter(all_ids_in_batch, dtype=np.int32, count=len(all_ids_in_batch))
            batch_arr = batch_arr[batch_arr < vocab_size]
            if len(batch_arr) == 0 or not np.any(counter[batch_arr] < per_token_samples):
                return

        for token_ids, row_idx, text in zip(batch_token_ids, batch_row_idxs, batch_texts):
            if not token_ids:
                continue
            arr = np.array(token_ids, dtype=np.int32)
            unique_ids = np.unique(arr)
            unique_ids = unique_ids[unique_ids < vocab_size]
            mask = counter[unique_ids] < per_token_samples
            contrib = unique_ids[mask]
            if len(contrib) == 0:
                continue
            counter[contrib] += 1
            if not dry_run:
                buf.add(
                    text=text,
                    source=source_name,
                    row_idx=row_idx,
                    n_tokens=len(token_ids),
                )
            selected += 1

    # ── 파이프라인: 토크나이징(GIL 해제)과 coverage(GIL 보유)를 overlap ──
    # 별도 스레드에서 토크나이징 → 결과 큐 → 메인 스레드에서 coverage 처리
    # encode_batch_fast가 GIL 해제하므로 두 단계가 실제 병렬 실행
    import queue as _queue, threading as _th

    tok_queue: _queue.Queue = _queue.Queue(maxsize=2)

    def _tokenize_worker():
        """토크나이징 스레드: 텍스트 배치 → token ID 배치"""
        try:
            for batch_idxs, batch_texts in iter_hf_batches(
                source_cfg, batch_size, max_rows, n_readers,
            ):
                batch_ids = _tokenize(batch_texts)
                tok_queue.put((batch_idxs, batch_texts, batch_ids))
        except Exception as e:
            log.error(f"토크나이징 스레드 오류: {e}")
        finally:
            tok_queue.put(None)

    tok_thread = _th.Thread(target=_tokenize_worker, daemon=True)
    tok_thread.start()

    while True:
        item = tok_queue.get()
        if item is None:
            break
        batch_idxs, batch_texts, batch_ids = item
        scanned += len(batch_texts)

        _process_batch(batch_ids, batch_idxs, batch_texts)

        # 전체 커버 완료 체크
        if np.all(counter[target_arr] >= per_token_samples):
            log.info("모든 토큰 커버 완료 — 조기 종료")
            break

        if scanned % log_interval < batch_size:
            elapsed = time.time() - t0
            rate = scanned / elapsed
            n_covered = int(np.sum(counter[target_arr] >= per_token_samples))
            pct = n_covered / total_target * 100
            log.info(
                f"[{source_name}] 스캔 {scanned:,} | 선택 {selected:,} | "
                f"커버 {n_covered:,}/{total_target:,} ({pct:.1f}%) | "
                f"{rate:.0f} rows/s"
            )

    tok_thread.join(timeout=5)

    elapsed = time.time() - t0
    n_covered = int(np.sum(counter[target_arr] >= per_token_samples))
    log.info(
        f"[{source_name}] 완료 — 스캔 {scanned:,}, 선택 {selected:,}, "
        f"커버 {n_covered:,}/{total_target:,}, {elapsed:.1f}s"
    )
    return scanned


def scan_local_parquet(
    parquet_path: str,
    rust_tokenizer,
    tokenizer_fallback,
    vocab_size: int,
    target_arr: "np.ndarray",
    counter: "np.ndarray",
    buf: ChunkBuffer,
    per_token_samples: int,
    dry_run: bool = False,
    max_rows: Optional[int] = None,
    log_interval: int = 10_000,
    batch_size: int = 512,
    n_readers: int = 4,
) -> int:
    """로컬 Parquet 파일을 소스로 재샘플링.

    기존 scan_source와 동일한 파이프라인이지만:
    - iter_local_parquet_batches로 로컬 파일에서 (source, row_idx, text) 읽기
    - 원본 source/row_idx를 그대로 buf에 기록 (추적성 유지)

    Returns:
        스캔한 총 row 수
    """
    import queue as _queue, threading as _th

    total_target = len(target_arr)
    scanned = 0
    selected = 0
    t0 = time.time()

    def _tokenize(texts: list[str]) -> list[list[int]]:
        if rust_tokenizer is not None:
            encs = rust_tokenizer.encode_batch_fast(texts, add_special_tokens=False)
            return [enc.ids for enc in encs]
        result = tokenizer_fallback(
            texts, add_special_tokens=False,
            return_attention_mask=False, return_token_type_ids=False,
        )
        return result["input_ids"]

    def _process_batch(batch_token_ids, batch_sources, batch_row_idxs, batch_texts):
        nonlocal selected

        all_ids_in_batch: set[int] = set()
        for ids in batch_token_ids:
            all_ids_in_batch.update(ids)
        if all_ids_in_batch:
            batch_arr = np.fromiter(all_ids_in_batch, dtype=np.int32, count=len(all_ids_in_batch))
            batch_arr = batch_arr[batch_arr < vocab_size]
            if len(batch_arr) == 0 or not np.any(counter[batch_arr] < per_token_samples):
                return

        for token_ids, src, row_idx, text in zip(
            batch_token_ids, batch_sources, batch_row_idxs, batch_texts,
        ):
            if not token_ids:
                continue
            arr = np.array(token_ids, dtype=np.int32)
            unique_ids = np.unique(arr)
            unique_ids = unique_ids[unique_ids < vocab_size]
            mask = counter[unique_ids] < per_token_samples
            contrib = unique_ids[mask]
            if len(contrib) == 0:
                continue
            counter[contrib] += 1
            if not dry_run:
                buf.add(text=text, source=src, row_idx=row_idx, n_tokens=len(token_ids))
            selected += 1

    tok_queue: _queue.Queue = _queue.Queue(maxsize=2)

    def _tokenize_worker():
        try:
            for srcs, idxs, texts in iter_local_parquet_batches(
                parquet_path, batch_size, max_rows, n_readers,
            ):
                batch_ids = _tokenize(texts)
                tok_queue.put((srcs, idxs, texts, batch_ids))
        except Exception as e:
            log.error(f"토크나이징 스레드 오류: {e}")
        finally:
            tok_queue.put(None)

    tok_thread = _th.Thread(target=_tokenize_worker, daemon=True)
    tok_thread.start()

    source_label = Path(parquet_path).name
    while True:
        item = tok_queue.get()
        if item is None:
            break
        srcs, idxs, texts, batch_ids = item
        scanned += len(texts)

        _process_batch(batch_ids, srcs, idxs, texts)

        if np.all(counter[target_arr] >= per_token_samples):
            log.info("모든 토큰 커버 완료 — 조기 종료")
            break

        if scanned % log_interval < batch_size:
            elapsed = time.time() - t0
            rate = scanned / elapsed
            n_covered = int(np.sum(counter[target_arr] >= per_token_samples))
            pct = n_covered / total_target * 100
            log.info(
                f"[{source_label}] 스캔 {scanned:,} | 선택 {selected:,} | "
                f"커버 {n_covered:,}/{total_target:,} ({pct:.1f}%) | "
                f"{rate:.0f} rows/s"
            )

    tok_thread.join(timeout=5)

    elapsed = time.time() - t0
    n_covered = int(np.sum(counter[target_arr] >= per_token_samples))
    log.info(
        f"[{source_label}] 완료 — 스캔 {scanned:,}, 선택 {selected:,}, "
        f"커버 {n_covered:,}/{total_target:,}, {elapsed:.1f}s"
    )
    return scanned


# ── 청크 병합 ──

def merge_chunks(
    buf: ChunkBuffer,
    output_path: str,
    dedup: bool = False,
) -> int:
    """버퍼 청크를 병합해 최종 Parquet 저장

    스트리밍 방식: 청크를 하나씩 읽어 ParquetWriter로 기록.
    메모리 사용량 = 한 청크 크기 (전체를 메모리에 올리지 않음).

    Note: scan_source는 각 (source, row_idx)를 정확히 한 번만 buf.add()에
    전달하므로 구조적으로 중복이 없다. --dedup은 안전장치로만 존재.

    Returns:
        최종 저장된 row 수
    """
    chunk_paths = buf.chunk_paths()
    if not chunk_paths:
        log.warning("병합할 청크 파일 없음")
        return 0

    log.info(f"청크 {len(chunk_paths)}개 병합 → {output_path}")

    if dedup:
        # 전체 concat 후 group_by dedup (메모리 많이 사용)
        tables = [pq.read_table(str(cp)) for cp in chunk_paths]
        combined = pa.concat_tables(tables)
        del tables
        combined = (
            combined
            .group_by(["source", "row_idx"])
            .aggregate([("text", "one"), ("n_tokens", "one")])
            .rename_columns(["source", "row_idx", "text", "n_tokens"])
            .select(["text", "source", "row_idx", "n_tokens"])
        )
        pq.write_table(combined, output_path, compression="snappy")
        total = len(combined)
    else:
        # 스트리밍 병합: 청크 하나씩 읽어 바로 기록 (메모리 효율적)
        writer = pq.ParquetWriter(output_path, schema=PARQUET_SCHEMA, compression="snappy")
        total = 0
        for cp in chunk_paths:
            table = pq.read_table(str(cp))
            writer.write_table(table)
            total += len(table)
        writer.close()

    log.info(f"최종 Parquet 저장: {output_path} ({total:,} rows)")
    return total


# ── 커버리지 리포트 ──

def write_coverage_report(
    vocab_info: dict,
    counter: "np.ndarray",
    target_arr: "np.ndarray",
    per_token_samples: int,
    output_path: str,
):
    """토큰별 커버리지 리포트 파일 저장

    Args:
        counter: np.ndarray[int32] — token_id별 수집 카운트
        target_arr: 커버리지 대상 token_id 배열
    """
    all_target = vocab_info["all_target"]
    byte_fallback = vocab_info["byte_fallback"]

    covered_mask = counter[target_arr] >= per_token_samples
    covered = set(target_arr[covered_mask].tolist())
    missing = all_target - covered
    n_covered = len(covered)

    lines = [
        "=== K-Exaone 토큰 커버리지 리포트 ===",
        f"목표 토큰 수: {len(all_target):,}",
        f"커버된 토큰: {n_covered:,} ({n_covered/len(all_target)*100:.2f}%)",
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
        missing_byte = {tid: byte_fallback[tid] for tid in missing if tid in byte_fallback}
        if missing_byte:
            lines.append(f"byte fallback 미커버 ({len(missing_byte)}개):")
            for tid, bval in sorted(missing_byte.items(), key=lambda x: x[1]):
                lines.append(f"  token_id={tid}, byte=0x{bval:02X}")

        missing_normal = missing - set(byte_fallback.keys())
        if missing_normal:
            lines.append(f"일반 토큰 미커버 ({len(missing_normal)}개):")
            for tid in sorted(missing_normal)[:200]:
                lines.append(f"  token_id={tid}, count={int(counter[tid])}")
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
        "--buf_dir", default=".tmp/.coverage_buf",
        help="임시 청크 버퍼 디렉토리 (기본: .tmp/.coverage_buf)",
    )
    p.add_argument(
        "--sources", nargs="+", default=DEFAULT_SOURCES,
        choices=list(SOURCE_CONFIGS.keys()),
        help=f"사용할 HF 소스 (기본: {DEFAULT_SOURCES}). --input_parquet 지정 시 무시",
    )
    p.add_argument(
        "--input_parquet", default=None,
        help="로컬 Parquet 파일에서 재샘플링 (HF 소스 대신). "
             "스키마: text, source, row_idx, n_tokens",
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
        "--dedup", action="store_true",
        help="최종 Parquet 병합 시 중복 row 제거 (기본: 안 함, 파이프라인이 중복 없이 설계됨)",
    )
    p.add_argument(
        "--log_interval", type=int, default=10_000,
        help="진행 로그 출력 간격 (기본: 10,000 rows)",
    )
    p.add_argument(
        "--batch_size", type=int, default=512,
        help="배치 토크나이징 크기 (기본: 512 rows/배치)",
    )
    p.add_argument(
        "--n_readers", type=int, default=4,
        help="Parquet 병렬 리더 스레드 수 (기본: 4)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. 토크나이저 로드 및 vocab 분류 ──
    tokenizer = load_tokenizer(args.model_id)
    vocab_info = classify_vocab(tokenizer)
    vocab_size = tokenizer.vocab_size
    all_target = vocab_info["all_target"]

    # Rust 백엔드 토크나이저 추출 (encode_batch_fast 지원)
    rust_tokenizer = None
    tokenizer_fallback = None
    if hasattr(tokenizer, "backend_tokenizer"):
        rust_tokenizer = tokenizer.backend_tokenizer
        log.info("Rust-native encode_batch_fast 사용 (단일 프로세스, rayon 멀티스레드)")
        del tokenizer  # Rust 백엔드만 유지, HF wrapper 해제
    else:
        log.warning("Rust 백엔드 없음 — HF tokenizer __call__ fallback 사용")
        tokenizer_fallback = tokenizer

    log.info(f"커버리지 목표: {len(all_target):,} 토큰, 토큰당 {args.per_token_samples:,} rows")

    # ── 2. 커버리지 추적 상태 (numpy 배열) ──
    # 비대상 토큰(special 등)은 이미 커버된 것으로 초기화 → per-row 필터링 불필요
    counter = np.full(vocab_size, args.per_token_samples, dtype=np.int32)
    target_arr = np.array(sorted(all_target), dtype=np.int32)
    counter[target_arr] = 0  # 대상 토큰만 0에서 시작

    # ── 3. 청크 버퍼 초기화 ──
    buf = ChunkBuffer(args.buf_dir, chunk_size=args.chunk_size)

    # ── 4. 스캔: 로컬 Parquet 재샘플링 OR HF 소스 ──
    total_scanned = 0
    if args.input_parquet:
        log.info(
            f"\n[로컬 Parquet 재샘플링] {args.input_parquet} "
            f"→ 토큰당 {args.per_token_samples:,} rows"
        )
        total_scanned = scan_local_parquet(
            parquet_path=args.input_parquet,
            rust_tokenizer=rust_tokenizer,
            tokenizer_fallback=tokenizer_fallback,
            vocab_size=vocab_size,
            target_arr=target_arr,
            counter=counter,
            buf=buf,
            per_token_samples=args.per_token_samples,
            dry_run=args.dry_run,
            max_rows=args.max_rows,
            log_interval=args.log_interval,
            batch_size=args.batch_size,
            n_readers=args.n_readers,
        )
    else:
        for source_name in args.sources:
            n_covered = int(np.sum(counter[target_arr] >= args.per_token_samples))
            if n_covered >= len(all_target):
                log.info("소스 스캔 전 이미 전체 커버 완료")
                break

            remaining = len(all_target) - n_covered
            log.info(
                f"\n[소스: {source_name}] 잔여 미커버 토큰: {remaining:,}"
            )

            n = scan_source(
                source_name=source_name,
                rust_tokenizer=rust_tokenizer,
                tokenizer_fallback=tokenizer_fallback,
                vocab_size=vocab_size,
                target_arr=target_arr,
                counter=counter,
                buf=buf,
                per_token_samples=args.per_token_samples,
                dry_run=args.dry_run,
                max_rows=args.max_rows,
                log_interval=args.log_interval,
                batch_size=args.batch_size,
                n_readers=args.n_readers,
            )
            total_scanned += n

    log.info(f"\n총 스캔: {total_scanned:,} rows")

    # ── 5. 버퍼 마무리 ──
    if not args.dry_run:
        n_buffered = buf.finalize()
        log.info(f"버퍼 총 저장: {n_buffered:,} rows")

        # ── 6. 청크 병합 → 최종 Parquet ──
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        n_final = merge_chunks(buf, args.output, dedup=args.dedup)
        log.info(f"최종 데이터셋: {args.output} ({n_final:,} rows)")
    else:
        log.info("[드라이런] 파일 저장 안 함")

    # ── 7. 커버리지 리포트 ──
    if not args.dry_run:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_coverage_report(
        vocab_info=vocab_info,
        counter=counter,
        target_arr=target_arr,
        per_token_samples=args.per_token_samples,
        output_path=args.report if not args.dry_run else "/dev/null",
    )


if __name__ == "__main__":
    main()
