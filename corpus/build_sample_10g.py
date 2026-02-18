#!/usr/bin/env python3
"""
build_sample_10g.py
===================
corpus/sample_ko.jsonl, sample_ja.jsonl, sample_en.jsonl 에서
한국어:일본어:영어 = 8:1:1 **바이트** 비율로 무작위 샘플링하여
총 ~10 GB 크기의 sample_10g.jsonl 을 생성합니다.

전략 (2-pass):
  Pass 1 — 각 파일을 순차 스캔하여 라인 수와 각 라인의 바이트 길이를 수집
  Pass 2 — 셔플된 라인 인덱스를 목표 바이트까지 선택, set으로 저장 →
           파일을 다시 순차 스캔하며 선택된 라인만 임시 파일에 기록
  최종   — 임시 파일들을 합치면서 언어 간 셔플

메모리: 라인 길이 배열(uint16 per line, ~250 MB for 130M lines)
I/O:   각 파일 2번 순차 읽기 → SSD/HDD 모두 빠름

사용법:
    python3 corpus/build_sample_10g.py
"""

import os
import sys
import random
import time
import array
import tempfile
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
TARGET_TOTAL_BYTES = 10 * (1024 ** 3)  # 10 GiB
RATIO = {"ko": 8, "ja": 1, "en": 1}
RATIO_SUM = sum(RATIO.values())

CORPUS_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {
    "ko": CORPUS_DIR / "sample_ko.jsonl",
    "ja": CORPUS_DIR / "sample_ja.jsonl",
    "en": CORPUS_DIR / "sample_en.jsonl",
}
OUTPUT_FILE = CORPUS_DIR / "sample_10g.jsonl"

SEED = 42
PROGRESS_LINES = 10_000_000


def scan_line_lengths(filepath: Path) -> array.array:
    """
    파일을 순차 읽기하여 각 라인의 바이트 길이를 array('I')에 저장.
    uint32 → 라인당 최대 ~4 GB (충분).
    """
    lengths = array.array("I")  # unsigned int, 4 bytes each
    file_size = filepath.stat().st_size
    print(f"  스캔: {filepath.name} ({file_size / 1e9:.2f} GB)")

    count = 0
    with open(filepath, "rb") as f:
        for line in f:
            lengths.append(len(line))
            count += 1
            if count % PROGRESS_LINES == 0:
                print(f"    ... {count:,} 라인")

    print(f"    완료: {count:,} 라인")
    return lengths


def select_indices(
    lengths: array.array,
    target_bytes: int,
    rng: random.Random,
) -> set[int]:
    """
    라인 인덱스를 셔플한 뒤, 목표 바이트에 도달할 때까지 선택.
    선택된 인덱스의 set을 반환.
    """
    n = len(lengths)
    indices = list(range(n))
    rng.shuffle(indices)

    selected = set()
    current_bytes = 0

    for idx in indices:
        ln = lengths[idx]
        if current_bytes + ln > target_bytes:
            break
        selected.add(idx)
        current_bytes += ln

    return selected, current_bytes


def extract_selected(
    filepath: Path,
    selected: set[int],
    tmp_path: Path,
    lang: str,
) -> int:
    """
    파일을 순차 스캔하며, selected에 포함된 라인만 tmp_path에 기록.
    기록한 바이트 수를 반환.
    """
    written = 0
    count = 0
    total_selected = len(selected)

    print(f"  [{lang}] {filepath.name} → {tmp_path.name}  ({total_selected:,} 라인 추출)")

    with open(filepath, "rb") as fin, open(tmp_path, "wb") as fout:
        for i, line in enumerate(fin):
            if i in selected:
                fout.write(line)
                written += len(line)
                count += 1
                if count % 5_000_000 == 0:
                    print(f"    ... {count:,} / {total_selected:,} ({written / 1e9:.2f} GB)")

    print(f"    [{lang}] 추출 완료: {count:,} 라인, {written / 1e9:.4f} GB")
    return written


def shuffle_merge(tmp_files: list[Path], output: Path):
    """
    여러 임시 파일의 라인들을 합치면서 셔플하여 최종 출력 파일에 기록.

    메모리 효율을 위해:
    1) 먼저 (파일 인덱스, 라인 오프셋) 목록을 만들어 셔플
    2) 셔플 순서대로 읽어 기록

    -- 하지만 각 임시 파일은 ~1-8 GB 라 전부 메모리에 올릴 수 없으므로 --
    대신: 라인 번호 기반. 각 파일의 라인 수만 알면 셔플 인덱스 생성 가능.
    """
    # 각 tmp 파일의 라인 수와 오프셋 수집
    file_line_counts = []
    for tf in tmp_files:
        count = 0
        with open(tf, "rb") as f:
            for _ in f:
                count += 1
        file_line_counts.append(count)
        print(f"    {tf.name}: {count:,} 라인")

    total_lines = sum(file_line_counts)
    print(f"  총 라인: {total_lines:,}")

    # (file_idx, line_no) 배열 생성 및 셔플 → 메모리 ~total_lines * 8 bytes
    # 대신 간단히: 각 파일의 라인을 오프셋 배열로 저장, 글로벌 셔플
    # 하지만 라인이 수천만개면 좀 큼.

    # 더 간단한 방법: 각 임시 파일을 mmap해서 오프셋 인덱스를 구축 후 셔플
    # 아니면, 그냥 글로벌 라인 번호를 셔플하고, 해당 파일의 라인으로 매핑

    rng = random.Random(SEED + 1)

    # 글로벌 인덱스 → (file_idx, local_line_no) 매핑
    # 글로벌 인덱스 = 0..total_lines-1
    # cumulative offsets
    cum = []
    c = 0
    for lc in file_line_counts:
        cum.append(c)
        c += lc

    global_indices = list(range(total_lines))
    print(f"  셔플 중... ({total_lines:,} 라인)")
    rng.shuffle(global_indices)

    # 각 파일의 라인 오프셋을 구축 (바이트 오프셋 목록)
    print(f"  각 파일 라인 오프셋 구축 중...")
    file_offsets: list[array.array] = []
    for tf in tmp_files:
        offsets = array.array("Q")  # uint64
        with open(tf, "rb") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line)
        file_offsets.append(offsets)

    # 셔플 순서대로 읽어서 출력
    print(f"  셔플 순서대로 {output.name} 에 기록 중...")
    # 파일 핸들을 열어둠
    file_handles = [open(tf, "rb") for tf in tmp_files]

    try:
        with open(output, "wb") as fout:
            written = 0
            for progress, gi in enumerate(global_indices):
                # gi → file_idx, local_line
                # binary search in cum
                fi = 0
                for j in range(len(cum) - 1, -1, -1):
                    if gi >= cum[j]:
                        fi = j
                        break
                local_line = gi - cum[fi]

                fh = file_handles[fi]
                fh.seek(file_offsets[fi][local_line])
                line = fh.readline()
                fout.write(line)
                written += len(line)

                if (progress + 1) % 5_000_000 == 0:
                    print(f"    ... {progress + 1:,} / {total_lines:,} ({written / 1e9:.2f} GB)")

        print(f"  기록 완료: {written / 1e9:.4f} GB")
    finally:
        for fh in file_handles:
            fh.close()


def main():
    t0 = time.time()
    rng = random.Random(SEED)

    # ──────────────────────────────────────────────────
    # Pass 1: 라인 길이 스캔
    # ──────────────────────────────────────────────────
    print("=" * 60)
    print("Pass 1: 라인 길이 스캔")
    print("=" * 60)
    all_lengths: dict[str, array.array] = {}
    for lang, fpath in SOURCE_FILES.items():
        if not fpath.exists():
            print(f"ERROR: {fpath} 파일을 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        all_lengths[lang] = scan_line_lengths(fpath)

    # ──────────────────────────────────────────────────
    # 목표 바이트 산출
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("비율 기반 목표 바이트 산출")
    print("=" * 60)

    targets: dict[str, int] = {}
    for lang in RATIO:
        targets[lang] = TARGET_TOTAL_BYTES * RATIO[lang] // RATIO_SUM

    # 파일 크기로 제한
    for lang in RATIO:
        file_size = SOURCE_FILES[lang].stat().st_size
        if targets[lang] > file_size:
            print(
                f"  주의: {lang} 목표({targets[lang]/1e9:.2f}GB)가 "
                f"파일 크기({file_size/1e9:.2f}GB)를 초과 → 파일 크기로 제한"
            )
            targets[lang] = file_size

    for lang in RATIO:
        print(f"  {lang}: {targets[lang] / 1e9:.2f} GB")
    print(f"  합계: {sum(targets.values()) / 1e9:.2f} GB")

    # ──────────────────────────────────────────────────
    # 인덱스 선택
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("인덱스 선택 (셔플 + 목표 바이트)")
    print("=" * 60)

    selected_sets: dict[str, set[int]] = {}
    for lang in RATIO:
        print(f"  [{lang}] {len(all_lengths[lang]):,} 라인 중 선택...")
        sel, sel_bytes = select_indices(all_lengths[lang], targets[lang], rng)
        selected_sets[lang] = sel
        print(f"    → {len(sel):,} 라인 선택 ({sel_bytes / 1e9:.4f} GB)")

    # 메모리 절약: lengths 해제
    del all_lengths

    # ──────────────────────────────────────────────────
    # Pass 2: 선택된 라인 추출
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Pass 2: 선택된 라인 추출")
    print("=" * 60)

    tmp_dir = CORPUS_DIR
    tmp_files: list[Path] = []
    for lang in RATIO:
        tmp_path = tmp_dir / f"_tmp_sample_{lang}.jsonl"
        tmp_files.append(tmp_path)
        extract_selected(
            SOURCE_FILES[lang],
            selected_sets[lang],
            tmp_path,
            lang,
        )

    # 메모리 해제
    del selected_sets

    # ──────────────────────────────────────────────────
    # 최종 셔플 + 병합
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("최종 셔플 및 병합")
    print("=" * 60)
    shuffle_merge(tmp_files, OUTPUT_FILE)

    # 임시 파일 정리
    for tf in tmp_files:
        tf.unlink()
        print(f"  삭제: {tf.name}")

    elapsed = time.time() - t0
    final_size = OUTPUT_FILE.stat().st_size
    print()
    print("=" * 60)
    print(f"완료! {OUTPUT_FILE.name}")
    print(f"  크기: {final_size / 1e9:.4f} GB ({final_size / (1024**3):.4f} GiB)")
    print(f"  소요 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
    print("=" * 60)


if __name__ == "__main__":
    main()
