#!/usr/bin/env python3
"""
build_sample_20g.py
===================
corpus/sample_ko.jsonl, sample_ja.jsonl, sample_en.jsonl 에서
한국어:일본어:영어 = 8:1:1 **바이트** 비율로 무작위 샘플링하여
총 ~20 GB 크기의 sample_20g.jsonl 을 생성합니다.

기존 sample_10g.jsonl 과 val_50k.jsonl 의 데이터는 제외하고 생성합니다.

전략 (2-pass):
  Pass 0 — 제외할 파일(sample_10g.jsonl, val_50k.jsonl) 스캔하여 라인 해시 수집
  Pass 1 — 각 파일을 순차 스캔하여 라인 수와 각 라인의 바이트 길이를 수집 (해시 캐시에 포함된 라인은 길이 0으로 마킹)
  Pass 2 — 셔플된 라인 인덱스를 목표 바이트까지 선택, set으로 저장 →
           파일을 다시 순차 스캔하며 선택된 라인만 임시 파일에 기록
  최종   — 임시 파일들을 합치면서 언어 간 셔플
"""

import os
import sys
import random
import time
import array
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
TARGET_TOTAL_BYTES = 20 * (1024 ** 3)  # 20 GiB
RATIO = {"ko": 8, "ja": 1, "en": 1}
RATIO_SUM = sum(RATIO.values())

CORPUS_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {
    "ko": CORPUS_DIR / "sample_ko.jsonl",
    "ja": CORPUS_DIR / "sample_ja.jsonl",
    "en": CORPUS_DIR / "sample_en.jsonl",
}
OUTPUT_FILE = CORPUS_DIR / "sample_20g.jsonl"
EXCLUDE_FILES = [
    CORPUS_DIR / "sample_10g.jsonl",
    CORPUS_DIR / "val_50k.jsonl",
]

SEED = 42
PROGRESS_LINES = 10_000_000


def collect_exclude_hashes(filepaths: list[Path]) -> set[int]:
    """
    제외할 파일들을 순차 읽기하여 각 라인의 해시값을 set에 저장.
    """
    exclude_hashes = set()
    for filepath in filepaths:
        if not filepath.exists():
            print(f"  주의: {filepath.name} 파일이 존재하지 않아 건너뜁니다.")
            continue
            
        file_size = filepath.stat().st_size
        print(f"  해시 수집: {filepath.name} ({file_size / 1e9:.2f} GB)")
        
        count = 0
        with open(filepath, "rb") as f:
            for line in f:
                exclude_hashes.add(hash(line))
                count += 1
                if count % PROGRESS_LINES == 0:
                    print(f"    ... {count:,} 라인")
                    
        print(f"    수집 완료: {count:,} 라인 (현재 누적 고유 해시: {len(exclude_hashes):,})")
        
    return exclude_hashes


def scan_line_lengths(filepath: Path, exclude_hashes: set[int]) -> array.array:
    """
    파일을 순차 읽기하여 각 라인의 바이트 길이를 array('I')에 저장.
    순서 보장을 위해 모든 라인에 대해 배열 항목을 추가하되, 
    제외할 라인의 길이는 0으로 저장.
    """
    lengths = array.array("I")  # unsigned int, 4 bytes each
    file_size = filepath.stat().st_size
    print(f"  스캔: {filepath.name} ({file_size / 1e9:.2f} GB)")

    count = 0
    excluded_count = 0
    with open(filepath, "rb") as f:
        for line in f:
            if hash(line) in exclude_hashes:
                lengths.append(0)
                excluded_count += 1
            else:
                lengths.append(len(line))
            
            count += 1
            if count % PROGRESS_LINES == 0:
                print(f"    ... {count:,} 라인 (제외됨: {excluded_count:,})")

    print(f"    완료: {count:,} 라인 (총 제외됨: {excluded_count:,})")
    return lengths


def select_indices(
    lengths: array.array,
    target_bytes: int,
    rng: random.Random,
) -> set[int]:
    """
    라인 인덱스를 셔플한 뒤, 목표 바이트에 도달할 때까지 선택.
    선택된 인덱스의 set을 반환. (결측처리된 길이 0은 스킵)
    """
    n = len(lengths)
    indices = list(range(n))
    rng.shuffle(indices)

    selected = set()
    current_bytes = 0

    for idx in indices:
        ln = lengths[idx]
        if ln == 0:
            continue
            
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
    """
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

    rng = random.Random(SEED + 1)
    
    cum = []
    c = 0
    for lc in file_line_counts:
        cum.append(c)
        c += lc

    global_indices = list(range(total_lines))
    print(f"  셔플 중... ({total_lines:,} 라인)")
    rng.shuffle(global_indices)

    print(f"  각 파일 라인 오프셋 구축 중...")
    file_offsets = []
    for tf in tmp_files:
        offsets = array.array("Q")  # uint64
        with open(tf, "rb") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line)
        file_offsets.append(offsets)

    print(f"  셔플 순서대로 {output.name} 에 기록 중...")
    file_handles = [open(tf, "rb") for tf in tmp_files]

    try:
        with open(output, "wb") as fout:
            written = 0
            len_cum = len(cum)
            for progress, gi in enumerate(global_indices):
                fi = 0
                for j in range(len_cum - 1, -1, -1):
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
    # Pass 0: 제외 파일들의 라인 해시 수집
    # ──────────────────────────────────────────────────
    print("=" * 60)
    print("Pass 0: 제외 파일 라인 해시 수집")
    print("=" * 60)
    exclude_hashes = collect_exclude_hashes(EXCLUDE_FILES)
    
    # ──────────────────────────────────────────────────
    # Pass 1: 라인 길이 스캔
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Pass 1: 원본 데이터스캔 (제외 분량 제외)")
    print("=" * 60)
    all_lengths = {}
    for lang, fpath in SOURCE_FILES.items():
        if not fpath.exists():
            print(f"ERROR: {fpath} 파일을 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        all_lengths[lang] = scan_line_lengths(fpath, exclude_hashes)

    # 메모리 절약: 제외 해시 세트는 더 이상 필요 없음
    del exclude_hashes

    # ──────────────────────────────────────────────────
    # 목표 바이트 산출
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("비율 기반 목표 바이트 산출")
    print("=" * 60)

    targets = {}
    for lang in RATIO:
        targets[lang] = TARGET_TOTAL_BYTES * RATIO[lang] // RATIO_SUM

    # 파일 크기로 제한할 때, 제외처리된 라인 제외한 순수 바이트 체크
    for lang in RATIO:
        # lengths 안에서 0이 아닌 값들의 합이 실제 사용할 수 있는 최대 크기입니다.
        valid_size = sum(ln for ln in all_lengths[lang] if ln > 0)
        if targets[lang] > valid_size:
            print(
                f"  주의: {lang} 목표({targets[lang]/1e9:.2f}GB)가 "
                f"제외 처리 후 유효한 파일 크기({valid_size/1e9:.2f}GB)를 초과 → 유효 크기로 제한"
            )
            targets[lang] = valid_size

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

    selected_sets = {}
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
    tmp_files = []
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
        try:
            tf.unlink()
            print(f"  삭제: {tf.name}")
        except FileNotFoundError:
            pass

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
