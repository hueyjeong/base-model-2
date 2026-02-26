#!/usr/bin/env python3
"""
build_sample_full.py
===================
한국어 전체(sample_ko.jsonl)를 기준으로,
한국어:일본어:영어 = 8:1:1 **바이트** 비율을 맞추어 무작위 샘플링 후
총 ~44 GB 크기(한국어 약 35GB + 일본어 4.4GB + 영어 4.4GB)의 sample_full.jsonl을 생성합니다.

- val_50k.jsonl 에 포함된 검증 세인트 라인은 훈련 데이터 오염(Leakage)을 막기 위해 제외합니다.
- 한국어 데이터는 샘플링 없이 사용되지만, 제외해야 할 라인이 있으므로 일본어/영어처럼 임시 파일을 생성하여 처리합니다.
"""

import os
import sys
import random
import time
import array
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────
RATIO = {"ko": 8, "ja": 1, "en": 1}

CORPUS_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {
    "ko": CORPUS_DIR / "sample_ko.jsonl",
    "ja": CORPUS_DIR / "sample_ja.jsonl",
    "en": CORPUS_DIR / "sample_en.jsonl",
}
OUTPUT_FILE = CORPUS_DIR / "sample_full.jsonl"
EXCLUDE_FILE = CORPUS_DIR / "val_50k.jsonl"

SEED = 42
PROGRESS_LINES = 10_000_000

def collect_exclude_hashes(filepath: Path) -> set[int]:
    """
    제외할 파일을 순차 읽기하여 각 라인의 해시값을 set에 저장.
    """
    exclude_hashes = set()
    if not filepath.exists():
        print(f"  주의: {filepath.name} 파일이 존재하지 않아 건너뜁니다.")
        return exclude_hashes
        
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
    제외할 라인의 길이는 0으로 마킹합니다.
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
                print(f"    ... {count:,} 라인 스캔 (제외됨: {excluded_count:,})")

    print(f"    완료: {count:,} 라인 (총 제외됨: {excluded_count:,})")
    return lengths

def select_indices(
    lengths: array.array,
    target_bytes: int,
    rng: random.Random,
    sample_all: bool = False
) -> set[int]:
    """
    라인 인덱스를 셔플한 뒤, 목표 바이트에 도달할 때까지 선택합니다.
    sample_all 이 True 인 경우, 길이가 0이 아닌 모든 라인을 선택합니다.
    (길이가 0인 라인은 val_50k 에 의해 제외된 라인입니다)
    """
    selected = set()
    
    if sample_all:
        for idx, ln in enumerate(lengths):
            if ln > 0:
                selected.add(idx)
        # 한국어처럼 모두 뽑을 때는 현재 바이트 계산을 생략합니다.
        return selected, -1

    # 목표 바이트까지 추출
    n = len(lengths)
    indices = list(range(n))
    rng.shuffle(indices)

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
    임시 파일들의 라인들을 합치면서 셔플하여 최종 출력 파일에 기록.
    """
    print(f"  각 파일 라인 오프셋 및 라인 수 구축 중...")
    file_offsets = []
    file_line_counts = []
    
    for tf in tmp_files:
        offsets = array.array("Q")  # uint64
        with open(tf, "rb") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line)
        file_offsets.append(offsets)
        file_line_counts.append(len(offsets))
        print(f"    {tf.name}: {len(offsets):,} 라인")

    total_lines = sum(file_line_counts)
    print(f"  총 라인: {total_lines:,}")

    rng = random.Random(SEED + 1)
    cum = []
    c = 0
    for lc in file_line_counts:
        cum.append(c)
        c += lc

    print(f"  셔플 중... ({total_lines:,} 라인) - 라인 수가 많아 약 1~2분 정도 소요될 수 있습니다.")
    st = time.time()
    global_indices = list(range(total_lines))
    rng.shuffle(global_indices)
    print(f"  셔플 완료! ({time.time() - st:.1f}초 소요)")

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

    # 파일 존재 확인
    for lang, fpath in SOURCE_FILES.items():
        if not fpath.exists():
            print(f"ERROR: {fpath} 파일을 찾을 수 없습니다.", file=sys.stderr)
            sys.exit(1)

    # ──────────────────────────────────────────────────
    # Pass 0: val_50k 제외 파일의 라인 해시 수집
    # ──────────────────────────────────────────────────
    print("=" * 60)
    print("Pass 0: val_50k 중복 방지 라인 해시 수집")
    print("=" * 60)
    exclude_hashes = collect_exclude_hashes(EXCLUDE_FILE)

    # ──────────────────────────────────────────────────
    # Pass 1: 스캔 (모든 언어 오염 제외처리)
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Pass 1: 라인 길이 스캔 (val_50k 제외)")
    print("=" * 60)
    all_lengths = {}
    for lang in ["ko", "ja", "en"]:
        all_lengths[lang] = scan_line_lengths(SOURCE_FILES[lang], exclude_hashes)

    del exclude_hashes

    # ──────────────────────────────────────────────────
    # 목표 바이트 산출 (한국어를 기준으로 타겟 계산)
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("목표 바이트 산출 (한국어 기준 8:1:1)")
    print("=" * 60)

    # val_50k 를 제외하고 남은 한국어 데이터의 실제 바이트 크기
    ko_valid_size = sum(ln for ln in all_lengths["ko"] if ln > 0)
    
    targets = {
        "ko": ko_valid_size,
        "ja": int(ko_valid_size * (RATIO["ja"] / RATIO["ko"])),
        "en": int(ko_valid_size * (RATIO["en"] / RATIO["ko"])),
    }

    # 파일 크기로 제한
    for lang in ["ja", "en"]:
        valid_size = sum(ln for ln in all_lengths[lang] if ln > 0)
        if targets[lang] > valid_size:
            print(f"  주의: {lang} 목표용량이 유효파일 크기를 초과하여 최대치로 사용합니다.")
            targets[lang] = valid_size

    for lang in ["ko", "ja", "en"]:
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
    for lang in ["ko", "ja", "en"]:
        print(f"  [{lang}] {len(all_lengths[lang]):,} 라인 중 선택...")
        
        # 한국어는 100% 사용하므로 길이 0을 제외한 나머지를 전체 선택
        is_sample_all = (lang == "ko")
        sel, sel_bytes = select_indices(all_lengths[lang], targets[lang], rng, sample_all=is_sample_all)
        selected_sets[lang] = sel
        
        if is_sample_all:
            print(f"    → {len(sel):,} 라인 선택 (분량 제한 없음, 유효라인 모두 사용)")
        else:
            print(f"    → {len(sel):,} 라인 선택 ({sel_bytes / 1e9:.4f} GB)")

    del all_lengths

    # ──────────────────────────────────────────────────
    # Pass 2: 선택된 라인 추출
    # ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Pass 2: 선택된 라인 임시 파일에 추출")
    print("=" * 60)

    # val_50k 가 섞여있는 원본을 피하기 위해 ko도 임시파일을 만들어야 합니다
    tmp_files = []

    for lang in ["ko", "ja", "en"]:
        tmp_path = CORPUS_DIR / f"_tmp_sample_full_{lang}.jsonl"
        extract_selected(SOURCE_FILES[lang], selected_sets[lang], tmp_path, lang)
        tmp_files.append(tmp_path)

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

    # 중단된 다른 스크립트 찌꺼기 파일도 함께 삭제해 줍니다.
    for lang in ["ja", "en"]:
        old_tmp_tf = CORPUS_DIR / f"_tmp_sample_{lang}.jsonl"
        if old_tmp_tf.exists():
            old_tmp_tf.unlink()

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
