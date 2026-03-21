"""
오류 분포 통계 검증 스크립트.

코퍼스에서 N개 문장을 샘플링하여:
A. 오류 유형별 분포 비교 (default vs realistic vs KoGEC 목표)
B. 모듈별 hit rate (성공률)
C. 전체 노이즈 적용 통계

실행:
    python error_generation/test_distribution.py --corpus corpus/val_50k.jsonl --n_samples 1000
"""

import sys
import os
import json
import random
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_generation import KoreanErrorGenerator, ERROR_GENERATORS
from error_generation.kagas_mapping import KAGAS_MAP, REVERSE_KAGAS_MAP, KOGEC_DISTRIBUTION
from training.noising import WEIGHT_PRESETS


def load_corpus(path: str, n: int = 1000, min_len: int = 10) -> list[str]:
    """코퍼스에서 n개 문장 로드."""
    texts = []
    for p in (path if isinstance(path, list) else [path]):
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if len(text) >= min_len:
                    texts.append(text)
                if len(texts) >= n:
                    break
        if len(texts) >= n:
            break
    return texts[:n]


def measure_distribution(texts: list[str], weights: dict, n_errors: int = 3, seed: int = 42):
    """프리셋 기준 오류 유형별 분포 측정.

    apply_single_error 내부의 가중치 기반 선택을 추적하여
    실제 적용된 오류 유형의 빈도를 집계.
    """
    gen = KoreanErrorGenerator(seed=seed, weights_override=weights)
    type_counter = Counter()
    total_attempts = 0

    for text in texts:
        current = text
        for _ in range(n_errors):
            total_attempts += 1
            # 가중치 기반 선택 + 적용 시도 추적
            applied = False
            for _retry in range(10):
                [chosen_fn] = gen._rng.choices(gen._fns, weights=gen._weights, k=1)
                result = chosen_fn(current, gen._rng)
                if result is not None:
                    idx = gen._fns.index(chosen_fn)
                    name = gen._names[idx]
                    type_counter[name] += 1
                    current = result
                    applied = True
                    break
            if not applied:
                type_counter["_NO_MATCH"] += 1

    return type_counter, total_attempts


def measure_hit_rate(texts: list[str], seed: int = 42):
    """모듈별 성공률 측정."""
    gen = KoreanErrorGenerator(seed=seed)
    results = {}

    for name in gen.error_types:
        hits = 0
        examples = []
        rng = random.Random(seed)
        for text in texts:
            result = gen.apply_single_error(text, error_type=name)
            if result is not None and result != text:
                hits += 1
                if len(examples) < 3:
                    examples.append((text[:40], result[:40]))
        results[name] = {
            "hit_rate": hits / len(texts) if texts else 0,
            "hits": hits,
            "examples": examples,
        }

    return results


def measure_overall_stats(texts: list[str], weights: dict, n_errors: int = 3, seed: int = 42):
    """전체 노이즈 적용 통계."""
    gen = KoreanErrorGenerator(seed=seed, weights_override=weights)
    changed = 0
    total_edit_dist = 0

    for text in texts:
        result = gen.apply_random_errors(text, n_errors=n_errors)
        if result != text:
            changed += 1
            # 문자 단위 편집 거리 (간단 방식: 다른 문자 수)
            total_edit_dist += sum(1 for a, b in zip(text, result) if a != b) + abs(len(text) - len(result))

    return {
        "total": len(texts),
        "changed": changed,
        "change_rate": changed / len(texts) if texts else 0,
        "avg_edit_dist": total_edit_dist / len(texts) if texts else 0,
    }


def print_distribution_report(texts: list[str], n_errors: int = 3, seed: int = 42):
    """default vs realistic 분포 비교 리포트."""
    print("=" * 80)
    print("  A. 오류 유형별 분포 비교 (KAGAS 11-type)")
    print("=" * 80)

    presets = {
        "default": WEIGHT_PRESETS.get("default", {}),
        "realistic": WEIGHT_PRESETS.get("realistic", {}),
    }

    preset_kagas = {}
    for preset_name, weights in presets.items():
        type_counter, total = measure_distribution(texts, weights, n_errors, seed)

        # KAGAS 그룹핑
        kagas_counter = Counter()
        for our_type, count in type_counter.items():
            if our_type == "_NO_MATCH":
                kagas_counter["_NO_MATCH"] += count
                continue
            kagas = REVERSE_KAGAS_MAP.get(our_type, "OTHER")
            kagas_counter[kagas] += count
        preset_kagas[preset_name] = (kagas_counter, total)

    # 헤더
    print(f"\n{'KAGAS 유형':>12s} | {'KoGEC 목표':>10s} | {'default':>10s} | {'realistic':>10s}")
    print("-" * 55)

    all_kagas = sorted(set(list(KAGAS_MAP.keys()) + ["OTHER"]))
    for kagas in all_kagas:
        target = KOGEC_DISTRIBUTION.get(kagas, 0) * 100
        default_count = preset_kagas["default"][0].get(kagas, 0)
        default_total = preset_kagas["default"][1]
        default_pct = default_count / default_total * 100 if default_total else 0
        realistic_count = preset_kagas["realistic"][0].get(kagas, 0)
        realistic_total = preset_kagas["realistic"][1]
        realistic_pct = realistic_count / realistic_total * 100 if realistic_total else 0
        print(f"{kagas:>12s} | {target:>9.1f}% | {default_pct:>9.1f}% | {realistic_pct:>9.1f}%")

    # 모듈별 상세 (default)
    print(f"\n--- 모듈별 상세 (default 프리셋) ---")
    default_counter = measure_distribution(texts, presets["default"], n_errors, seed + 100)[0]
    default_total = sum(default_counter.values())
    for name, count in default_counter.most_common():
        pct = count / default_total * 100 if default_total else 0
        kagas = REVERSE_KAGAS_MAP.get(name, "OTHER")
        print(f"  {name:25s} [{kagas:>12s}] {count:5d} ({pct:5.1f}%)")


def print_hit_rate_report(texts: list[str], seed: int = 42):
    """모듈별 hit rate 리포트."""
    print("\n" + "=" * 80)
    print("  B. 모듈별 Hit Rate (성공률)")
    print("=" * 80)

    results = measure_hit_rate(texts, seed)

    print(f"\n{'모듈':25s} | {'hit rate':>8s} | {'hits':>5s}/{len(texts)} | 예시")
    print("-" * 80)

    for name, data in sorted(results.items(), key=lambda x: -x[1]["hit_rate"]):
        hr = data["hit_rate"] * 100
        hits = data["hits"]
        ex = ""
        if data["examples"]:
            orig, changed = data["examples"][0]
            ex = f'"{orig}" → "{changed}"'
        print(f"  {name:23s} | {hr:>7.1f}% | {hits:>5d} | {ex}")


def print_overall_report(texts: list[str], n_errors: int = 3, seed: int = 42):
    """전체 노이즈 적용 통계 리포트."""
    print("\n" + "=" * 80)
    print("  C. 전체 노이즈 적용 통계")
    print("=" * 80)

    for preset_name in ["default", "realistic"]:
        weights = WEIGHT_PRESETS.get(preset_name, {})
        stats = measure_overall_stats(texts, weights, n_errors, seed)
        print(f"\n  [{preset_name}] n_errors={n_errors}")
        print(f"    문장 수: {stats['total']}")
        print(f"    변경된 문장: {stats['changed']} ({stats['change_rate']*100:.1f}%)")
        print(f"    평균 편집 거리: {stats['avg_edit_dist']:.1f} chars")


def main():
    parser = argparse.ArgumentParser(description="오류 분포 통계 검증")
    parser.add_argument("--corpus", type=str, nargs="+", required=True,
                        help="JSONL 코퍼스 경로")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="샘플링할 문장 수")
    parser.add_argument("--n_errors", type=int, default=3,
                        help="문장당 적용할 오류 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"코퍼스 로딩: {args.corpus} (n={args.n_samples})")
    texts = load_corpus(args.corpus, args.n_samples)
    print(f"로딩 완료: {len(texts)}개 문장\n")

    print_distribution_report(texts, args.n_errors, args.seed)
    print_hit_rate_report(texts, args.seed)
    print_overall_report(texts, args.n_errors, args.seed)

    # KAGAS 매핑 무결성
    print("\n" + "=" * 80)
    print("  D. KAGAS 매핑 무결성")
    print("=" * 80)
    names = {n for n, _, _ in ERROR_GENERATORS}
    mapped = set(REVERSE_KAGAS_MAP.keys())
    print(f"  매핑 완료: {len(mapped & names)}/{len(names)}")
    missing = names - mapped
    if missing:
        print(f"  미매핑 (OTHER로 분류): {missing}")


if __name__ == "__main__":
    main()
