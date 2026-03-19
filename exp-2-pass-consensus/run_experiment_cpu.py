"""합의 기반 2단계 반복 교정 실험 — Rust CPU ternary 엔진 버전

Gumbel noise (temperature) 기반 stochastic inference로
V1(single-pass), V2(2-pass), V3(consensus-2), V4(2-stage consensus) 비교.

사용법:
    source .venv/bin/activate
    OMP_NUM_THREADS=32 python exp-2-pass-consensus/run_experiment_cpu.py \
        --ckpt dense_mamba2_d640_step_26000.pt \
        --config exp-2-pass-consensus/exported_step26000/config.json \
        --model exp-2-pass-consensus/exported_step26000/model.bmmq \
        --corpus corpus/val_50k.jsonl \
        --n_samples 2000 --n_repeats 5 --temperature 0.3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.edit_tags import apply_edit_tags, TAG_KEEP
from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
from training.noising import DenoisingNoiser, NoiseConfig

# C++ 가속 Levenshtein
try:
    from training.editor_dataset import compute_edit_tags
except ImportError:
    from model.edit_tags import compute_edit_tags


RUST_BINARY = str(PROJECT_ROOT / "inference_dense/target/release/dense-editor-inference")


# ── Rust 엔진 호출 ──────────────────────────────────────────────────────

def call_rust_engine(
    all_ids: list[list[int]],
    config_path: str,
    model_path: str,
    temperature: float,
    seed: int,
    omp_threads: int,
) -> list[list[int]]:
    """Rust 엔진을 subprocess로 호출하여 batch 추론

    stdin으로 JSON Lines 전송, stdout에서 tags 수신.
    모델은 프로세스 내에서 1회 로드, 모든 시퀀스를 순차 처리.
    """
    # stdin 데이터 준비
    input_lines = []
    for ids in all_ids:
        input_lines.append(json.dumps({"ids": ids}))
    input_text = "\n".join(input_lines) + "\n"

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = [
        RUST_BINARY,
        "--infer",
        "--config", config_path,
        "--model", model_path,
        "--temperature", str(temperature),
        "--seed", str(seed),
    ]

    proc = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
    )

    if proc.returncode != 0:
        print(f"Rust 엔진 오류: {proc.stderr}", file=sys.stderr)
        raise RuntimeError(f"Rust 엔진 실패 (exit={proc.returncode})")

    # stderr에 프로파일 정보 → 첫 호출만 표시
    lines = proc.stderr.strip().split("\n") if proc.stderr.strip() else []
    for line in lines[:3]:
        if "profile" in line.lower() or "로드" in line or "BMMQ" in line:
            print(f"  [rust] {line}", file=sys.stderr)

    # stdout에서 tags 파싱
    all_tags = []
    for line in proc.stdout.strip().split("\n"):
        if not line.strip():
            continue
        result = json.loads(line)
        all_tags.append(result["tags"])

    if len(all_tags) != len(all_ids):
        raise RuntimeError(
            f"Rust 출력 수 불일치: {len(all_tags)} vs {len(all_ids)} expected"
        )

    return all_tags


# ── 평가 데이터 준비 ───────────────────────────────────────────────────

def prepare_eval_data(
    corpus_path: str,
    tokenizer: KeyboardTokenizer,
    n_samples: int,
    noise_seed: int,
) -> list[tuple[list[int], list[int]]]:
    """val JSONL에서 (noised_ids, clean_ids) 쌍 생성"""
    noise_cfg = NoiseConfig(
        korean_error_prob=0.5,
        korean_error_count=3,
        token_mask_ratio=0.0,
        token_delete_ratio=0.0,
        text_infill_ratio=0.0,
    )
    noiser = DenoisingNoiser(tokenizer, noise_cfg, seed=noise_seed)

    pairs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= n_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                text = line
            if len(text) < 10:
                continue

            noised_ids, clean_ids, _ = noiser(text)
            if len(noised_ids) > 1024 or len(clean_ids) > 1024:
                continue
            pairs.append((noised_ids, clean_ids))

    print(f"평가 데이터: {len(pairs)}개 문장 ({corpus_path})")
    return pairs


# ── Consensus ──────────────────────────────────────────────────────────

def consensus_tags(tags_a: list[int], tags_b: list[int]) -> list[int]:
    """두 tag 시퀀스의 합의: 동일한 tag만 유지, 불일치 → TAG_KEEP(0)"""
    assert len(tags_a) == len(tags_b), \
        f"consensus 길이 불일치: {len(tags_a)} vs {len(tags_b)}"
    return [a if a == b else TAG_KEEP for a, b in zip(tags_a, tags_b)]


# ── Variation 실행기 ───────────────────────────────────────────────────

def apply_tags_all(
    all_ids: list[list[int]],
    all_tags: list[list[int]],
    vocab_size: int,
) -> list[list[int]]:
    return [
        apply_edit_tags(ids, tags, vocab_size)
        for ids, tags in zip(all_ids, all_tags)
    ]


def run_variation(
    variation: str,
    all_noised: list[list[int]],
    config_path: str,
    model_path: str,
    temperature: float,
    base_seed: int,
    omp_threads: int,
    vocab_size: int,
) -> tuple[list[list[int]], dict]:
    """variation 실행 → (최종 출력 IDs, 타이밍 정보)"""

    timings = {}

    if variation == "v1":
        # Single-pass: gec(x)
        t0 = time.time()
        tags = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed, omp_threads,
        )
        timings["pass1"] = time.time() - t0
        finals = apply_tags_all(all_noised, tags, vocab_size)
        return finals, timings

    elif variation == "v2":
        # 2-pass: gec(gec(x))
        t0 = time.time()
        tags_1 = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed, omp_threads,
        )
        timings["pass1"] = time.time() - t0
        intermediates = apply_tags_all(all_noised, tags_1, vocab_size)

        t0 = time.time()
        tags_2 = call_rust_engine(
            intermediates, config_path, model_path,
            temperature, base_seed + 100, omp_threads,
        )
        timings["pass2"] = time.time() - t0
        finals = apply_tags_all(intermediates, tags_2, vocab_size)
        return finals, timings

    elif variation == "v3":
        # Consensus-2: consensus(gec_a(x), gec_b(x))
        t0 = time.time()
        tags_a = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed, omp_threads,
        )
        timings["pass_a"] = time.time() - t0

        t0 = time.time()
        tags_b = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed + 1, omp_threads,
        )
        timings["pass_b"] = time.time() - t0

        cons = [consensus_tags(a, b) for a, b in zip(tags_a, tags_b)]
        finals = apply_tags_all(all_noised, cons, vocab_size)
        return finals, timings

    elif variation == "v4":
        # 2-stage consensus: y=cons(a,b), final=apply(y, cons(a',b'))
        # Stage 1
        t0 = time.time()
        tags_a = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed, omp_threads,
        )
        timings["s1_a"] = time.time() - t0

        t0 = time.time()
        tags_b = call_rust_engine(
            all_noised, config_path, model_path,
            temperature, base_seed + 1, omp_threads,
        )
        timings["s1_b"] = time.time() - t0

        cons_1 = [consensus_tags(a, b) for a, b in zip(tags_a, tags_b)]
        y_list = apply_tags_all(all_noised, cons_1, vocab_size)

        # Stage 2: consensus on y
        t0 = time.time()
        tags_a2 = call_rust_engine(
            y_list, config_path, model_path,
            temperature, base_seed + 10, omp_threads,
        )
        timings["s2_a"] = time.time() - t0

        t0 = time.time()
        tags_b2 = call_rust_engine(
            y_list, config_path, model_path,
            temperature, base_seed + 11, omp_threads,
        )
        timings["s2_b"] = time.time() - t0

        cons_2 = [consensus_tags(a, b) for a, b in zip(tags_a2, tags_b2)]
        finals = apply_tags_all(y_list, cons_2, vocab_size)
        return finals, timings

    raise ValueError(f"알 수 없는 variation: {variation}")


# ── 평가 메트릭 ────────────────────────────────────────────────────────

def evaluate_all(
    eval_data: list[tuple[list[int], list[int]]],
    finals: list[list[int]],
    vocab_size: int,
    gold_tags_cache: list[list[int]],
) -> dict:
    tp_exact = 0
    fp = 0
    fn = 0
    total_pred_edits = 0
    total_gold_edits = 0
    total_sentences = len(eval_data)
    changed_sentences = 0

    for i, ((noised_ids, clean_ids), final_ids) in enumerate(zip(eval_data, finals)):
        gold_tags = gold_tags_cache[i]
        pred_tags = compute_edit_tags(noised_ids, final_ids, vocab_size)

        n_pred = sum(1 for t in pred_tags if t != TAG_KEEP)
        n_gold = sum(1 for t in gold_tags if t != TAG_KEEP)
        total_pred_edits += n_pred
        total_gold_edits += n_gold
        if n_pred > 0:
            changed_sentences += 1

        for g, p in zip(gold_tags, pred_tags):
            g_edit = (g != TAG_KEEP)
            p_edit = (p != TAG_KEEP)
            if g_edit and p_edit:
                if g == p:
                    tp_exact += 1
                else:
                    fp += 1
                    fn += 1
            elif p_edit and not g_edit:
                fp += 1
            elif g_edit and not p_edit:
                fn += 1

    p = tp_exact / max(tp_exact + fp, 1)
    r = tp_exact / max(tp_exact + fn, 1)
    f05 = _f_beta(p, r, 0.5)
    f1 = _f_beta(p, r, 1.0)

    return {
        "precision": p,
        "recall": r,
        "f05": f05,
        "f1": f1,
        "tp": tp_exact,
        "fp": fp,
        "fn": fn,
        "total_pred_edits": total_pred_edits,
        "total_gold_edits": total_gold_edits,
        "avg_edits_per_sent": total_pred_edits / max(total_sentences, 1),
        "changed_sent_ratio": changed_sentences / max(total_sentences, 1),
        "n_sentences": total_sentences,
    }


def _f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


# ── 결과 출력 ─────────────────────────────────────────────────────────

VARIATION_NAMES = {
    "v1": "V1 single-pass",
    "v2": "V2 2-pass",
    "v3": "V3 consensus-2",
    "v4": "V4 2-stage consensus",
}


def print_summary(all_results: dict[str, list[dict]]):
    import statistics

    sep = "=" * 100
    print(f"\n{sep}")
    print("Summary (mean ± std)")
    print(sep)
    print(f"{'Variation':<25} | {'Precision':>13} | {'Recall':>13} | "
          f"{'F0.5':>13} | {'F1':>13} | {'Edits/sent':>13} | {'Time(s)':>10}")
    print("-" * 100)

    for var_key in ["v1", "v2", "v3", "v4"]:
        if var_key not in all_results:
            continue
        runs = all_results[var_key]
        name = VARIATION_NAMES.get(var_key, var_key)

        def ms(key):
            vals = [r[key] for r in runs]
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:.4f}±{s:.4f}"

        def ms_time():
            vals = [r.get("total_time", 0) for r in runs]
            m = statistics.mean(vals)
            return f"{m:.1f}"

        print(f"{name:<25} | {ms('precision'):>13} | {ms('recall'):>13} | "
              f"{ms('f05'):>13} | {ms('f1'):>13} | {ms('avg_edits_per_sent'):>13} | "
              f"{ms_time():>10}")

    print(sep)

    # 해석
    print("\n[ 해석 요약 ]")
    best_var = None
    best_f05 = -1
    for var_key, runs in all_results.items():
        mean_f05 = statistics.mean([r["f05"] for r in runs])
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_var = var_key

    print(f"  F0.5 기준 최고: {VARIATION_NAMES.get(best_var, best_var)} (F0.5={best_f05:.4f})")

    if "v1" in all_results:
        v1_p = statistics.mean([r["precision"] for r in all_results["v1"]])
        v1_r = statistics.mean([r["recall"] for r in all_results["v1"]])
        v1_f05 = statistics.mean([r["f05"] for r in all_results["v1"]])

        for var_key in ["v2", "v3", "v4"]:
            if var_key not in all_results:
                continue
            runs = all_results[var_key]
            name = VARIATION_NAMES.get(var_key, var_key)
            p = statistics.mean([r["precision"] for r in runs])
            r = statistics.mean([r["recall"] for r in runs])
            f05 = statistics.mean([r["f05"] for r in runs])
            dp = p - v1_p
            dr = r - v1_r
            df = f05 - v1_f05
            print(f"  {name} vs V1: P {dp:+.4f}, R {dr:+.4f}, F0.5 {df:+.4f}")


# ── 메인 ─────────────────────────────────────────────────────────────

def run_experiment(args):
    tokenizer = KeyboardTokenizer()
    vocab_size = tokenizer.vocab_size

    # 데이터 준비
    eval_data = prepare_eval_data(args.corpus, tokenizer, args.n_samples, args.seed)

    # Gold tags 사전 계산
    print("Gold tags 사전 계산...")
    t0 = time.time()
    gold_tags_cache = [
        compute_edit_tags(noised_ids, clean_ids, vocab_size)
        for noised_ids, clean_ids in eval_data
    ]
    print(f"  완료 ({time.time() - t0:.1f}s)")

    all_noised = [pair[0] for pair in eval_data]
    total_tokens = sum(len(ids) for ids in all_noised)
    print(f"  총 {total_tokens:,} tokens, 평균 {total_tokens / len(eval_data):.0f} tok/sent")

    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    print(f"\nRust 엔진: {RUST_BINARY}")
    print(f"OMP_NUM_THREADS: {omp_threads}")
    print(f"Temperature: {args.temperature}")
    print(f"Variations: {args.variations}")
    print(f"Repeats: {args.n_repeats}")

    all_results: dict[str, list[dict]] = {}

    for var in args.variations:
        print(f"\n{'='*60}")
        print(f"Variation: {VARIATION_NAMES.get(var, var)}")
        print(f"{'='*60}")

        var_runs = []

        for repeat_idx in range(args.n_repeats):
            # 각 repeat마다 다른 base_seed → Gumbel noise 다양화
            base_seed = args.seed + repeat_idx * 1000

            t0 = time.time()
            finals, timings = run_variation(
                var, all_noised, args.config, args.model,
                args.temperature, base_seed, omp_threads, vocab_size,
            )
            total_time = time.time() - t0

            metrics = evaluate_all(eval_data, finals, vocab_size, gold_tags_cache)
            metrics["repeat"] = repeat_idx
            metrics["base_seed"] = base_seed
            metrics["total_time"] = round(total_time, 1)
            metrics["timings"] = timings

            var_runs.append(metrics)

            timing_str = " + ".join(f"{k}={v:.1f}s" for k, v in timings.items())
            print(
                f"  repeat {repeat_idx}: "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                f"F0.5={metrics['f05']:.4f} F1={metrics['f1']:.4f} "
                f"edits/sent={metrics['avg_edits_per_sent']:.2f} "
                f"({timing_str}, total={total_time:.1f}s)"
            )

        all_results[var] = var_runs

    print_summary(all_results)

    # JSON 저장
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_results_cpu.json"
    output = {
        "config": {
            "ckpt": args.ckpt,
            "config_path": args.config,
            "model_path": args.model,
            "corpus": args.corpus,
            "n_samples": len(eval_data),
            "n_repeats": args.n_repeats,
            "temperature": args.temperature,
            "seed": args.seed,
            "omp_threads": omp_threads,
            "variations": args.variations,
            "stochasticity": "gumbel_noise",
            "engine": "rust_ternary_cpu",
        },
        "results": all_results,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {raw_path}")


def main():
    parser = argparse.ArgumentParser(
        description="합의 기반 2단계 반복 교정 실험 — Rust CPU ternary 엔진",
    )
    parser.add_argument("--ckpt", required=True, help="원본 체크포인트 (메타 정보용)")
    parser.add_argument("--config", required=True, help="BMMQ config.json 경로")
    parser.add_argument("--model", required=True, help="BMMQ model.bmmq 경로")
    parser.add_argument("--corpus", default="corpus/val_50k.jsonl")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument(
        "--variations", nargs="+", default=["v1", "v2", "v3", "v4"],
        choices=["v1", "v2", "v3", "v4"],
    )
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Gumbel noise temperature (0=결정론적)")
    parser.add_argument("--output_dir", default="exp-2-pass-consensus/results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
