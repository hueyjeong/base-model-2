"""INT8 CUDA 최적화 A/B 벤치

배치/스텝별 조합 성능을 비교한다.
기본 벤치는 non-graph 조합만 수행하고, CUDA Graph는 실험 옵션으로 분리한다.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from statistics import mean


BASE_CMD = [
    sys.executable,
    "-m",
    "training.pretrain",
    "--tokenizer", "keyboard",
    "--corpus", "corpus/sample_10g.jsonl",
    "--batch_size", "1",
    "--grad_accum_steps", "1",
    "--num_workers", "0",
    "--bf16",
    "--int8",
    "--int8_backend", "cuda",
    "--log_every", "1",
]


CASES = [
    {
        "name": "baseline_no_graph",
        "env": {
            "BITLINEAR_CUDA_BACKWARD": "fp32_tf32",
            "BITLINEAR_CUDA_GRADW_LT": "0",
            "BITLINEAR_CUDA_FUSED_ACT": "0",
            "BITLINEAR_CUDA_FUSED_WEIGHT": "0",
        },
        "cuda_graph": False,
    },
    {
        "name": "gradw_lt_only",
        "env": {
            "BITLINEAR_CUDA_BACKWARD": "fp32_tf32",
            "BITLINEAR_CUDA_GRADW_LT": "1",
            "BITLINEAR_CUDA_FUSED_ACT": "0",
            "BITLINEAR_CUDA_FUSED_WEIGHT": "0",
        },
        "cuda_graph": False,
    },
    {
        "name": "fused_quant_only",
        "env": {
            "BITLINEAR_CUDA_BACKWARD": "fp32_tf32",
            "BITLINEAR_CUDA_GRADW_LT": "0",
            "BITLINEAR_CUDA_FUSED_ACT": "1",
            "BITLINEAR_CUDA_FUSED_WEIGHT": "1",
        },
        "cuda_graph": False,
    },
    {
        "name": "all_optimizations",
        "env": {
            "BITLINEAR_CUDA_BACKWARD": "fp32_tf32",
            "BITLINEAR_CUDA_GRADW_LT": "1",
            "BITLINEAR_CUDA_FUSED_ACT": "1",
            "BITLINEAR_CUDA_FUSED_WEIGHT": "1",
        },
        "cuda_graph": False,
    },
    {
        "name": "all_optimizations_graph_exp",
        "env": {
            "BITLINEAR_CUDA_BACKWARD": "fp32_tf32",
            "BITLINEAR_CUDA_GRADW_LT": "1",
            "BITLINEAR_CUDA_FUSED_ACT": "1",
            "BITLINEAR_CUDA_FUSED_WEIGHT": "1",
        },
        "cuda_graph": True,
    },
]


TOKS_RE = re.compile(r"step\s+(\d+).*?\|\s+(\d+)\s+tok/s", re.IGNORECASE)


def run_case(case: dict, size: str, max_steps: int, batch_size: int) -> tuple[float, str]:
    cmd = list(BASE_CMD)
    cmd.extend(["--size", size])
    cmd.extend(["--max_steps", str(max_steps), "--batch_size", str(batch_size)])
    if case["cuda_graph"]:
        cmd.append("--cuda_graph")

    env = os.environ.copy()
    env.update(case["env"])

    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(f"case={case['name']} failed\n{out}")

    toks = []
    for line in out.splitlines():
        m = TOKS_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        tok_s = int(m.group(2))
        if step >= 2:
            toks.append(tok_s)

    if not toks:
        raise RuntimeError(f"case={case['name']} has no tok/s lines\n{out}")

    return mean(toks), out


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA ablation benchmark")
    parser.add_argument("--size", type=str, default="8M")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--graph_max_batch", type=int, default=2)
    parser.add_argument("--include_graph_experimental", action="store_true",
                        help="실험용 CUDA Graph 케이스를 추가 실행")
    args = parser.parse_args()

    for batch_size in args.batch_sizes:
        results = []
        print(f"\n[Ablation] size={args.size}, max_steps={args.max_steps}, batch_size={batch_size}, avg tok/s(step>=2)")
        for case in CASES:
            if case["cuda_graph"] and not args.include_graph_experimental:
                continue
            if case["cuda_graph"] and batch_size > args.graph_max_batch:
                print(f"- {case['name']:<18}: SKIP (batch>{args.graph_max_batch})")
                continue
            avg_toks, _ = run_case(case, size=args.size, max_steps=args.max_steps, batch_size=batch_size)
            results.append((case["name"], avg_toks))
            print(f"- {case['name']:<18}: {avg_toks:.1f} tok/s")

        base = dict(results).get("baseline_no_graph", None)
        if base is not None:
            print("\n[Delta vs baseline_no_graph]")
            for name, val in results:
                delta = (val - base) / base * 100.0
                print(f"- {name:<18}: {delta:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
