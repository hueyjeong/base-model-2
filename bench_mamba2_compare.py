"""Mamba-2 d_state/d_model 비교 품질 벤치마크

Mamba-1 기준 대비 Mamba-2의 d_state(16/32/64/128) 및 d_model(704, 12L) 변형 비교.
2000 steps, seq=512, 오버핏 테스트.

Usage:
    source .venv/bin/activate
    export BITLINEAR_CUDA_BACKWARD=bf16_tc
    export BITLINEAR_CUDA_GRADW_LT=1
    export BITLINEAR_CUDA_FUSED_ACT=1
    export BITLINEAR_CUDA_FUSED_WEIGHT=1

    python bench_mamba2_compare.py --corpus corpus/val_50k.jsonl
    python bench_mamba2_compare.py --corpus corpus/val_50k.jsonl --variants ds16 ds64 wide64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from bench_quality import bench_quality_one

# 테스트 변형 목록
# (이름, mixing_type, d_model, config_overrides)
VARIANTS = {
    # Mamba-1 기준선 (동일 조건 비교용)
    "mamba1": ("mamba", 640, {}),
    # Mamba-2 d_state=64 (기본)
    "ds64":   ("mamba2", 640, {"mamba2_d_state": 64}),
}


def main():
    parser = argparse.ArgumentParser(description="Mamba-2 d_state/d_model 비교 벤치마크")
    parser.add_argument("--corpus", type=str, nargs="+", required=True)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--target_params", type=int, default=128_000_000)
    parser.add_argument("--variants", nargs="+",
                        default=list(VARIANTS.keys()),
                        choices=list(VARIANTS.keys()))
    args = parser.parse_args()

    eff_batch = args.batch_size * args.grad_accum
    print(f"=== Mamba-2 d_state/d_model 비교 벤치마크 ===")
    print(f"seq={args.seq_len}, batch={args.batch_size}, grad_accum={args.grad_accum}, "
          f"effective_batch={eff_batch}, steps={args.max_steps}")
    print(f"variants: {args.variants}\n")

    all_results = {}

    for name in args.variants:
        mixing_type, d_model, overrides = VARIANTS[name]
        ds = overrides.get("mamba2_d_state", 64)
        print(f"--- {name} (mamba2 ds={ds} d={d_model}) ---")
        try:
            r = bench_quality_one(
                mixing_type, d_model, args.corpus, args.text_key,
                args.max_steps, args.log_interval, args.seq_len, args.batch_size,
                args.bf16, args.target_params, grad_accum=args.grad_accum,
                **overrides,
            )
            all_results[name] = r
            f = r["final"]
            print(f"  => {r['n_layers']}L {r['n_params']/1e6:.1f}M | "
                  f"loss={f.get('loss',0):.4f} P={f.get('edit_p',0):.2%} R={f.get('edit_r',0):.2%}\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            print()

    # 최종 비교
    print(f"\n{'='*70}")
    print(f"{'Name':<10} {'Type':<8} {'d':>4} {'L':>3} {'Params':>7} "
          f"{'Loss':>8} {'EditP':>7} {'EditR':>7}")
    print("-" * 70)
    for name in args.variants:
        if name in all_results:
            r = all_results[name]
            f = r["final"]
            mt, d, ov = VARIANTS[name]
            ds = ov.get("mamba2_d_state", 64)
            print(f"{name:<10} ds={ds:<4} {d:>4} {r['n_layers']:>3} {r['n_params']/1e6:>6.1f}M "
                  f"{f.get('loss',0):>8.4f} {f.get('edit_p',0):>6.2%} {f.get('edit_r',0):>6.2%}")


if __name__ == "__main__":
    main()
