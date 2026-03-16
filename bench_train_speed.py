"""DenseEditor GPU 학습 속도 벤치마크

7종 mixing layer의 학습 속도(tok/s), 메모리 사용량, step 시간을 비교.
더미 데이터로 실제 forward + backward + optimizer step 수행.

Usage:
    python bench_train_speed.py --d_model 640 --max_steps 30 --batch_size 1 --bf16
"""
import argparse
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dense_editor_config import make_config
from model.dense_editor import DenseEditor

MIXING_TYPES = ["xlstm", "mlstm", "rwkv", "retnet", "mamba", "fnet", "tcn"]


def bench_one(mixing_type: str, d_model: int, seq_len: int, batch_size: int,
              max_steps: int, warmup: int, use_bf16: bool, grad_ckpt: bool,
              target_params: int):
    """단일 아키텍처 학습 속도 벤치마크"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = make_config(mixing_type, d_model=d_model, target_params=target_params,
                      max_seq_len=seq_len)
    model = DenseEditor(cfg).to(device)
    if grad_ckpt:
        model.gradient_checkpointing = True

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 더미 데이터
    input_ids = torch.randint(1, cfg.vocab_size, (batch_size, seq_len), device=device)
    edit_tags = torch.randint(0, cfg.n_tags, (batch_size, seq_len), device=device)
    pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else torch.enable_grad()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    model.train()
    step_times = []

    for step in range(max_steps):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(input_ids, pad_mask)
            targets = torch.where(pad_mask, edit_tags, torch.tensor(-100, device=device))
            loss = criterion(logits.view(-1, cfg.n_tags), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if step >= warmup:
            step_times.append(dt)

    # 메트릭
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    avg_step = sum(step_times) / len(step_times) if step_times else 0
    tok_per_step = batch_size * seq_len
    tok_s = tok_per_step / avg_step if avg_step > 0 else 0

    # 정리
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mixing_type": mixing_type,
        "d_model": d_model,
        "n_layers": cfg.n_layers,
        "n_params": n_params,
        "step_ms": avg_step * 1000,
        "tok_s": tok_s,
        "peak_mem_gb": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="DenseEditor GPU 학습 속도 벤치마크")
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--target_params", type=int, default=128_000_000)
    parser.add_argument("--mixing_types", nargs="+", default=MIXING_TYPES)
    args = parser.parse_args()

    print(f"=== DenseEditor GPU 학습 속도 벤치마크 ===")
    print(f"d_model={args.d_model}, seq_len={args.seq_len}, batch={args.batch_size}")
    print(f"bf16={args.bf16}, grad_ckpt={args.grad_ckpt}, warmup={args.warmup}, steps={args.max_steps}\n")

    print(f"{'Arch':<10} {'d':>4} {'Layers':>6} {'Params':>7} {'Step(ms)':>10} {'tok/s':>8} {'Mem(GB)':>8}")
    print("-" * 60)

    for mt in args.mixing_types:
        try:
            r = bench_one(mt, args.d_model, args.seq_len, args.batch_size,
                          args.max_steps, args.warmup, args.bf16, args.grad_ckpt,
                          args.target_params)
            print(f"{r['mixing_type']:<10} {r['d_model']:>4} {r['n_layers']:>6} "
                  f"{r['n_params']/1e6:>6.1f}M {r['step_ms']:>10.1f} {r['tok_s']:>8.0f} "
                  f"{r['peak_mem_gb']:>8.2f}")
        except Exception as e:
            print(f"{mt:<10} ERROR: {e}")


if __name__ == "__main__":
    main()
