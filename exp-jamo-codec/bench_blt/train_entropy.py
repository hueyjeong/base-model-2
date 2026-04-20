"""1M byte-level entropy LM 학습.

사용:
    source .venv/bin/activate
    export PYTHONPATH=exp-jamo-codec/bench_blt
    python exp-jamo-codec/bench_blt/train_entropy.py \
        --train_parquet corpus/jamo-codec-v3/train.parquet \
        --val_parquet corpus/jamo-codec-v3/val.parquet \
        --out_dir exp-jamo-codec/bench_blt/ckpt_entropy \
        --max_steps 5000 --batch_size 32 --byte_seq 512
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from byte_data import VOCAB, ByteStreamDataset, collate_bytes
from entropy_lm import EntropyByteLM


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(progress, 1.0)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet", default="corpus/jamo-codec-v3/train.parquet")
    p.add_argument("--val_parquet", default="corpus/jamo-codec-v3/val.parquet")
    p.add_argument("--out_dir", default="exp-jamo-codec/bench_blt/ckpt_entropy")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--byte_seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=384)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"out: {out}")

    model = EntropyByteLM(
        vocab=VOCAB,
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        window=args.window,
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    train_ds = ByteStreamDataset(
        args.train_parquet, byte_seq=args.byte_seq + 1, seed=args.seed, shuffle=True
    )
    val_ds = ByteStreamDataset(
        args.val_parquet, byte_seq=args.byte_seq + 1, seed=0, shuffle=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_bytes, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=1,
        collate_fn=collate_bytes, pin_memory=True,
    )

    @torch.no_grad()
    def run_val() -> tuple[float, float]:
        model.eval()
        loss_sum = 0.0
        n_sum = 0
        vit = iter(val_dl)
        for _ in range(args.val_batches):
            try:
                b = next(vit)
            except StopIteration:
                break
            b = b.to(device, non_blocking=True)
            inp = b[:, :-1]
            tgt = b[:, 1:]
            logits = model(inp)
            l = F.cross_entropy(
                logits.reshape(-1, VOCAB).float(), tgt.reshape(-1).long(),
                reduction="sum",
            )
            loss_sum += l.item()
            n_sum += tgt.numel()
        model.train()
        loss = loss_sum / max(1, n_sum)
        bpb = loss / math.log(2)
        return loss, bpb

    step = 0
    t_last = time.time()
    loss_accum = 0.0
    loss_n = 0
    model.train()
    log_path = out / "train_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"=== entropy LM train start: {n_params/1e6:.2f}M params ===")

    for batch in train_dl:
        if step >= args.max_steps:
            break
        lr = cosine_lr(step, args.warmup, args.max_steps, args.lr, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = batch.to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1).long())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        loss_accum += loss.item()
        loss_n += 1
        step += 1

        if step % args.log_every == 0:
            avg = loss_accum / loss_n
            bpb = avg / math.log(2)
            dt = time.time() - t_last
            t_last = time.time()
            log(
                f"[step {step:6d}] loss={avg:.4f} bpb={bpb:.3f} lr={lr:.2e} "
                f"{args.log_every / dt:.1f} step/s"
            )
            loss_accum = 0.0
            loss_n = 0

        if step % args.val_every == 0 or step == args.max_steps:
            vl, vbpb = run_val()
            log(f"[val  {step:6d}] val_loss={vl:.4f} val_bpb={vbpb:.3f}")

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "step": step,
            }
            path = out / f"entropy_step_{step}.pt"
            torch.save(ckpt, path)
            log(f"[save {step:6d}] -> {path.name}")

    log("=== done ===")


if __name__ == "__main__":
    main()
