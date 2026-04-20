"""BBPE Base 학습 — BLT 비교용.

bpb (bits-per-byte) 정확 계산:
  bpb = sum(-log p(token)) / (sum(byte_per_token[targets]) * ln(2))

사용:
    source .venv/bin/activate
    export PYTHONPATH=exp-jamo-codec/bench_blt
    python exp-jamo-codec/bench_blt/train_bbpe.py \
        --tokenizer checkpoints/bbpe_35k \
        --train_parquet corpus/jamo-codec-v3/train.parquet \
        --val_parquet corpus/jamo-codec-v3/val.parquet \
        --out_dir exp-jamo-codec/bench_blt/ckpt_bbpe \
        --max_steps 10000 --batch_size 2 --seq_len 290
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbpe_data import BBPEStreamDataset, collate_bbpe, precompute_byte_lengths
from bbpe_lm import BBPELM


def cosine_lr(step, warmup, max_steps, peak, min_ratio):
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="checkpoints/bbpe_35k")
    p.add_argument("--train_parquet", default="corpus/jamo-codec-v3/train.parquet")
    p.add_argument("--val_parquet", default="corpus/jamo-codec-v3/val.parquet")
    p.add_argument("--out_dir", default="exp-jamo-codec/bench_blt/ckpt_bbpe")
    p.add_argument("--seq_len", type=int, default=290)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--val_batch_size", type=int, default=4)
    # model
    p.add_argument("--vocab", type=int, default=35000)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--d_ff", type=int, default=3072)
    p.add_argument("--no_tie_embed", action="store_true")
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=30)
    p.add_argument("--save_every", type=int, default=5000)
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

    # byte length table (bpb 계산용)
    print("computing byte lengths...")
    byte_len = precompute_byte_lengths(args.tokenizer).to(device)
    print(f"byte_len: total={byte_len.sum().item()} mean_nonzero={byte_len[byte_len>0].float().mean().item():.2f}")

    # model
    model = BBPELM(
        vocab=args.vocab,
        hidden=args.hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        tie_embed=not args.no_tie_embed,
    ).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BBPELM params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    # data
    train_ds = BBPEStreamDataset(
        args.train_parquet, args.tokenizer, seq_len=args.seq_len, seed=args.seed, shuffle=True
    )
    val_ds = BBPEStreamDataset(
        args.val_parquet, args.tokenizer, seq_len=args.seq_len, seed=0, shuffle=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_bbpe, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=1,
        collate_fn=collate_bbpe, pin_memory=True,
    )

    log_path = out / "train_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"=== BBPE Base train start: {n_params/1e6:.2f}M ===")
    log(f"seq_len={args.seq_len} batch={args.batch_size} lr={args.lr} max_steps={args.max_steps}")

    @torch.no_grad()
    def run_val() -> tuple[float, float, float]:
        model.eval()
        nat_sum = 0.0
        tok_sum = 0
        byte_sum = 0
        it = iter(val_dl)
        for _ in range(args.val_batches):
            try:
                b = next(it)
            except StopIteration:
                break
            b = b.to(device, non_blocking=True)
            inp = b[:, :-1]
            tgt = b[:, 1:]
            logits = model(inp)
            l = F.cross_entropy(
                logits.reshape(-1, args.vocab).float(), tgt.reshape(-1).long(),
                reduction="sum",
            )
            nat_sum += l.item()
            tok_sum += tgt.numel()
            byte_sum += byte_len[tgt].sum().item()
        model.train()
        loss_per_tok = nat_sum / max(1, tok_sum)
        bpb = (nat_sum / max(1, byte_sum)) / math.log(2)
        return loss_per_tok, bpb, byte_sum / max(1, tok_sum)

    step = 0
    t_last = time.time()
    nat_accum = 0.0
    tok_accum = 0
    byte_accum = 0
    n_samples = 0
    model.train()

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
        loss = F.cross_entropy(logits.reshape(-1, args.vocab).float(), tgt.reshape(-1).long())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        nat_accum += loss.item() * tgt.numel()
        tok_accum += tgt.numel()
        byte_accum += byte_len[tgt].sum().item()
        n_samples += 1
        step += 1

        if step % args.log_every == 0:
            avg = nat_accum / max(1, tok_accum)
            bpb = (nat_accum / max(1, byte_accum)) / math.log(2)
            avg_bpt = byte_accum / max(1, tok_accum)
            dt = time.time() - t_last
            t_last = time.time()
            log(
                f"[step {step:6d}] loss={avg:.4f} bpb={bpb:.3f} bpt={avg_bpt:.2f} "
                f"lr={lr:.2e} {args.log_every / dt:.2f} step/s"
            )
            nat_accum = 0.0
            tok_accum = 0
            byte_accum = 0
            n_samples = 0

        if step % args.val_every == 0 or step == args.max_steps:
            vl, vbpb, vbpt = run_val()
            log(f"[val  {step:6d}] val_loss={vl:.4f} val_bpb={vbpb:.3f} val_bpt={vbpt:.2f}")

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {"model": model.state_dict(), "args": vars(args), "step": step}
            path = out / f"bbpe_step_{step}.pt"
            torch.save(ckpt, path)
            log(f"[save {step:6d}] -> {path.name}")

    log("=== done ===")


if __name__ == "__main__":
    main()
