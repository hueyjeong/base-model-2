"""BLT 128M 학습 — entropy LM (frozen) patching 사용.

사용:
    source .venv/bin/activate
    export PYTHONPATH=exp-jamo-codec/bench_blt
    python exp-jamo-codec/bench_blt/train_blt.py \
        --entropy_ckpt exp-jamo-codec/bench_blt/ckpt_entropy/entropy_step_5000.pt \
        --train_parquet corpus/jamo-codec-v3/train.parquet \
        --val_parquet corpus/jamo-codec-v3/val.parquet \
        --out_dir exp-jamo-codec/bench_blt/ckpt_blt \
        --max_steps 2000 --batch_size 2 --byte_seq 1024 \
        --target_patch_size 6
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

from blt_full import BLT128M
from byte_data import BOS, VOCAB, ByteStreamDataset, collate_bytes
from entropy_lm import EntropyByteLM


def load_entropy_lm(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> EntropyByteLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = EntropyByteLM(
        vocab=VOCAB,
        hidden=a.get("hidden", 128),
        n_layers=a.get("n_layers", 6),
        n_heads=a.get("n_heads", 4),
        d_ff=a.get("d_ff", 384),
        window=a.get("window", 256),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def compute_entropy_batch(
    entropy: EntropyByteLM, byte_ids: torch.Tensor, chunk: int = 1024
) -> torch.Tensor:
    """큰 입력을 chunk 별로 끊어서 H 계산. [B, L] → [B, L]."""
    return entropy.entropy(byte_ids)


@torch.no_grad()
def calibrate_threshold(
    entropy: EntropyByteLM,
    dl: DataLoader,
    device: torch.device,
    target_patch_size: int,
    n_batches: int = 10,
) -> float:
    """평균 patch size 가 target 이 되도록 global threshold 추정.
    boundary_ratio ≈ 1 / target_patch_size
    → threshold = (1 - 1/target) 분위수
    """
    ratio = 1.0 / max(1, target_patch_size)
    pct = 1.0 - ratio
    h_pool: list[torch.Tensor] = []
    it = iter(dl)
    for _ in range(n_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        b = b.to(device, non_blocking=True)
        H = compute_entropy_batch(entropy, b)
        h_pool.append(H.flatten().float().cpu())
    allh = torch.cat(h_pool)
    return float(torch.quantile(allh, pct).item())


def make_boundaries(
    byte_ids: torch.Tensor, H: torch.Tensor, threshold: float
) -> torch.Tensor:
    """[B, L] + H [B, L] + threshold → boundaries [B, L] bool."""
    bound = H > threshold
    bound[:, 0] = True  # 첫 위치 강제
    bound = bound | (byte_ids == BOS)  # BOS 위치 강제
    return bound


def cosine_lr(step: int, warmup: int, max_steps: int, peak: float, min_ratio: float) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(progress, 1.0)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entropy_ckpt", required=True)
    p.add_argument("--train_parquet", default="corpus/jamo-codec-v3/train.parquet")
    p.add_argument("--val_parquet", default="corpus/jamo-codec-v3/val.parquet")
    p.add_argument("--out_dir", default="exp-jamo-codec/bench_blt/ckpt_blt")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--byte_seq", type=int, default=1024)
    p.add_argument("--target_patch_size", type=int, default=6)
    p.add_argument("--threshold", type=float, default=None, help="수동 threshold. None 이면 자동 calibration")
    p.add_argument("--calib_batches", type=int, default=10)
    # BLT config
    p.add_argument("--h_enc", type=int, default=384)
    p.add_argument("--h_lat", type=int, default=768)
    p.add_argument("--h_dec", type=int, default=384)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--lat_layers", type=int, default=12)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--enc_heads", type=int, default=6)
    p.add_argument("--lat_heads", type=int, default=12)
    p.add_argument("--dec_heads", type=int, default=6)
    p.add_argument("--enc_ff", type=int, default=1024)
    p.add_argument("--lat_ff", type=int, default=3072)
    p.add_argument("--dec_ff", type=int, default=1024)
    p.add_argument("--hash_buckets", type=int, default=50_000)
    p.add_argument("--hash_dim", type=int, default=64)
    p.add_argument("--swa_window", type=int, default=512)
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    # logging
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=30)
    p.add_argument("--save_every", type=int, default=1000)
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

    # Entropy LM
    print(f"\nloading entropy LM: {args.entropy_ckpt}")
    entropy = load_entropy_lm(args.entropy_ckpt, device, dtype)
    nE = sum(p.numel() for p in entropy.parameters())
    print(f"entropy params: {nE/1e6:.2f}M (frozen)")

    # BLT
    print()
    blt = BLT128M(
        byte_vocab=VOCAB,
        hash_buckets=args.hash_buckets,
        hash_dim=args.hash_dim,
        h_enc=args.h_enc,
        h_lat=args.h_lat,
        h_dec=args.h_dec,
        enc_layers=args.enc_layers,
        lat_layers=args.lat_layers,
        dec_layers=args.dec_layers,
        enc_heads=args.enc_heads,
        lat_heads=args.lat_heads,
        dec_heads=args.dec_heads,
        enc_ff=args.enc_ff,
        lat_ff=args.lat_ff,
        dec_ff=args.dec_ff,
        swa_window=args.swa_window,
    ).to(device=device, dtype=dtype)
    nB = sum(p.numel() for p in blt.parameters())
    print(f"BLT params: {nB/1e6:.2f}M")

    opt = torch.optim.AdamW(
        blt.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    # Data
    train_ds = ByteStreamDataset(
        args.train_parquet, byte_seq=args.byte_seq, seed=args.seed, shuffle=True
    )
    val_ds = ByteStreamDataset(
        args.val_parquet, byte_seq=args.byte_seq, seed=0, shuffle=False
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_bytes, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.val_batch_size, num_workers=1,
        collate_fn=collate_bytes, pin_memory=True,
    )

    # Calibration
    if args.threshold is None:
        calib_dl = DataLoader(
            train_ds, batch_size=args.batch_size, num_workers=0,
            collate_fn=collate_bytes, pin_memory=True,
        )
        print(f"\ncalibrating threshold (target patch_size={args.target_patch_size})...")
        th = calibrate_threshold(
            entropy, calib_dl, device,
            target_patch_size=args.target_patch_size, n_batches=args.calib_batches,
        )
        print(f"calibrated threshold: {th:.4f}")
    else:
        th = args.threshold
        print(f"\nusing threshold: {th:.4f}")

    log_path = out / "train_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(f"=== BLT train start ===")
    log(f"entropy: {args.entropy_ckpt}")
    log(f"BLT params: {nB/1e6:.2f}M  threshold={th:.4f}  target_ps={args.target_patch_size}")
    log(f"batch={args.batch_size} byte_seq={args.byte_seq} lr={args.lr} max_steps={args.max_steps}")

    @torch.no_grad()
    def run_val() -> tuple[float, float, float]:
        blt.eval()
        loss_sum = 0.0
        n_sum = 0
        ps_sum = 0.0
        ps_n = 0
        it = iter(val_dl)
        for _ in range(args.val_batches):
            try:
                b = next(it)
            except StopIteration:
                break
            b = b.to(device, non_blocking=True)
            H = entropy.entropy(b)
            bound = make_boundaries(b, H, th)
            inp = b[:, :-1]
            bnd = bound[:, :-1]
            tgt = b[:, 1:]
            logits = blt(inp, bnd)
            l = F.cross_entropy(
                logits.reshape(-1, VOCAB).float(), tgt.reshape(-1).long(),
                reduction="sum",
            )
            loss_sum += l.item()
            n_sum += tgt.numel()
            # patch size
            n_patches = (bnd.long().sum(dim=1)).float()
            ps = (bnd.shape[1] / n_patches).mean().item()
            ps_sum += ps
            ps_n += 1
        blt.train()
        loss = loss_sum / max(1, n_sum)
        bpb = loss / math.log(2)
        avg_ps = ps_sum / max(1, ps_n)
        return loss, bpb, avg_ps

    step = 0
    t_last = time.time()
    loss_accum = 0.0
    loss_n = 0
    ps_accum = 0.0
    blt.train()
    grad_accum = max(1, args.grad_accum)
    microstep = 0

    for batch in train_dl:
        if step >= args.max_steps:
            break

        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            H = entropy.entropy(batch)
            bound = make_boundaries(batch, H, th)
        inp = batch[:, :-1]
        bnd = bound[:, :-1]
        tgt = batch[:, 1:]

        logits = blt(inp, bnd)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1).long())
        loss = loss / grad_accum
        loss.backward()

        # patch size metric
        n_p = bnd.long().sum(dim=1).float()
        ps_accum += (bnd.shape[1] / n_p).mean().item()

        microstep += 1
        if microstep % grad_accum != 0:
            continue

        lr = cosine_lr(step, args.warmup, args.max_steps, args.lr, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        torch.nn.utils.clip_grad_norm_(blt.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        loss_accum += loss.item() * grad_accum
        loss_n += 1
        step += 1

        if step % args.log_every == 0:
            avg = loss_accum / loss_n
            bpb = avg / math.log(2)
            ps = ps_accum / (loss_n * grad_accum)
            dt = time.time() - t_last
            t_last = time.time()
            log(
                f"[step {step:6d}] loss={avg:.4f} bpb={bpb:.3f} ps={ps:.2f} "
                f"lr={lr:.2e} {args.log_every / dt:.2f} step/s"
            )
            loss_accum = 0.0
            loss_n = 0
            ps_accum = 0.0

        if step % args.val_every == 0 or step == args.max_steps:
            vl, vbpb, vps = run_val()
            log(f"[val  {step:6d}] val_loss={vl:.4f} val_bpb={vbpb:.3f} val_ps={vps:.2f}")

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {
                "model": blt.state_dict(),
                "args": vars(args),
                "step": step,
                "threshold": th,
            }
            path = out / f"blt_step_{step}.pt"
            torch.save(ckpt, path)
            log(f"[save {step:6d}] -> {path.name}")

    log("=== done ===")


if __name__ == "__main__":
    main()
