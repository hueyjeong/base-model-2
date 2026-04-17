"""KoELECTRA Small v3 + SimpleCodec 사전학습 스크립트.

사용 예시 (4 GPU DDP):
    torchrun --nproc_per_node=4 -m exp-jamo-codec.koelectra.train \
        --codec_ckpt checkpoints/simple_codec_final.pt \
        --train_parquet corpus/k-exaone_random_coverage_1000_len4096.parquet \
        --val_parquet   corpus/k-exaone_coverage_5_len1000.parquet \
        --max_patches 512 --max_jamo_per_token 32 \
        --batch_size 128 --grad_accum_steps 1 \
        --lr 5e-4 --warmup_steps 10000 --max_steps 800000 \
        --mask_ratio 0.20 --gen_loss_weight 50.0 \
        --save_every 10000 --val_every 5000 \
        --rclone_remote "gdrive:exp-jamo-codec-koelectra/small/" \
        --keep_latest_n 3 \
        --bf16

Codec 은 freeze. Transformer + proj + head 만 학습.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# sys.path: exp-jamo-codec/ + 리포 루트
_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXP_ROOT, ".."))
for p in (_EXP_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402

from koelectra.data.bbpe_token_dataset import (  # noqa: E402
    BBPETokenDataset, load_bbpe_tokenizer, _worker_init_fn,
)
from koelectra.data.masking import make_patch_mask, apply_mask  # noqa: E402
from koelectra.model.electra import JamoKoElectra  # noqa: E402
from koelectra.upload import upload_checkpoint_bundle  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# DDP
# ──────────────────────────────────────────────────────────────────────────
def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": device}


def is_rank0(rank: int) -> bool:
    return rank == 0


def cleanup_ddp(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────
# LR 스케줄러
# ──────────────────────────────────────────────────────────────────────────
def linear_lr(step: int, warmup: int, max_lr: float, max_steps: int, min_lr: float = 0.0) -> float:
    """ELECTRA 원논문 스케줄: linear warmup → linear decay to min_lr."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    remaining = max_steps - warmup
    progress = (step - warmup) / max(remaining, 1)
    progress = min(max(progress, 0.0), 1.0)
    return max_lr * (1 - progress) + min_lr * progress


def apply_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ──────────────────────────────────────────────────────────────────────────
# Flash Attention 검증
# ──────────────────────────────────────────────────────────────────────────
def check_flash_attention(device, dtype, rank: int):
    if not is_rank0(rank):
        return
    if device.type != "cuda":
        print("[Flash] CPU 환경 — skip")
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"[Flash] OK (dtype={dtype}, out.shape={tuple(out.shape)})")
    except Exception as e:
        print(f"[Flash] 경고: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────
def collate_batch(samples):
    """BBPETokenDataset 배치화."""
    jamo_ids = torch.stack([s["jamo_ids"] for s in samples])
    jamo_mask = torch.stack([s["jamo_mask"] for s in samples])
    token_pad_mask = torch.stack([s["token_pad_mask"] for s in samples])
    special_token_mask = torch.stack([s["special_token_mask"] for s in samples])
    n_tokens = torch.tensor([s["n_tokens"] for s in samples], dtype=torch.long)
    return {
        "jamo_ids": jamo_ids,
        "jamo_mask": jamo_mask,
        "token_pad_mask": token_pad_mask,
        "special_token_mask": special_token_mask,
        "n_tokens": n_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────────────────────────────────
def unwrap(model):
    m = model
    for _ in range(2):
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        elif isinstance(m, DDP):
            m = m.module
        else:
            break
    return m


def _rng_sidecar_path(ckpt_path: str, rank: int) -> str:
    base, ext = os.path.splitext(ckpt_path)
    return f"{base}.rng_rank{rank}{ext}"


def _snapshot_rng():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def save_checkpoint(path, step, model, optimizer, dataset, args, rank: int, extra=None):
    if is_rank0(rank):
        state = {
            "step": step,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_state": dataset.state_dict() if dataset is not None else {},
            "args": vars(args),
        }
        if extra:
            state.update(extra)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)

    rng_path = _rng_sidecar_path(path, rank)
    tmp = rng_path + ".tmp"
    torch.save(_snapshot_rng(), tmp)
    os.replace(tmp, rng_path)


def _restore_rng(rng: dict, rank: int):
    if rng.get("torch") is not None:
        try:
            torch_state = rng["torch"]
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch_state.cpu().to(torch.uint8)
            torch.set_rng_state(torch_state)
        except Exception as e:
            print(f"[Resume rank{rank}] torch RNG skip: {e}")
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            cuda_states = [s.cpu().to(torch.uint8) if torch.is_tensor(s) else s
                           for s in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            print(f"[Resume rank{rank}] cuda RNG skip: {e}")
    if rng.get("python") is not None:
        try:
            random.setstate(rng["python"])
        except Exception:
            pass
    if rng.get("numpy") is not None:
        try:
            np.random.set_state(rng["numpy"])
        except Exception:
            pass


def load_checkpoint(path, model, optimizer, dataset, device, rank: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if dataset is not None and "data_state" in ckpt:
        dataset.load_state_dict(ckpt["data_state"])

    rng_path = _rng_sidecar_path(path, rank)
    if os.path.exists(rng_path):
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        _restore_rng(rng, rank)
        if is_rank0(rank):
            print(f"[Resume] RNG sidecar 복원: {rng_path}")
    elif is_rank0(rank):
        print(f"[Resume] RNG sidecar 없음 — rank 별 재초기화")
    return ckpt.get("step", 0)


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(model, val_dataset, args, device, amp_dtype, rank, world_size,
                   n_batches: int = 500):
    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=0,
        collate_fn=collate_batch,
    )
    gen_sum = torch.zeros(1, device=device)
    disc_sum = torch.zeros(1, device=device)
    total_sum = torch.zeros(1, device=device)
    acc_sum = torch.zeros(1, device=device)
    util_sum = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
        special_token_mask = batch["special_token_mask"].to(device, non_blocking=True)
        n_tokens = batch["n_tokens"].to(device, non_blocking=True)

        masked_patch_mask = make_patch_mask(
            n_tokens, max_patches=args.max_patches,
            mask_ratio=args.mask_ratio,
            special_patch_mask=special_token_mask,
        )
        masked_jamo_ids, masked_jamo_mask, per_jamo_mask = apply_mask(
            jamo_ids, jamo_mask, masked_patch_mask,
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model(jamo_ids, jamo_mask, token_pad_mask,
                        masked_jamo_ids, masked_jamo_mask,
                        masked_patch_mask, per_jamo_mask)
        gen_sum += out["gen_loss"].detach().float()
        disc_sum += out["disc_loss"].detach().float()
        total_sum += out["total_loss"].detach().float()
        acc_sum += out["disc_acc"].detach().float()
        util_sum += out["patch_util"].detach().float()
        count += 1.0

    if world_size > 1:
        dist.all_reduce(gen_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(disc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(util_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    count = count.clamp(min=1)
    model.train()
    return {
        "val/gen_loss": (gen_sum / count).item(),
        "val/disc_loss": (disc_sum / count).item(),
        "val/total_loss": (total_sum / count).item(),
        "val/disc_acc": (acc_sum / count).item(),
        "val/patch_util": (util_sum / count).item(),
    }


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # Codec & Model
    ap.add_argument("--codec_ckpt", type=str,
                    default="checkpoints/simple_codec_final.pt")
    ap.add_argument("--codec_d_model", type=int, default=256)
    ap.add_argument("--codec_n_enc_layers", type=int, default=5)
    ap.add_argument("--codec_n_dec_layers", type=int, default=5)
    ap.add_argument("--codec_kernel_size", type=int, default=5)
    ap.add_argument("--max_jamo_per_token", type=int, default=32)
    ap.add_argument("--embedding_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--gen_layers", type=int, default=12)
    ap.add_argument("--disc_layers", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_patches", type=int, default=512)
    ap.add_argument("--gen_loss_weight", type=float, default=50.0)

    # 데이터
    ap.add_argument("--train_parquet", type=str, nargs="+", required=True)
    ap.add_argument("--val_parquet", type=str, nargs="+", default=None)
    ap.add_argument("--text_key", type=str, default="text")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_length", type=int, default=10)

    # 마스킹
    ap.add_argument("--mask_ratio", type=float, default=0.20)

    # 학습
    ap.add_argument("--batch_size", type=int, default=128, help="per-GPU")
    ap.add_argument("--val_batch_size", type=int, default=64)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_lr", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=800000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_eps", type=float, default=1e-6)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--no_tf32", action="store_true")

    # 체크포인트 & 로깅
    ap.add_argument("--out_dir", type=str, default="exp-jamo-codec/koelectra/checkpoints")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_batches", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--rclone_remote", type=str, default=None)
    ap.add_argument("--keep_latest_n", type=int, default=3)
    ap.add_argument("--log_file", type=str, default=None)

    args = ap.parse_args()

    # TF32
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # DDP
    ddp = setup_ddp()
    rank, world_size, device = ddp["rank"], ddp["world_size"], ddp["device"]

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16 and device.type == "cuda"

    if is_rank0(rank):
        print(f"[Setup] rank={rank}/{world_size}, device={device}, bf16={args.bf16}")
        print(f"[Setup] max_patches={args.max_patches}, "
              f"max_jamo_per_token={args.max_jamo_per_token}")
        os.makedirs(args.out_dir, exist_ok=True)

    check_flash_attention(device, amp_dtype, rank)

    # ── 토크나이저 ──
    if is_rank0(rank):
        print("[Tok] BBPE(K-EXAONE) + JamoTokenizer 로드")
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    # ── Dataset ──
    train_ds = BBPETokenDataset(
        file_paths=args.train_parquet,
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_patches=args.max_patches,
        max_jamo_per_token=args.max_jamo_per_token,
        text_key=args.text_key,
        min_length=args.min_length,
        rank=rank, world_size=world_size,
    )
    val_ds = None
    if args.val_parquet:
        val_ds = BBPETokenDataset(
            file_paths=args.val_parquet,
            bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
            max_patches=args.max_patches,
            max_jamo_per_token=args.max_jamo_per_token,
            text_key=args.text_key,
            min_length=args.min_length,
            rank=rank, world_size=world_size,
        )
        val_ds._prewarm_cache(verbose=is_rank0(rank))

    # ── 모델 ──
    model = JamoKoElectra(
        codec_d_model=args.codec_d_model,
        codec_n_enc_layers=args.codec_n_enc_layers,
        codec_n_dec_layers=args.codec_n_dec_layers,
        codec_kernel_size=args.codec_kernel_size,
        max_jamo_per_token=args.max_jamo_per_token,
        codec_dropout=args.dropout,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        gen_layers=args.gen_layers,
        disc_layers=args.disc_layers,
        dropout=args.dropout,
        max_patches=args.max_patches,
        gen_loss_weight=args.gen_loss_weight,
    ).to(device)

    load_info = model.load_codec_pretrained(args.codec_ckpt, map_location=device)
    if is_rank0(rank):
        total = sum(p.numel() for p in model.parameters())
        codec_n = sum(p.numel() for p in model.codec_parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] total={total/1e6:.2f}M | "
              f"codec(frozen)={codec_n/1e6:.2f}M | "
              f"trainable={trainable/1e6:.2f}M")
        print(f"[Codec load] missing={len(load_info['missing'])}, "
              f"unexpected={len(load_info['unexpected'])}")

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[ddp["local_rank"]], find_unused_parameters=False)

    # torch.compile
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 64
        _dynamo.config.accumulated_cache_size_limit = 256
        model = torch.compile(model, mode=args.compile_mode)
        if is_rank0(rank):
            print(f"[Compile] torch.compile mode={args.compile_mode}")

    # ── Optimizer (codec freeze — non_codec_parameters 만) ──
    trainable_params = list(unwrap(model).non_codec_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # ── Resume ──
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        if is_rank0(rank):
            print(f"[Resume] {args.resume}")
        global_step = load_checkpoint(args.resume, model, optimizer, train_ds, device, rank)
        if is_rank0(rank):
            print(f"[Resume] step={global_step}")

    # ── DataLoader ──
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_batch, persistent_workers=(args.num_workers > 0),
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )

    # 로그 파일
    if is_rank0(rank):
        log_path = args.log_file or os.path.join(args.out_dir, "train.log")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    else:
        log_path = None
    log_file = open(log_path, "a") if log_path else None
    if is_rank0(rank):
        print(f"[Log] {log_path}")

    model.train()
    t0 = time.time()
    acc_total = acc_gen = acc_disc = acc_acc = acc_rep = acc_mask = acc_util = 0.0
    acc_count = 0
    last_log_step = global_step

    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    upload_threads: list = []

    while global_step < args.max_steps:
        for micro in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
            token_pad_mask = batch["token_pad_mask"].to(device, non_blocking=True)
            special_token_mask = batch["special_token_mask"].to(device, non_blocking=True)
            n_tokens = batch["n_tokens"].to(device, non_blocking=True)

            masked_patch_mask = make_patch_mask(
                n_tokens, max_patches=args.max_patches,
                mask_ratio=args.mask_ratio,
                special_patch_mask=special_token_mask,
            )
            masked_jamo_ids, masked_jamo_mask, per_jamo_mask = apply_mask(
                jamo_ids, jamo_mask, masked_patch_mask,
            )

            is_last_micro = (micro == args.grad_accum_steps - 1)
            sync_ctx = (model.no_sync() if (isinstance(model, DDP) and not is_last_micro)
                        else _nullctx())
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(jamo_ids, jamo_mask, token_pad_mask,
                                masked_jamo_ids, masked_jamo_mask,
                                masked_patch_mask, per_jamo_mask)
                    loss = out["total_loss"] / args.grad_accum_steps
                loss.backward()

            acc_total += out["total_loss"].detach().float().item()
            acc_gen += out["gen_loss"].detach().float().item()
            acc_disc += out["disc_loss"].detach().float().item()
            acc_acc += out["disc_acc"].detach().float().item()
            acc_rep += out["replaced_rate"].detach().float().item()
            acc_mask += out["masked_tokens"].detach().float().item()
            acc_util += out["patch_util"].detach().float().item()
            acc_count += 1

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        lr_now = linear_lr(global_step, args.warmup_steps, args.lr, args.max_steps, args.min_lr)
        apply_lr(optimizer, lr_now)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        if is_rank0(rank) and global_step % args.log_every == 0:
            elapsed = time.time() - t0
            steps_since = global_step - last_log_step
            steps_per_s = steps_since / max(elapsed, 1e-6) if elapsed > 0 else 0.0
            n = max(acc_count, 1)
            msg = (f"step {global_step:>7d} | "
                   f"total {acc_total/n:.3f} | "
                   f"gen {acc_gen/n:.3f} | "
                   f"disc {acc_disc/n:.3f} | "
                   f"disc_acc {acc_acc/n:.3f} | "
                   f"rep {acc_rep/n:.3f} | "
                   f"mtok {acc_mask/n:.0f} | "
                   f"util {acc_util/n:.3f} | "
                   f"lr {lr_now:.2e} | "
                   f"{steps_per_s:.2f} step/s")
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device) / 1024**3
                msg += f" | mem {mem:.1f}GB"
            print(msg, flush=True)
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()
            t0 = time.time()
            last_log_step = global_step
            acc_total = acc_gen = acc_disc = acc_acc = acc_rep = acc_mask = acc_util = 0.0
            acc_count = 0

        # Validation
        if val_ds is not None and global_step % args.val_every == 0:
            val_ds.load_state_dict({"line_counter": 0})
            val_metrics = run_validation(
                model, val_ds, args, device, amp_dtype, rank, world_size,
                n_batches=args.val_batches,
            )
            if is_rank0(rank):
                msg = f"[Val] step {global_step} | " + " | ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                print(msg, flush=True)
                if log_file:
                    log_file.write(msg + "\n")
                    log_file.flush()

        # Checkpoint
        if global_step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"electra_step_{global_step}.pt")
            save_checkpoint(ckpt_path, global_step, model, optimizer, train_ds, args, rank)
            if world_size > 1:
                dist.barrier()
            if is_rank0(rank):
                print(f"[Ckpt] saved → {ckpt_path}", flush=True)
                if args.rclone_remote:
                    t = upload_checkpoint_bundle(
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        world_size=world_size,
                        remote_dest=args.rclone_remote,
                        keep_latest_n=args.keep_latest_n,
                        blocking=False,
                    )
                    if t is not None:
                        upload_threads.append(t)
                        upload_threads[:] = [u for u in upload_threads if u.is_alive()]

    # 최종 체크포인트
    final_path = os.path.join(args.out_dir, f"electra_step_{global_step}_final.pt")
    save_checkpoint(final_path, global_step, model, optimizer, train_ds, args, rank)
    if world_size > 1:
        dist.barrier()
    if is_rank0(rank):
        print(f"[Done] final → {final_path}")
        for t in upload_threads:
            t.join(timeout=300)
        if args.rclone_remote:
            if log_file:
                log_file.flush()
            upload_checkpoint_bundle(
                ckpt_path=final_path,
                log_path=log_path,
                world_size=world_size,
                remote_dest=args.rclone_remote,
                keep_latest_n=args.keep_latest_n,
                blocking=True,
            )
        if log_file:
            log_file.close()

    cleanup_ddp(world_size)


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
