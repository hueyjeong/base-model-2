"""KoELECTRA Small v3 + Jamo-Codec 사전학습 스크립트.

사용 예시 (4 GPU DDP):
    torchrun --nproc_per_node=4 -m exp-jamo-codec.koelectra.train \
        --codec_ckpt exp-jamo-codec/checkpoints/composition_6L_step600000.pt \
        --train_parquet corpus/jamo-codec-v3/train.parquet \
        --val_parquet   corpus/jamo-codec-v3/val.parquet \
        --max_seq_len 2048 --max_patches 512 \
        --batch_size 128 --grad_accum_steps 1 \
        --lr 5e-4 --codec_lr_ratio 0.1 \
        --warmup_steps 10000 --max_steps 800000 \
        --mask_ratio 0.20 --gen_loss_weight 50.0 \
        --save_every 10000 --val_every 5000 \
        --rclone_remote "gdrive:exp-jamo-codec-koelectra/small/" \
        --keep_latest_n 3 \
        --bf16
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

# 경로 설정: exp-jamo-codec/ 를 sys.path에 등록하여 codec/, tok/, data/ import 가능
_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_EXP_ROOT, ".."))
for p in (_EXP_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer  # noqa: E402
from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402

from koelectra.model.electra import JamoKoElectra  # noqa: E402
from koelectra.data.masking import make_patch_mask, apply_mask  # noqa: E402

# training/upload_gdrive.py 재사용
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "training"))
from upload_gdrive import upload_and_cleanup  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# DDP
# ──────────────────────────────────────────────────────────────────────────
def setup_ddp():
    """환경변수 기반 DDP 초기화. 비-DDP 환경에서도 동작."""
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
# LR 스케줄러 (linear warmup + linear decay)
# ──────────────────────────────────────────────────────────────────────────
def linear_lr(step: int, warmup: int, max_lr: float, max_steps: int, min_lr: float = 0.0) -> float:
    """ELECTRA 원논문 스케줄: linear warmup → linear decay to 0."""
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    remaining = max_steps - warmup
    progress = (step - warmup) / max(remaining, 1)
    progress = min(max(progress, 0.0), 1.0)
    return max_lr * (1 - progress) + min_lr * progress


def apply_lr(optimizer, lr_main: float, lr_codec: float):
    for pg in optimizer.param_groups:
        if pg.get("name") == "codec":
            pg["lr"] = lr_codec
        else:
            pg["lr"] = lr_main


# ──────────────────────────────────────────────────────────────────────────
# Flash Attention 검증
# ──────────────────────────────────────────────────────────────────────────
def check_flash_attention(device, dtype, rank: int):
    """SDPBackend.FLASH_ATTENTION 컨텍스트에서 더미 forward. 실패 시 경고."""
    if not is_rank0(rank):
        return
    if device.type != "cuda":
        print("[Flash] CPU 환경 — Flash Attention skip")
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        k = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        v = torch.randn(2, 4, 64, 64, device=device, dtype=dtype)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"[Flash] FLASH_ATTENTION backend 활성 OK (dtype={dtype}, out.shape={tuple(out.shape)})")
    except Exception as e:
        print(f"[Flash] 경고: FLASH_ATTENTION 강제 실패 → {type(e).__name__}: {e}")
        print("[Flash] 학습은 계속 진행 (기본 SDPA가 적절한 backend 자동 선택)")


# ──────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────
def collate_batch(samples):
    """BBPEJamoDataset 배치화."""
    jamo_ids = torch.stack([s["jamo_ids"] for s in samples])
    jamo_mask = torch.stack([s["jamo_mask"] for s in samples])
    segment_ids = torch.stack([s["segment_ids"] for s in samples])
    n_segments = torch.tensor([s["n_segments"] for s in samples], dtype=torch.long)
    line_counters = [s.get("_line_counter", 0) for s in samples]
    return {
        "jamo_ids": jamo_ids,
        "jamo_mask": jamo_mask,
        "segment_ids": segment_ids,
        "n_segments": n_segments,
        "line_counters": line_counters,
    }


# ──────────────────────────────────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────────────────────────────────
def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def _rng_sidecar_path(ckpt_path: str, rank: int) -> str:
    """체크포인트 본체와 같은 디렉터리에 rank별 RNG sidecar 파일.

    예: electra_step_10000.pt → electra_step_10000.rng_rank2.pt
    `electra_step_*` 글롭에 걸리므로 upload_and_cleanup이 함께 정리한다.
    """
    base, ext = os.path.splitext(ckpt_path)  # (.../electra_step_10000, .pt)
    return f"{base}.rng_rank{rank}{ext}"


def _snapshot_rng():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def save_checkpoint(path, step, model, optimizer, dataset, args, rank: int, extra=None):
    """체크포인트 저장.

    - rank0만 본체(model/optimizer/step/data_state/args)를 `path`에 저장
    - 모든 rank가 자기 RNG를 sidecar 파일에 저장
    """
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

    # 모든 rank가 자기 RNG 저장 (독립 파일)
    rng_path = _rng_sidecar_path(path, rank)
    tmp = rng_path + ".tmp"
    torch.save(_snapshot_rng(), tmp)
    os.replace(tmp, rng_path)


def _restore_rng(rng: dict, rank: int):
    """rank별 RNG 딕셔너리에서 torch/cuda/python/numpy RNG 복원."""
    if rng.get("torch") is not None:
        try:
            torch_state = rng["torch"]
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch_state.cpu().to(torch.uint8)
            torch.set_rng_state(torch_state)
        except Exception as e:
            print(f"[Resume rank{rank}] torch RNG 복원 skip: {e}")
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            cuda_states = [s.cpu().to(torch.uint8) if torch.is_tensor(s) else s
                           for s in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            print(f"[Resume rank{rank}] cuda RNG 복원 skip: {e}")
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
    """체크포인트 본체(모든 rank) + 자기 rank의 RNG sidecar 로드."""
    # 본체: 모든 rank가 동일하게 읽어야 model/optimizer가 동기화됨
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        # optimizer state의 tensor를 device로 이동
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if dataset is not None and "data_state" in ckpt:
        dataset.load_state_dict(ckpt["data_state"])

    # RNG: rank별 sidecar 파일에서 로드
    rng_path = _rng_sidecar_path(path, rank)
    if os.path.exists(rng_path):
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        _restore_rng(rng, rank)
        if is_rank0(rank):
            print(f"[Resume] RNG sidecar 복원 (rank0): {rng_path}")
    else:
        # 구버전 체크포인트 호환: 본체에 rng_state가 있었던 경우
        legacy = ckpt.get("rng_state")
        if legacy is not None and is_rank0(rank):
            _restore_rng(legacy, rank)
            print(f"[Resume] rank0 RNG를 구버전 본체에서 복원 "
                  f"(rank>0는 seed+rank로 재초기화)")
        elif is_rank0(rank):
            print(f"[Resume] RNG sidecar 없음: {rng_path} — "
                  f"rank별 RNG는 seed+rank 초기값으로 진행")
    return ckpt.get("step", 0)


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(model, val_dataset, args, device, amp_dtype, rank, world_size,
                   n_batches: int = 500):
    """Val loss 집계 (all_reduce 평균)."""
    unwrap(model).eval()
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
    count = torch.zeros(1, device=device)

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
        jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)
        n_segments = batch["n_segments"].to(device, non_blocking=True)

        masked_patch_mask = make_patch_mask(n_segments, max_patches=args.max_patches,
                                            mask_ratio=args.mask_ratio)
        masked_jamo_ids, per_jamo_mask = apply_mask(
            jamo_ids, segment_ids, jamo_mask, masked_patch_mask
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model(jamo_ids, jamo_mask, segment_ids, n_segments,
                        masked_jamo_ids, per_jamo_mask, masked_patch_mask)
        gen_sum += out["gen_loss"].detach().float()
        disc_sum += out["disc_loss"].detach().float()
        total_sum += out["total_loss"].detach().float()
        acc_sum += out["disc_acc"].detach().float()
        count += 1.0

    if world_size > 1:
        dist.all_reduce(gen_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(disc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    count = count.clamp(min=1)
    unwrap(model).train()
    return {
        "val/gen_loss": (gen_sum / count).item(),
        "val/disc_loss": (disc_sum / count).item(),
        "val/total_loss": (total_sum / count).item(),
        "val/disc_acc": (acc_sum / count).item(),
    }


# ──────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # Codec & Model
    ap.add_argument("--codec_ckpt", type=str, required=True)
    ap.add_argument("--codec_d_model", type=int, default=256)
    ap.add_argument("--codec_n_layers", type=int, default=6)
    ap.add_argument("--codec_kernel_size", type=int, default=7)
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
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_length", type=int, default=10)

    # 마스킹
    ap.add_argument("--mask_ratio", type=float, default=0.20)

    # 학습
    ap.add_argument("--batch_size", type=int, default=128, help="per-GPU")
    ap.add_argument("--val_batch_size", type=int, default=64)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--codec_lr_ratio", type=float, default=0.1)
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

    # 체크포인트 & 로깅
    ap.add_argument("--out_dir", type=str, default="exp-jamo-codec/koelectra/checkpoints")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_batches", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--rclone_remote", type=str, default=None,
                    help="예: gdrive:exp-jamo-codec-koelectra/small/")
    ap.add_argument("--keep_latest_n", type=int, default=3)

    args = ap.parse_args()

    # DDP
    ddp = setup_ddp()
    rank, world_size, device = ddp["rank"], ddp["world_size"], ddp["device"]

    # 시드 (rank별로 다르게 하면 DDP 샘플 다양성↑)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16 and device.type == "cuda"

    if is_rank0(rank):
        print(f"[Setup] rank={rank}/{world_size}, device={device}, bf16={args.bf16}")
        os.makedirs(args.out_dir, exist_ok=True)

    # Flash Attention 검증
    check_flash_attention(device, amp_dtype, rank)

    # ── 토크나이저 ──
    if is_rank0(rank):
        print("[Tok] BBPE(K-EXAONE) + JamoTokenizer 로드")
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    # ── Dataset ──
    train_ds = BBPEJamoDataset(
        file_paths=args.train_parquet,
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_seq_len=args.max_seq_len,
        max_jamo_per_token=args.max_jamo_per_token,
        text_key=args.text_key,
        min_length=args.min_length,
        rank=rank, world_size=world_size,
        max_patches=args.max_patches,
    )
    val_ds = None
    if args.val_parquet:
        val_ds = BBPEJamoDataset(
            file_paths=args.val_parquet,
            bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
            max_seq_len=args.max_seq_len,
            max_jamo_per_token=args.max_jamo_per_token,
            text_key=args.text_key,
            min_length=args.min_length,
            rank=rank, world_size=world_size,
            max_patches=args.max_patches,
        )

    # ── 모델 ──
    model = JamoKoElectra(
        codec_d_model=args.codec_d_model,
        codec_n_layers=args.codec_n_layers,
        codec_kernel_size=args.codec_kernel_size,
        max_jamo_per_token=args.max_jamo_per_token,
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
        total_params = sum(p.numel() for p in model.parameters())
        codec_params = sum(p.numel() for p in model.codec_parameters())
        print(f"[Model] total={total_params/1e6:.2f}M "
              f"(codec={codec_params/1e6:.2f}M, tf+proj={(total_params-codec_params)/1e6:.2f}M)")
        print(f"[Codec load] missing(enc/dec)={len(load_info['encoder_missing'])}/"
              f"{len(load_info['decoder_missing'])}, "
              f"unexpected={len(load_info['encoder_unexpected'])}/"
              f"{len(load_info['decoder_unexpected'])}")

    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[ddp["local_rank"]], find_unused_parameters=False)

    # ── Optimizer: codec은 lr * codec_lr_ratio ──
    codec_params_list = list(unwrap(model).codec_parameters())
    non_codec_params_list = list(unwrap(model).non_codec_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": codec_params_list, "name": "codec",
             "lr": args.lr * args.codec_lr_ratio,
             "weight_decay": args.weight_decay},
            {"params": non_codec_params_list, "name": "main",
             "lr": args.lr,
             "weight_decay": args.weight_decay},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

    # ── Resume ──
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        if is_rank0(rank):
            print(f"[Resume] {args.resume}")
        global_step = load_checkpoint(args.resume, model, optimizer, train_ds, device, rank)
        if is_rank0(rank):
            print(f"[Resume] step={global_step}, "
                  f"dataset.line_counter={train_ds.state_dict().get('line_counter')}")

    # ── DataLoader ──
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_batch, persistent_workers=(args.num_workers > 0),
        pin_memory=(device.type == "cuda"),
    )

    # ── 학습 루프 ──
    log_path = os.path.join(args.out_dir, f"train_rank{rank}.log") if is_rank0(rank) else None
    log_file = open(log_path, "a") if log_path else None

    model.train()
    t0 = time.time()
    acc_total = 0.0
    acc_gen = 0.0
    acc_disc = 0.0
    acc_acc = 0.0
    acc_rep = 0.0
    acc_mask = 0.0
    acc_count = 0
    last_log_step = global_step

    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        # grad accumulation 루프
        for micro in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            jamo_mask = batch["jamo_mask"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            n_segments = batch["n_segments"].to(device, non_blocking=True)

            masked_patch_mask = make_patch_mask(
                n_segments, max_patches=args.max_patches, mask_ratio=args.mask_ratio
            )
            masked_jamo_ids, per_jamo_mask = apply_mask(
                jamo_ids, segment_ids, jamo_mask, masked_patch_mask
            )

            # DDP: gradient all_reduce는 backward에서 자동. micro 스텝에선 no_sync 사용 가능
            is_last_micro = (micro == args.grad_accum_steps - 1)
            sync_ctx = (model.no_sync() if (isinstance(model, DDP) and not is_last_micro)
                        else _nullctx())
            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(jamo_ids, jamo_mask, segment_ids, n_segments,
                                masked_jamo_ids, per_jamo_mask, masked_patch_mask)
                    loss = out["total_loss"] / args.grad_accum_steps
                loss.backward()

            acc_total += out["total_loss"].detach().float().item()
            acc_gen += out["gen_loss"].detach().float().item()
            acc_disc += out["disc_loss"].detach().float().item()
            acc_acc += out["disc_acc"].detach().float().item()
            acc_rep += out["replaced_rate"].detach().float().item()
            acc_mask += out["masked_tokens"].detach().float().item()
            acc_count += 1

        # Optimizer step
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                unwrap(model).parameters(), args.max_grad_norm
            )
        lr_main = linear_lr(global_step, args.warmup_steps, args.lr, args.max_steps, args.min_lr)
        lr_codec = lr_main * args.codec_lr_ratio
        apply_lr(optimizer, lr_main, lr_codec)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

        # ── 로깅 ──
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
                   f"lr_m {lr_main:.2e} | lr_c {lr_codec:.2e} | "
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
            acc_total = acc_gen = acc_disc = acc_acc = acc_rep = acc_mask = 0.0
            acc_count = 0

        # ── Validation ──
        if val_ds is not None and global_step % args.val_every == 0:
            # 매번 동일한 처음 N 배치를 평가 (val loss 비교 가능성 확보)
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

        # ── 체크포인트 ──
        if global_step % args.save_every == 0:
            # 모든 rank가 save_checkpoint 호출 (본체는 rank0, RNG sidecar는 각자)
            ckpt_path = os.path.join(args.out_dir, f"electra_step_{global_step}.pt")
            save_checkpoint(ckpt_path, global_step, model, optimizer, train_ds, args, rank)
            if is_rank0(rank):
                print(f"[Ckpt] saved → {ckpt_path} (+ rank RNG sidecars)", flush=True)
                if args.rclone_remote:
                    upload_and_cleanup(
                        ckpt_path=ckpt_path,
                        log_path=log_path,
                        remote_dest=args.rclone_remote,
                        keep_latest_n=args.keep_latest_n,
                    )

    # 종료: 모든 rank가 final 체크포인트 + RNG sidecar 저장
    final_path = os.path.join(args.out_dir, f"electra_step_{global_step}_final.pt")
    save_checkpoint(final_path, global_step, model, optimizer, train_ds, args, rank)
    if is_rank0(rank):
        print(f"[Done] final ckpt → {final_path}")
        if args.rclone_remote:
            upload_and_cleanup(final_path, log_path, args.rclone_remote, args.keep_latest_n)
        if log_file:
            log_file.close()

    cleanup_ddp(world_size)


# null context (Python 3.7+ contextlib.nullcontext 대체 — 호환성)
class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
