"""SimpleCodec 학습 — per-token 포맷.

설계: 각 BBPE 토큰을 독립적으로 encode/decode.
배치 = T 개 토큰 stack ([T, max_jamo]). GPU 가 모든 토큰 병렬 처리.
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.simple_codec import SimpleCodec
from data.simple_dataset import SimpleJamoDataset, load_bbpe_tokenizer
from tok.jamo_tokenizer import JamoTokenizer


def _unwrap_state_dict(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    sd = m.state_dict()
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


@torch.no_grad()
def validate(codec, val_loader, device, max_samples=10000, world_size=1):
    codec.eval()
    total_correct = 0
    total_slots = 0
    total_correct_valid = 0
    total_valid = 0
    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    per_rank_max = max(1, max_samples // max(world_size, 1))

    for batch in val_loader:
        jamo_ids = batch["jamo_ids"].to(device)
        mask = batch["mask"].to(device)

        out = codec(jamo_ids, mask)
        pred = out["logits"].argmax(dim=-1)

        # 전 슬롯 정확도 (PAD 포함)
        target_all = jamo_ids.clone()
        target_all[~mask] = 0
        total_correct += (pred == target_all).sum().item()
        total_slots += target_all.numel()

        # 유효 슬롯 정확도 (실자모만)
        total_correct_valid += ((pred == jamo_ids) & mask).sum().item()
        total_valid += mask.sum().item()

        total_loss += out["loss"].item()
        n_batches += 1
        n_samples += jamo_ids.size(0)

        if n_samples >= per_rank_max:
            break

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([total_correct, total_slots, total_correct_valid,
                              total_valid, total_loss, n_batches, n_samples],
                             dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_correct = int(stats[0].item())
        total_slots = int(stats[1].item())
        total_correct_valid = int(stats[2].item())
        total_valid = int(stats[3].item())
        total_loss = float(stats[4].item())
        n_batches = int(stats[5].item())
        n_samples = int(stats[6].item())

    codec.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_acc_all": total_correct / max(total_slots, 1) * 100,
        "val_acc_valid": total_correct_valid / max(total_valid, 1) * 100,
        "val_samples": n_samples,
    }


def train(args):
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device: {device}" + (f" (DDP {world_size})" if is_distributed else ""))

    # 토크나이저
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    if rank == 0:
        print(f"BBPE vocab: {bbpe.vocab_size:,}, Jamo vocab: {jamo.vocab_size}")

    # 모델
    codec = SimpleCodec(
        jamo_vocab=jamo.vocab_size,
        d_model=args.d_model,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        kernel_size=args.kernel_size,
        max_jamo=args.max_jamo,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in codec.parameters())
    if rank == 0:
        print(f"SimpleCodec: d={args.d_model}, enc_L={args.n_enc_layers}, "
              f"dec_L={args.n_dec_layers}, k={args.kernel_size}, "
              f"max_jamo={args.max_jamo}, params={n_params/1e6:.2f}M")

    if args.compile:
        if rank == 0:
            print("torch.compile...")
        codec = torch.compile(codec)

    if is_distributed:
        codec = DDP(codec, device_ids=[rank])

    # 데이터셋
    dataset = SimpleJamoDataset(
        file_paths=args.corpus,
        bbpe_tokenizer=bbpe,
        jamo_tokenizer=jamo,
        max_jamo=args.max_jamo,
        text_key=args.text_key,
        rank=rank,
        world_size=world_size,
    )

    def _worker_init(worker_id):
        w_info = torch.utils.data.get_worker_info()
        if w_info is None:
            return
        ds = w_info.dataset
        ds._prewarm_cache(verbose=(worker_id == 0))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_worker_init if args.num_workers > 0 else None,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    else:
        dataset._prewarm_cache(verbose=(rank == 0))
    loader = DataLoader(dataset, **loader_kwargs)

    # Validation loader
    val_loader = None
    if args.val_corpus:
        val_dataset = SimpleJamoDataset(
            file_paths=args.val_corpus,
            bbpe_tokenizer=bbpe,
            jamo_tokenizer=jamo,
            max_jamo=args.max_jamo,
            text_key=args.text_key,
            rank=rank,
            world_size=world_size,
        )
        val_dataset._prewarm_cache(verbose=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # Optimizer + scheduler (warmup + cosine)
    optimizer = torch.optim.AdamW(
        codec.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0

    # Resume / init_from
    if args.init_from:
        if rank == 0:
            print(f"Init from: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ckpt["model"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        raw = codec.module if hasattr(codec, "module") else codec
        if hasattr(raw, "_orig_mod"):
            missing, unexpected = raw._orig_mod.load_state_dict(sd, strict=False)
        else:
            missing, unexpected = raw.load_state_dict(sd, strict=False)
        if rank == 0 and missing:
            print(f"  missing: {len(missing)} (새로 초기화)")
        if rank == 0 and unexpected:
            print(f"  unexpected: {len(unexpected)} (무시)")

    if args.resume:
        if rank == 0:
            print(f"Resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"]
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        raw = codec.module if hasattr(codec, "module") else codec
        if hasattr(raw, "_orig_mod"):
            raw._orig_mod.load_state_dict(sd, strict=False)
        else:
            raw.load_state_dict(sd, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("step", 0)
        if args.num_workers == 0:
            data_state = ckpt.get("data_state")
            if isinstance(data_state, dict):
                dataset.load_state_dict(data_state)

    # 학습 루프
    os.makedirs(args.out_dir, exist_ok=True)
    use_amp = args.bf16 and device.type == "cuda"
    grad_accum = args.grad_accum_steps
    accum_loss = 0.0
    accum_correct_all = 0
    accum_slots_all = 0
    accum_correct_valid = 0
    accum_valid = 0
    micro_step = 0
    t_start = time.time()

    if rank == 0:
        print(f"\n학습 시작: max_steps={args.max_steps}, batch={args.batch_size}×accum{grad_accum} (토큰 수)")
        print(f"    step            loss           acc_valid           acc_all         lr    tok/s")

    for batch in loader:
        if global_step >= args.max_steps:
            break
        jamo_ids = batch["jamo_ids"].to(device)
        mask = batch["mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = codec(jamo_ids, mask)
            loss = out["loss"] / grad_accum

        loss.backward()

        # 통계
        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            target_all = jamo_ids.clone()
            target_all[~mask] = 0
            accum_correct_all += (pred == target_all).sum().item()
            accum_slots_all += target_all.numel()
            accum_correct_valid += ((pred == jamo_ids) & mask).sum().item()
            accum_valid += mask.sum().item()

        accum_loss += loss.item() * grad_accum
        micro_step += 1

        if micro_step % grad_accum != 0:
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.log_every == 0 and rank == 0:
            dt = time.time() - t_start
            avg_loss = accum_loss / args.log_every
            acc_all = accum_correct_all / max(accum_slots_all, 1) * 100
            acc_valid = accum_correct_valid / max(accum_valid, 1) * 100
            tok_s = accum_slots_all / max(dt, 1e-6)
            if is_distributed:
                tok_s *= world_size
            lr = scheduler.get_last_lr()[0]
            progress = global_step / args.max_steps * 100
            print(f"{global_step:8d} {avg_loss:15.12f} {acc_valid:16.12f}% {acc_all:16.12f}% "
                  f"{lr:10.2e} {tok_s:8.0f}  {progress:.1f}%")
            accum_loss = 0.0
            accum_correct_all = accum_slots_all = 0
            accum_correct_valid = accum_valid = 0
            t_start = time.time()

        # Validation
        if val_loader is not None and args.val_every > 0 and global_step % args.val_every == 0:
            val_m = validate(codec, val_loader, device, args.val_samples, world_size)
            if rank == 0:
                print(f"  [VAL] loss={val_m['val_loss']:.12f}, "
                      f"acc_valid={val_m['val_acc_valid']:.12f}%, "
                      f"acc_all={val_m['val_acc_all']:.12f}%, "
                      f"samples={val_m['val_samples']}")

        # Save + GDrive 업로드
        if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
            path = os.path.join(args.out_dir, f"simple_codec_step{global_step}.pt")
            ckpt = {
                "model": _unwrap_state_dict(codec),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "args": vars(args),
            }
            torch.save(ckpt, path)
            print(f"  → 체크포인트 저장: {path}")
            # GDrive 업로드 (체크포인트 + 로그, 백그라운드 + 이전 ckpt 삭제)
            gdrive = os.environ.get("GDRIVE")
            if gdrive:
                from training.upload_gdrive import upload_and_cleanup
                log_path = os.environ.get("LOG_PATH",
                    os.path.join(os.path.dirname(args.out_dir), "simple_train_log.txt"))
                upload_and_cleanup(path, log_path, gdrive)
                print(f"  → GDrive 업로드 시작 (백그라운드): {gdrive}")

    # 최종
    if rank == 0:
        path = os.path.join(args.out_dir, "simple_codec_final.pt")
        ckpt = {
            "model": _unwrap_state_dict(codec),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step,
            "args": vars(args),
        }
        torch.save(ckpt, path)
        print(f"\n학습 완료: {global_step} steps\n최종 저장: {path}")
        # 최종 GDrive 업로드 (동기)
        gdrive = os.environ.get("GDRIVE")
        if gdrive:
            import subprocess, shlex
            log_path = os.environ.get("LOG_PATH",
                os.path.join(os.path.dirname(args.out_dir), "simple_train_log.txt"))
            print(f"  → GDrive 최종 업로드 중... ({gdrive})")
            try:
                subprocess.run(
                    f"rclone copy {shlex.quote(path)} {shlex.quote(gdrive)}",
                    shell=True, check=True,
                )
                if os.path.exists(log_path):
                    subprocess.run(
                        f"rclone copy {shlex.quote(log_path)} {shlex.quote(gdrive)}",
                        shell=True, check=True,
                    )
                print(f"  → GDrive 업로드 완료")
            except Exception as e:
                print(f"  → GDrive 업로드 실패: {e}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    # 데이터
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--text_key", default="text")
    p.add_argument("--val_corpus", nargs="+", default=None)

    # 모델
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_enc_layers", type=int, default=5)
    p.add_argument("--n_dec_layers", type=int, default=5)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--max_jamo", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)

    # 학습
    p.add_argument("--batch_size", type=int, default=512, help="배치당 토큰 수")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")

    # 인프라
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--no_pin_memory", action="store_true")

    # 로깅/저장/재개
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_samples", type=int, default=10000)
    p.add_argument("--out_dir", default="exp-jamo-codec/checkpoints_simple")
    p.add_argument("--resume", default=None)
    p.add_argument("--init_from", default=None)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
