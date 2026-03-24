"""ELECTRA RTD 사전학습 스크립트

커스텀 Diamond 인코더(Discriminator) + 소형 Transformer(Generator)를 joint training.
DDP, BF16 AMP, torch.compile, AdamW, cosine/wsd LR schedule 지원.
체크포인트에 dataset state 포함 → resume 시 동일 에포크/위치에서 재개.
rclone 업로드 + 이전 체크포인트 자동 삭제.

Usage:
    # 단일 GPU
    python -m exp-electra-gec.phase2-keyboard-electra.pretrain_rtd \
        --preset small --corpus corpus/sample_full.jsonl --text_key text \
        --bf16 --batch_size 32 --lr 5e-4 --schedule wsd --max_steps 100000

    # DDP (4 GPU)
    torchrun --nproc_per_node=4 \
        -m exp-electra-gec.phase2-keyboard-electra.pretrain_rtd \
        --preset small --corpus corpus/sample_full.jsonl --text_key text \
        --bf16 --batch_size 32 --lr 5e-4 --schedule wsd --max_steps 100000
"""
import argparse
import gc
import io
import math
import os
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import ElectraConfig, make_small_config, make_128m_config
from electra_rtd import ElectraRTD
from rtd_dataset import RTDDataset
from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
from training.upload_gdrive import upload_and_cleanup


# ── 유틸 ──

def get_lr(
    step: int, warmup: int, max_lr: float, max_steps: int,
    min_lr_ratio: float = 0.01, schedule: str = "wsd",
) -> float:
    """LR 스케줄: cosine 또는 wsd"""
    min_lr = max_lr * min_lr_ratio
    if step < warmup:
        return min_lr + (max_lr - min_lr) * step / max(warmup, 1)

    if schedule == "wsd":
        remaining = max_steps - warmup
        stable_end = warmup + int(remaining * 0.8)
        if step < stable_end:
            return max_lr
        decay_progress = (step - stable_end) / max(max_steps - stable_end, 1)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
    else:
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def format_chars(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class GPUPrefetcher:
    """다음 배치를 별도 CUDA stream에서 미리 GPU로 전송"""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._prefetch()
        while self._next is not None:
            batch = self._next
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self._prefetch()
            yield batch

    def _prefetch(self):
        try:
            batch = next(self._iter)
            with torch.cuda.stream(self.stream):
                self._next = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
        except StopIteration:
            self._next = None


@torch.no_grad()
def validate_rtd(model, val_loader, device, n_batches=50, use_amp=False):
    """검증: disc_loss, rtd_acc, gen_loss 측정

    model은 raw_model(DDP unwrap 전)을 전달받아야 함.
    """
    model.eval()
    raw = model

    total_disc_loss = 0.0
    total_gen_loss = 0.0
    total_rtd_acc = 0.0
    total_gen_acc = 0.0
    total_replaced = 0.0
    count = 0

    val_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = raw(input_ids, pad_mask)

        total_disc_loss += outputs["disc_loss"].item()
        total_gen_loss += outputs["gen_loss"].item()
        total_rtd_acc += outputs["rtd_acc"].item()
        total_gen_acc += outputs["gen_acc"].item()
        total_replaced += outputs["replaced_ratio"].item()
        count += 1

    model.train()
    if count == 0:
        return {}

    return {
        "disc_loss": total_disc_loss / count,
        "gen_loss": total_gen_loss / count,
        "rtd_acc": total_rtd_acc / count,
        "gen_acc": total_gen_acc / count,
        "replaced_ratio": total_replaced / count,
    }


def main():
    parser = argparse.ArgumentParser(description="ELECTRA RTD Pretrain")
    # 모델
    parser.add_argument("--preset", choices=["small", "128m"], default="small")
    # 데이터
    parser.add_argument("--corpus", required=True, nargs="+")
    parser.add_argument("--text_key", default=None)
    parser.add_argument("--val_corpus", default=None, nargs="*")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    # 학습
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--schedule", choices=["cosine", "wsd"], default="wsd")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    # ELECTRA
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--disc_loss_weight", type=float, default=50.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    # 저장/로깅
    parser.add_argument("--save_dir", default="checkpoints/electra_rtd")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--log_file", default=None)
    parser.add_argument("--gdrive_remote", default=None,
                        help="rclone 원격지 (예: gdrive:electra-ckpts/)")
    parser.add_argument("--resume", default=None)
    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    # ── DDP 초기화 ──
    ddp = int(os.environ.get("RANK", -1)) >= 0
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)

    # ── Config ──
    if args.preset == "small":
        cfg = make_small_config(
            mask_prob=args.mask_prob,
            disc_loss_weight=args.disc_loss_weight,
            temperature=args.temperature,
        )
    else:
        cfg = make_128m_config(
            mask_prob=args.mask_prob,
            disc_loss_weight=args.disc_loss_weight,
            temperature=args.temperature,
        )
    cfg.disc.max_seq_len = args.max_seq_len
    cfg.gen.max_seq_len = args.max_seq_len

    # ── 모델 ──
    model = ElectraRTD(cfg).to(device)
    raw_model = model

    if args.gradient_checkpointing:
        raw_model.discriminator.gradient_checkpointing = True

    total_params = sum(p.numel() for p in model.parameters())
    gen_params = model.generator.count_parameters()
    disc_params = sum(p.numel() for p in model.discriminator.parameters())

    if is_main:
        print(f"=== ELECTRA RTD Pretrain ({args.preset}) ===")
        print(f"Generator: {gen_params:,} ({gen_params/1e6:.2f}M)")
        print(f"Discriminator: {disc_params:,} ({disc_params/1e6:.2f}M)")
        print(f"총: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"layer_spec: {cfg.disc.layer_spec}")
        print(f"mask_prob={cfg.mask_prob}, disc_weight={cfg.disc_loss_weight}, "
              f"temp={cfg.temperature}")
        print(f"LR={args.lr}, schedule={args.schedule}, warmup={args.warmup_steps}")
        print(f"batch={args.batch_size}×{args.grad_accum_steps}, "
              f"max_steps={args.max_steps}")
        print()

    # DDP
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # torch.compile
    if args.compile:
        if is_main:
            print(f"torch.compile 적용 (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    # ── Optimizer ──
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98), weight_decay=args.weight_decay,
        fused=use_fused,
    )

    # ── 토크나이저 + 데이터셋 ──
    tokenizer = KeyboardTokenizer()

    train_ds = RTDDataset(
        args.corpus, tokenizer,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
        rank=rank, world_size=world_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if args.val_corpus:
        val_ds = RTDDataset(
            args.val_corpus, tokenizer,
            max_seq_len=args.max_seq_len,
            text_key=args.text_key,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, num_workers=2,
            drop_last=True, prefetch_factor=4, persistent_workers=True,
        )

    # ── Resume ──
    start_step = 0
    _current_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        _current_epoch = ckpt.get("current_epoch", 0)

        # 데이터셋 state 복원 (동일 에포크/위치에서 재개)
        data_state = ckpt.get("dataset_state")
        if data_state is not None:
            train_ds.load_state_dict(data_state)
            if is_main:
                print(f"체크포인트 복원: step {start_step}, epoch {_current_epoch} "
                      f"(dataset state 포함, line={data_state.get('line_counter', '?')})")
        else:
            if is_main:
                print(f"체크포인트 복원: step {start_step} (dataset state 없음)")

        del ckpt
        gc.collect()

    # ── 학습 루프 ──
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    os.makedirs(args.save_dir, exist_ok=True)

    # 로그 파일
    log_file = args.log_file
    if log_file and is_main:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        if log_file and is_main:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    if is_main:
        log(f"학습 시작: step {start_step} → {args.max_steps}")

    model.train()
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 비동기 체크포인트 저장
    _save_thread: threading.Thread | None = None
    _prev_ckpt_path: str | None = None

    step = start_step
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    accum_metrics = {"gen_loss": 0.0, "disc_loss": 0.0, "rtd_acc": 0.0,
                     "gen_acc": 0.0, "replaced_ratio": 0.0}
    accum_count = 0
    _max_line_counter = 0
    log_tokens = 0

    use_prefetch = torch.cuda.is_available()

    def make_data_iter():
        if use_prefetch:
            return iter(GPUPrefetcher(train_loader, device))
        return iter(train_loader)

    data_iter = make_data_iter()

    for step in range(start_step, args.max_steps):
        # LR 스케줄
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps,
                     args.min_lr_ratio, args.schedule)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                _current_epoch += 1
                if is_main:
                    log(f"\n에포크 {_current_epoch} 완료 (step {step})")
                # 에포크 전환: dataset 위치 리셋 → 처음부터 다시 읽기
                train_ds._line_counter = 0
                train_ds._resume_line = 0
                data_iter = make_data_iter()
                batch = next(data_iter)

            if use_prefetch:
                # GPUPrefetcher가 이미 device로 전송
                input_ids = batch["input_ids"]
                pad_mask = batch["pad_mask"]
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                pad_mask = batch["pad_mask"].to(device, non_blocking=True)

            # DDP no_sync
            is_last_accum = (accum_step == args.grad_accum_steps - 1)
            ctx = model.no_sync() if (ddp and not is_last_accum) else nullcontext()

            with ctx:
                with amp_ctx:
                    outputs = raw_model(input_ids, pad_mask)
                    total_loss = raw_model.get_total_loss(outputs)
                    loss_scaled = total_loss / args.grad_accum_steps

                loss_scaled.backward()

            for k in accum_metrics:
                accum_metrics[k] += outputs[k].item()
            accum_count += 1
            log_tokens += pad_mask.sum().item()
            if "_line_counter" in batch:
                lc = batch["_line_counter"]
                lc_val = lc.max().item() if isinstance(lc, torch.Tensor) else max(lc) if isinstance(lc, (list, tuple)) else lc
                _max_line_counter = max(_max_line_counter, int(lc_val))

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if is_main and (step + 1) % args.log_every == 0:
            n = accum_count
            elapsed = time.time() - t0
            avg = {k: v / n for k, v in accum_metrics.items()}
            tok_s = log_tokens / max(elapsed, 1e-6)
            mem_str = ""
            if step + 1 == args.log_every and torch.cuda.is_available():
                alloc = torch.cuda.max_memory_allocated() / 1024**3
                mem_str = f" | mem {alloc:.1f}G"
            log(
                f"[{step+1:>6d}/{args.max_steps}] "
                f"d_loss={avg['disc_loss']:.4f} g_loss={avg['gen_loss']:.4f} "
                f"rtd={avg['rtd_acc']:.3f} g_acc={avg['gen_acc']:.3f} "
                f"repl={avg['replaced_ratio']:.3f} "
                f"lr={lr:.2e} ep={_current_epoch} line={_max_line_counter} "
                f"{tok_s:,.0f} tok/s {fmt_time(elapsed)}{mem_str}"
            )
            accum_metrics = {k: 0.0 for k in accum_metrics}
            accum_count = 0
            log_tokens = 0
            t0 = time.time()

        # Validation
        if val_loader is not None and (step + 1) % args.val_every == 0 and is_main:
            val_metrics = validate_rtd(raw_model, val_loader, device, use_amp=use_amp)
            if val_metrics:
                log(
                    f"  [VAL] d_loss={val_metrics['disc_loss']:.4f} "
                    f"g_loss={val_metrics['gen_loss']:.4f} "
                    f"rtd={val_metrics['rtd_acc']:.3f} "
                    f"g_acc={val_metrics['gen_acc']:.3f} "
                    f"repl={val_metrics['replaced_ratio']:.3f}"
                )
            raw_model.train()
            # eval→train 전환 후 Int8Linear 양자화 캐시 무효화
            # (eval에서 .round(), train에서 _ste_round 사용 — 캐시 불일치 방지)
            from model.bitlinear import Int8Linear as _I8L
            for m in raw_model.modules():
                if isinstance(m, _I8L):
                    m._weight_version = -1

        # 체크포인트 저장
        if (step + 1) % args.save_every == 0 and is_main:
            # worker→main _line_counter 동기화
            train_ds._line_counter = _max_line_counter

            ckpt_path = os.path.join(
                args.save_dir,
                f"electra_{args.preset}_step_{step+1}.pt",
            )

            # 이전 저장 스레드 완료 대기
            if _save_thread is not None:
                _save_thread.join()

            # 메모리 버퍼에 직렬화
            ckpt_buf = io.BytesIO()
            torch.save({
                "step": step + 1,
                "current_epoch": _current_epoch,
                "config": asdict(cfg),
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dataset_state": train_ds.state_dict(),
            }, ckpt_buf)

            # 백그라운드 스레드: 디스크 기록 + 이전 삭제 + 업로드
            _prev = _prev_ckpt_path
            _gdrive = args.gdrive_remote
            _logf = args.log_file

            def _save_task(buf, path, prev, gdrive, logf):
                buf.seek(0)
                with open(path, "wb") as f:
                    f.write(buf.getvalue())
                print(f"  체크포인트 저장 완료: {path}", flush=True)
                if prev and os.path.exists(prev):
                    try:
                        os.remove(prev)
                        print(f"  이전 체크포인트 삭제: {prev}", flush=True)
                    except OSError:
                        pass
                if gdrive:
                    upload_and_cleanup(path, logf, gdrive, keep_latest_n=1)

            _save_thread = threading.Thread(
                target=_save_task,
                args=(ckpt_buf, ckpt_path, _prev, _gdrive, _logf),
                daemon=True,
            )
            _save_thread.start()
            _prev_ckpt_path = ckpt_path

    # ── 최종 저장 ──
    if _save_thread is not None:
        _save_thread.join()

    final_step = step + 1
    if is_main:
        final_path = os.path.join(args.save_dir, f"electra_{args.preset}_final.pt")
        torch.save({
            "step": final_step,
            "current_epoch": _current_epoch,
            "config": asdict(cfg),
            "model_state_dict": raw_model.state_dict(),
        }, final_path)
        log(f"\n학습 완료! 최종 체크포인트: {final_path} (step {final_step})")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
