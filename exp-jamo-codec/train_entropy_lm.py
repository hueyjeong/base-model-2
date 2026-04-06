"""SmallLM 단독 NTP 학습 — 엔트로피 모델 스케일링 실험

SmallLM (RMSNorm + SwiGLU causal Transformer)을 next-token prediction으로 학습.
크기별 perplexity/BPB를 비교하여 EntropyPatchCodec에 탑재할 적정 크기 탐색.
DDP 지원 (torchrun --nproc_per_node=N).
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

from codec.entropy_codec import SmallLM
from train_codec import CodecDataset, load_tokenizer, _unwrap_state_dict


def train(args):
    # DDP 초기화
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
        print(f"Device: {device}" + (f" (DDP {world_size} GPUs)" if is_distributed else ""))

    # 토크나이저
    tokenizer = load_tokenizer(args.tokenizer)
    if rank == 0:
        print(f"토크나이저: {args.tokenizer} (vocab={tokenizer.vocab_size})")

    # 모델
    model = SmallLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.entropy_d_model,
        n_layers=args.entropy_n_layers,
        n_heads=args.entropy_n_heads,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"SmallLM: d={args.entropy_d_model}, L={args.entropy_n_layers}, "
              f"H={args.entropy_n_heads}, params={n_params/1e6:.2f}M")

    # torch.compile
    if args.compile:
        if rank == 0:
            print("torch.compile 적용 중...")
        model = torch.compile(model)
        if rank == 0:
            print("torch.compile 완료")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # 데이터
    dataset = CodecDataset(
        file_paths=args.corpus,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR schedule: warmup → cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # BF16
    use_amp = args.bf16 and device.type == "cuda"

    # 학습
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_tokens = 0
    t_start = time.time()

    grad_accum = args.grad_accum_steps
    if rank == 0:
        eff_batch = args.batch_size * grad_accum
        print(f"\n학습 시작: max_steps={args.max_steps}, batch={args.batch_size}"
              + (f"×{grad_accum}={eff_batch}" if grad_accum > 1 else "")
              + f", seq_len={args.max_seq_len}"
              + (f", DDP {world_size} GPUs" if is_distributed else ""))
        print(f"{'step':>8} {'loss':>8} {'ppl':>8} {'bpb':>7} {'lr':>10} {'tok/s':>8}")
        print("-" * 58)

    micro_step = 0
    for batch in loader:
        if global_step >= args.max_steps:
            break

        ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids)  # [B, L, V]
            # NTP: logits[t] predicts ids[t+1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = ids[:, 1:].contiguous()
            shift_mask = pad_mask[:, 1:]  # target 위치의 유효성

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=0,  # PAD
                reduction="none",
            )
            # 유효 토큰만 loss 계산
            valid = shift_mask.reshape(-1) & (shift_targets.reshape(-1) != 0)
            loss = loss[valid].mean() / grad_accum

        loss.backward()

        with torch.no_grad():
            n_valid = valid.sum().item()
            accum_loss += loss.item() * grad_accum
            accum_tokens += n_valid

        micro_step += 1
        if micro_step % grad_accum != 0:
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # 로깅
        if global_step % args.log_every == 0 and rank == 0:
            dt = time.time() - t_start
            avg_loss = accum_loss / args.log_every
            ppl = math.exp(min(avg_loss, 20))  # overflow 방지
            bpb = avg_loss / math.log(2)
            tok_s = accum_tokens / max(dt, 1e-6)
            if is_distributed:
                tok_s *= world_size
            lr = scheduler.get_last_lr()[0]

            print(f"{global_step:8d} {avg_loss:8.4f} {ppl:8.2f} {bpb:7.4f} {lr:10.2e} {tok_s:8.0f}")

            accum_loss = 0.0
            accum_tokens = 0
            t_start = time.time()

        # 체크포인트
        if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
            model_sd = _unwrap_state_dict(model)
            tag = f"entropy_lm_{args.entropy_d_model}d_{args.entropy_n_layers}L"
            save_path = os.path.join(args.out_dir, f"{tag}_step{global_step}.pt")
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "model": model_sd,
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "args": vars(args),
            }, save_path)
            print(f"  → 체크포인트 저장: {save_path}")

    if rank == 0:
        print(f"\n학습 완료: {global_step} steps")

    # 최종 저장
    if args.out_dir and rank == 0:
        model_sd = _unwrap_state_dict(model)
        tag = f"entropy_lm_{args.entropy_d_model}d_{args.entropy_n_layers}L"
        save_path = os.path.join(args.out_dir, f"{tag}_final.pt")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model": model_sd,
            "step": global_step,
            "args": vars(args),
        }, save_path)
        print(f"최종 저장: {save_path}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="SmallLM (엔트로피 모델) NTP 학습")

    # 데이터
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="byte")
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 모델
    parser.add_argument("--entropy_d_model", type=int, default=128)
    parser.add_argument("--entropy_n_layers", type=int, default=2)
    parser.add_argument("--entropy_n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 학습
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # 로깅/저장
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--out_dir", default="exp-jamo-codec/checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
