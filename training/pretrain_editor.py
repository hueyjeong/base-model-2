"""BitEditor 사전학습 스크립트

Usage:
    python -m training.pretrain_editor \
        --size 128M \
        --corpus corpus/sample_10g.jsonl \
        --text_key text \
        --bf16 \
        --max_steps 100000

    # DDP (multi-GPU)
    torchrun --nproc_per_node=2 -m training.pretrain_editor \
        --size 128M --corpus corpus/sample_10g.jsonl --text_key text --bf16
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.editor_config import BitEditorConfig
from model.editor import BitEditor
from model.edit_tags import compute_edit_tags, apply_edit_tags, TAG_KEEP
from training.noising import DenoisingNoiser, NoiseConfig
from training.editor_dataset import EditorDataset

# ── 토크나이저 프리셋 (pretrain.py 재사용) ──

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

TOKENIZER_PRESETS = {
    "keyboard": {
        "module": "keyboard_tokenizer.keyboard_wrapper",
        "class": "KeyboardTokenizer",
        "default_path": os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"),
    },
    "char": {
        "module": "char_tokenizer.char_wrapper",
        "class": "CharTokenizer",
        "default_path": os.path.join(PROJECT_ROOT, "char_tokenizer", "char_level_tokenizer.json"),
    },
}


def load_tokenizer(name: str, path: str | None = None):
    if name not in TOKENIZER_PRESETS:
        raise ValueError(f"지원하지 않는 토크나이저: {name}\n  사용 가능: {list(TOKENIZER_PRESETS.keys())}")
    preset = TOKENIZER_PRESETS[name]
    import importlib
    mod = importlib.import_module(preset["module"])
    cls = getattr(mod, preset["class"])
    return cls(path or preset["default_path"])


# ── 모델 사이즈 프리셋 ──

EDITOR_CONFIGS = {
    "8M": dict(
        d_model=192, n_rwkv_layers=6,
        d_inner=192, n_heads=6, headdim=32,
        d_ff=256, n_experts=8, top_k=1,
        n_attn_heads=12, attn_insertion_points=(2, 4),
        lora_rank=8,
    ),
    "128M": dict(
        d_model=384, n_rwkv_layers=10,
        d_inner=384, n_heads=12, headdim=32,
        d_ff=512, n_experts=16, top_k=1,
        n_attn_heads=24, attn_insertion_points=(3, 7, 9),
        lora_rank=16,
    ),
}


def get_lr(step: int, warmup: int, max_lr: float, max_steps: int) -> float:
    """Linear warmup + cosine decay"""
    min_lr = max_lr * 0.01
    if step < warmup:
        return min_lr + (max_lr - min_lr) * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def train(args):
    # DDP
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if global_rank == 0:
        print(f"Device: {device} | World Size: {world_size}")

    # 토크나이저
    tokenizer = load_tokenizer(args.tokenizer)
    if global_rank == 0:
        print(f"토크나이저: {args.tokenizer}, vocab_size={tokenizer.vocab_size}")

    # 모델 설정
    if args.size not in EDITOR_CONFIGS:
        print(f"지원하지 않는 사이즈: {args.size}, 사용 가능: {list(EDITOR_CONFIGS.keys())}")
        return 1

    model_kwargs = dict(EDITOR_CONFIGS[args.size])
    model_kwargs["vocab_size"] = tokenizer.vocab_size
    model_kwargs["n_tags"] = 2 + 2 * tokenizer.vocab_size
    model_kwargs["max_seq_len"] = args.max_seq_len
    model_kwargs["pad_id"] = tokenizer.pad_id
    model_kwargs["bos_id"] = tokenizer.bos_id
    model_kwargs["n_iterations"] = args.n_iterations

    config = BitEditorConfig(**model_kwargs)

    if global_rank == 0:
        print(f"\n모델 설정: BitEditor {args.size}")
        print(f"  d_model={config.d_model}, n_rwkv_layers={config.n_rwkv_layers}, "
              f"n_experts={config.n_experts}, top_k={config.top_k}")
        print(f"  n_tags={config.n_tags}, n_iterations={config.n_iterations}")

    # 모델 생성
    model = BitEditor(config).to(device)
    params = model.count_parameters()
    active = model.estimate_active_params()

    if global_rank == 0:
        print(f"  총 파라미터: {format_params(params['total'])}")
        print(f"  활성 파라미터: {format_params(active)}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_distributed else model

    # Gradient checkpointing
    if args.grad_ckpt:
        raw_model.gradient_checkpointing = True

    # 노이즈 설정 (토큰 레벨 비활성화)
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0,
        token_delete_ratio=0.0,
        text_infill_ratio=0.0,
    )
    noiser = DenoisingNoiser(
        tokenizer, noise_cfg,
        seed=args.seed + global_rank,
        use_korean_errors=True,
    )

    # 데이터셋
    dataset = EditorDataset(
        args.corpus, tokenizer, noiser,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
        lang_key=args.lang_key,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if global_rank == 0:
        print(f"\n데이터셋: 스트리밍 (max_seq_len={args.max_seq_len})")

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01,
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # AMP
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    scaler = None  # BF16은 scaler 불필요

    # 체크포인트 복원
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if global_rank == 0:
            print(f"\n체크포인트 복원: step {start_step}")
        del ckpt
        gc.collect()

    # 학습 루프
    if global_rank == 0:
        print(f"\n학습 시작: step {start_step} → {args.max_steps}")
        print(f"  batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}")
        print(f"  lr={args.lr}, warmup={args.warmup_steps}")
        print(f"  n_iterations={config.n_iterations}")
        print()

    model.train()
    data_iter = iter(loader)
    log_interval = args.log_interval
    save_interval = args.save_interval

    running_loss = 0.0
    running_aux = 0.0
    running_tokens = 0
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # LR 스케줄
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = torch.tensor(0.0, device=device)

        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            edit_tags = batch["edit_tags"].to(device)
            pad_mask = batch["pad_mask"].to(device)
            original_ids = batch["original_ids"].to(device)

            # Iterative refinement 학습
            current_ids = input_ids
            iter_loss = torch.tensor(0.0, device=device)

            for it in range(config.n_iterations):
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        tag_logits, aux_loss = model(current_ids, pad_mask)
                else:
                    tag_logits, aux_loss = model(current_ids, pad_mask)

                # 현재 iteration의 태그에 대한 CE loss
                # PAD 위치는 -100으로 ignore
                targets = edit_tags.clone()
                targets[~pad_mask] = -100

                ce_loss = criterion(
                    tag_logits.view(-1, config.n_tags),
                    targets.view(-1),
                )

                loss = (ce_loss + aux_loss) / (config.n_iterations * args.grad_accum_steps)
                loss.backward()

                iter_loss = iter_loss + ce_loss.detach()

                # 다음 iteration 준비: 예측 태그 적용 → 새 편집 태그 계산
                if it < config.n_iterations - 1:
                    with torch.no_grad():
                        pred_tags = tag_logits.argmax(dim=-1)  # (B, T)
                        B, T = current_ids.shape

                        # 배치별로 태그 적용
                        new_ids_list = []
                        new_tags_list = []
                        for b in range(B):
                            valid = pad_mask[b]
                            src = current_ids[b][valid].tolist()
                            tags_b = pred_tags[b][valid].tolist()

                            # 태그 적용
                            modified = apply_edit_tags(src, tags_b, config.vocab_size)
                            # truncate/pad to max_seq_len
                            modified = modified[:config.max_seq_len]
                            pad_len = config.max_seq_len - len(modified)
                            modified_padded = modified + [config.pad_id] * pad_len

                            # 새 편집 태그 계산
                            orig = original_ids[b][original_ids[b] != config.pad_id].tolist()
                            new_tags = compute_edit_tags(modified, orig, config.vocab_size)
                            new_tags = new_tags + [TAG_KEEP] * pad_len

                            new_ids_list.append(modified_padded)
                            new_tags_list.append(new_tags)

                        current_ids = torch.tensor(new_ids_list, dtype=torch.long, device=device)
                        edit_tags = torch.tensor(new_tags_list, dtype=torch.long, device=device)
                        # pad_mask 업데이트
                        pad_mask = (current_ids != config.pad_id)

            total_loss = total_loss + iter_loss / config.n_iterations
            n_tok = batch["pad_mask"].sum().item()
            running_tokens += n_tok

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += total_loss.item() / args.grad_accum_steps
        running_aux += aux_loss.item()

        # 로깅
        if (step + 1) % log_interval == 0 and global_rank == 0:
            dt = time.time() - t0
            avg_loss = running_loss / log_interval
            avg_aux = running_aux / log_interval
            tok_s = running_tokens / max(dt, 1e-6)
            print(f"step {step + 1:>6d} | loss {avg_loss:.4f} | aux {avg_aux:.4f} | "
                  f"lr {lr:.2e} | {tok_s:.0f} tok/s")
            running_loss = 0.0
            running_aux = 0.0
            running_tokens = 0
            t0 = time.time()

        # 체크포인트
        if (step + 1) % save_interval == 0 and global_rank == 0:
            ckpt_path = os.path.join(
                args.save_dir,
                f"editor_{args.size}_step{step + 1}.pt"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "step": step + 1,
                "config": asdict(config),
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  체크포인트 저장: {ckpt_path}")

    if global_rank == 0:
        print("\n학습 완료!")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="BitEditor 사전학습")

    # 모델
    parser.add_argument("--size", type=str, default="128M",
                        choices=list(EDITOR_CONFIGS.keys()))
    parser.add_argument("--tokenizer", type=str, default="keyboard",
                        choices=list(TOKENIZER_PRESETS.keys()))
    parser.add_argument("--n_iterations", type=int, default=3,
                        help="Iterative refinement 반복 횟수")

    # 데이터
    parser.add_argument("--corpus", type=str, nargs="+", required=True)
    parser.add_argument("--text_key", type=str, default=None)
    parser.add_argument("--lang_key", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 학습
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # 로깅/저장
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
