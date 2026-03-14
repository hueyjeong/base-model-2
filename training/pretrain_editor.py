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
from contextlib import nullcontext
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
from training.upload_gdrive import upload_and_cleanup

# C++ Levenshtein 확장 (editor_dataset에서 JIT 빌드한 것 재사용)
_LEVENSHTEIN_CPP = None
try:
    from training.editor_dataset import _lev_ext as _LEVENSHTEIN_CPP
except (ImportError, AttributeError):
    pass

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


def get_lr(
    step: int, warmup: int, max_lr: float, max_steps: int,
    min_lr_ratio: float = 0.1, schedule: str = "cosine",
) -> float:
    """학습률 스케줄러

    Args:
        schedule: "cosine" (warmup + cosine decay) 또는
                  "wsd" (Warmup-Stable-Decay: warmup → 80% stable → 20% decay)
        min_lr_ratio: 최소 LR = max_lr × min_lr_ratio
    """
    min_lr = max_lr * min_lr_ratio
    if step < warmup:
        return min_lr + (max_lr - min_lr) * step / max(warmup, 1)

    if schedule == "wsd":
        # WSD: warmup 이후 80%는 peak LR 유지, 마지막 20% decay
        remaining = max_steps - warmup
        stable_end = warmup + int(remaining * 0.8)
        if step < stable_end:
            return max_lr
        decay_progress = (step - stable_end) / max(max_steps - stable_end, 1)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
    else:
        # Cosine decay
        progress = (step - warmup) / max(max_steps - warmup, 1)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_chars(n: int) -> str:
    """문자 수를 읽기 좋게 포맷 (500M, 1.2B 등)"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def validate_editor(model, val_loader, criterion, config, device, use_amp, n_steps,
                     amp_dtype=torch.bfloat16):
    """검증 루프: n_steps 배치에 대해 loss, 태그 정확도, 편집 precision/recall 계산"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    edit_tp = 0  # non-KEEP 정답
    edit_fp = 0  # KEEP인데 non-KEEP으로 예측
    edit_fn = 0  # non-KEEP인데 KEEP으로 예측
    val_iter = iter(val_loader)

    with torch.no_grad():
        for _ in range(n_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)
            edit_tags = batch["edit_tags"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    tag_logits, aux_loss = model(input_ids, pad_mask)
            else:
                tag_logits, aux_loss = model(input_ids, pad_mask)

            targets = edit_tags.clone()
            targets[~pad_mask] = -100
            loss = criterion(
                tag_logits.view(-1, config.n_tags),
                targets.view(-1),
            )

            valid = pad_mask
            n_tok = valid.sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

            preds = tag_logits.argmax(dim=-1)
            total_correct += (preds[valid] == edit_tags[valid]).sum().item()

            # Precision/Recall: non-KEEP 태그 (TAG_KEEP=0)
            pred_edit = preds[valid] != TAG_KEEP
            true_edit = edit_tags[valid] != TAG_KEEP
            edit_tp += (pred_edit & true_edit).sum().item()
            edit_fp += (pred_edit & ~true_edit).sum().item()
            edit_fn += (~pred_edit & true_edit).sum().item()

    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    val_loss = total_loss / total_tokens
    tag_acc = total_correct / total_tokens
    edit_precision = edit_tp / max(edit_tp + edit_fp, 1)
    edit_recall = edit_tp / max(edit_tp + edit_fn, 1)
    return val_loss, tag_acc, edit_precision, edit_recall


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

    # INT8 텐서코어 BitLinear 교체
    if args.int8:
        if args.int8_backend == "cuda":
            try:
                from model.cuda_bitlinear import replace_bitlinear_with_cuda
                model = replace_bitlinear_with_cuda(model)
            except Exception as e:
                if global_rank == 0:
                    print(f"CUDA BitLinear 로드 실패: {e}, triton fallback")
                from model.triton_bitlinear import replace_bitlinear_with_triton
                model = replace_bitlinear_with_triton(model)
        else:
            from model.triton_bitlinear import replace_bitlinear_with_triton
            model = replace_bitlinear_with_triton(model)
        if global_rank == 0:
            print(f"  INT8 backend: {args.int8_backend}")

    # raw_model 참조 저장 (wrapping 전 — state_dict, grad_ckpt, validate에서 사용)
    raw_model = model

    # Gradient checkpointing
    if args.grad_ckpt:
        raw_model.gradient_checkpointing = True

    # torch.compile (DDP 전에 — compile은 로컬 계산만 최적화, DDP는 통신을 독립 관리)
    if args.compile:
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 256
        if global_rank == 0:
            print("torch.compile 적용 중... (첫 step 느림, 이후 빠름)")
        model = torch.compile(model)

    # DDP (compile 후에)
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,  # gradient 복사 제거 → 메모리 절약 + 속도
            static_graph=True,             # 통신/계산 오버랩 최적화 (MoE 모듈 set 고정)
        )

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

    # 데이터셋 — n_iterations > 1이면 패킹 비활성 (per-doc original_ids 필요)
    use_pack = (config.n_iterations <= 1)
    dataset = EditorDataset(
        args.corpus, tokenizer, noiser,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
        lang_key=args.lang_key,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
        pack=use_pack,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # 검증 데이터셋
    val_loader = None
    if args.val_corpus:
        val_noiser = DenoisingNoiser(
            tokenizer, noise_cfg,
            seed=args.seed + 1,
            use_korean_errors=True,
        )
        val_dataset = EditorDataset(
            args.val_corpus, tokenizer, val_noiser,
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_seq_len,
            text_key=args.text_key,
            lang_key=args.lang_key,
            seed=args.seed + 1,
            rank=global_rank,
            world_size=world_size,
            pack=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=max(args.num_workers, 1),
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    if global_rank == 0:
        print(f"\n데이터셋: 스트리밍 (max_seq_len={args.max_seq_len})")
        if args.val_corpus:
            print(f"검증 데이터: {args.val_corpus}")

    # 옵티마이저 (CUDA: fused 단일 커널로 optimizer step)
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01,
        fused=use_fused,
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # AMP
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    scaler = None  # BF16은 scaler 불필요

    # 체크포인트 복원
    start_step = 0
    restored_total_chars = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        restored_total_chars = ckpt.get("total_chars", 0)

        # 데이터 RNG state 복원 (동일 데이터 순서 재현)
        data_state = ckpt.get("data_state")
        if isinstance(data_state, dict):
            if "noiser_state" in data_state:
                noiser.load_state_dict(data_state["noiser_state"])
            if "dataset_state" in data_state:
                dataset.load_state_dict(data_state["dataset_state"])
            if global_rank == 0:
                print(f"\n체크포인트 복원: step {start_step}, chars {format_chars(restored_total_chars)} (data state 포함)")
        else:
            if global_rank == 0:
                print(f"\n체크포인트 복원: step {start_step}, chars {format_chars(restored_total_chars)} (data state 없음 — 데이터 처음부터)")

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
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+

    # 메모리 진단 (step 1 후)
    if global_rank == 0 and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(loader)
    log_interval = args.log_interval
    save_interval = args.save_interval

    # GPU 스칼라 프리얼로케이션 (매 step torch.tensor() 생성 회피)
    running_loss_t = torch.zeros(1, device=device)
    running_aux_t = torch.zeros(1, device=device)
    running_tokens_t = torch.zeros(1, dtype=torch.long, device=device)
    log_chars = torch.zeros(1, dtype=torch.long, device=device)
    total_chars = torch.zeros(1, dtype=torch.long, device=device) + restored_total_chars
    _total_loss = torch.zeros(1, device=device)
    _iter_loss = torch.zeros(1, device=device)
    _ignore_idx = torch.tensor(-100, dtype=torch.long, device=device)
    _max_line_counter = 0  # worker→main _line_counter 추적
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # LR 스케줄
        lr = get_lr(step, args.warmup_steps, args.lr, args.max_steps,
                    min_lr_ratio=args.min_lr_ratio, schedule=args.schedule)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        _total_loss.zero_()

        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            edit_tags = batch["edit_tags"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)
            original_ids = batch["original_ids"].to(device, non_blocking=True)

            # Iterative refinement 학습
            current_ids = input_ids
            _iter_loss.zero_()

            # DDP: 마지막 accum step에서만 gradient sync (중간 step은 no_sync)
            is_last_accum = (accum_step == args.grad_accum_steps - 1)
            ctx = model.no_sync() if (is_distributed and not is_last_accum) else nullcontext()

            with ctx:
              for it in range(config.n_iterations):
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        tag_logits, aux_loss = model(current_ids, pad_mask)
                else:
                    tag_logits, aux_loss = model(current_ids, pad_mask)

                # 현재 iteration의 태그에 대한 CE loss (PAD → -100)
                targets = torch.where(pad_mask, edit_tags, _ignore_idx)

                ce_loss = criterion(
                    tag_logits.view(-1, config.n_tags),
                    targets.view(-1),
                )

                loss = (ce_loss + aux_loss) / (config.n_iterations * args.grad_accum_steps)
                loss.backward()

                _iter_loss += ce_loss.detach()

                # 다음 iteration 준비: 예측 태그 적용 → 새 편집 태그 계산
                if it < config.n_iterations - 1:
                    with torch.no_grad():
                        pred_tags = tag_logits.argmax(dim=-1)  # (B, T)

                        if _LEVENSHTEIN_CPP is not None:
                            # C++ OpenMP 가속 (배치 병렬 처리)
                            new_ids, new_tags_t, new_mask = _LEVENSHTEIN_CPP.batch_refinement_step(
                                current_ids, pred_tags, original_ids, pad_mask,
                                config.vocab_size, config.pad_id, config.max_seq_len,
                            )
                            current_ids = new_ids
                            edit_tags = new_tags_t
                            pad_mask = new_mask
                        else:
                            # Python 폴백
                            B, T = current_ids.shape
                            new_ids_list = []
                            new_tags_list = []
                            for b in range(B):
                                valid = pad_mask[b]
                                src = current_ids[b][valid].tolist()
                                tags_b = pred_tags[b][valid].tolist()

                                modified = apply_edit_tags(src, tags_b, config.vocab_size)
                                modified = modified[:config.max_seq_len]
                                pad_len = config.max_seq_len - len(modified)
                                modified_padded = modified + [config.pad_id] * pad_len

                                orig = original_ids[b][original_ids[b] != config.pad_id].tolist()
                                new_tags = compute_edit_tags(modified, orig, config.vocab_size)
                                new_tags = new_tags + [TAG_KEEP] * pad_len

                                new_ids_list.append(modified_padded)
                                new_tags_list.append(new_tags)

                            current_ids = torch.tensor(new_ids_list, dtype=torch.long, device=device)
                            edit_tags = torch.tensor(new_tags_list, dtype=torch.long, device=device)
                            pad_mask = (current_ids != config.pad_id)

            _total_loss += _iter_loss / config.n_iterations
            running_tokens_t += batch["pad_mask"].sum()
            batch_chars = batch["n_chars"].sum().to(device)
            log_chars += batch_chars
            # total_chars는 log_interval마다 all_reduce 후 갱신 (DDP 정확도)
            if "_line_counter" in batch:
                _max_line_counter = max(_max_line_counter, batch["_line_counter"].max().item())

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss_t += _total_loss / args.grad_accum_steps
        running_aux_t += aux_loss.detach()

        # 로깅 (DDP: all_reduce로 global 통계 → 정확한 throughput 표시)
        if (step + 1) % log_interval == 0:
            # DDP: loss, aux, tokens를 한 번에 all_reduce (통신 1회)
            if is_distributed:
                log_stats = torch.stack([
                    running_loss_t.squeeze(),
                    running_aux_t.squeeze(),
                    running_tokens_t.float().squeeze(),
                    log_chars.float().squeeze(),
                ])
                dist.all_reduce(log_stats)
                total_chars += log_stats[3].long()
            else:
                total_chars += log_chars

            if global_rank == 0:
                dt = time.time() - t0
                if is_distributed:
                    avg_loss = log_stats[0].item() / (log_interval * world_size)
                    avg_aux = log_stats[1].item() / (log_interval * world_size)
                    tok_s = log_stats[2].item() / max(dt, 1e-6)
                else:
                    avg_loss = running_loss_t.item() / log_interval
                    avg_aux = running_aux_t.item() / log_interval
                    tok_s = running_tokens_t.item() / max(dt, 1e-6)
                _total_chars = total_chars.item()
                # 메모리 정보 (첫 로그에만)
                mem_str = ""
                if step + 1 == log_interval and torch.cuda.is_available():
                    alloc = torch.cuda.max_memory_allocated() / 1024**3
                    resv = torch.cuda.max_memory_reserved() / 1024**3
                    mem_str = f" | mem {alloc:.1f}G/{resv:.1f}G"
                gpu_str = f" ({world_size}GPU)" if world_size > 1 else ""
                print(f"step {step + 1:>6d} | loss {avg_loss:.4f} | aux {avg_aux:.4f} | "
                      f"chars {format_chars(_total_chars)} | "
                      f"lr {lr:.2e} | {tok_s:.0f} tok/s{gpu_str} | {dt:.1f}s{mem_str}", flush=True)
            running_loss_t.zero_()
            running_aux_t.zero_()
            running_tokens_t.zero_()
            log_chars.zero_()
            t0 = time.time()

        # 검증 (DDP: 전체 GPU가 참여 → all_reduce로 평균)
        if (val_loader is not None and args.val_every
                and (step + 1) >= args.warmup_steps
                and (step + 1 - args.warmup_steps) % args.val_every == 0):
            val_loss, tag_acc, edit_p, edit_r = validate_editor(
                raw_model, val_loader, criterion, config, device,
                use_amp, args.val_steps, amp_dtype=amp_dtype,
            )
            if is_distributed:
                val_stats = torch.tensor([val_loss, tag_acc, edit_p, edit_r], device=device)
                dist.all_reduce(val_stats)
                val_stats /= world_size
                val_loss, tag_acc, edit_p, edit_r = val_stats.tolist()
            if global_rank == 0:
                print(f"  val step {step + 1:>6d} | val_loss {val_loss:.4f} | "
                      f"tag_acc {tag_acc:.2%} | edit_P {edit_p:.2%} | edit_R {edit_r:.2%}", flush=True)

        # 체크포인트 (DDP: all_reduce는 모든 rank 참여 필요)
        if (step + 1) % save_interval == 0:
            # 잔여 log_chars flush (save_interval ≠ log_interval 배수일 때 대비)
            if is_distributed:
                flush = log_chars.float().clone()
                dist.all_reduce(flush)
                total_chars += flush.long()
            else:
                total_chars += log_chars
            log_chars.zero_()

            # _line_counter: worker→main 전파 (DDP: 모든 rank의 max)
            if is_distributed:
                lc_t = torch.tensor(_max_line_counter, dtype=torch.long, device=device)
                dist.all_reduce(lc_t, op=dist.ReduceOp.MAX)
                _max_line_counter = lc_t.item()
            dataset._line_counter = _max_line_counter

            if global_rank == 0:
                ckpt_path = os.path.join(
                    args.save_dir,
                    f"editor_{args.size}_step_{step + 1}.pt"
                )
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    "step": step + 1,
                    "total_chars": int(total_chars),
                    "config": asdict(config),
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "data_state": {
                        "noiser_state": noiser.state_dict(),
                        "dataset_state": dataset.state_dict(),
                    },
                }, ckpt_path)
                print(f"  체크포인트 저장: {ckpt_path}", flush=True)

                if args.gdrive_remote:
                    upload_and_cleanup(ckpt_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

    # 최종 저장 전 잔여 chars flush
    if is_distributed:
        flush = log_chars.float().clone()
        dist.all_reduce(flush)
        total_chars += flush.long()
    else:
        total_chars += log_chars
    log_chars.zero_()

    if global_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        final_path = os.path.join(args.save_dir, f"editor_{args.size}_final.pt")
        torch.save({
            "step": args.max_steps,
            "total_chars": int(total_chars),
            "config": asdict(config),
            "model": raw_model.state_dict(),
        }, final_path)
        print(f"\n최종 모델 저장: {final_path}")
        print(f"학습 완료! (총 {args.max_steps} 스텝, {format_chars(int(total_chars))} chars)")

        if args.gdrive_remote:
            upload_and_cleanup(final_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="BitEditor 사전학습")

    # 모델
    parser.add_argument("--size", type=str, default="128M",
                        choices=list(EDITOR_CONFIGS.keys()))
    parser.add_argument("--tokenizer", type=str, default="keyboard",
                        choices=list(TOKENIZER_PRESETS.keys()))
    parser.add_argument("--n_iterations", type=int, default=1,
                        help="Iterative refinement 반복 횟수 (초기 학습 1, fine-tuning 2-3)")

    # 데이터
    parser.add_argument("--corpus", type=str, nargs="+", required=True)
    parser.add_argument("--text_key", type=str, default=None)
    parser.add_argument("--lang_key", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 학습
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="최소 LR = lr × min_lr_ratio (default 0.1)")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"],
                        help="LR 스케줄: cosine 또는 wsd (Warmup-Stable-Decay)")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 텐서코어 BitLinear (dp4a/cublasLt)")
    parser.add_argument("--int8_backend", default="cuda", choices=["triton", "cuda"],
                        help="INT8 backend 선택")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 적용 (커널 fusion, 첫 step 느림)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # 검증
    parser.add_argument("--val_corpus", type=str, nargs="+", default=None,
                        help="검증 코퍼스 파일 경로")
    parser.add_argument("--val_every", type=int, default=500,
                        help="검증 주기 (스텝)")
    parser.add_argument("--val_steps", type=int, default=20,
                        help="검증 시 평가할 배치 수")

    # 로깅/저장
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gdrive_remote", default=None,
                        help="체크포인트 업로드용 rclone 대상 폴더 (예: 'gdrive:my_checkpoints/')")
    parser.add_argument("--log_file", default=None,
                        help="동기화할 로그 파일명")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
