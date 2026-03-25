"""DenseEditor 사전학습 스크립트

7종 mixing layer (Mamba/FNet/TCN/RWKV/RetNet/sLSTM/mLSTM) 실험용.
Dense 128M 모델, d_model/mixing_type 지정 가능.

Usage:
    python -m training.pretrain_dense_editor \
        --mixing_type xlstm --d_model 640 \
        --corpus corpus/sample_10g.jsonl --bf16

    # DDP
    torchrun --nproc_per_node=2 -m training.pretrain_dense_editor \
        --mixing_type xlstm --d_model 640 --corpus corpus/sample_10g.jsonl --bf16
"""
import argparse
import gc
import io
import json
import math
import os

# ── 기본 환경 변수 (DDP/메모리 최적화) ──
_DEFAULT_ENV = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "OMP_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
for _k, _v in _DEFAULT_ENV.items():
    os.environ.setdefault(_k, _v)
import sys
import threading
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

from model.dense_editor_config import DenseEditorConfig, make_config
from model.dense_editor import DenseEditor
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


VALID_MIXING_TYPES = ["mamba", "mamba2", "fnet", "tcn", "rwkv", "retnet", "xlstm", "mlstm", "attention", "hybrid"]


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

            torch.compiler.cudagraph_mark_step_begin()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    tag_logits = model(input_ids, pad_mask)
            else:
                tag_logits = model(input_ids, pad_mask)

            targets = edit_tags.clone()
            targets[~pad_mask] = -100
            loss = criterion(
                tag_logits.float().view(-1, config.n_tags),
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
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    val_loss = total_loss / total_tokens
    tag_acc = total_correct / total_tokens
    edit_precision = edit_tp / max(edit_tp + edit_fp, 1)
    edit_recall = edit_tp / max(edit_tp + edit_fn, 1)
    beta_sq = 0.5 ** 2  # F0.5: precision 가중
    edit_f05 = ((1 + beta_sq) * edit_precision * edit_recall
                / max(beta_sq * edit_precision + edit_recall, 1e-8))
    return val_loss, tag_acc, edit_precision, edit_recall, edit_f05


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
    if args.mixing_type not in VALID_MIXING_TYPES:
        print(f"지원하지 않는 mixing_type: {args.mixing_type}, 사용 가능: {VALID_MIXING_TYPES}")
        return 1

    config_overrides = dict(
        vocab_size=tokenizer.vocab_size,
        n_tags=2 + 2 * tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
    )
    if args.n_layers is not None:
        config_overrides["n_layers"] = args.n_layers
    if args.d_ff is not None:
        config_overrides["d_ff"] = args.d_ff
    if args.mamba_expand is not None:
        config_overrides["mamba_expand"] = args.mamba_expand
    if args.mamba_d_conv is not None:
        config_overrides["mamba_d_conv"] = args.mamba_d_conv
    if args.bitlinear_mamba:
        config_overrides["bitlinear_mamba"] = True
    if args.int8_qat:
        config_overrides["int8_qat"] = True
    if args.mamba2_in_proj_rank is not None:
        config_overrides["mamba2_in_proj_rank"] = args.mamba2_in_proj_rank
    config = make_config(
        mixing_type=args.mixing_type,
        d_model=args.d_model,
        target_params=args.target_params,
        **config_overrides,
    )

    if global_rank == 0:
        print(f"\n모델 설정: DenseEditor ({args.mixing_type})")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}, d_ff={config.d_ff}")
        print(f"  mixing_type={config.mixing_type}, n_tags={config.n_tags}")

    # 모델 생성
    model = DenseEditor(config).to(device)
    params = model.count_parameters()

    if global_rank == 0:
        print(f"  총 파라미터: {format_params(params['total'])}")

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

    # INT8 QAT 모드: Int8Linear → Int8LinearCuda 교체 (torch.compile graph break 제거)
    if args.int8_qat:
        try:
            from model.cuda_bitlinear import replace_int8linear_with_cuda
            model = replace_int8linear_with_cuda(model)
        except Exception as e:
            if global_rank == 0:
                print(f"Int8LinearCuda 교체 실패: {e}, 기존 Int8Linear 유지")

    # raw_model 참조 저장 (wrapping 전 — state_dict, grad_ckpt, validate에서 사용)
    raw_model = model

    # Gradient checkpointing
    if args.grad_ckpt:
        raw_model.gradient_checkpointing = True

    # DDP
    if is_distributed:
        # compile + allow_in_graph 사용 시 autograd graph이 정상 추적되므로
        # find_unused_parameters=False 가능 → DDP 오버헤드 감소
        _find_unused = not args.compile
        model = DDP(
            model,
            device_ids=[local_rank],
            gradient_as_bucket_view=True,
            static_graph=False,
            find_unused_parameters=_find_unused,
        )

    # torch.compile (DDP 후에 — DDP 래핑 후 compile해야 reducer hook 충돌 방지)
    if args.compile:
        # RMSNorm: Triton @disable → PyTorch ops (torch.compile이 자체 fusion)
        from model.encoder import RMSNorm as _RMSNorm
        for m in raw_model.modules():
            if isinstance(m, _RMSNorm):
                m.use_triton = False

        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 256
        # max-autotune-no-cudagraphs: CUDA graph 없이 matmul autotune만 적용
        # 양자화 캐시(BitLinear/Int8Linear)가 CUDA graph와 호환 불가
        compile_mode = args.compile_mode
        if compile_mode == "max-autotune":
            compile_mode = "max-autotune-no-cudagraphs"
        if global_rank == 0:
            print(f"torch.compile 적용 중 (mode={compile_mode})... (첫 step 느림, 이후 빠름)")
        model = torch.compile(model, mode=compile_mode)

    # 노이즈 설정 (토큰 레벨 비활성화, 한국어 오류 증강 CLI 제어)
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0,
        token_delete_ratio=0.0,
        text_infill_ratio=0.0,
        korean_error_prob=args.error_prob,
        korean_error_count=args.error_count,
        weight_preset=args.noise_preset,
    )
    if global_rank == 0:
        print(f"  noise: error_prob={args.error_prob}, error_count={args.error_count}, preset={args.noise_preset}")
    noiser = DenoisingNoiser(
        tokenizer, noise_cfg,
        seed=args.seed + global_rank,
        use_korean_errors=True,
    )

    # 데이터셋 — n_iterations > 1이면 패킹 비활성 (per-doc original_ids 필요)
    use_pack = (args.n_iterations <= 1)
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

    # Loss (label smoothing + 편집 태그 가중치)
    ce_weight = None
    if args.edit_loss_weight != 1.0:
        # TAG_KEEP(0)=1.0, 나머지 편집 태그=edit_loss_weight
        ce_weight = torch.ones(config.n_tags, device=device)
        ce_weight[1:] = args.edit_loss_weight
        if global_rank == 0:
            print(f"  edit_loss_weight={args.edit_loss_weight} (non-KEEP 태그 가중치)")
    # AMP
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    # ce_weight는 float32 유지 — cross_entropy는 항상 .float() 로짓과 사용
    criterion = nn.CrossEntropyLoss(
        weight=ce_weight,
        ignore_index=-100,
        label_smoothing=args.label_smoothing,
    )
    if args.label_smoothing > 0 and global_rank == 0:
        print(f"  label_smoothing={args.label_smoothing}")
    scaler = None  # BF16은 scaler 불필요

    # 체크포인트 복원
    start_step = 0
    restored_total_chars = 0
    _restored_epoch_state = None
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

        # 에포크 상태 복원
        epoch_state = ckpt.get("epoch_state")
        if isinstance(epoch_state, dict):
            _restored_epoch_state = epoch_state
        else:
            _restored_epoch_state = None

        del ckpt
        gc.collect()

    # 학습 루프
    if global_rank == 0:
        epoch_str = f", epochs={args.epochs}" if args.epochs is not None else ""
        print(f"\n학습 시작: step {start_step} → {args.max_steps}{epoch_str}")
        print(f"  batch_size={args.batch_size}, grad_accum={args.grad_accum_steps}")
        print(f"  lr={args.lr}, warmup={args.warmup_steps}")
        print(f"  n_iterations={args.n_iterations}")
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
    running_tokens_t = torch.zeros(1, dtype=torch.long, device=device)
    log_chars = torch.zeros(1, dtype=torch.long, device=device)
    total_chars = torch.zeros(1, dtype=torch.long, device=device) + restored_total_chars
    _total_loss = torch.zeros(1, device=device)
    _iter_loss = torch.zeros(1, device=device)
    _ignore_idx = torch.tensor(-100, dtype=torch.long, device=device)
    _max_line_counter = 0  # worker→main _line_counter 추적
    _current_epoch = 0
    _steps_per_epoch = None
    _epoch_done = False
    _lr_decay_start = None  # 에포크 1 종료 후 cosine decay 시작점
    _lr_decay_end = None

    # 비동기 체크포인트 저장
    _save_thread: threading.Thread | None = None
    _prev_ckpt_path: str | None = None

    # 에포크 상태 복원
    if _restored_epoch_state is not None:
        _current_epoch = _restored_epoch_state.get("current_epoch", 0)
        _steps_per_epoch = _restored_epoch_state.get("steps_per_epoch")
        _lr_decay_start = _restored_epoch_state.get("lr_decay_start")
        _lr_decay_end = _restored_epoch_state.get("lr_decay_end")
        if global_rank == 0 and _steps_per_epoch is not None:
            print(f"  에포크 상태 복원: epoch={_current_epoch}, steps_per_epoch={_steps_per_epoch}")
            if _lr_decay_start is not None:
                print(f"  lr decay: step {_lr_decay_start} → {_lr_decay_end}")
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # LR 스케줄
        if _lr_decay_start is not None and step >= _lr_decay_start:
            # 에포크 1 이후: decay_start → decay_end 구간에서 cosine decay
            min_lr = args.lr * args.min_lr_ratio
            progress = (step - _lr_decay_start) / max(_lr_decay_end - _lr_decay_start, 1)
            progress = min(progress, 1.0)
            lr = min_lr + (args.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
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
                _current_epoch += 1

                if _steps_per_epoch is None:
                    _steps_per_epoch = step  # step 0부터의 절대 스텝 = 에포크 길이
                if global_rank == 0:
                    print(f"\n에포크 {_current_epoch} 완료 ({_steps_per_epoch} steps/epoch)")

                # 에포크 1 종료 시: 나머지 에포크에 대한 cosine decay 설정
                if _lr_decay_start is None and args.epochs is not None and args.epochs > 1:
                    _lr_decay_start = step
                    _lr_decay_end = step + _steps_per_epoch // 3
                    if global_rank == 0:
                        print(f"  → cosine decay: step {_lr_decay_start} → {_lr_decay_end}")

                if args.epochs is not None and _current_epoch >= args.epochs:
                    _epoch_done = True
                    break
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
              # CUDA graph 호환: reduce-overhead compile 시 step 경계 표시
              # grad_accum + n_iterations 각 forward마다 호출 필요
              if args.compile:
                  torch.compiler.cudagraph_mark_step_begin()
              for it in range(args.n_iterations):
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        tag_logits = model(current_ids, pad_mask)
                else:
                    tag_logits = model(current_ids, pad_mask)

                # 현재 iteration의 태그에 대한 CE loss (PAD → -100)
                targets = torch.where(pad_mask, edit_tags, _ignore_idx)

                ce_loss = criterion(
                    tag_logits.float().view(-1, config.n_tags),
                    targets.view(-1),
                )

                loss = ce_loss / (args.n_iterations * args.grad_accum_steps)

                # Ternary proximity regularization (모든 linear 가중치의 ternary 근접성 유지)
                if args.quant_reg_weight > 0:
                    from model.bitlinear import BitLinear as _BL, quantize_weights_158 as _qw
                    quant_loss = torch.tensor(0.0, device=device)
                    for m in raw_model.modules():
                        if isinstance(m, _BL):
                            quant_loss = quant_loss + m.quantization_loss()
                        elif isinstance(m, nn.Linear) and m.weight.requires_grad:
                            # Mamba2 nn.Linear projection도 ternary 근접성 유지
                            with torch.no_grad():
                                gamma = m.weight.abs().mean().clamp(min=1e-5)
                                target = gamma * (m.weight / gamma).clamp(-1.0, 1.0).round()
                            quant_loss = quant_loss + ((m.weight - target) ** 2).mean()
                    loss = loss + args.quant_reg_weight * quant_loss

                loss.backward()

                _iter_loss += ce_loss.detach()

                # 다음 iteration 준비: 예측 태그 적용 → 새 편집 태그 계산
                if it < args.n_iterations - 1:
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

            _total_loss += _iter_loss / args.n_iterations
            running_tokens_t += batch["pad_mask"].sum()
            batch_chars = batch["n_chars"].sum().to(device)
            log_chars += batch_chars
            # total_chars는 log_interval마다 all_reduce 후 갱신 (DDP 정확도)
            if "_line_counter" in batch:
                _max_line_counter = max(_max_line_counter, batch["_line_counter"].max().item())

        if _epoch_done:
            break

        # Gradient step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss_t += _total_loss / args.grad_accum_steps

        # 로깅 (DDP: all_reduce로 global 통계 → 정확한 throughput 표시)
        if (step + 1) % log_interval == 0:
            # DDP: loss, aux, tokens를 한 번에 all_reduce (통신 1회)
            if is_distributed:
                log_stats = torch.stack([
                    running_loss_t.squeeze(),
                    running_tokens_t.float().squeeze(),
                    log_chars.float().squeeze(),
                ])
                dist.all_reduce(log_stats)
                total_chars += log_stats[2].long()
            else:
                total_chars += log_chars

            if global_rank == 0:
                dt = time.time() - t0
                if is_distributed:
                    avg_loss = log_stats[0].item() / (log_interval * world_size)
                    tok_s = log_stats[1].item() / max(dt, 1e-6)
                else:
                    avg_loss = running_loss_t.item() / log_interval
                    tok_s = running_tokens_t.item() / max(dt, 1e-6)
                _total_chars = total_chars.item()
                # 메모리 정보 (첫 로그에만)
                mem_str = ""
                if step + 1 == log_interval and torch.cuda.is_available():
                    alloc = torch.cuda.max_memory_allocated() / 1024**3
                    resv = torch.cuda.max_memory_reserved() / 1024**3
                    mem_str = f" | mem {alloc:.1f}G/{resv:.1f}G"
                gpu_str = f" ({world_size}GPU)" if world_size > 1 else ""
                print(f"step {step + 1:>6d} | loss {avg_loss:.4f} | "
                      f"chars {format_chars(_total_chars)} | "
                      f"lr {lr:.2e} | {tok_s:.0f} tok/s{gpu_str} | {dt:.1f}s{mem_str}", flush=True)
            running_loss_t.zero_()
            running_tokens_t.zero_()
            log_chars.zero_()
            t0 = time.time()

        # 검증 (DDP: 전체 GPU가 참여 → all_reduce로 평균)
        if (val_loader is not None and args.val_every
                and (step + 1) >= args.warmup_steps
                and (step + 1 - args.warmup_steps) % args.val_every == 0):
            val_loss, tag_acc, edit_p, edit_r, edit_f05 = validate_editor(
                raw_model, val_loader, criterion, config, device,
                use_amp, args.val_steps, amp_dtype=amp_dtype,
            )
            if is_distributed:
                val_stats = torch.tensor([val_loss, tag_acc, edit_p, edit_r], device=device)
                dist.all_reduce(val_stats)
                val_stats /= world_size
                val_loss, tag_acc, edit_p, edit_r = val_stats.tolist()
                # F0.5는 평균 P,R로 재계산 (harmonic mean은 평균 불가)
                beta_sq = 0.5 ** 2
                edit_f05 = ((1 + beta_sq) * edit_p * edit_r
                            / max(beta_sq * edit_p + edit_r, 1e-8))
            if global_rank == 0:
                print(f"  val step {step + 1:>6d} | val_loss {val_loss:.4f} | "
                      f"tag_acc {tag_acc:.2%} | P {edit_p:.2%} R {edit_r:.2%} F0.5 {edit_f05:.2%}",
                      flush=True)

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
                    f"dense_{args.mixing_type}_d{args.d_model}_step_{step + 1}.pt"
                )
                os.makedirs(args.save_dir, exist_ok=True)

                # 이전 저장 스레드 완료 대기
                if _save_thread is not None:
                    _save_thread.join()

                # 메모리 버퍼에 직렬화 (GPU 텐서 안전하게 캡처)
                ckpt_buf = io.BytesIO()
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
                    "epoch_state": {
                        "current_epoch": _current_epoch,
                        "steps_per_epoch": _steps_per_epoch,
                        "lr_decay_start": _lr_decay_start,
                        "lr_decay_end": _lr_decay_end,
                    },
                }, ckpt_buf)

                # 백그라운드 스레드: 디스크 기록 + 이전 체크포인트 삭제 + 업로드
                _prev = _prev_ckpt_path
                _gdrive = args.gdrive_remote
                _logf = args.log_file

                def _save_task(buf, path, prev, gdrive, logf):
                    buf.seek(0)
                    with open(path, "wb") as f:
                        f.write(buf.getvalue())
                    print(f"  체크포인트 저장 완료: {path}", flush=True)
                    # 이전 체크포인트 삭제
                    if prev and os.path.exists(prev):
                        try:
                            os.remove(prev)
                            print(f"  이전 체크포인트 삭제: {prev}", flush=True)
                        except OSError:
                            pass
                    # 업로드
                    if gdrive:
                        upload_and_cleanup(path, logf, gdrive, keep_latest_n=1)

                _save_thread = threading.Thread(
                    target=_save_task,
                    args=(ckpt_buf, ckpt_path, _prev, _gdrive, _logf),
                    daemon=True,
                )
                _save_thread.start()
                _prev_ckpt_path = ckpt_path

    # 최종 저장 전 잔여 chars flush
    if is_distributed:
        flush = log_chars.float().clone()
        dist.all_reduce(flush)
        total_chars += flush.long()
    else:
        total_chars += log_chars
    log_chars.zero_()

    final_step = step + 1 if _epoch_done else args.max_steps

    # 진행 중인 비동기 저장 완료 대기
    if _save_thread is not None:
        _save_thread.join()

    if global_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        final_path = os.path.join(args.save_dir, f"dense_{args.mixing_type}_d{args.d_model}_final.pt")
        torch.save({
            "step": final_step,
            "total_chars": int(total_chars),
            "config": asdict(config),
            "model": raw_model.state_dict(),
        }, final_path)
        print(f"\n최종 모델 저장: {final_path}")
        print(f"학습 완료! (총 {final_step} 스텝, {format_chars(int(total_chars))} chars)")
        if _steps_per_epoch is not None:
            print(f"  steps_per_epoch = {_steps_per_epoch}")
            if args.epochs == 1:
                print(f"  → 다음 실행 시 --max_steps {_steps_per_epoch} 으로 설정하면 cosine decay 적용됨")

        if args.gdrive_remote:
            upload_and_cleanup(final_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DenseEditor 사전학습")

    # 모델
    parser.add_argument("--mixing_type", type=str, default="xlstm",
                        choices=VALID_MIXING_TYPES)
    parser.add_argument("--d_model", type=int, default=640,
                        help="모델 히든 차원 (headdim=32의 배수)")
    parser.add_argument("--target_params", type=int, default=128_000_000,
                        help="타겟 파라미터 수")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="레이어 수 직접 지정 (미지정 시 target_params로 자동 계산)")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN 히든 차원 (미지정 시 d_model*8/3)")
    parser.add_argument("--mamba_expand", type=int, default=None,
                        help="Mamba expand factor (기본 2)")
    parser.add_argument("--mamba_d_conv", type=int, default=None,
                        help="Mamba conv kernel size (기본 4)")
    parser.add_argument("--bitlinear_mamba", action="store_true",
                        help="Mamba-2 in/out_proj를 BitLinear로 교체 (QAT)")
    parser.add_argument("--int8_qat", action="store_true",
                        help="전체 INT8 QAT (BitLinear→Int8Linear, Mamba proj 포함)")
    parser.add_argument("--mamba2_in_proj_rank", type=int, default=None,
                        help="Mamba-2 in_proj 저랭크 차원 (미지정 시 full rank)")
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
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="최소 LR = lr × min_lr_ratio (default 0.01)")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "wsd"],
                        help="LR 스케줄: cosine 또는 wsd (Warmup-Stable-Decay)")
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=None,
                        help="에포크 수 (미지정 시 max_steps로만 제어)")
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 텐서코어 BitLinear (dp4a/cublasLt)")
    parser.add_argument("--int8_backend", default="cuda", choices=["triton", "cuda"],
                        help="INT8 backend 선택")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 적용 (커널 fusion, 첫 step 느림)")
    parser.add_argument("--compile_mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile 모드 (default, reduce-overhead, max-autotune)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing 계수 (0=비활성)")
    parser.add_argument("--edit_loss_weight", type=float, default=2.0,
                        help="non-KEEP 편집 태그 loss 가중치 (1.0=균등)")
    parser.add_argument("--quant_reg_weight", type=float, default=0.0,
                        help="BitLinear proximity regularization weight (0=비활성, 권장 0.01-0.1)")
    parser.add_argument("--error_prob", type=float, default=0.5,
                        help="한국어 오류 주입 확률 (NoiseConfig.korean_error_prob)")
    parser.add_argument("--error_count", type=int, default=3,
                        help="오류 주입 시 오류 수 (NoiseConfig.korean_error_count)")
    parser.add_argument("--noise_preset", type=str, default="default",
                        choices=["default", "realistic"],
                        help="한국어 오류 가중치 프리셋 (default | realistic)")
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
