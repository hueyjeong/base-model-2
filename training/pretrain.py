"""BART 스타일 디노이징 사전학습 스크립트

Usage:
    python -m training.pretrain \
        --size 8M \
        --corpus corpus/sample_10g.jsonl \
        --text_key text \
        --grad_accum_steps 32 \
        --lr 5e-4 \
        --max_steps 100000
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from training.noising import DenoisingNoiser, NoiseConfig
from training.dataset import StreamingPackedDataset
try:
    from training.upload_gdrive import upload_and_cleanup
except ImportError:
    # 모듈이 없을 경우를 대비 (테스트 환경 등)
    def upload_and_cleanup(*args, **kwargs): pass

# ── 토크나이저 프리셋 ──────────────────────────────────────────────────

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

TOKENIZER_PRESETS = {
    "bbpe": {
        "module": "bbpe_tokenizer.bbpe_wrapper",
        "class": "BBPETokenizer",
        "default_path": os.path.join(PROJECT_ROOT, "bbpe_tokenizer", "bbpe.json"),
    },
    "mecab_bbpe": {
        "module": "mecab_bbpe_tokenizer.mecab_bbpe_wrapper",
        "class": "MeCabBBPETokenizer",
        "default_path": os.path.join(PROJECT_ROOT, "mecab_bbpe_tokenizer", "mecab_bbpe.json"),
    },
    "nfd": {
        "module": "nfd_tokenizer.tokenizer_wrapper",
        "class": "NFDTokenizer",
        "default_path": os.path.join(PROJECT_ROOT, "nfd_tokenizer", "custom_gec_tokenizer_manual.json"),
    },
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
    """토크나이저 이름으로 동적 로드"""
    if name not in TOKENIZER_PRESETS:
        raise ValueError(f"지원하지 않는 토크나이저: {name}\n"
                         f"  사용 가능: {list(TOKENIZER_PRESETS.keys())}")
    preset = TOKENIZER_PRESETS[name]
    import importlib
    mod = importlib.import_module(preset["module"])
    cls = getattr(mod, preset["class"])
    tok_path = path or preset["default_path"]
    return cls(tok_path)


# ── 모델 사이즈 프리셋 ────────────────────────────────────────────────

SMALL_CONFIGS = {
    "8M": dict(
        d_model=288, d_inner=576, d_ff=544,
        n_encoder_layers=3, n_decoder_layers=5,
        n_heads=8, n_kv_heads=4, dt_rank=24,
        d_state=16, d_conv=4,
    ),
    "16M": dict(
        d_model=352, d_inner=704, d_ff=640,
        n_encoder_layers=4, n_decoder_layers=7,
        n_heads=8, n_kv_heads=4, dt_rank=24,
        d_state=16, d_conv=4,
    ),
    "32M": dict(
        d_model=448, d_inner=896, d_ff=768,
        n_encoder_layers=5, n_decoder_layers=9,
        n_heads=8, n_kv_heads=4, dt_rank=32,
        d_state=16, d_conv=4,
    ),
    "64M": dict(
        d_model=576, d_inner=1152, d_ff=1088,
        n_encoder_layers=6, n_decoder_layers=10,
        n_heads=8, n_kv_heads=4, dt_rank=40,
        d_state=16, d_conv=4,
    ),
    "128M": dict(
        d_model=768, d_inner=1536, d_ff=1280,
        n_encoder_layers=7, n_decoder_layers=12,
        n_heads=12, n_kv_heads=4, dt_rank=48,
        d_state=16, d_conv=4,
    ),
}


# ── Fused Cross-Entropy (liger-kernel) ──────────────────────────────
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    FUSED_CE_AVAILABLE = True
except ImportError:
    FUSED_CE_AVAILABLE = False

def validate(model, val_loader, criterion, config, device, use_amp, n_steps,
             fused_ce_loss=None, amp_dtype=torch.float16, source_bias=0.0,
             tokenizer=None):
    """검증 루프: n_steps 배치에 대해 평균 loss 및 BPC, CER 계산"""
    model.eval()
    import editdistance
    total_loss = 0.0
    total_tokens = 0
    total_chars = 0
    total_edit_distance = 0
    total_target_chars_for_cer = 0
    val_iter = iter(val_loader)

    with torch.no_grad():
        for i in range(n_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            n_chars = batch["n_chars"]

            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            if fused_ce_loss is not None:
                # Fused CE: logits 텐서 미생성
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        encoder_out = model.encode(src_ids, src_mask)
                        hidden = model.decode(tgt_input, encoder_out,
                                              src_mask, return_hidden=True, src_ids=src_ids, source_bias=source_bias)
                        loss = fused_ce_loss(
                            model.lm_head.weight.float(),
                            hidden.view(-1, hidden.size(-1)).float(),
                            tgt_target.reshape(-1),
                        )
                else:
                    encoder_out = model.encode(src_ids, src_mask)
                    hidden = model.decode(tgt_input, encoder_out,
                                          src_mask, return_hidden=True, src_ids=src_ids, source_bias=source_bias)
                    loss = fused_ce_loss(
                        model.lm_head.weight.float(),
                        hidden.view(-1, hidden.size(-1)).float(),
                        tgt_target.reshape(-1),
                    )
                if tokenizer is not None:
                    # logits for argmax (without requiring gradients, and chunked inference if very large, but usually fine)
                    logits = F.linear(hidden.view(-1, hidden.size(-1)).float(), model.lm_head.weight.float())
                    preds = logits.argmax(dim=-1).view(tgt_target.shape)
            else:
                # 기존 방식
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
                        loss = criterion(
                            logits.view(-1, config.vocab_size),
                            tgt_target.reshape(-1),
                        )
                else:
                    logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
                    loss = criterion(
                        logits.view(-1, config.vocab_size),
                        tgt_target.reshape(-1),
                    )
                if tokenizer is not None:
                    preds = logits.argmax(dim=-1).view(tgt_target.shape)

            if tokenizer is not None:
                for b in range(tgt_target.shape[0]):
                    valid_mask = tgt_target[b] != config.pad_id
                    target_ids_b = tgt_target[b][valid_mask].tolist()
                    pred_ids_b = preds[b][valid_mask].tolist()
                    
                    target_text = tokenizer.decode(target_ids_b).replace("<s>", "").replace("</s>", "")
                    pred_text = tokenizer.decode(pred_ids_b).replace("<s>", "").replace("</s>", "")
                    
                    total_edit_distance += editdistance.eval(target_text, pred_text)
                    total_target_chars_for_cer += len(target_text)

            n_tok = (tgt_target != config.pad_id).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
            total_chars += n_chars

    model.train()
    if total_tokens == 0:
        return float('nan'), float('nan'), float('nan')
    avg_loss = total_loss / total_tokens
    bpc = (avg_loss * total_tokens) / (max(total_chars, 1) * math.log(2))
    cer = (total_edit_distance / max(total_target_chars_for_cer, 1)) if tokenizer is not None else float('nan')
    return avg_loss, bpc, cer


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

def get_lr(step: int, warmup: int, max_lr: float, max_steps: int) -> float:
    """Linear warmup + cosine decay 스케줄"""
    min_lr = max_lr * 0.01  # 최소 학습률 (max의 1%)
    if step < warmup:
        return min_lr + (max_lr - min_lr) * step / max(warmup, 1)
    # cosine decay
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))


def train(args):
    # DDP 초기화
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
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

    # ── 토크나이저 ──
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"토크나이저 로드: {args.tokenizer}, vocab_size={tokenizer.vocab_size}")

    # ── 모델 생성 ──
    if args.size not in SMALL_CONFIGS:
        print(f"❌ 지원하지 않는 사이즈: {args.size}")
        print(f"   사용 가능: {list(SMALL_CONFIGS.keys())}")
        return 1

    model_kwargs = dict(SMALL_CONFIGS[args.size])
    model_kwargs["vocab_size"] = tokenizer.vocab_size
    model_kwargs["use_copy_gate"] = args.use_copy_gate
    config = BitMambaSeq2SeqConfig(**model_kwargs)
    model = BitMambaSeq2Seq(config).to(device)
    # 임베딩만 FP32로 (FP16 임베딩 gradient NaN 방지, 나머지는 native dtype 유지)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()

    counts = model.count_parameters()
    if global_rank == 0:
        print(f"\n모델: {args.size}")
        print(f"  d_model={config.d_model}, enc={config.n_encoder_layers}, "
              f"dec={config.n_decoder_layers}")
        print(f"  임베딩 제외 파라미터: {format_params(counts['total_excl_embedding'])}")
        print(f"  전체 파라미터: {format_params(counts['total'])}")

    # INT8 텐서코어 BitLinear 교체
    if args.int8:
        if args.int8_backend == "cuda":
            try:
                from model.cuda_bitlinear import replace_bitlinear_with_cuda
                model = replace_bitlinear_with_cuda(model)
            except Exception as e:
                print(f"⚠️ CUDA BitLinear 로드 실패: {e}")
                print("   triton backend로 fallback 합니다.")
                from model.triton_bitlinear import replace_bitlinear_with_triton
                model = replace_bitlinear_with_triton(model)
        else:
            from model.triton_bitlinear import replace_bitlinear_with_triton
            model = replace_bitlinear_with_triton(model)

    # Gradient Checkpointing
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing = True
        model.decoder.gradient_checkpointing = True
        print("  Gradient Checkpointing: ✔")

    # ── 데이터셋 (스트리밍 + 패킹) ──
    noiser = DenoisingNoiser(
        tokenizer, NoiseConfig(), seed=args.seed + global_rank, use_korean_errors=True,
    )
    dataset = StreamingPackedDataset(
        args.corpus, tokenizer, noiser,
        pack_size=args.pack_size,
        text_key=args.text_key, lang_key=args.lang_key,
        seed=args.seed,
        rank=global_rank,
        world_size=world_size,
    )
    if global_rank == 0:
        print(f"\n데이터셋: 스트리밍 (pack_size={args.pack_size})")

    # ── 검증 데이터셋 ──
    val_loader = None
    if args.val_corpus:
        val_dataset = StreamingPackedDataset(
            args.val_corpus, tokenizer, noiser,
            pack_size=args.pack_size,
            text_key=args.text_key, lang_key=args.lang_key,
            seed=args.seed + 1,  # 학습과 다른 시드
            rank=global_rank,
            world_size=world_size,
        )
        if global_rank == 0:
            print(f"검증 데이터: {args.val_corpus}")

    # Fused Cross-Entropy 설정
    fused_ce_loss = None
    if args.fused_ce:
        if not FUSED_CE_AVAILABLE:
            print("⚠️  liger-kernel 미설치. pip install liger-kernel 후 재시도")
            print("   기존 CE 방식으로 대체합니다.")
        else:
            fused_ce_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=config.pad_id,
                reduction="mean",
            )
            if global_rank == 0:
                print(f"💫 Fused Cross-Entropy 활성화 (logits 메모리 0)")

    # DataLoader: IterableDataset이므로 shuffle 불필요
    def collate_packed(batch):
        """패킹된 샘플들을 배치로 묶기 (길이가 다를 수 있으므로 패딩)"""
        from torch.nn.utils.rnn import pad_sequence
        src_list = [b["src_ids"] for b in batch]
        tgt_list = [b["tgt_ids"] for b in batch]
        n_chars = sum(b["n_chars"] for b in batch)
        src_ids = pad_sequence(src_list, batch_first=True, padding_value=0)
        tgt_ids = pad_sequence(tgt_list, batch_first=True, padding_value=0)
        # 패딩 위치 마스크 (True=유효, False=패딩)
        src_mask = src_ids != 0
        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "src_mask": src_mask,
                "n_chars": n_chars}

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_packed,
        persistent_workers=args.num_workers > 0,
    )
    if args.val_corpus:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            collate_fn=collate_packed,
            persistent_workers=args.num_workers > 0,
        )

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98), weight_decay=args.weight_decay,
    )
    # Loss: cross-entropy (PAD 무시)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # AMP 설정 (BF16 우선, FP16 fallback)
    use_amp = (args.bf16 or args.amp) and torch.cuda.is_available()
    if args.bf16 and torch.cuda.is_available():
        amp_dtype = torch.bfloat16
        scaler = None  # BF16은 GradScaler 불필요 (FP32 동일 dynamic range)
        if global_rank == 0:
            print(f"⚡ BF16 Mixed Precision 활성화 (GradScaler 없음)")
    elif args.amp and torch.cuda.is_available():
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        if global_rank == 0:
            print(f"⚡ FP16 Mixed Precision 활성화 (GradScaler 사용)")
    else:
        amp_dtype = None
        scaler = None

    # DDP Wrapping 모델
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module
    else:
        raw_model = model

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        if global_rank == 0:
            print(f"\n체크포인트 로드: {args.resume}")
        # DDP 환경에서는 map_location에 device를 지정하여 바로 해당 GPU 램에 로드하는 것이 효율적입니다.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if global_rank == 0:
            print(f"  스텝 {start_step}부터 재시작")

    # CUDA Graph (실험): 고정 shape + 단일 GPU + 비스케일러 경로에서만 활성화
    use_cuda_graph = (
        args.cuda_graph
        and torch.cuda.is_available()
        and (not is_distributed)
        and (not args.grad_ckpt)
        and (not args.compile)
        and (not args.fused_ce)
        and (scaler is None)
    )
    if args.cuda_graph and not use_cuda_graph and global_rank == 0:
        print("⚠️  CUDA Graph 조건 미충족(단일GPU/grad_ckpt off/compile off/fused_ce off/FP16 scaler off). 비활성화합니다.")
    if use_cuda_graph:
        try:
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
            if global_rank == 0:
                print("ℹ️  CUDA Graph 모드: AccumulateGrad stream mismatch 경고 비활성화")
        except Exception:
            pass

    # ── 학습 루프 ──
    max_chars = args.max_chars
    stop_mode = "chars" if max_chars else "steps"
    if global_rank == 0:
        print(f"\n학습 시작 (max_steps={args.max_steps}, "
              f"max_chars={format_chars(max_chars) if max_chars else 'unlimited'}, "
              f"grad_accum={args.grad_accum_steps})")
        print(f"  effective batch = {args.batch_size} × {args.grad_accum_steps} × {world_size} = {args.batch_size * args.grad_accum_steps * world_size} packs")
        print(f"  종료 기준: {stop_mode}")
        print("=" * 60)

    # torch.compile (커널 fusion으로 속도 향상)
    if args.compile:
        if args.int8:
            print("⚠️  --int8과 --compile은 동시 사용 불가 (custom autograd). --compile 건너뜀")
        else:
            print("🔧 torch.compile 적용 중... (첫 step 느림, 이후 빠름)")
            model = torch.compile(model)

    model.train()
    optimizer.zero_grad()

    global_step = start_step
    total_chars = args.start_chars
    accum_loss = 0.0
    accum_tokens = 0
    log_loss = 0.0
    log_tokens = 0
    log_chars = 0
    t_start = time.time()

    data_iter = iter(loader)
    epoch = 1
    training_done = False

    def _next_batch():
        nonlocal data_iter, epoch, global_step
        try:
            return next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            b = next(data_iter)
            if global_rank == 0:
                print(f"  ── epoch {epoch} 시작 (step {global_step}) ──", flush=True)
            return b

    graph_runner = None
    if use_cuda_graph:
        warm = _next_batch()
        warm_src = warm["src_ids"].to(device)
        warm_tgt = warm["tgt_ids"].to(device)
        warm_mask = warm["src_mask"].to(device)
        del warm

        static_src = torch.empty_like(warm_src)
        static_tgt = torch.empty_like(warm_tgt)
        static_mask = torch.empty_like(warm_mask)
        static_loss = torch.zeros((), device=device)

        static_src.copy_(warm_src)
        static_tgt.copy_(warm_tgt)
        static_mask.copy_(warm_mask)

        tgt_input_static = static_tgt[:, :-1]
        tgt_target_static = static_tgt[:, 1:]

        # 캡처 전 eager 워밍업 (JIT/커널 초기화 및 lazy path 소거)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                warm_logits = model(static_src, tgt_input_static, static_mask, source_bias=args.source_bias)
                warm_loss = criterion(
                    warm_logits.view(-1, config.vocab_size),
                    tgt_target_static.reshape(-1),
                )
                warm_loss = warm_loss / args.grad_accum_steps
        else:
            warm_logits = model(static_src, tgt_input_static, static_mask, source_bias=args.source_bias)
            warm_loss = criterion(
                warm_logits.view(-1, config.vocab_size),
                tgt_target_static.reshape(-1),
            )
            warm_loss = warm_loss / args.grad_accum_steps
        warm_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del warm_logits, warm_loss
        torch.cuda.synchronize()

        optimizer.zero_grad(set_to_none=True)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model(static_src, tgt_input_static, static_mask, source_bias=args.source_bias)
                    loss = criterion(
                        logits.view(-1, config.vocab_size),
                        tgt_target_static.reshape(-1),
                    )
                    loss = loss / args.grad_accum_steps
            else:
                logits = model(static_src, tgt_input_static, static_mask, source_bias=args.source_bias)
                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    tgt_target_static.reshape(-1),
                )
                loss = loss / args.grad_accum_steps
            loss.backward()
            static_loss.copy_(loss.detach())

        if global_rank == 0:
            print("🚀 CUDA Graph 캡처 완료 (기본 경로: non-fused CE)")

        graph_runner = {
            "graph": g,
            "static_src": static_src,
            "static_tgt": static_tgt,
            "static_mask": static_mask,
            "static_loss": static_loss,
            "shape": (tuple(warm_src.shape), tuple(warm_tgt.shape), tuple(warm_mask.shape)),
        }

    while global_step < args.max_steps and not training_done:
        # gradient accumulation 루프
        for accum_i in range(args.grad_accum_steps):
            batch = _next_batch()

            src_ids = batch["src_ids"].to(device)  # (B, src_len)
            tgt_ids = batch["tgt_ids"].to(device)  # (B, tgt_len)
            src_mask = batch["src_mask"].to(device)  # (B, src_len)
            batch_chars = batch["n_chars"]
            del batch  # CPU 텐서 참조 제거

            # Teacher forcing: 디코더 입력은 tgt_ids[:-1], 타겟은 tgt_ids[1:]
            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            used_graph = False
            if graph_runner is not None:
                cur_shape = (tuple(src_ids.shape), tuple(tgt_ids.shape), tuple(src_mask.shape))
                if cur_shape == graph_runner["shape"]:
                    graph_runner["static_src"].copy_(src_ids)
                    graph_runner["static_tgt"].copy_(tgt_ids)
                    graph_runner["static_mask"].copy_(src_mask)
                    graph_runner["graph"].replay()
                    loss_val = graph_runner["static_loss"].item()
                    used_graph = True
                else:
                    if global_rank == 0:
                        print("⚠️  CUDA Graph shape 불일치 감지. 해당 스텝은 eager 경로로 실행")

            if (not used_graph) and fused_ce_loss is not None:
                # ── Fused Cross-Entropy (logits 미생성) ──
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        encoder_out = raw_model.encode(src_ids, src_mask)
                        hidden = raw_model.decode(tgt_input, encoder_out,
                                              src_mask, return_hidden=True, src_ids=src_ids, source_bias=args.source_bias)
                        loss = fused_ce_loss(
                            raw_model.lm_head.weight.float(),
                            hidden.view(-1, hidden.size(-1)).float(),
                            tgt_target.reshape(-1),
                        )
                        loss = loss / args.grad_accum_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    encoder_out = raw_model.encode(src_ids, src_mask)
                    hidden = raw_model.decode(tgt_input, encoder_out,
                                          src_mask, return_hidden=True, src_ids=src_ids, source_bias=args.source_bias)
                    loss = fused_ce_loss(
                        raw_model.lm_head.weight.float(),
                        hidden.view(-1, hidden.size(-1)).float(),
                        tgt_target.reshape(-1),
                    )
                    loss = loss / args.grad_accum_steps
                    loss.backward()
            else:
                # ── 기존 방식 ──
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        logits = model(src_ids, tgt_input, src_mask, source_bias=args.source_bias)
                        loss = criterion(
                            logits.view(-1, config.vocab_size),
                            tgt_target.reshape(-1),
                        )
                        loss = loss / args.grad_accum_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    logits = model(src_ids, tgt_input, src_mask, source_bias=args.source_bias)
                    loss = criterion(
                        logits.view(-1, config.vocab_size),
                        tgt_target.reshape(-1),
                    )
                    loss = loss / args.grad_accum_steps
                    loss.backward()

            if not used_graph:
                loss_val = loss.item()

            n_tokens = tgt_target.numel()

            # 분산 환경일 경우 손실과 통계를 동기화(reduce)하여 로깅 정확성 보장
            if is_distributed:
                loss_info = torch.tensor([loss.item(), n_tokens, batch_chars], device=device)
                dist.all_reduce(loss_info, op=dist.ReduceOp.SUM)
                sync_loss = loss_info[0].item() / world_size
                sync_tokens = int(loss_info[1].item())
                sync_chars = int(loss_info[2].item())
            else:
                sync_loss = loss_val
                sync_tokens = n_tokens
                sync_chars = batch_chars

            accum_loss += sync_loss * args.grad_accum_steps
            accum_tokens += sync_tokens
            log_loss += sync_loss * args.grad_accum_steps
            log_tokens += sync_tokens
            log_chars += sync_chars
            total_chars += sync_chars

            # 메모리 해제: 계산 그래프 참조 제거
            if used_graph:
                del src_ids, tgt_ids, src_mask, tgt_input, tgt_target
            else:
                del loss, src_ids, tgt_ids, src_mask, tgt_input, tgt_target

        # Optimizer step
        lr = get_lr(global_step, args.warmup_steps, args.lr, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # 로그
        if global_step % args.log_every == 0 and global_rank == 0:
            elapsed = time.time() - t_start
            avg_loss = log_loss / max(args.log_every, 1)
            tok_per_sec = log_tokens / max(elapsed, 1e-6)
            bpc = (log_loss / max(args.log_every, 1) * log_tokens) / (max(log_chars, 1) * math.log(2)) if log_chars > 0 else 0.0
            print(f"  step {global_step:>7d} | loss {avg_loss:.4f} | bpc {bpc:.3f} | "
                  f"chars {format_chars(total_chars)} | "
                  f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | "
                  f"{elapsed:.1f}s", flush=True)
            log_loss = 0.0
            log_tokens = 0
            log_chars = 0
            t_start = time.time()

        # max_chars 체크
        if max_chars and total_chars >= max_chars:
            if global_rank == 0:
                print(f"\n  ✅ 문자 예산 도달: {format_chars(total_chars)} >= {format_chars(max_chars)}")
            training_done = True

        # 검증
        if val_loader is not None and args.val_every and global_step % args.val_every == 0:
            # TODO: 다중 GPU 환경에서 검증셋을 분산/수집하는 방법이 있지만, 우선 메인 프로세스에서만 검증
            if global_rank == 0:
                val_loss, val_bpc, val_cer = validate(
                    raw_model, val_loader, criterion, config, device,
                    use_amp, args.val_steps, fused_ce_loss=fused_ce_loss,
                    amp_dtype=amp_dtype, source_bias=args.source_bias,
                    tokenizer=tokenizer,
                )
                print(f"  📊 val step {global_step:>7d} | val_loss {val_loss:.4f} | val_bpc {val_bpc:.3f} | val_cer {val_cer:.4f}", flush=True)
            # 다른 프로세스 동기화 
            if is_distributed:
                dist.barrier()

        # 체크포인트 저장
        if args.save_dir and global_step % args.save_every == 0 and global_rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step}.pt")
            ckpt = {
                "step": global_step,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_kwargs,
                "args": vars(args),
            }
            if scaler:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"  💾 체크포인트 저장: {ckpt_path}")
            
            # Google Drive 자동 업로드 및 최신 체크포인트 보존
            if args.gdrive_remote:
                upload_and_cleanup(ckpt_path, args.log_file, args.gdrive_remote, keep_latest_n=1)

    # 최종 저장
    if args.save_dir and global_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        final_path = os.path.join(args.save_dir, "final.pt")
        torch.save({
            "step": global_step,
            "model": raw_model.state_dict(),
            "config": model_kwargs,
        }, final_path)
        print(f"\n최종 모델 저장: {final_path}")
        print(f"\n학습 완료! (총 {global_step} 스텝)")
        
        # 마지막 모델과 로그 업로드 수행
        if args.gdrive_remote:
            upload_and_cleanup(final_path, args.log_file, args.gdrive_remote, keep_latest_n=1)
        
    if is_distributed:
        dist.destroy_process_group()
        
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="BitNet-Mamba Seq2Seq BART 스타일 사전학습",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 모델
    parser.add_argument("--size", default="8M",
                        choices=list(SMALL_CONFIGS.keys()),
                        help="모델 사이즈 프리셋")
    parser.add_argument("--tokenizer", default="bbpe",
                        choices=list(TOKENIZER_PRESETS.keys()),
                        help="토크나이저 종류")

    # 데이터
    parser.add_argument("--corpus", required=True, help="코퍼스 파일 경로")
    parser.add_argument("--text_key", default=None, help="JSONL 텍스트 필드")
    parser.add_argument("--lang_key", default=None, help="JSONL 언어 필드")

    # 학습
    parser.add_argument("--lr", type=float, default=5e-4, help="최대 학습률")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="워밍업 스텝")
    parser.add_argument("--max_steps", type=int, default=100000, help="최대 학습 스텝")
    parser.add_argument("--max_chars", type=int, default=None,
                        help="총 문자 예산 (예: 500_000_000). 설정 시 이 문자 수 도달하면 종료")
    parser.add_argument("--start_chars", type=int, default=0,
                        help="이미 학습한 문자 수 (resume 시 사용)")
    parser.add_argument("--grad_accum_steps", type=int, default=32,
                        help="Gradient accumulation 스텝 (effective batch size)")
    parser.add_argument("--pack_size", type=int, default=4096,
                        help="패킹 목표 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="DataLoader 배치 크기 (pack 단위)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--fused_ce", action="store_true",
                        help="Fused Cross-Entropy (liger-kernel). logits 메모리 0으로 절약")
    parser.add_argument("--amp", action="store_true", help="FP16 Mixed precision (AMP)")
    parser.add_argument("--bf16", action="store_true",
                        help="BF16 Mixed precision (FP32 동일 범위, scaler 불필요, 더 안정적)")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Gradient checkpointing (활성화 시 활성화 메모리 3~4배 절약)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 적용 (커널 fusion, 첫 step 느리나 이후 1.3~2x 빠름)")
    parser.add_argument("--cuda_graph", action="store_true",
                        help="(실험) CUDA Graph 캡처 사용: 단일GPU 고정-shape eager 경로")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 tensor core BitLinear로 교체")
    parser.add_argument("--int8_backend", default="triton", choices=["triton", "cuda"],
                        help="INT8 BitLinear backend 선택")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수 (0=메인 프로세스만)")

    # 원문 참조력 실험 (Trial A & B 기본 적용)
    parser.add_argument("--source_bias", type=float, default=0.5,
                        help="Trial A: 원문 등장 토큰 Logit 가산 강도 (기본: 0.5)")
    parser.add_argument("--no_copy_gate", action="store_false", dest="use_copy_gate",
                        help="Trial B: Copy Gate 비활성화")

    # 저장
    parser.add_argument("--save_dir", default=None, help="체크포인트 저장 디렉토리")
    parser.add_argument("--save_every", type=int, default=1000, help="저장 주기 (스텝)")
    parser.add_argument("--log_every", type=int, default=50, help="로그 주기 (스텝)")
    parser.add_argument("--resume", default=None, help="재시작 체크포인트 경로")
    
    # Google Drive 업로드 동기화
    parser.add_argument("--gdrive_remote", default=None, help="체크포인트 업로드용 rclone 대상 폴더 (예: 'gdrive:my_checkpoints/')")
    parser.add_argument("--log_file", default=None, help="동기화할 로그 파일명 (예: 'training_run_v1.log')")

    # 검증
    parser.add_argument("--val_corpus", default=None, help="검증 코퍼스 파일 경로")
    parser.add_argument("--val_every", type=int, default=200, help="검증 주기 (스텝)")
    parser.add_argument("--val_steps", type=int, default=20, help="검증 시 평가할 배치 수")

    args = parser.parse_args()
    return train(args)

if __name__ == "__main__":
    sys.exit(main())
