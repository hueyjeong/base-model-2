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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from training.noising import DenoisingNoiser, NoiseConfig
from training.dataset import StreamingPackedDataset

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
        d_model=256, d_inner=512, d_ff=448,
        n_encoder_layers=3, n_decoder_layers=5,
        n_heads=8, n_kv_heads=4, dt_rank=16,
        d_state=16, d_conv=4,
    ),
    "16M": dict(
        d_model=320, d_inner=640, d_ff=640,
        n_encoder_layers=4, n_decoder_layers=6,
        n_heads=8, n_kv_heads=4, dt_rank=20,
        d_state=16, d_conv=4,
    ),
    "32M": dict(
        d_model=448, d_inner=896, d_ff=768,
        n_encoder_layers=4, n_decoder_layers=7,
        n_heads=8, n_kv_heads=4, dt_rank=28,
        d_state=16, d_conv=4,
    ),
    "64M": dict(
        d_model=512, d_inner=1024, d_ff=896,
        n_encoder_layers=6, n_decoder_layers=10,
        n_heads=8, n_kv_heads=4, dt_rank=32,
        d_state=16, d_conv=4,
    ),
    "128M": dict(
        d_model=768, d_inner=1536, d_ff=1280,
        n_encoder_layers=6, n_decoder_layers=10,
        n_heads=12, n_kv_heads=4, dt_rank=48,
        d_state=16, d_conv=4,
    ),
}


def validate(model, val_loader, criterion, config, device, use_amp, n_steps):
    """검증 루프: n_steps 배치에 대해 평균 loss 및 BPC 계산"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_chars = 0
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

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(src_ids, tgt_input, src_mask)
                    loss = criterion(
                        logits.view(-1, config.vocab_size),
                        tgt_target.reshape(-1),
                    )
            else:
                logits = model(src_ids, tgt_input, src_mask)
                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    tgt_target.reshape(-1),
                )

            n_tokens = (tgt_target != criterion.ignore_index).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            total_chars += n_chars

    model.train()
    if total_tokens == 0:
        return float('nan'), float('nan')
    avg_loss = total_loss / total_tokens
    bpc = (avg_loss * total_tokens) / (max(total_chars, 1) * math.log(2))
    return avg_loss, bpc


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    config = BitMambaSeq2SeqConfig(**model_kwargs)
    model = BitMambaSeq2Seq(config).to(device)
    # 임베딩만 FP32로 (FP16 임베딩 gradient NaN 방지, 나머지는 native dtype 유지)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()

    counts = model.count_parameters()
    print(f"\n모델: {args.size}")
    print(f"  d_model={config.d_model}, enc={config.n_encoder_layers}, "
          f"dec={config.n_decoder_layers}")
    print(f"  임베딩 제외 파라미터: {format_params(counts['total_excl_embedding'])}")
    print(f"  전체 파라미터: {format_params(counts['total'])}")

    # Gradient Checkpointing
    if args.grad_ckpt:
        model.encoder.gradient_checkpointing = True
        model.decoder.gradient_checkpointing = True
        print("  Gradient Checkpointing: ✔")

    # ── 데이터셋 (스트리밍 + 패킹) ──
    noiser = DenoisingNoiser(
        tokenizer, NoiseConfig(), seed=args.seed, use_korean_errors=True,
    )
    dataset = StreamingPackedDataset(
        args.corpus, tokenizer, noiser,
        pack_size=args.pack_size,
        text_key=args.text_key, lang_key=args.lang_key,
        seed=args.seed,
    )
    print(f"\n데이터셋: 스트리밍 (pack_size={args.pack_size})")

    # ── 검증 데이터셋 ──
    val_loader = None
    if args.val_corpus:
        val_dataset = StreamingPackedDataset(
            args.val_corpus, tokenizer, noiser,
            pack_size=args.pack_size,
            text_key=args.text_key, lang_key=args.lang_key,
            seed=args.seed + 1,  # 학습과 다른 시드
        )
        print(f"검증 데이터: {args.val_corpus}")

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

    # AMP scaler
    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── 체크포인트 재시작 ──
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n체크포인트 로드: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  스텝 {start_step}부터 재시작")

    # ── 학습 루프 ──
    max_chars = args.max_chars
    stop_mode = "chars" if max_chars else "steps"
    print(f"\n학습 시작 (max_steps={args.max_steps}, "
          f"max_chars={format_chars(max_chars) if max_chars else 'unlimited'}, "
          f"grad_accum={args.grad_accum_steps})")
    print(f"  effective batch = {args.batch_size} × {args.grad_accum_steps} = {args.batch_size * args.grad_accum_steps} packs")
    print(f"  종료 기준: {stop_mode}")
    print("=" * 60)

    model.train()
    optimizer.zero_grad()

    global_step = start_step
    total_chars = 0
    accum_loss = 0.0
    accum_tokens = 0
    log_loss = 0.0
    log_tokens = 0
    log_chars = 0
    t_start = time.time()

    data_iter = iter(loader)
    epoch = 1
    training_done = False

    while global_step < args.max_steps and not training_done:
        # gradient accumulation 루프
        for accum_i in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(loader)
                batch = next(data_iter)
                print(f"  ── epoch {epoch} 시작 (step {global_step}) ──")

            src_ids = batch["src_ids"].to(device)  # (B, src_len)
            tgt_ids = batch["tgt_ids"].to(device)  # (B, tgt_len)
            src_mask = batch["src_mask"].to(device)  # (B, src_len)
            batch_chars = batch["n_chars"]
            del batch  # CPU 텐서 참조 제거

            # Teacher forcing: 디코더 입력은 tgt_ids[:-1], 타겟은 tgt_ids[1:]
            tgt_input = tgt_ids[:, :-1]
            tgt_target = tgt_ids[:, 1:]

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(src_ids, tgt_input, src_mask)
                    loss = criterion(
                        logits.view(-1, config.vocab_size),
                        tgt_target.reshape(-1),
                    )
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits = model(src_ids, tgt_input, src_mask)
                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    tgt_target.reshape(-1),
                )
                loss = loss / args.grad_accum_steps
                loss.backward()

            n_tokens = tgt_target.numel()
            accum_loss += loss.item() * args.grad_accum_steps
            accum_tokens += n_tokens
            log_loss += loss.item() * args.grad_accum_steps
            log_tokens += n_tokens
            log_chars += batch_chars
            total_chars += batch_chars

            # 메모리 해제: 계산 그래프 참조 제거
            del logits, loss, src_ids, tgt_ids, src_mask, tgt_input, tgt_target

        # Optimizer step
        lr = get_lr(global_step, args.warmup_steps, args.lr, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if use_amp:
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
        if global_step % args.log_every == 0:
            elapsed = time.time() - t_start
            avg_loss = log_loss / max(args.log_every, 1)
            tok_per_sec = log_tokens / max(elapsed, 1e-6)
            bpc = (log_loss / max(args.log_every, 1) * log_tokens) / (max(log_chars, 1) * math.log(2)) if log_chars > 0 else 0.0
            print(f"  step {global_step:>7d} | loss {avg_loss:.4f} | bpc {bpc:.3f} | "
                  f"chars {format_chars(total_chars)} | "
                  f"lr {lr:.2e} | {tok_per_sec:.0f} tok/s | "
                  f"{elapsed:.1f}s")
            log_loss = 0.0
            log_tokens = 0
            log_chars = 0
            t_start = time.time()

        # max_chars 체크
        if max_chars and total_chars >= max_chars:
            print(f"\n  ✅ 문자 예산 도달: {format_chars(total_chars)} >= {format_chars(max_chars)}")
            training_done = True

        # 검증
        if val_loader is not None and args.val_every and global_step % args.val_every == 0:
            val_loss, val_bpc = validate(
                model, val_loader, criterion, config, device,
                use_amp, args.val_steps,
            )
            print(f"  📊 val step {global_step:>7d} | val_loss {val_loss:.4f} | val_bpc {val_bpc:.3f}")

        # 체크포인트 저장
        if args.save_dir and global_step % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, f"step_{global_step}.pt")
            ckpt = {
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": model_kwargs,
                "args": vars(args),
            }
            if scaler:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            print(f"  💾 체크포인트 저장: {ckpt_path}")

    # 최종 저장
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        final_path = os.path.join(args.save_dir, "final.pt")
        torch.save({
            "step": global_step,
            "model": model.state_dict(),
            "config": model_kwargs,
        }, final_path)
        print(f"\n최종 모델 저장: {final_path}")

    print(f"\n학습 완료! (총 {global_step} 스텝)")
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
    parser.add_argument("--grad_accum_steps", type=int, default=32,
                        help="Gradient accumulation 스텝 (effective batch size)")
    parser.add_argument("--pack_size", type=int, default=2048,
                        help="패킹 목표 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="DataLoader 배치 크기 (pack 단위)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (AMP)")
    parser.add_argument("--grad_ckpt", action="store_true",
                        help="Gradient checkpointing (활성화 시 활성화 메모리 3~4배 절약)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수 (0=메인 프로세스만)")

    # 저장
    parser.add_argument("--save_dir", default=None, help="체크포인트 저장 디렉토리")
    parser.add_argument("--save_every", type=int, default=5000, help="저장 주기 (스텝)")
    parser.add_argument("--log_every", type=int, default=50, help="로그 주기 (스텝)")
    parser.add_argument("--resume", default=None, help="재시작 체크포인트 경로")

    # 검증
    parser.add_argument("--val_corpus", default=None, help="검증 코퍼스 파일 경로")
    parser.add_argument("--val_every", type=int, default=100, help="검증 주기 (스텝)")
    parser.add_argument("--val_steps", type=int, default=20, help="검증 시 평가할 배치 수")

    args = parser.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
