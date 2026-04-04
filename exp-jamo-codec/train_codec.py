"""Conv Codec 단독 학습 스크립트

토크나이저 3종(byte/jamo/keyboard) × 압축률(2/4/8) sweep.
코퍼스에서 스트리밍으로 읽어 codec reconstruction 학습.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.conv_codec import ConvCodec


# ── 데이터셋 ──────────────────────────────────────────────────────────

class CodecDataset(IterableDataset):
    """JSONL/txt 코퍼스 → 토큰 패킹 스트리밍 데이터셋

    RTDDataset 패턴 재사용: 여러 텍스트를 max_seq_len까지 연결.
    """

    def __init__(
        self,
        file_paths,
        tokenizer,
        max_seq_len: int = 512,
        text_key: str = None,
        min_length: int = 10,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.min_length = min_length

    def _iter_texts(self):
        """파일에서 텍스트 스트리밍"""
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")

            if is_parquet:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)
                text_col = self.text_key or "text"
                for batch in pf.iter_batches(batch_size=65536, columns=[text_col]):
                    for text in batch[text_col].to_pylist():
                        if text and len(text) >= self.min_length:
                            yield text
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < self.min_length:
                        continue
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                    else:
                        text = line
                    if len(text) >= self.min_length:
                        yield text

    def __iter__(self):
        """패킹: 여러 텍스트를 BOS...EOS 단위로 연결"""
        pad_id = self.tokenizer.pad_id
        buf = []

        for text in self._iter_texts():
            ids = self.tokenizer.encode(text, add_special=True)
            if not ids:
                continue

            remaining = self.max_seq_len - len(buf)
            if len(ids) > remaining:
                if buf:
                    yield self._make_sample(buf, pad_id)
                buf = []

            remaining = self.max_seq_len - len(buf)
            buf.extend(ids[:remaining])

        if buf:
            yield self._make_sample(buf, pad_id)

    def _make_sample(self, buf, pad_id):
        seq_len = len(buf)
        pad_len = self.max_seq_len - seq_len
        return {
            "input_ids": torch.tensor(buf + [pad_id] * pad_len, dtype=torch.long),
            "pad_mask": torch.tensor(
                [True] * seq_len + [False] * pad_len, dtype=torch.bool,
            ),
        }


# ── 토크나이저 로딩 ────────────────────────────────────────────────────

def load_tokenizer(name: str):
    """토크나이저 이름으로 로드"""
    if name == "byte":
        from tokenizers.byte_tokenizer import ByteTokenizer
        return ByteTokenizer()
    elif name == "jamo":
        from tokenizers.jamo_tokenizer import JamoTokenizer
        return JamoTokenizer()
    elif name == "keyboard":
        from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
        return KeyboardTokenizer()
    else:
        raise ValueError(f"알 수 없는 토크나이저: {name}")


# ── 학습 루프 ──────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 토크나이저
    tokenizer = load_tokenizer(args.tokenizer)
    print(f"토크나이저: {args.tokenizer} (vocab={tokenizer.vocab_size})")

    # 모델
    codec = ConvCodec(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        stride=args.stride,
        n_layers=args.n_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in codec.parameters())
    print(f"ConvCodec: d={args.d_model}, stride={args.stride}, "
          f"layers={args.n_layers}, params={n_params/1e6:.2f}M")

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
        codec.parameters(),
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
    scaler = None  # BF16은 GradScaler 불필요

    # 학습
    codec.train()
    global_step = 0
    accum_loss = 0.0
    accum_correct = 0
    accum_total = 0
    t_start = time.time()

    print(f"\n학습 시작: max_steps={args.max_steps}, batch={args.batch_size}, "
          f"seq_len={args.max_seq_len}")
    print(f"{'step':>8} {'loss':>8} {'acc':>8} {'lr':>10} {'tok/s':>8}")
    print("-" * 50)

    for batch in loader:
        if global_step >= args.max_steps:
            break

        ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = codec(ids, pad_mask)
            loss = out["loss"]

        loss.backward()

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # 통계
        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            valid = pad_mask & (ids != 0)  # PAD 제외
            correct = ((pred == ids) & valid).sum().item()
            total = valid.sum().item()

        accum_loss += loss.item()
        accum_correct += correct
        accum_total += total
        global_step += 1

        # 로깅
        if global_step % args.log_every == 0:
            dt = time.time() - t_start
            avg_loss = accum_loss / args.log_every
            avg_acc = accum_correct / max(accum_total, 1) * 100
            tok_s = accum_total / max(dt, 1e-6)
            lr = scheduler.get_last_lr()[0]

            print(f"{global_step:8d} {avg_loss:8.4f} {avg_acc:7.2f}% {lr:10.2e} {tok_s:8.0f}")

            accum_loss = 0.0
            accum_correct = 0
            accum_total = 0
            t_start = time.time()

        # 체크포인트
        if args.save_every > 0 and global_step % args.save_every == 0:
            save_path = os.path.join(
                args.out_dir, f"codec_{args.tokenizer}_s{args.stride}_step{global_step}.pt"
            )
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "model": codec.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": global_step,
                "args": vars(args),
            }, save_path)
            print(f"  → 체크포인트 저장: {save_path}")

    print(f"\n학습 완료: {global_step} steps")

    # 최종 저장
    if args.out_dir:
        save_path = os.path.join(
            args.out_dir, f"codec_{args.tokenizer}_s{args.stride}_final.pt"
        )
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model": codec.state_dict(),
            "step": global_step,
            "args": vars(args),
        }, save_path)
        print(f"최종 저장: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Conv Codec 학습")

    # 데이터
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="jamo")
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 모델
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 학습
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # 로깅/저장
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--out_dir", default="exp-jamo-codec/checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
