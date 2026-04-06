"""Codec 학습 스크립트 (Conv / XAttn / Entropy)

토크나이저 3종(byte/jamo/keyboard) × 압축률(2/4/8) sweep.
코퍼스에서 스트리밍으로 읽어 codec reconstruction 학습.
DDP 지원 (torchrun --nproc_per_node=N).
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.conv_codec import ConvCodec
from codec.xattn_codec import CrossAttentionCodec
from codec.entropy_codec import EntropyPatchCodec


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
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size

    def _open_remote_parquet(self, url):
        """HTTP parquet 파일 열기 (재시도 포함)"""
        import fsspec
        import pyarrow.parquet as pq
        max_retries = 5
        for attempt in range(max_retries):
            try:
                f = fsspec.open(url, "rb").open()
                pf = pq.ParquetFile(f)
                return f, pf
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[rank {self.rank}] HTTP open 실패 ({e}), {wait}s 후 재시도...")
                    import time
                    time.sleep(wait)
                else:
                    raise

    def _iter_remote_parquet(self, url, text_col):
        """HTTP parquet row group 단위 스트리밍 (연결 끊김 시 재시도)"""
        import pyarrow.parquet as pq
        f, pf = self._open_remote_parquet(url)
        n_row_groups = pf.metadata.num_row_groups
        rg_idx = 0

        while rg_idx < n_row_groups:
            try:
                table = pf.read_row_group(rg_idx, columns=[text_col])
                for text in table[text_col].to_pylist():
                    if text and len(text) >= self.min_length:
                        yield text
                rg_idx += 1
            except Exception as e:
                print(f"[rank {self.rank}] row_group {rg_idx}/{n_row_groups} 읽기 실패 ({e}), 재연결...")
                try:
                    f.close()
                except Exception:
                    pass
                import time
                time.sleep(2)
                f, pf = self._open_remote_parquet(url)

        try:
            f.close()
        except Exception:
            pass

    def _iter_texts(self):
        """파일에서 텍스트 스트리밍 (HTTP URL 지원, 재시도)"""
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")
            is_remote = fpath.startswith("http://") or fpath.startswith("https://")

            if is_parquet:
                if is_remote:
                    text_col = self.text_key or "text"
                    yield from self._iter_remote_parquet(fpath, text_col)
                    continue
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

    def _iter_sharded_texts(self):
        """DDP rank별 interleaving으로 텍스트 스트리밍"""
        for i, text in enumerate(self._iter_texts()):
            if self.world_size > 1 and i % self.world_size != self.rank:
                continue
            yield text

    def __iter__(self):
        """패킹: 여러 텍스트를 BOS...EOS 단위로 연결 (무한 순환)"""
        pad_id = self.tokenizer.pad_id

        while True:
            buf = []
            for text in self._iter_sharded_texts():
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
        from tok.byte_tokenizer import ByteTokenizer
        return ByteTokenizer()
    elif name == "jamo":
        from tok.jamo_tokenizer import JamoTokenizer
        return JamoTokenizer()
    elif name == "keyboard":
        from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
        return KeyboardTokenizer()
    else:
        raise ValueError(f"알 수 없는 토크나이저: {name}")


def _unwrap_state_dict(model):
    """DDP/compile 래핑을 벗긴 state_dict 반환"""
    m = model
    if hasattr(m, "module"):
        m = m.module
    sd = m.state_dict()
    # torch.compile이 붙이는 _orig_mod. 접두사 제거
    prefix = "_orig_mod."
    cleaned = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return cleaned


# ── 학습 루프 ──────────────────────────────────────────────────────────

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
    if args.codec == "conv":
        codec = ConvCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            stride=args.stride,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            use_hash_ngram=args.use_hash_ngram,
        ).to(device)
        codec_name = "ConvCodec"
    elif args.codec == "xattn":
        codec = CrossAttentionCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            stride=args.stride,
            n_local_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
        ).to(device)
        codec_name = "XAttnCodec"
    elif args.codec == "entropy_conv":
        codec = EntropyPatchCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            encoder_type="conv",
            entropy_threshold=args.entropy_threshold,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
        codec_name = "EntropyConv"
    elif args.codec == "entropy_xattn":
        codec = EntropyPatchCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            encoder_type="xattn",
            entropy_threshold=args.entropy_threshold,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
        ).to(device)
        codec_name = "EntropyXAttn"
    else:
        raise ValueError(f"알 수 없는 codec: {args.codec}")

    n_params = sum(p.numel() for p in codec.parameters())
    if rank == 0:
        print(f"{codec_name}: d={args.d_model}, stride={args.stride}, "
              f"layers={args.n_layers}, params={n_params/1e6:.2f}M")

    # torch.compile
    if args.compile:
        if rank == 0:
            print("torch.compile 적용 중...")
        codec = torch.compile(codec)
        if rank == 0:
            print("torch.compile 완료")

    if is_distributed:
        codec = DDP(codec, device_ids=[rank])

    # 데이터
    dataset = CodecDataset(
        file_paths=args.corpus,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
        rank=rank,
        world_size=world_size,
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

    grad_accum = args.grad_accum_steps
    if rank == 0:
        eff_batch = args.batch_size * grad_accum * world_size
        batch_desc = f"batch={args.batch_size}"
        if grad_accum > 1 or world_size > 1:
            parts = [str(args.batch_size)]
            if grad_accum > 1:
                parts.append(f"accum{grad_accum}")
            if world_size > 1:
                parts.append(f"{world_size}gpu")
            batch_desc = f"batch={'×'.join(parts)}={eff_batch}"
        print(f"\n학습 시작: max_steps={args.max_steps}, {batch_desc}"
              + f", seq_len={args.max_seq_len}")
        print(f"{'step':>8} {'loss':>8} {'bpb':>7} {'acc':>8} {'lr':>10} {'tok/s':>8}")
        print("-" * 58)

    micro_step = 0
    for batch in loader:
        if global_step >= args.max_steps:
            break

        ids = batch["input_ids"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = codec(ids, pad_mask)
            loss = out["loss"] / grad_accum

        loss.backward()

        # 통계 (스케일 전 loss 기록)
        with torch.no_grad():
            pred = out["logits"].argmax(dim=-1)
            valid = pad_mask & (ids != 0)
            correct = ((pred == ids) & valid).sum().item()
            total = valid.sum().item()

        accum_loss += loss.item() * grad_accum  # 원래 스케일로 복원
        accum_correct += correct
        accum_total += total
        micro_step += 1

        if micro_step % grad_accum != 0:
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # 로깅
        if global_step % args.log_every == 0 and rank == 0:
            dt = time.time() - t_start
            avg_loss = accum_loss / args.log_every
            bpb = avg_loss / math.log(2)  # bits-per-byte (CE nats → bits)
            avg_acc = accum_correct / max(accum_total, 1) * 100
            tok_s = accum_total / max(dt, 1e-6)
            if is_distributed:
                tok_s *= world_size  # 전체 GPU 합산
            lr = scheduler.get_last_lr()[0]

            print(f"{global_step:8d} {avg_loss:8.4f} {bpb:7.4f} {avg_acc:7.2f}% {lr:10.2e} {tok_s:8.0f}")

            accum_loss = 0.0
            accum_correct = 0
            accum_total = 0
            t_start = time.time()

        # 체크포인트
        if args.save_every > 0 and global_step % args.save_every == 0 and rank == 0:
            model_sd = _unwrap_state_dict(codec)
            save_path = os.path.join(
                args.out_dir, f"{args.codec}_{args.tokenizer}_s{args.stride}_step{global_step}.pt"
            )
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
        model_sd = codec.module.state_dict() if is_distributed else codec.state_dict()
        save_path = os.path.join(
            args.out_dir, f"{args.codec}_{args.tokenizer}_s{args.stride}_final.pt"
        )
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "model": model_sd,
            "step": global_step,
            "args": vars(args),
        }, save_path)
        print(f"최종 저장: {save_path}")

    if is_distributed:
        dist.destroy_process_group()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Codec 학습")

    # 데이터
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="jamo")
    parser.add_argument("--max_seq_len", type=int, default=512)

    # 모델
    parser.add_argument("--codec", choices=["conv", "xattn", "entropy_conv", "entropy_xattn"], default="conv")
    parser.add_argument("--entropy_threshold", type=float, default=8.0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_hash_ngram", action="store_true", help="Hash n-gram embedding 적용 (Conv only)")

    # 학습
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile 적용")
    parser.add_argument("--num_workers", type=int, default=2)

    # 로깅/저장
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--out_dir", default="exp-jamo-codec/checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
