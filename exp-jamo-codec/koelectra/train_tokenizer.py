"""35K BBPE 토크나이저 학습 (byte fallback + 특수 토큰).

사용:
    python -m koelectra.train_tokenizer \
        --train_parquet corpus/jamo-codec-v3/train.parquet \
        --max_rows 5000000 \
        --vocab_size 35000 \
        --out_dir checkpoints/bbpe_35k

결과: HuggingFace PreTrainedTokenizerFast 포맷으로 저장 →
    AutoTokenizer.from_pretrained("checkpoints/bbpe_35k") 로 로드 가능.

Binary ELECTRA 입력에 맞게:
- bos_token="[BOS]" (id 보장: 1 — special_tokens 순서)
- eos_token="[EOS]"
- pad_token="[PAD]" (id 0)
- unk_token="[UNK]"
- mask_token="[MASK]"
- byte_fallback=True — OOV UTF-8 바이트까지 커버
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MASK]"]


def iter_texts(paths, max_rows=None, min_length=10, batch_size=8192):
    """parquet 스트리밍으로 text 만 yield."""
    n = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
            col = batch["text"]
            for v in col:
                s = v.as_py()
                if s and len(s) >= min_length:
                    yield s
                    n += 1
                    if max_rows and n >= max_rows:
                        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", nargs="+", required=True)
    ap.add_argument("--vocab_size", type=int, default=35000)
    ap.add_argument("--max_rows", type=int, default=5_000_000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--min_length", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="checkpoints/bbpe_35k")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[Tokenizer] vocab_size={args.vocab_size}, max_rows={args.max_rows}")
    print(f"[Data] {args.train_parquet}")

    # BPE 모델 + byte fallback + Gpt2 스타일 byte-level pre-tokenizer/decoder
    tok = Tokenizer(BPE(byte_fallback=True, unk_token="[UNK]", fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,  # → [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, [MASK]=4
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 byte 포함
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    t0 = time.time()
    print("[Train] 시작...")
    tok.train_from_iterator(
        iter_texts(args.train_parquet,
                   max_rows=args.max_rows,
                   min_length=args.min_length),
        trainer=trainer,
        length=args.max_rows,  # progress bar 용
    )
    dt = time.time() - t0
    print(f"[Train] 완료: {dt:.1f}s, vocab={tok.get_vocab_size():,}")

    # 샘플 encode 확인
    sample_texts = [
        "안녕하세요, 반갑습니다.",
        "이것은 한국어 BBPE 토크나이저 테스트입니다.",
        "English mixed with 한국어 and 漢字.",
        "🎉 이모지도 byte fallback 으로 처리됩니다.",
    ]
    print("\n[Sample encodings]")
    for s in sample_texts:
        enc = tok.encode(s)
        decoded = tok.decode(enc.ids)
        print(f"  src: {s!r}")
        print(f"  ids: {enc.ids[:15]}{'...' if len(enc.ids) > 15 else ''}  (n={len(enc.ids)})")
        print(f"  dec: {decoded!r}")
        print(f"  match: {s == decoded}")

    # HuggingFace wrapper 로 저장 (AutoTokenizer 호환)
    wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    wrapper.save_pretrained(args.out_dir)
    print(f"\n[Saved] {args.out_dir}")
    print(f"  vocab_size={wrapper.vocab_size}")
    print(f"  bos={wrapper.bos_token_id}, eos={wrapper.eos_token_id}, "
          f"pad={wrapper.pad_token_id}, unk={wrapper.unk_token_id}, "
          f"mask={wrapper.mask_token_id}")


if __name__ == "__main__":
    main()
