"""체크포인트 기준 source_bias 스윕 평가 스크립트

사용 예시:
    python eval_ckpt_bias.py \
        --ckpt checkpoints/run_v1/step_22600.pt \
        --biases 0.0,0.2,0.5,0.8,1.0
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.pretrain import load_tokenizer, validate
from training.dataset import StreamingPackedDataset
from training.noising import DenoisingNoiser, NoiseConfig
from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq


def collate_packed(batch):
    from torch.nn.utils.rnn import pad_sequence

    src_list = [b["src_ids"] for b in batch]
    tgt_list = [b["tgt_ids"] for b in batch]
    n_chars = sum(b["n_chars"] for b in batch)

    src_ids = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_ids = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    src_mask = src_ids != 0

    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_mask": src_mask,
        "n_chars": n_chars,
    }


def parse_biases(raw: str) -> list[float]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("bias 목록이 비어 있습니다. 예: --biases 0.0,0.5,1.0")
    return out


def make_val_loader(
    tokenizer,
    val_corpus: str,
    pack_size: int,
    text_key: str | None,
    lang_key: str | None,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    noiser = DenoisingNoiser(tokenizer, NoiseConfig(), seed=seed, use_korean_errors=True)
    dataset = StreamingPackedDataset(
        val_corpus,
        tokenizer,
        noiser,
        pack_size=pack_size,
        text_key=text_key,
        lang_key=lang_key,
        seed=seed + 1,
        rank=0,
        world_size=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_packed,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader


def main() -> int:
    parser = argparse.ArgumentParser(description="체크포인트 source_bias 스윕 평가")
    parser.add_argument("--ckpt", required=True, help="평가할 체크포인트 경로")
    parser.add_argument("--biases", default="0.0,0.5", help="콤마 구분 bias 목록")
    parser.add_argument("--val_steps", type=int, default=None, help="평가 배치 수 override")
    parser.add_argument("--batch_size", type=int, default=None, help="검증 배치 크기 override")
    parser.add_argument("--num_workers", type=int, default=0, help="검증 DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="검증 노이즈/데이터 seed")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {args.ckpt}")

    biases = parse_biases(args.biases)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    cfg_dict = ckpt["config"]

    tokenizer_name = train_args.get("tokenizer", "keyboard")
    tokenizer = load_tokenizer(tokenizer_name)

    config = BitMambaSeq2SeqConfig(**cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BitMambaSeq2Seq(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_corpus = train_args.get("val_corpus", "corpus/val_50k.jsonl")
    pack_size = int(train_args.get("pack_size", 4096))
    text_key = train_args.get("text_key", "text")
    lang_key = train_args.get("lang_key", None)
    val_steps = int(args.val_steps if args.val_steps is not None else train_args.get("val_steps", 20))
    batch_size = int(args.batch_size if args.batch_size is not None else train_args.get("batch_size", 2))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"[CKPT] {args.ckpt}")
    print(f"[Device] {device}")
    print(f"[Tokenizer] {tokenizer_name}")
    print(f"[Val Corpus] {val_corpus}")
    print(f"[Val Steps] {val_steps}, [Batch Size] {batch_size}, [Workers] {args.num_workers}")
    print(f"[Biases] {biases}")
    print("-" * 80)

    results = []
    for bias in biases:
        # bias 비교 공정성을 위해 매 bias마다 동일 seed로 val_loader 재생성
        val_loader = make_val_loader(
            tokenizer=tokenizer,
            val_corpus=val_corpus,
            pack_size=pack_size,
            text_key=text_key,
            lang_key=lang_key,
            batch_size=batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        val_loss, val_bpc, val_cer = validate(
            model,
            val_loader,
            criterion,
            config,
            device,
            use_amp=False,
            n_steps=val_steps,
            fused_ce_loss=None,
            amp_dtype=torch.float16,
            source_bias=bias,
            tokenizer=tokenizer,
        )

        results.append((bias, val_loss, val_bpc, val_cer))
        print(
            f"bias={bias:>4.2f} | val_loss={val_loss:.4f} | "
            f"val_bpc={val_bpc:.3f} | val_cer={val_cer:.4f}"
        )

    # CER 기준 정렬 출력
    print("-" * 80)
    print("[CER 낮은 순]")
    for bias, val_loss, val_bpc, val_cer in sorted(results, key=lambda x: x[3]):
        print(
            f"bias={bias:>4.2f} | val_cer={val_cer:.4f} | "
            f"val_loss={val_loss:.4f} | val_bpc={val_bpc:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
