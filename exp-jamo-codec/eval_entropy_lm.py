"""SmallLM 엔트로피 모델 평가

학습된 SmallLM의 NTP 품질(perplexity, BPB) + 패치 경계 품질 분석.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.entropy_codec import SmallLM, compute_patch_boundaries
from train_codec import CodecDataset, load_tokenizer


# ── 한국어 예시 텍스트 ──
SAMPLE_TEXTS = [
    "대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다.",
    "오늘 날씨가 참 좋습니다. 산책하러 나가볼까요?",
    "맞춤법을 확인해 주세요. 감사합니다.",
    "인공지능이 빠르게 발전하고 있다. 특히 자연어 처리 분야에서 큰 성과를 거두고 있다.",
    "김철수 씨가 프로그래밍을 배우기 시작했습니다.",
]


def evaluate_ntp(model, tokenizer, corpus_paths, text_key, max_seq_len,
                 batch_size, device, max_samples=None):
    """NTP perplexity, BPB 측정"""
    dataset = CodecDataset(
        file_paths=corpus_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_key=text_key,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_seqs = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            logits = model(ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = ids[:, 1:].contiguous()
            shift_mask = pad_mask[:, 1:]

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
            valid = shift_mask.reshape(-1) & (shift_targets.reshape(-1) != 0)
            total_loss += loss[valid].sum().item()
            total_tokens += valid.sum().item()
            n_seqs += ids.size(0)

            if max_samples and n_seqs >= max_samples:
                break

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    bpb = avg_loss / math.log(2)

    return {
        "avg_loss": avg_loss,
        "perplexity": ppl,
        "bpb": bpb,
        "n_tokens": total_tokens,
        "n_sequences": n_seqs,
    }


def analyze_patches(model, tokenizer, device, thresholds=(4.0, 6.0, 8.0, 12.0, 16.0)):
    """한국어 예시로 패치 경계 품질 분석"""
    model.eval()

    print(f"\n{'='*60}")
    print("패치 경계 분석 (한국어 예시)")
    print(f"{'='*60}")

    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special=True)
        ids_t = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            entropy = model.compute_entropy(ids_t)  # [1, L]

        # 토큰별 entropy 출력
        tokens_decoded = []
        for tid in ids:
            tokens_decoded.append(tokenizer.decode([tid], skip_special=True))

        print(f"\n  텍스트: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  토큰 수: {len(ids)}")

        # entropy top-5 위치
        ent_vals = entropy[0].cpu().tolist()
        indexed = sorted(enumerate(ent_vals), key=lambda x: -x[1])[:5]
        print(f"  높은 entropy top-5:")
        for pos, val in indexed:
            tok_str = tokens_decoded[pos] if pos < len(tokens_decoded) else "?"
            print(f"    pos={pos:3d} ent={val:.2f} tok='{tok_str}'")

        # threshold별 패치 수
        print(f"  threshold별 패치: ", end="")
        for thr in thresholds:
            boundaries = compute_patch_boundaries(
                entropy, threshold=thr, min_patch=2, max_patch=32,
            )
            n_patches = boundaries[0].sum().item()
            avg_size = len(ids) / max(n_patches, 1)
            print(f"thr={thr:.0f}→{n_patches}p({avg_size:.1f}tok) ", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="SmallLM 엔트로피 모델 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", nargs="+", default=None)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="byte")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    d = saved_args.get("entropy_d_model", 128)
    nl = saved_args.get("entropy_n_layers", 2)
    nh = saved_args.get("entropy_n_heads", 4)

    model = SmallLM(
        vocab_size=tokenizer.vocab_size, d_model=d, n_layers=nl, n_heads=nh,
    ).to(device)

    # _orig_mod. 접두사 제거
    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    model.load_state_dict(sd)

    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt.get("step", "?")
    print(f"모델: d={d}, L={nl}, H={nh}, params={n_params/1e6:.2f}M (step {step})")

    # 1. NTP 평가
    if args.corpus:
        print(f"\n=== NTP 평가 ===")
        metrics = evaluate_ntp(
            model, tokenizer, args.corpus, args.text_key,
            args.max_seq_len, args.batch_size, device, args.max_samples,
        )
        print(f"  Loss:       {metrics['avg_loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
        print(f"  BPB:        {metrics['bpb']:.4f}")
        print(f"  토큰:       {metrics['n_tokens']:,}")
        print(f"  시퀀스:     {metrics['n_sequences']:,}")

    # 2. 패치 경계 분석
    analyze_patches(model, tokenizer, device)


if __name__ == "__main__":
    main()
