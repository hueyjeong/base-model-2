"""CompositionCodec 평가 — 복원 정확도 + 오류 분석

체크포인트를 로드해서 test.parquet으로 복원 정확도 측정 + 틀린 토큰 샘플 출력.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.composition_codec import CompositionCodec
from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer
from tok.jamo_tokenizer import JamoTokenizer


def evaluate(codec, loader, device, jamo_tok, max_samples=None, show_errors=20):
    """복원 정확도 + 오류 샘플 출력"""
    codec.eval()
    total_correct = 0
    total_jamo = 0
    total_segments = 0
    seg_exact = 0
    n_samples = 0
    errors = []

    with torch.no_grad():
        for batch in loader:
            jamo_ids = batch["jamo_ids"].to(device)
            jamo_mask = batch["jamo_mask"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            n_segments = batch["n_segments"].to(device)

            out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
            pred = out["logits"].argmax(dim=-1)  # [B, L]

            B, L = jamo_ids.shape
            for b in range(B):
                mask = jamo_mask[b]
                gt = jamo_ids[b][mask].cpu().tolist()
                pr = pred[b][mask].cpu().tolist()
                segs = segment_ids[b][mask].cpu().tolist()

                correct = sum(1 for g, p in zip(gt, pr) if g == p)
                total_correct += correct
                total_jamo += len(gt)

                # 세그먼트(토큰)별 exact match
                cur_seg = -1
                seg_gt = []
                seg_pr = []
                for g, p, s in zip(gt, pr, segs):
                    if s != cur_seg:
                        if cur_seg >= 0:
                            total_segments += 1
                            if seg_gt == seg_pr:
                                seg_exact += 1
                            elif len(errors) < show_errors:
                                gt_str = jamo_tok.decode(seg_gt, skip_special=False)
                                pr_str = jamo_tok.decode(seg_pr, skip_special=False)
                                errors.append((gt_str, pr_str))
                        seg_gt = []
                        seg_pr = []
                        cur_seg = s
                    seg_gt.append(g)
                    seg_pr.append(p)
                # 마지막 세그먼트
                if seg_gt:
                    total_segments += 1
                    if seg_gt == seg_pr:
                        seg_exact += 1
                    elif len(errors) < show_errors:
                        gt_str = jamo_tok.decode(seg_gt, skip_special=False)
                        pr_str = jamo_tok.decode(seg_pr, skip_special=False)
                        errors.append((gt_str, pr_str))

            n_samples += B
            if max_samples and n_samples >= max_samples:
                break

    jamo_acc = total_correct / max(total_jamo, 1) * 100
    seg_em = seg_exact / max(total_segments, 1) * 100

    print(f"\n=== 복원 정확도 ===")
    print(f"  자모 정확도:    {jamo_acc:.4f}%")
    print(f"  토큰 EM:       {seg_em:.4f}%")
    print(f"  총 자모:        {total_jamo:,}")
    print(f"  총 토큰:        {total_segments:,}")
    print(f"  평가 샘플:      {n_samples:,}")

    if errors:
        print(f"\n=== 오류 샘플 (최대 {show_errors}개) ===")
        for i, (gt, pr) in enumerate(errors):
            print(f"  [{i+1}] 정답: '{gt}' → 예측: '{pr}'")

    return {"jamo_acc": jamo_acc, "seg_em": seg_em}


def main():
    parser = argparse.ArgumentParser(description="CompositionCodec 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--show_errors", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    d = saved_args.get("d_model", 256)
    nl = saved_args.get("n_layers", 5)
    k = saved_args.get("kernel_size", 7)

    codec = CompositionCodec(
        jamo_vocab=jamo.vocab_size, d_model=d, n_layers=nl, kernel_size=k,
    ).to(device)

    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in sd):
        sd = {key[len(prefix):] if key.startswith(prefix) else key: v for key, v in sd.items()}
    codec.load_state_dict(sd)

    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in codec.parameters())
    print(f"모델: d={d}, L={nl}, k={k}, params={n_params/1e6:.2f}M (step {step})")

    # 데이터
    dataset = BBPEJamoDataset(
        file_paths=args.corpus,
        bbpe_tokenizer=bbpe,
        jamo_tokenizer=jamo,
        max_seq_len=args.max_seq_len,
        text_key=args.text_key,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    evaluate(codec, loader, device, jamo, args.max_samples, args.show_errors)


if __name__ == "__main__":
    main()
