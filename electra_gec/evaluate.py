"""KoELECTRA GECToR 텍스트 레벨 평가

체크포인트 로드 → 노이즈 입력 → 교정 → 문자 레벨 P/R/F0.5.
Iterative refinement (N-pass) + keep_bias/conf_threshold 튜닝 지원.

Usage:
    python -m electra_gec.evaluate \
        --checkpoint electra_gec/checkpoints/best.pt \
        --corpus corpus/val_50k.jsonl --text_key text \
        --n_samples 500 --n_passes 3
"""
import argparse
import difflib
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from electra_gec.model import (
    KoELECTRAGECToR,
    apply_two_head_tags,
    ACTION_KEEP,
)
from training.noising import DenoisingNoiser, NoiseConfig


def char_edits(src: str, tgt: str) -> set[tuple[int, str, str]]:
    """문자 레벨 diff → 편집 집합 {(pos, old_char, new_char)}"""
    edits = set()
    sm = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        edits.add((i1, src[i1:i2], tgt[j1:j2]))
    return edits


def compute_prf(tp: int, fp: int, fn: int, beta: float = 0.5):
    """P/R/F_beta 계산"""
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    b2 = beta * beta
    f = (1 + b2) * p * r / max(b2 * p + r, 1e-8) if (p + r) > 0 else 0.0
    return p, r, f


def correct_text(
    model: KoELECTRAGECToR,
    tokenizer: AutoTokenizer,
    noised_text: str,
    device: torch.device,
    n_passes: int = 1,
    keep_bias: float = 0.0,
    conf_threshold: float = 0.0,
    max_seq_len: int = 512,
) -> str:
    """노이즈 텍스트 → 교정 텍스트 (iterative refinement)"""
    current = noised_text

    for _ in range(n_passes):
        enc = tokenizer(
            current, max_length=max_seq_len, truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        actions, contents, _ = model.predict(
            input_ids, attn_mask,
            keep_bias=keep_bias, conf_threshold=conf_threshold,
        )

        # CLS/SEP 제외
        ids_list = input_ids[0].tolist()
        acts_list = actions[0].tolist()
        conts_list = contents[0].tolist()

        # 유효 토큰만 (PAD 제외)
        valid_len = attn_mask[0].sum().item()
        ids_list = ids_list[1:valid_len - 1]    # CLS, SEP 제외
        acts_list = acts_list[1:valid_len - 1]
        conts_list = conts_list[1:valid_len - 1]

        # 편집 없으면 조기 종료
        if all(a == ACTION_KEEP for a in acts_list):
            break

        corrected_ids = apply_two_head_tags(ids_list, acts_list, conts_list)
        current = tokenizer.decode(corrected_ids, skip_special_tokens=True)

    return current


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ──
    model = KoELECTRAGECToR(args.model_name).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"체크포인트 로드: {args.checkpoint}")
        if "metrics" in ckpt:
            print(f"  저장 시 메트릭: {ckpt['metrics']}")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── 노이즈 ──
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    kb_tok_path = os.path.join(
        os.path.dirname(__file__), "..", "keyboard_tokenizer", "keyboard_tokenizer.json"
    )
    kb_tok = KeyboardTokenizer(kb_tok_path)
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        korean_error_prob=args.error_prob, korean_error_count=args.error_count,
        weight_preset=args.noise_preset,
    )
    noiser = DenoisingNoiser(kb_tok, noise_cfg, seed=args.seed, use_korean_errors=True)

    # ── 코퍼스 로드 ──
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.text_key:
                try:
                    obj = json.loads(line)
                    text = obj.get(args.text_key, "")
                except json.JSONDecodeError:
                    continue
            else:
                text = line
            if len(text) >= 10:
                texts.append(text)
            if len(texts) >= args.n_samples:
                break
    print(f"평가 샘플: {len(texts)}개")

    # ── 평가 ──
    if args.tune_thresholds:
        _tune_thresholds(model, tokenizer, noiser, texts, device, args)
    else:
        _evaluate_once(
            model, tokenizer, noiser, texts, device,
            n_passes=args.n_passes,
            keep_bias=args.keep_bias,
            conf_threshold=args.conf_threshold,
            max_seq_len=args.max_seq_len,
        )


def _evaluate_once(model, tokenizer, noiser, texts, device,
                    n_passes=1, keep_bias=0.0, conf_threshold=0.0,
                    max_seq_len=512, verbose=True):
    """단일 설정으로 평가"""
    tp = fp = fn = 0
    n_no_edit = 0
    t0 = time.time()

    for i, text in enumerate(texts):
        lang = noiser._detect_lang(text)
        noised = noiser._apply_text_noise(text, lang)

        corrected = correct_text(
            model, tokenizer, noised, device,
            n_passes=n_passes, keep_bias=keep_bias,
            conf_threshold=conf_threshold, max_seq_len=max_seq_len,
        )

        # 문자 레벨 편집 비교
        gold_edits = char_edits(noised, text)      # 정답 편집
        pred_edits = char_edits(noised, corrected)  # 예측 편집

        tp += len(gold_edits & pred_edits)
        fp += len(pred_edits - gold_edits)
        fn += len(gold_edits - pred_edits)

        if not pred_edits:
            n_no_edit += 1

        if verbose and i < 5:
            print(f"\n[{i}] 원본: {text[:60]}...")
            print(f"    노이즈: {noised[:60]}...")
            print(f"    교정: {corrected[:60]}...")
            print(f"    gold={len(gold_edits)} pred={len(pred_edits)} tp={len(gold_edits & pred_edits)}")

    dt = time.time() - t0
    p, r, f05 = compute_prf(tp, fp, fn, beta=0.5)

    if verbose:
        print(f"\n{'='*50}")
        print(f"n_passes={n_passes} keep_bias={keep_bias:.1f} conf_threshold={conf_threshold:.2f}")
        print(f"P={p:.4f}  R={r:.4f}  F0.5={f05:.4f}")
        print(f"TP={tp} FP={fp} FN={fn} | 무편집={n_no_edit}/{len(texts)}")
        print(f"{dt:.1f}s ({len(texts)/dt:.1f} sent/s)")

    return p, r, f05


def _tune_thresholds(model, tokenizer, noiser, texts, device, args):
    """keep_bias × conf_threshold 그리드 서치"""
    print("\n=== Threshold 튜닝 ===")
    best_f05 = 0
    best_params = (0, 0, 1)

    for n_passes in [1, 2, 3]:
        for kb in [0, 0.5, 1.0, 2.0, 3.0]:
            for ct in [0, 0.1, 0.3, 0.5]:
                p, r, f05 = _evaluate_once(
                    model, tokenizer, noiser, texts[:200], device,
                    n_passes=n_passes, keep_bias=kb, conf_threshold=ct,
                    max_seq_len=args.max_seq_len, verbose=False,
                )
                mark = " ★" if f05 > best_f05 else ""
                print(f"  pass={n_passes} kb={kb:.1f} ct={ct:.1f} → P={p:.3f} R={r:.3f} F0.5={f05:.3f}{mark}")
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = (kb, ct, n_passes)

    kb, ct, np = best_params
    print(f"\n최적: keep_bias={kb:.1f}, conf_threshold={ct:.1f}, n_passes={np} → F0.5={best_f05:.3f}")
    print("\n=== 최적 설정으로 전체 평가 ===")
    _evaluate_once(
        model, tokenizer, noiser, texts, device,
        n_passes=np, keep_bias=kb, conf_threshold=ct,
        max_seq_len=args.max_seq_len,
    )


def main():
    p = argparse.ArgumentParser(description="KoELECTRA GECToR 평가")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--corpus", required=True)
    p.add_argument("--text_key", default=None)
    p.add_argument("--model_name", default="monologg/koelectra-base-v3-discriminator")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--n_passes", type=int, default=1)
    p.add_argument("--keep_bias", type=float, default=0.0)
    p.add_argument("--conf_threshold", type=float, default=0.0)
    p.add_argument("--tune_thresholds", action="store_true")
    p.add_argument("--noise_preset", default="realistic")
    p.add_argument("--error_prob", type=float, default=0.5)
    p.add_argument("--error_count", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    main_args = p.parse_args()
    evaluate(main_args)


if __name__ == "__main__":
    main()
