"""DenseEditor 체크포인트 전체 검증 스크립트

Usage:
    # 기본 평가
    python eval_dense_ckpt.py --ckpt checkpoints/dense_mamba2_d640_step_25000.pt

    # confidence threshold 스윕
    python eval_dense_ckpt.py --ckpt ... --thresholds 0.0,0.5,0.7,0.8,0.9,0.95

    # multi-pass threshold consensus (교집합)
    python eval_dense_ckpt.py --ckpt ... --multi_pass_thresholds 0.7,0.8,0.9

    # V3 Gumbel consensus
    python eval_dense_ckpt.py --ckpt ... --consensus --consensus_temp 0.3,0.5
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from model.dense_editor_config import make_config
from model.dense_editor import DenseEditor
from model.edit_tags import TAG_KEEP, apply_edit_tags
from training.noising import DenoisingNoiser, NoiseConfig
from training.editor_dataset import EditorDataset
from torch.utils.data import DataLoader


def edit_distance(a, b):
    """Levenshtein 편집 거리 (문자/어절 리스트 범용, 1행 DP)"""
    n, m = len(a), len(b)
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def unpack_sentences(input_ids, edit_tags, preds, valid_len, bos_id, eos_id):
    """패킹된 배치 행에서 [BOS]...[EOS] 단위로 개별 문장 추출

    Returns:
        list of (src_ids, pred_tags, gold_tags) — BOS/EOS 포함
    """
    sentences = []
    i = 0
    while i < valid_len:
        if input_ids[i] == bos_id:
            j = i + 1
            while j < valid_len and input_ids[j] != eos_id:
                j += 1
            if j < valid_len:  # EOS 발견
                sentences.append((
                    input_ids[i:j + 1],
                    preds[i:j + 1],
                    edit_tags[i:j + 1],
                ))
                i = j + 1
            else:
                break  # 불완전 문장 — 스킵
        else:
            i += 1
    return sentences


def forward_with_gumbel(model, input_ids, pad_mask, use_amp, temperature):
    """Gumbel-max trick으로 stochastic 추론 → argmax tags"""
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, pad_mask)
    else:
        logits = model(input_ids, pad_mask)

    logits = logits.float()
    if temperature > 0:
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits) * 0.9998 + 0.0001
        ))
        logits = logits / temperature + gumbel_noise

    return logits.argmax(dim=-1)


def forward_with_threshold(model, input_ids, pad_mask, use_amp, threshold):
    """softmax confidence threshold 적용 → non-KEEP 중 낮은 확신은 KEEP으로"""
    if use_amp:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, pad_mask)
    else:
        logits = model(input_ids, pad_mask)

    probs = torch.softmax(logits.float(), dim=-1)
    max_prob, preds = probs.max(dim=-1)

    if threshold > 0:
        low_conf = (preds != TAG_KEEP) & (max_prob < threshold)
        preds[low_conf] = TAG_KEEP

    return preds


def consensus_tags(tags_a, tags_b):
    """두 tag 텐서의 합의: 동일한 tag만 유지, 불일치 → TAG_KEEP"""
    return torch.where(tags_a == tags_b, tags_a, torch.full_like(tags_a, TAG_KEEP))


def compute_metrics(preds, edit_tags, pad_mask):
    """P/R/F0.5 계산"""
    valid = pad_mask
    pred_edit = preds[valid] != TAG_KEEP
    true_edit = edit_tags[valid] != TAG_KEEP
    tp = (pred_edit & true_edit).sum().item()
    fp = (pred_edit & ~true_edit).sum().item()
    fn = (~pred_edit & true_edit).sum().item()
    return tp, fp, fn


PROJECT_ROOT = os.path.dirname(__file__)

def load_tokenizer(name="keyboard"):
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    return KeyboardTokenizer(os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"))


def main():
    parser = argparse.ArgumentParser(description="DenseEditor 체크포인트 전체 검증")
    parser.add_argument("--ckpt", required=True, help="체크포인트 경로")
    parser.add_argument("--corpus", default="corpus/val_50k.jsonl", help="검증 코퍼스")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--noise_preset", default="default", choices=["default", "realistic"])
    parser.add_argument("--max_batches", type=int, default=None, help="최대 배치 수 (None=전체)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="confidence threshold 스윕 (콤마 구분, 예: 0.0,0.5,0.8,0.9,0.95,0.99)")
    parser.add_argument("--multi_pass_thresholds", type=str, default=None,
                        help="멀티 패스 threshold consensus (콤마 구분, 예: 0.7,0.8,0.9)")
    parser.add_argument("--consensus", action="store_true",
                        help="V3 Gumbel consensus 평가")
    parser.add_argument("--consensus_temps", type=str, default="0.3,0.5",
                        help="Gumbel consensus temperature (콤마 구분)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    step = ckpt.get("step", "?")
    print(f"[Checkpoint] {args.ckpt} (step {step})")

    # 토크나이저
    tokenizer = load_tokenizer()

    # 모델 설정 복원
    from model.dense_editor_config import DenseEditorConfig
    config = DenseEditorConfig(**cfg_dict)

    # 모델 로드
    model = DenseEditor(config)

    # 학습 시 INT8 QAT 사용 여부 복원
    use_int8_qat = cfg_dict.get("int8_qat", False)
    if use_int8_qat:
        try:
            from model.cuda_bitlinear import replace_int8linear_with_cuda
            model = replace_int8linear_with_cuda(model)
            print(f"[INT8] Int8Linear CUDA 교체 완료")
        except Exception as e:
            print(f"[INT8] CUDA 교체 실패: {e}")
    else:
        print(f"[INT8] 미사용 (기본 BitLinear)")

    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {config.mixing_type} d={config.d_model} L={config.n_layers} ({n_params/1e6:.1f}M params)")

    # 노이즈 + 데이터셋
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        weight_preset=args.noise_preset,
    )
    noiser = DenoisingNoiser(tokenizer, noise_cfg, seed=42, use_korean_errors=True)

    dataset = EditorDataset(
        args.corpus, tokenizer, noiser,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
        text_key=args.text_key,
        seed=42, rank=0, world_size=1, pack=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                        pin_memory=True, drop_last=False, prefetch_factor=4,
                        persistent_workers=True)

    # 가중치 기반 loss (학습과 동일)
    edit_loss_weight = 2.0
    tag_weights = torch.ones(config.n_tags, device=device)
    tag_weights[1:] = edit_loss_weight
    criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=tag_weights)

    # threshold 스윕 모드
    thresholds = None
    if args.thresholds:
        thresholds = sorted(set(float(t) for t in args.thresholds.split(",")))
        print(f"[Thresholds] {thresholds}")

    # 검증 루프 — threshold별 TP/FP/FN 누적
    if thresholds:
        # threshold별 카운터
        th_tp = {t: 0 for t in thresholds}
        th_fp = {t: 0 for t in thresholds}
        th_fn = {t: 0 for t in thresholds}
    else:
        thresholds = None

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    edit_tp = 0
    edit_fp = 0
    edit_fn = 0
    n_batches = 0
    all_sentences = []  # 글자/어절 평가용 (src_ids, pred_tags, gold_tags) 누적
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    vocab_size = config.vocab_size

    use_amp = device.type == "cuda"
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            edit_tags = batch["edit_tags"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

            # 기본 P/R (argmax, threshold 없음)
            pred_edit = preds[valid] != TAG_KEEP
            true_edit = edit_tags[valid] != TAG_KEEP
            edit_tp += (pred_edit & true_edit).sum().item()
            edit_fp += (pred_edit & ~true_edit).sum().item()
            edit_fn += (~pred_edit & true_edit).sum().item()

            # 글자/어절 평가용 문장 추출
            input_cpu = input_ids.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            tags_cpu = edit_tags.cpu().tolist()
            mask_lens = pad_mask.sum(dim=1).tolist()
            for b in range(input_ids.size(0)):
                sents = unpack_sentences(
                    input_cpu[b], tags_cpu[b], preds_cpu[b],
                    int(mask_lens[b]), bos_id, eos_id)
                all_sentences.extend(sents)

            # threshold 스윕
            if thresholds:
                probs = torch.softmax(tag_logits.float(), dim=-1)
                max_prob, _ = probs.max(dim=-1)
                valid_preds = preds[valid]
                valid_tags = edit_tags[valid]
                valid_max_prob = max_prob[valid]
                valid_true_edit = true_edit

                for t in thresholds:
                    # non-KEEP 예측인데 confidence < threshold → KEEP으로 되돌림
                    th_preds = valid_preds.clone()
                    low_conf = (th_preds != TAG_KEEP) & (valid_max_prob < t)
                    th_preds[low_conf] = TAG_KEEP

                    th_pred_edit = th_preds != TAG_KEEP
                    th_tp[t] += (th_pred_edit & valid_true_edit).sum().item()
                    th_fp[t] += (th_pred_edit & ~valid_true_edit).sum().item()
                    th_fn[t] += (~th_pred_edit & valid_true_edit).sum().item()

            n_batches += 1
            if n_batches % 100 == 0:
                elapsed = time.time() - t0
                p = edit_tp / max(edit_tp + edit_fp, 1)
                r = edit_tp / max(edit_tp + edit_fn, 1)
                print(f"  batch {n_batches}: tokens={total_tokens:,} P={p:.2%} R={r:.2%} ({elapsed:.1f}s)")

            if args.max_batches and n_batches >= args.max_batches:
                break

    elapsed = time.time() - t0

    # 최종 메트릭 (argmax 기준)
    val_loss = total_loss / max(total_tokens, 1)
    tag_acc = total_correct / max(total_tokens, 1)
    edit_precision = edit_tp / max(edit_tp + edit_fp, 1)
    edit_recall = edit_tp / max(edit_tp + edit_fn, 1)
    beta_sq = 0.5 ** 2
    edit_f05 = ((1 + beta_sq) * edit_precision * edit_recall
                / max(beta_sq * edit_precision + edit_recall, 1e-8))

    print(f"\n{'='*60}")
    print(f"[Result] step={step}, {n_batches} batches, {total_tokens:,} tokens, {elapsed:.1f}s")
    print(f"  val_loss   = {val_loss:.4f}")
    print(f"  tag_acc    = {tag_acc:.2%}")
    print(f"  Precision  = {edit_precision:.2%}")
    print(f"  Recall     = {edit_recall:.2%}")
    print(f"  F0.5       = {edit_f05:.2%}")
    print(f"  TP={edit_tp:,}  FP={edit_fp:,}  FN={edit_fn:,}")

    # ── 글자/어절 단위 평가 ──
    if all_sentences:
        t_span = time.time()
        print(f"\n  --- Text-level Metrics ({len(all_sentences):,} sentences) ---")
        for granularity, label in [("char", "Char"), ("word", "Word")]:
            tp, fp, fn = 0.0, 0.0, 0.0
            for src_ids, pred_tags, gold_tags in all_sentences:
                hyp_ids = apply_edit_tags(src_ids, pred_tags, vocab_size)
                gold_ids = apply_edit_tags(src_ids, gold_tags, vocab_size)
                src_text = tokenizer.decode(src_ids, skip_special=True)
                hyp_text = tokenizer.decode(hyp_ids, skip_special=True)
                gold_text = tokenizer.decode(gold_ids, skip_special=True)

                if granularity == "char":
                    s, h, g = list(src_text), list(hyp_text), list(gold_text)
                else:
                    s, h, g = src_text.split(), hyp_text.split(), gold_text.split()

                d_sg = edit_distance(s, g)
                d_sh = edit_distance(s, h)
                d_hg = edit_distance(h, g)
                tp += (d_sg + d_sh - d_hg) / 2
                fp += (d_sh + d_hg - d_sg) / 2
                fn += (d_sg + d_hg - d_sh) / 2

            p = tp / max(tp + fp, 1e-8)
            r = tp / max(tp + fn, 1e-8)
            f05 = ((1 + beta_sq) * p * r / max(beta_sq * p + r, 1e-8))
            print(f"  [{label}] P={p:.2%}  R={r:.2%}  F0.5={f05:.2%}  "
                  f"TP={tp:.0f}  FP={fp:.0f}  FN={fn:.0f}")
        print(f"  (span eval: {time.time() - t_span:.1f}s)")

    # threshold 스윕 결과
    if thresholds:
        print(f"\n{'='*60}")
        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F0.5':>10} {'TP':>10} {'FP':>10} {'FN':>10}")
        print("-" * 70)
        for t in thresholds:
            tp, fp, fn = th_tp[t], th_fp[t], th_fn[t]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f05 = ((1 + beta_sq) * p * r / max(beta_sq * p + r, 1e-8))
            print(f"{t:>10.2f} {p:>10.2%} {r:>10.2%} {f05:>10.2%} {tp:>10,} {fp:>10,} {fn:>10,}")

    # ── 멀티 패스 threshold consensus ──
    if args.multi_pass_thresholds:
        mp_thresholds = sorted(float(t) for t in args.multi_pass_thresholds.split(","))
        print(f"\n{'='*60}")
        print(f"[Multi-pass Threshold Consensus] thresholds={mp_thresholds}")
        print(f"각 threshold로 독립 예측 → 모든 패스가 동의한 편집만 유지")

        mp_tp, mp_fp, mp_fn = 0, 0, 0
        n_b = 0
        t1 = time.time()

        with torch.no_grad():
            loader2 = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                                 pin_memory=True, drop_last=False, prefetch_factor=4,
                                 persistent_workers=True)
            for batch in loader2:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                edit_tags = batch["edit_tags"].to(device, non_blocking=True)
                pad_mask = batch["pad_mask"].to(device, non_blocking=True)

                # 각 threshold로 예측
                all_preds = []
                for t in mp_thresholds:
                    preds_t = forward_with_threshold(model, input_ids, pad_mask, use_amp, t)
                    all_preds.append(preds_t)

                # consensus: 모든 패스가 동의한 것만 유지
                final_preds = all_preds[0]
                for p in all_preds[1:]:
                    final_preds = consensus_tags(final_preds, p)

                tp, fp, fn = compute_metrics(final_preds, edit_tags, pad_mask)
                mp_tp += tp; mp_fp += fp; mp_fn += fn

                n_b += 1
                if n_b % 100 == 0:
                    elapsed = time.time() - t1
                    p = mp_tp / max(mp_tp + mp_fp, 1)
                    r = mp_tp / max(mp_tp + mp_fn, 1)
                    print(f"  batch {n_b}: P={p:.2%} R={r:.2%} ({elapsed:.1f}s)")

                if args.max_batches and n_b >= args.max_batches:
                    break

        p = mp_tp / max(mp_tp + mp_fp, 1)
        r = mp_tp / max(mp_tp + mp_fn, 1)
        f05 = ((1 + beta_sq) * p * r / max(beta_sq * p + r, 1e-8))
        print(f"\n  [Multi-pass Consensus] thresholds={mp_thresholds}")
        print(f"  Precision  = {p:.2%}")
        print(f"  Recall     = {r:.2%}")
        print(f"  F0.5       = {f05:.2%}")
        print(f"  TP={mp_tp:,}  FP={mp_fp:,}  FN={mp_fn:,}")

    # ── V3 Gumbel Consensus ──
    if args.consensus:
        temps = [float(t) for t in args.consensus_temps.split(",")]
        print(f"\n{'='*60}")
        print(f"[V3 Gumbel Consensus] temperatures={temps}")
        print(f"2회 Gumbel stochastic 추론 → 동일 tag만 유지")

        for temp in temps:
            c_tp, c_fp, c_fn = 0, 0, 0
            n_b = 0
            t1 = time.time()

            with torch.no_grad():
                loader3 = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                                     pin_memory=True, drop_last=False, prefetch_factor=4,
                                     persistent_workers=True)
                for batch in loader3:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    edit_tags = batch["edit_tags"].to(device, non_blocking=True)
                    pad_mask = batch["pad_mask"].to(device, non_blocking=True)

                    # 2회 독립 Gumbel 추론
                    preds_a = forward_with_gumbel(model, input_ids, pad_mask, use_amp, temp)
                    preds_b = forward_with_gumbel(model, input_ids, pad_mask, use_amp, temp)

                    # consensus
                    final_preds = consensus_tags(preds_a, preds_b)

                    tp, fp, fn = compute_metrics(final_preds, edit_tags, pad_mask)
                    c_tp += tp; c_fp += fp; c_fn += fn

                    n_b += 1
                    if n_b % 100 == 0:
                        elapsed = time.time() - t1
                        p = c_tp / max(c_tp + c_fp, 1)
                        r = c_tp / max(c_tp + c_fn, 1)
                        print(f"  T={temp} batch {n_b}: P={p:.2%} R={r:.2%} ({elapsed:.1f}s)")

                    if args.max_batches and n_b >= args.max_batches:
                        break

            p = c_tp / max(c_tp + c_fp, 1)
            r = c_tp / max(c_tp + c_fn, 1)
            f05 = ((1 + beta_sq) * p * r / max(beta_sq * p + r, 1e-8))
            print(f"\n  [V3 Consensus T={temp}]")
            print(f"  Precision  = {p:.2%}")
            print(f"  Recall     = {r:.2%}")
            print(f"  F0.5       = {f05:.2%}")
            print(f"  TP={c_tp:,}  FP={c_fp:,}  FN={c_fn:,}")


if __name__ == "__main__":
    main()
