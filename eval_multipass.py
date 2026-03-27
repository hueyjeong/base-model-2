"""DenseEditor 멀티 패스 실험

매 패스: 모델 추론 → confidence threshold 적용 → 교정 → 다음 패스 입력
Consensus: 2회 Gumbel 추론 → 합의 → threshold 적용 → 교정 → 반복

Usage:
    # threshold 멀티 패스 (10k)
    python eval_multipass.py --ckpt checkpoints/dense_mamba2_d640_step_25000.pt \
        --n_samples 10000 --n_passes 3 --thresholds 0.0,0.5,0.7,0.8,0.9

    # consensus + threshold 멀티 패스
    python eval_multipass.py --ckpt ... --n_samples 10000 --n_passes 3 \
        --thresholds 0.0,0.7,0.8 --consensus --consensus_temps 0.3,0.5
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from model.dense_editor import DenseEditor
from model.edit_tags import TAG_KEEP, apply_edit_tags, compute_edit_tags

PROJECT_ROOT = os.path.dirname(__file__)


def load_tokenizer():
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    return KeyboardTokenizer(os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"))


def load_model(ckpt_path, device):
    """체크포인트에서 모델 로드"""
    from model.dense_editor_config import DenseEditorConfig

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    step = ckpt.get("step", "?")
    config = DenseEditorConfig(**cfg_dict)

    model = DenseEditor(config)
    if cfg_dict.get("int8_qat", False):
        try:
            from model.cuda_bitlinear import replace_int8linear_with_cuda
            model = replace_int8linear_with_cuda(model)
            print(f"[INT8] Int8Linear CUDA 교체 완료")
        except Exception as e:
            print(f"[INT8] CUDA 교체 실패: {e}")

    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, step


def prepare_sentences(corpus_path, tokenizer, noiser, n_samples, text_key, max_seq_len):
    """코퍼스에서 문장 로드 → 노이즈 적용 → (noised_ids, clean_ids) 쌍 생성"""
    sentences = []
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    max_content = max_seq_len - 2

    with open(corpus_path) as f:
        for line in f:
            if len(sentences) >= n_samples:
                break
            obj = json.loads(line)
            text = obj[text_key]
            if not text.strip():
                continue

            clean_ids = tokenizer.encode(text, add_special=False)
            if not clean_ids:
                continue

            lang = noiser._detect_lang(text)
            noised_text = noiser._apply_text_noise(text, lang)
            noised_ids = tokenizer.encode(noised_text, add_special=False)
            if not noised_ids:
                continue

            noised_ids = noised_ids[:max_content]
            clean_ids = clean_ids[:max_content]

            # BOS/EOS 감싸기
            noised_ids = [bos_id] + noised_ids + [eos_id]
            clean_ids = [bos_id] + clean_ids + [eos_id]

            sentences.append({
                "noised_ids": noised_ids,
                "clean_ids": clean_ids,
            })

    return sentences


def _make_batch_tensors(batch_ids, pad_id, device):
    """문장 리스트 → 패딩된 텐서 + 마스크 (배치 한 번에 생성)"""
    max_len = max(len(ids) for ids in batch_ids)
    B = len(batch_ids)
    input_tensor = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, ids in enumerate(batch_ids):
        input_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        pad_mask[i, :len(ids)] = True
    return input_tensor, pad_mask


def _forward_batch(model, input_tensor, pad_mask, use_amp):
    """배치 forward → float logits"""
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tensor, pad_mask)
        else:
            logits = model(input_tensor, pad_mask)
    return logits.float()


def batch_predict_threshold(model, all_ids, device, use_amp, pad_id, threshold, batch_size=64):
    """배치 추론 → threshold 적용 → tags 리스트 (전부 텐서 연산)"""
    all_tags = []

    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start:start + batch_size]
        input_tensor, pad_mask = _make_batch_tensors(batch_ids, pad_id, device)
        logits = _forward_batch(model, input_tensor, pad_mask, use_amp)

        probs = torch.softmax(logits, dim=-1)
        max_prob, preds = probs.max(dim=-1)

        if threshold > 0:
            low_conf = (preds != TAG_KEEP) & (max_prob < threshold)
            preds[low_conf] = TAG_KEEP

        # 각 문장 길이만큼 잘라서 리스트로
        preds_cpu = preds.cpu()
        for i, ids in enumerate(batch_ids):
            all_tags.append(preds_cpu[i, :len(ids)].tolist())

    return all_tags


def batch_predict_consensus(model, all_ids, device, use_amp, pad_id,
                            temperature, threshold, batch_size=64):
    """배치 추론 → 2회 Gumbel consensus → threshold → tags 리스트"""
    all_tags = []

    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start:start + batch_size]
        input_tensor, pad_mask = _make_batch_tensors(batch_ids, pad_id, device)
        logits = _forward_batch(model, input_tensor, pad_mask, use_amp)

        # 2회 Gumbel sampling (배치 텐서 연산)
        def gumbel_argmax(logits, temp):
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) * 0.9998 + 0.0001))
            return (logits / temp + gumbel).argmax(dim=-1)

        preds_a = gumbel_argmax(logits, temperature)
        preds_b = gumbel_argmax(logits, temperature)

        # consensus
        consensus = torch.where(preds_a == preds_b, preds_a,
                                torch.full_like(preds_a, TAG_KEEP))

        # threshold
        if threshold > 0:
            probs = torch.softmax(logits, dim=-1)
            max_prob, _ = probs.max(dim=-1)
            low_conf = (consensus != TAG_KEEP) & (max_prob < threshold)
            consensus[low_conf] = TAG_KEEP

        consensus_cpu = consensus.cpu()
        for i, ids in enumerate(batch_ids):
            all_tags.append(consensus_cpu[i, :len(ids)].tolist())

    return all_tags


def evaluate(noised_ids_list, current_ids_list, clean_ids_list, vocab_size):
    """교정 결과 평가 — 토큰 위치별 비교

    각 위치에서:
    - needed = (noised != clean): 편집이 필요했는가
    - correct = (current == clean): 결과가 맞는가
    - changed = (current != noised): 편집했는가

    TP: needed & correct, FP: !needed & changed, FN: needed & !correct
    """
    tp = fp = fn = 0
    exact_match = 0

    for noised, current, clean in zip(noised_ids_list, current_ids_list, clean_ids_list):
        if current == clean:
            exact_match += 1

        max_len = max(len(current), len(clean), len(noised))
        cur_p = current + [0] * (max_len - len(current))
        cln_p = clean + [0] * (max_len - len(clean))
        noi_p = noised + [0] * (max_len - len(noised))

        for c, g, n in zip(cur_p, cln_p, noi_p):
            needed = (n != g)
            correct = (c == g)
            changed = (c != n)

            if needed and correct:
                tp += 1
            elif needed and not correct:
                fn += 1
            elif not needed and changed:
                fp += 1

    n = max(len(current_ids_list), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta_sq = 0.25
    f05 = ((1 + beta_sq) * precision * recall
           / max(beta_sq * precision + recall, 1e-8))

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "exact_match": exact_match,
        "exact_match_rate": exact_match / n,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="DenseEditor 멀티 패스 실험")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--corpus", default="corpus/val_50k.jsonl")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_passes", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--thresholds", type=str, default="0.0,0.5,0.7,0.8,0.9",
                        help="threshold 목록 (콤마 구분)")
    parser.add_argument("--consensus", action="store_true",
                        help="V3 Gumbel consensus 활성화")
    parser.add_argument("--consensus_temps", type=str, default="0.3,0.5",
                        help="Gumbel consensus temperature 목록")
    parser.add_argument("--noise_preset", default="default")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[Device] {device}")

    # 모델 로드
    model, config, step = load_model(args.ckpt, device)
    print(f"[Checkpoint] step={step}")
    print(f"[Model] {config.mixing_type} d={config.d_model} L={config.n_layers}")

    # 토크나이저 + 노이저
    tokenizer = load_tokenizer()
    from training.noising import DenoisingNoiser, NoiseConfig
    noise_cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        weight_preset=args.noise_preset,
    )
    noiser = DenoisingNoiser(tokenizer, noise_cfg, seed=42, use_korean_errors=True)

    # 데이터 준비
    print(f"[Data] {args.corpus} → {args.n_samples} 문장 로드 중...")
    sentences = prepare_sentences(args.corpus, tokenizer, noiser,
                                  args.n_samples, args.text_key, args.seq_len)
    print(f"  로드 완료: {len(sentences)} 문장")

    noised_ids_list = [s["noised_ids"] for s in sentences]
    clean_ids_list = [s["clean_ids"] for s in sentences]
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id

    # 초기 상태 (교정 전)
    init_m = evaluate(noised_ids_list, noised_ids_list, clean_ids_list, vocab_size)
    print(f"\n[초기 상태] exact_match={init_m['exact_match_rate']:.2%}, "
          f"교정 필요 문자={init_m['fn']:,}")

    thresholds = [float(t) for t in args.thresholds.split(",")]

    # ── Threshold 멀티 패스 ──
    print(f"\n{'='*75}")
    print(f"[Threshold Multi-pass] thresholds={thresholds}, n_passes={args.n_passes}")
    print(f"{'Pass':>5} {'Thresh':>7} {'Precision':>10} {'Recall':>8} {'F0.5':>8} "
          f"{'ExMatch':>8} {'FP':>8} {'FN':>8}")
    print("-" * 75)

    for threshold in thresholds:
        current_ids = [ids[:] for ids in noised_ids_list]

        for pass_idx in range(args.n_passes):
            t0 = time.time()

            all_tags = batch_predict_threshold(
                model, current_ids, device, use_amp, pad_id, threshold)

            current_ids = [apply_edit_tags(ids, tags, vocab_size)
                           for ids, tags in zip(current_ids, all_tags)]

            elapsed = time.time() - t0

            m = evaluate(noised_ids_list, current_ids, clean_ids_list, vocab_size)

            print(f"{pass_idx+1:>5} {threshold:>7.2f} {m['precision']:>10.2%} "
                  f"{m['recall']:>8.2%} {m['f05']:>8.2%} "
                  f"{m['exact_match_rate']:>8.2%} {m['fp']:>8,} "
                  f"{m['fn']:>8,}  ({elapsed:.1f}s)")

        print()

    # ── Consensus + Threshold 멀티 패스 ──
    if args.consensus:
        temps = [float(t) for t in args.consensus_temps.split(",")]
        print(f"\n{'='*80}")
        print(f"[V3 Gumbel Consensus Multi-pass] temps={temps}, thresholds={thresholds}")
        print(f"{'Pass':>5} {'T':>5} {'Thresh':>7} {'Precision':>10} {'Recall':>8} {'F0.5':>8} "
              f"{'ExMatch':>8} {'FP':>8} {'FN':>8}")
        print("-" * 80)

        for temp in temps:
            for threshold in thresholds:
                current_ids = [ids[:] for ids in noised_ids_list]

                for pass_idx in range(args.n_passes):
                    t0 = time.time()

                    all_tags = batch_predict_consensus(
                        model, current_ids, device, use_amp, pad_id,
                        temp, threshold)

                    current_ids = [apply_edit_tags(ids, tags, vocab_size)
                                   for ids, tags in zip(current_ids, all_tags)]

                    elapsed = time.time() - t0

                    m = evaluate(noised_ids_list, current_ids, clean_ids_list, vocab_size)

                    print(f"{pass_idx+1:>5} {temp:>5.1f} {threshold:>7.2f} "
                          f"{m['precision']:>10.2%} {m['recall']:>8.2%} "
                          f"{m['f05']:>8.2%} {m['exact_match_rate']:>8.2%} "
                          f"{m['fp']:>8,} {m['fn']:>8,}  ({elapsed:.1f}s)")

                print()


if __name__ == "__main__":
    main()
