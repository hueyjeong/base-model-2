"""DenseEditor → KAGAS M2 평가 파이프라인

오류문 입력 → 모델 교정 → 디코딩 → system_output.txt 생성

Usage:
    python eval_kagas.py --ckpt checkpoints/dense_mamba2_d640_step_100000.pt \
        --input /tmp/nikl_orig.txt --output /tmp/system_output.txt
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

from model.dense_editor import DenseEditor
from model.edit_tags import TAG_KEEP, apply_edit_tags

PROJECT_ROOT = os.path.dirname(__file__)


def load_tokenizer():
    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    return KeyboardTokenizer(os.path.join(PROJECT_ROOT, "keyboard_tokenizer", "keyboard_tokenizer.json"))


def load_model(ckpt_path, device):
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


def correct_sentences(model, config, tokenizer, sentences, device, batch_size=64, threshold=0.0):
    """문장 리스트 → 모델 교정 → 교정된 문장 리스트"""
    use_amp = device.type == "cuda"
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    vocab_size = config.vocab_size

    corrected = []
    for start in range(0, len(sentences), batch_size):
        batch_texts = sentences[start:start + batch_size]

        # 토크나이즈
        all_ids = []
        for text in batch_texts:
            ids = tokenizer.encode(text, add_special=False)
            ids = [bos_id] + ids + [eos_id]
            all_ids.append(ids)

        # 패딩
        max_len = max(len(ids) for ids in all_ids)
        input_tensor = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long, device=device)
        pad_mask = torch.zeros(len(all_ids), max_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(all_ids):
            input_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            pad_mask[i, :len(ids)] = True

        # 추론
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_tensor, pad_mask)
            else:
                logits = model(input_tensor, pad_mask)

            if threshold > 0:
                probs = torch.softmax(logits.float(), dim=-1)
                max_prob, preds = probs.max(dim=-1)
                low_conf = (preds != TAG_KEEP) & (max_prob < threshold)
                preds[low_conf] = TAG_KEEP
            else:
                preds = logits.argmax(dim=-1)

        # 편집 적용 + 디코딩
        preds_cpu = preds.cpu().tolist()
        for i, ids in enumerate(all_ids):
            pred_tags = preds_cpu[i][:len(ids)]
            corrected_ids = apply_edit_tags(ids, pred_tags, vocab_size)
            # BOS/EOS 제거 후 디코딩
            inner = corrected_ids[1:-1] if len(corrected_ids) >= 2 else corrected_ids
            text = tokenizer.decode(inner)
            corrected.append(text)

        if (start // batch_size + 1) % 10 == 0:
            print(f"  {start + len(batch_texts)}/{len(sentences)} 문장 처리됨")

    return corrected


def main():
    parser = argparse.ArgumentParser(description="DenseEditor KAGAS 평가용 교정")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True, help="오류문 파일 (한 줄에 한 문장)")
    parser.add_argument("--output", required=True, help="교정 결과 출력 파일")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model, config, step = load_model(args.ckpt, device)
    print(f"[Checkpoint] step={step}")

    tokenizer = load_tokenizer()

    # 입력 로드
    with open(args.input) as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"[Input] {len(sentences)} 문장")

    # 교정
    t0 = time.time()
    corrected = correct_sentences(model, config, tokenizer, sentences, device,
                                   batch_size=args.batch_size, threshold=args.threshold)
    elapsed = time.time() - t0
    print(f"[Done] {len(corrected)} 문장 교정, {elapsed:.1f}s ({len(corrected)/elapsed:.0f} sent/s)")

    # 저장
    with open(args.output, "w") as f:
        for text in corrected:
            f.write(text + "\n")
    print(f"[Saved] {args.output}")

    # 샘플 출력
    print("\n--- 샘플 (원문 → 교정) ---")
    for i in range(min(5, len(sentences))):
        if sentences[i] != corrected[i]:
            print(f"  IN:  {sentences[i][:80]}")
            print(f"  OUT: {corrected[i][:80]}")
            print()


if __name__ == "__main__":
    main()
