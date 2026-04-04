"""Conv Codec 평가 스크립트

학습된 codec의 복원 정확도, 시퀀스 정합도, 속도를 측정.
체크포인트 없이 실행하면 학습 직후 모델로 평가 가능.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.conv_codec import ConvCodec
from train_codec import CodecDataset, load_tokenizer


def evaluate(codec, tokenizer, corpus_paths, text_key, max_seq_len,
             batch_size, device, max_samples=None):
    """codec 복원 정확도 평가

    Returns:
        dict: token_acc, char_acc, seq_exact_match, avg_loss
    """
    dataset = CodecDataset(
        file_paths=corpus_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_key=text_key,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0,
    )

    codec.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0
    n_batches = 0

    # 문자 단위 평가용
    char_correct = 0
    char_total = 0
    seq_exact = 0
    seq_total = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            out = codec(ids, pad_mask)
            pred = out["logits"].argmax(dim=-1)

            # 토큰 정확도
            valid = pad_mask & (ids != 0)
            correct = ((pred == ids) & valid).sum().item()
            total = valid.sum().item()
            total_correct += correct
            total_tokens += total
            total_loss += out["loss"].item()
            n_batches += 1

            # 문자 단위 정확도 (디코드 후 비교)
            for i in range(ids.size(0)):
                valid_len = pad_mask[i].sum().item()
                orig_ids = ids[i, :valid_len].cpu().tolist()
                pred_ids = pred[i, :valid_len].cpu().tolist()

                orig_text = tokenizer.decode(orig_ids, skip_special=True)
                pred_text = tokenizer.decode(pred_ids, skip_special=True)

                # 문자 단위
                min_len = min(len(orig_text), len(pred_text))
                max_len = max(len(orig_text), len(pred_text))
                if max_len > 0:
                    matches = sum(
                        1 for a, b in zip(orig_text, pred_text) if a == b
                    )
                    char_correct += matches
                    char_total += max_len

                # 시퀀스 정합
                if orig_text == pred_text:
                    seq_exact += 1
                seq_total += 1

            if max_samples and seq_total >= max_samples:
                break

    token_acc = total_correct / max(total_tokens, 1)
    char_acc = char_correct / max(char_total, 1)
    seq_em = seq_exact / max(seq_total, 1)
    avg_loss = total_loss / max(n_batches, 1)

    return {
        "token_acc": token_acc,
        "char_acc": char_acc,
        "seq_exact_match": seq_em,
        "avg_loss": avg_loss,
        "n_sequences": seq_total,
        "n_tokens": total_tokens,
    }


def measure_latency(codec, tokenizer, device, seq_len=512, n_runs=100):
    """encode/decode 레이턴시 측정"""
    ids = torch.randint(1, tokenizer.vocab_size, (1, seq_len)).to(device)

    codec.eval()
    with torch.no_grad():
        # warmup
        for _ in range(10):
            z = codec.encode(ids)
            _ = codec.decode(z, target_len=seq_len)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # encode
        t0 = time.perf_counter()
        for _ in range(n_runs):
            z = codec.encode(ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        enc_time = (time.perf_counter() - t0) / n_runs * 1000  # ms

        # decode
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = codec.decode(z, target_len=seq_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dec_time = (time.perf_counter() - t0) / n_runs * 1000  # ms

    return {
        "encode_ms": enc_time,
        "decode_ms": dec_time,
        "total_ms": enc_time + dec_time,
        "seq_len": seq_len,
        "compressed_len": z.size(1),
    }


def show_samples(codec, tokenizer, corpus_paths, text_key, device,
                 max_seq_len=512, n_samples=5):
    """복원 예시 출력"""
    dataset = CodecDataset(
        file_paths=corpus_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_key=text_key,
    )

    codec.eval()
    print("\n--- 복원 예시 ---")
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= n_samples:
                break
            ids = sample["input_ids"].unsqueeze(0).to(device)
            pad_mask = sample["pad_mask"].unsqueeze(0).to(device)
            valid_len = pad_mask.sum().item()

            pred = codec.reconstruct(ids)

            orig_ids = ids[0, :valid_len].cpu().tolist()
            pred_ids = pred[0, :valid_len].cpu().tolist()

            orig_text = tokenizer.decode(orig_ids, skip_special=True)
            pred_text = tokenizer.decode(pred_ids, skip_special=True)

            match = orig_text == pred_text
            print(f"\n[{i+1}] {'MATCH' if match else 'DIFF'}")
            print(f"  원문: {orig_text[:100]}{'...' if len(orig_text) > 100 else ''}")
            print(f"  복원: {pred_text[:100]}{'...' if len(pred_text) > 100 else ''}")
            if not match:
                # 첫 번째 차이 위치
                for j, (a, b) in enumerate(zip(orig_text, pred_text)):
                    if a != b:
                        ctx_start = max(0, j - 5)
                        ctx_end = min(len(orig_text), j + 10)
                        print(f"  첫 차이 위치 {j}: "
                              f"'{orig_text[ctx_start:ctx_end]}' vs "
                              f"'{pred_text[ctx_start:ctx_end]}'")
                        break


def main():
    parser = argparse.ArgumentParser(description="Conv Codec 평가")

    parser.add_argument("--checkpoint", default=None, help="체크포인트 경로")
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="jamo")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--n_show", type=int, default=5)

    # 모델 (체크포인트 없을 때)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)

    # 모델 로드
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        codec = ConvCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=saved_args.get("d_model", args.d_model),
            stride=saved_args.get("stride", args.stride),
            n_layers=saved_args.get("n_layers", args.n_layers),
            kernel_size=saved_args.get("kernel_size", args.kernel_size),
        ).to(device)
        codec.load_state_dict(ckpt["model"])
        step = ckpt.get("step", "?")
        print(f"체크포인트 로드: {args.checkpoint} (step {step})")
    else:
        codec = ConvCodec(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            stride=args.stride,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
        ).to(device)
        print("체크포인트 없음 — 랜덤 초기화 모델로 평가")

    n_params = sum(p.numel() for p in codec.parameters())
    print(f"모델: d={codec.d_model}, stride={codec.stride}, params={n_params/1e6:.2f}M")
    print(f"토크나이저: {args.tokenizer} (vocab={tokenizer.vocab_size})")

    # 1. 복원 정확도
    print("\n=== 복원 정확도 ===")
    metrics = evaluate(
        codec, tokenizer, args.corpus, args.text_key,
        args.max_seq_len, args.batch_size, device, args.max_samples,
    )
    print(f"  토큰 정확도:   {metrics['token_acc']*100:.2f}%")
    print(f"  문자 정확도:   {metrics['char_acc']*100:.2f}%")
    print(f"  시퀀스 EM:     {metrics['seq_exact_match']*100:.2f}%")
    print(f"  평균 loss:     {metrics['avg_loss']:.4f}")
    print(f"  평가 시퀀스:   {metrics['n_sequences']}")
    print(f"  평가 토큰:     {metrics['n_tokens']}")

    # 2. 레이턴시
    print("\n=== 레이턴시 (GPU) ===" if device.type == "cuda" else "\n=== 레이턴시 (CPU) ===")
    for seq_len in [256, 512, 1024, 2048]:
        lat = measure_latency(codec, tokenizer, device, seq_len=seq_len)
        print(f"  seq={seq_len}: encode {lat['encode_ms']:.2f}ms, "
              f"decode {lat['decode_ms']:.2f}ms, "
              f"total {lat['total_ms']:.2f}ms "
              f"({seq_len}→{lat['compressed_len']})")

    # 3. 복원 예시
    if args.n_show > 0:
        show_samples(
            codec, tokenizer, args.corpus, args.text_key, device,
            args.max_seq_len, args.n_show,
        )


if __name__ == "__main__":
    main()
