"""Codec encoder 이웃 토큰 리크 측정.

아이디어: 문장의 토큰 벡터들을 codec encoder로 얻은 뒤, 특정 토큰 벡터 하나만
남기고 나머지를 0으로 만들어 codec decoder에 넣어 복원 시도. 복원된 자모가
이웃 토큰의 실제 자모와 얼마나 일치하는지로 conv receptive field 리크 측정.

비교:
  - Self-reconstruction: vec_i → 토큰 i 자모 복원 (원래 codec task)
  - Leak 측정: vec_i → 토큰 i±1, i±2, ... 자모 복원 (리크 여부)

리크 없으면 이웃 토큰 복원 정확도는 random(1/330 ≈ 0.3%) 수준이어야 한다.
높으면 hidden에 이웃 토큰 정보가 "쓸 만한 형태로" 실려있다는 뜻.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_THIS = os.path.abspath(os.path.dirname(__file__))
_EXP_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)

from data.bbpe_jamo_dataset import BBPEJamoDataset, load_bbpe_tokenizer  # noqa: E402
from tok.jamo_tokenizer import JamoTokenizer  # noqa: E402
from codec.composition_codec import CompositionEncoder, CompositionDecoder  # noqa: E402


def run_leak_test(
    codec_ckpt: str,
    parquet: str,
    n_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offsets: tuple = (-4, -3, -2, -1, 0, 1, 2, 3, 4),
):
    """Codec에서 이웃 토큰 리크 측정.

    Args:
        codec_ckpt: composition_6L_step*.pt 경로
        parquet: 테스트 데이터
        n_samples: 몇 문서로 평균 낼지
        offsets: 중심 토큰에서 복원하려는 이웃 오프셋 (0=self)

    Returns:
        dict {offset: accuracy}
    """
    # Codec 로드
    enc = CompositionEncoder(
        jamo_vocab=330, d_model=256, n_layers=6, kernel_size=7,
        dropout=0.0, max_jamo_per_token=32,
    ).to(device)
    dec = CompositionDecoder(
        jamo_vocab=330, d_model=256, n_layers=6, kernel_size=7,
        dropout=0.0, max_jamo_per_token=32,
    ).to(device)

    ckpt = torch.load(codec_ckpt, map_location=device, weights_only=False)
    sd = ckpt["model"]
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                         if k.startswith("encoder.")})
    dec.load_state_dict({k[len("decoder."):]: v for k, v in sd.items()
                         if k.startswith("decoder.")})
    enc.eval(); dec.eval()

    # Dataset (single-document으로 테스트 — atomic packing 없이)
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    ds = BBPEJamoDataset([parquet], bbpe, jamo,
                          max_seq_len=2048, max_patches=512,
                          text_key="text")

    # offset별 정확도 누적
    correct = {o: 0 for o in offsets}
    total = {o: 0 for o in offsets}
    full_recon_correct = 0  # vec_i만으로 토큰 i 복원 (self)
    full_recon_total = 0

    with torch.no_grad():
        seen = 0
        for sample in ds:
            if seen >= n_samples:
                break
            jamo_ids = sample["jamo_ids"].unsqueeze(0).to(device)  # [1, L]
            jamo_mask = sample["jamo_mask"].unsqueeze(0).to(device)
            segment_ids = sample["segment_ids"].unsqueeze(0).to(device)
            n_seg = sample["n_segments"]
            if n_seg < 10:
                continue  # 짧은 문서 제외

            # Encoder forward
            z = enc(jamo_ids, jamo_mask, segment_ids,
                    torch.tensor([n_seg], device=device))  # [1, max_seg, D]

            # 각 토큰 i를 중심으로 offset 이웃 복원
            L = jamo_ids.size(1)
            for i in range(n_seg):
                # "이웃 조사": vec_i만 남기고 나머지 0
                z_masked = torch.zeros_like(z)
                z_masked[0, i] = z[0, i]

                # Decoder로 복원 (전체 자모 위치에 대해 logits)
                logits = dec(z_masked, segment_ids, L, jamo_mask)  # [1, L, V]
                preds = logits.argmax(-1)[0]  # [L]

                # 각 offset에 대해 정답 비교
                for o in offsets:
                    j = i + o
                    if j < 0 or j >= n_seg:
                        continue
                    # 토큰 j에 속한 자모 위치들
                    token_j_mask = (segment_ids[0] == j) & jamo_mask[0]
                    if not token_j_mask.any():
                        continue
                    tgt = jamo_ids[0][token_j_mask]
                    pred = preds[token_j_mask]
                    correct[o] += (pred == tgt).sum().item()
                    total[o] += tgt.numel()

            # Self reconstruction (전체 z 사용, codec 원래 task)
            logits_full = dec(z, segment_ids, L, jamo_mask)
            preds_full = logits_full.argmax(-1)[0]
            valid = jamo_mask[0]
            full_recon_correct += (preds_full[valid] == jamo_ids[0][valid]).sum().item()
            full_recon_total += valid.sum().item()

            seen += 1

    results = {o: correct[o] / max(total[o], 1) for o in offsets}
    full_acc = full_recon_correct / max(full_recon_total, 1)

    print(f"\n=== Conv receptive field 리크 측정 ===")
    print(f"(codec: {codec_ckpt})")
    print(f"(samples: {seen} 문서)")
    print(f"\nSelf reconstruction (전체 z → 전체 자모): {full_acc:.4%}")
    print(f"Random baseline (1/330 vocab): {1/330:.4%}")
    print(f"\n[Leak] vec_i만 주고 토큰 i+offset 자모 복원 정확도:")
    print(f"  {'offset':>7} | {'accuracy':>10} | {'n_tokens':>10} | {'해석':<30}")
    print(f"  {'-'*7} | {'-'*10} | {'-'*10} | {'-'*30}")
    for o in offsets:
        acc = results[o]
        n = total[o]
        if o == 0:
            interp = "self (vec_i 자기 토큰 복원)"
        elif abs(o) == 1:
            interp = "바로 옆 토큰 — 가장 큰 리크"
        elif abs(o) <= 3:
            interp = "RF 내부 — 리크 가능"
        else:
            interp = "RF 경계 — 리크 미약"
        print(f"  {o:>+7d} | {acc:>9.2%} | {n:>10d} | {interp}")
    print(f"\n해석 가이드:")
    print(f"  - offset=0에서 99%+ 여야 codec self-recon 정상")
    print(f"  - offset=±1에서 10%+ 이면 이웃 토큰 정보가 상당히 유출")
    print(f"  - offset=±1이 {1/330*100:.2f}%(random) 근처면 리크 없음")
    print(f"  - offset=±1이 5%~30%면 'blurry context' 수준 (걱정 적음)")
    print(f"  - offset=±1이 50%+면 'identity 유출' (걱정 큼, 구조 수정 권장)")
    return results, full_acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec_ckpt", type=str,
                    default="exp-jamo-codec/checkpoints/composition_6L_step600000.pt")
    ap.add_argument("--parquet", type=str, default="corpus/jamo-codec-v3/val.parquet")
    ap.add_argument("--n_samples", type=int, default=100)
    args = ap.parse_args()
    run_leak_test(args.codec_ckpt, args.parquet, args.n_samples)
