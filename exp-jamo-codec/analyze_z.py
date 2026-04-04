"""z 공간 분석 — 오타 쌍의 거리, 의미 구조 검증

핵심 질문:
1. 오타와 정답의 z 거리가 가까운가?
2. 압축률이 높을수록 z 공간에서 오타/정답이 더 가까워지는가?
3. 의미가 다른 문장 간 z 거리 vs 오타 쌍 z 거리 — 분리되는가?
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.conv_codec import ConvCodec
from codec.xattn_codec import CrossAttentionCodec
from train_codec import load_tokenizer


# ── 테스트 쌍 ──────────────────────────────────────────────────────────

# (정답, 오타, 설명)
TYPO_PAIRS = [
    ("맞춤법", "맞춤뻡", "종성 오타"),
    ("습니다", "스빈다", "전형적 오타 패턴"),
    ("확인해", "확인헤", "모음 오타"),
    ("대한민국", "대한민귝", "종성 추가"),
    ("감사합니다", "감사함니다", "자음 오타"),
    ("안녕하세요", "안녕하새요", "모음 오타"),
    ("프로그래밍", "프로그레밍", "모음 오타"),
    ("인공지능", "인공지늉", "종성 오타"),
    ("김철수", "김철쑤", "쌍자음 오류"),
    ("고마워요", "고마워욬", "종성 추가 오류"),
]

# 의미가 다른 쌍 (거리가 멀어야 정상)
DIFFERENT_PAIRS = [
    ("맞춤법을 확인해 주세요", "오늘 날씨가 좋습니다"),
    ("대한민국은 민주공화국이다", "사과는 빨간색이다"),
    ("프로그래밍을 배우고 있습니다", "저녁에 치킨을 먹었다"),
    ("감사합니다", "안녕하세요"),
    ("인공지능이 발전하고 있다", "고양이가 귀엽다"),
]


def encode_text(codec, tokenizer, text, device):
    """텍스트 → z 벡터"""
    ids = tokenizer.encode(text, add_special=True)
    ids_t = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        z = codec.encode(ids_t)  # [1, L//s, d]
    return z.squeeze(0)  # [L//s, d]


def cosine_sim(z1, z2):
    """두 z 시퀀스의 평균 코사인 유사도"""
    # 길이 맞추기
    min_len = min(z1.size(0), z2.size(0))
    z1 = z1[:min_len]
    z2 = z2[:min_len]
    # 위치별 코사인 유사도의 평균
    sim = F.cosine_similarity(z1, z2, dim=-1)
    return sim.mean().item()


def l2_dist(z1, z2):
    """두 z 시퀀스의 평균 L2 거리"""
    min_len = min(z1.size(0), z2.size(0))
    z1 = z1[:min_len]
    z2 = z2[:min_len]
    dist = (z1 - z2).norm(dim=-1)
    return dist.mean().item()


def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    codec_type = saved_args.get("codec", "conv")
    d = saved_args.get("d_model", args.d_model)
    s = saved_args.get("stride", args.stride)
    if codec_type == "xattn":
        codec = CrossAttentionCodec(
            vocab_size=tokenizer.vocab_size, d_model=d, stride=s,
            n_local_layers=saved_args.get("n_layers", 2),
            n_heads=saved_args.get("n_heads", 4),
        ).to(device)
    else:
        codec = ConvCodec(
            vocab_size=tokenizer.vocab_size, d_model=d, stride=s,
            n_layers=saved_args.get("n_layers", 3),
            kernel_size=saved_args.get("kernel_size", 5),
        ).to(device)
    codec.load_state_dict(ckpt["model"])
    codec.eval()

    stride = saved_args.get("stride", args.stride)
    print(f"모델: stride={stride}, d={saved_args.get('d_model', args.d_model)}")
    print(f"토크나이저: {args.tokenizer}")

    # 1. 오타 쌍 분석
    print(f"\n{'='*60}")
    print("오타 쌍 z 거리")
    print(f"{'='*60}")
    print(f"{'정답':<12} {'오타':<12} {'cos_sim':>8} {'L2':>8} {'설명'}")
    print("-" * 60)

    typo_cos_sims = []
    typo_l2_dists = []
    for correct, typo, desc in TYPO_PAIRS:
        z_c = encode_text(codec, tokenizer, correct, device)
        z_t = encode_text(codec, tokenizer, typo, device)
        cos = cosine_sim(z_c, z_t)
        l2 = l2_dist(z_c, z_t)
        typo_cos_sims.append(cos)
        typo_l2_dists.append(l2)
        print(f"{correct:<12} {typo:<12} {cos:>8.4f} {l2:>8.4f} {desc}")

    avg_typo_cos = sum(typo_cos_sims) / len(typo_cos_sims)
    avg_typo_l2 = sum(typo_l2_dists) / len(typo_l2_dists)
    print(f"\n평균 오타 쌍: cos_sim={avg_typo_cos:.4f}, L2={avg_typo_l2:.4f}")

    # 2. 다른 의미 쌍 분석
    print(f"\n{'='*60}")
    print("다른 의미 쌍 z 거리 (멀어야 정상)")
    print(f"{'='*60}")

    diff_cos_sims = []
    diff_l2_dists = []
    for text_a, text_b in DIFFERENT_PAIRS:
        z_a = encode_text(codec, tokenizer, text_a, device)
        z_b = encode_text(codec, tokenizer, text_b, device)
        cos = cosine_sim(z_a, z_b)
        l2 = l2_dist(z_a, z_b)
        diff_cos_sims.append(cos)
        diff_l2_dists.append(l2)
        a_short = text_a[:15] + "..." if len(text_a) > 15 else text_a
        b_short = text_b[:15] + "..." if len(text_b) > 15 else text_b
        print(f"  {a_short:<18} vs {b_short:<18} cos={cos:.4f} L2={l2:.4f}")

    avg_diff_cos = sum(diff_cos_sims) / len(diff_cos_sims)
    avg_diff_l2 = sum(diff_l2_dists) / len(diff_l2_dists)
    print(f"\n평균 다른 의미: cos_sim={avg_diff_cos:.4f}, L2={avg_diff_l2:.4f}")

    # 3. 요약
    print(f"\n{'='*60}")
    print("요약")
    print(f"{'='*60}")
    print(f"  오타 쌍 평균 cos_sim:   {avg_typo_cos:.4f}")
    print(f"  다른 의미 평균 cos_sim: {avg_diff_cos:.4f}")
    print(f"  분리도 (차이):          {avg_typo_cos - avg_diff_cos:.4f}")
    print()
    print(f"  오타 쌍 평균 L2:        {avg_typo_l2:.4f}")
    print(f"  다른 의미 평균 L2:      {avg_diff_l2:.4f}")
    print(f"  분리도 (비율):          {avg_diff_l2 / max(avg_typo_l2, 1e-6):.2f}x")

    if avg_typo_cos > avg_diff_cos:
        print("\n  → 오타 쌍이 다른 의미보다 z 공간에서 더 가까움 (원하는 성질)")
    else:
        print("\n  → 오타 쌍과 다른 의미의 z 거리가 구분되지 않음 (z가 의미를 인코딩하지 못함)")


def main():
    parser = argparse.ArgumentParser(description="z 공간 분석")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="jamo")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
