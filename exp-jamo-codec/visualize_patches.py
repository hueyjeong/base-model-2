"""엔트로피 패치 경계 시각화

체크포인트를 로드해서 예시 텍스트를 슬래시(/)로 잘라 보여줌.
모델 크기별 비교 + threshold별 비교 지원.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codec.entropy_codec import SmallLM, compute_patch_boundaries
from train_codec import load_tokenizer


SAMPLE_TEXTS = [
    "대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다.",
    "오늘 날씨가 참 좋습니다. 산책하러 나가볼까요?",
    "맞춤법을 확인해 주세요. 감사합니다.",
    "인공지능이 빠르게 발전하고 있다. 특히 자연어 처리 분야에서 큰 성과를 거두고 있다.",
    "김철수 씨가 프로그래밍을 배우기 시작했습니다.",
    "맞춤뻡을 확인해 주세요.",
    "안녕하세요 반갑습니다.",
    "서울특별시 강남구 테헤란로 123번지에 위치한 주식회사 가나다라입니다.",
]


def load_model(checkpoint_path, tokenizer, device):
    """체크포인트에서 SmallLM 로드"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    d = saved_args.get("entropy_d_model", 128)
    nl = saved_args.get("entropy_n_layers", 2)
    nh = saved_args.get("entropy_n_heads", 4)

    model = SmallLM(
        vocab_size=tokenizer.vocab_size, d_model=d, n_layers=nl, n_heads=nh,
    ).to(device)

    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    step = ckpt.get("step", "?")
    return model, f"d={d} L={nl} ({n_params/1e6:.1f}M, step {step})"


def visualize(model, tokenizer, device, thresholds=(4.0, 6.0, 8.0, 12.0)):
    """텍스트를 슬래시로 잘라서 출력"""
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special=True)
        ids_t = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            entropy = model.compute_entropy(ids_t)

        print(f"\n  원문: {text}")

        for thr in thresholds:
            boundaries = compute_patch_boundaries(
                entropy, threshold=thr, min_patch=2, max_patch=32,
            )
            boundary_positions = boundaries[0].cpu().tolist()

            # 바이트를 패치별로 묶어서 디코드
            patches = []
            current_patch_ids = []
            for pos, tid in enumerate(ids):
                if boundary_positions[pos] and current_patch_ids:
                    patches.append(current_patch_ids)
                    current_patch_ids = []
                current_patch_ids.append(tid)
            if current_patch_ids:
                patches.append(current_patch_ids)

            # 각 패치를 디코드
            patch_strs = []
            for patch_ids in patches:
                decoded = tokenizer.decode(patch_ids, skip_special=True)
                patch_strs.append(decoded)

            slashed = "/".join(patch_strs)
            print(f"  thr={thr:>4.0f} ({len(patches):>2d}p): {slashed}")


def get_slashed(model, tokenizer, device, text, thr):
    """텍스트를 슬래시로 잘라서 반환"""
    ids = tokenizer.encode(text, add_special=True)
    ids_t = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        entropy = model.compute_entropy(ids_t)

    boundaries = compute_patch_boundaries(
        entropy, threshold=thr, min_patch=2, max_patch=32,
    )
    boundary_positions = boundaries[0].cpu().tolist()

    patches = []
    current_patch_ids = []
    for pos, tid in enumerate(ids):
        if boundary_positions[pos] and current_patch_ids:
            patches.append(current_patch_ids)
            current_patch_ids = []
        current_patch_ids.append(tid)
    if current_patch_ids:
        patches.append(current_patch_ids)

    patch_strs = []
    for patch_ids in patches:
        decoded = tokenizer.decode(patch_ids, skip_special=True)
        patch_strs.append(decoded)

    return "/".join(patch_strs), len(patches)


def main():
    parser = argparse.ArgumentParser(description="엔트로피 패치 경계 시각화")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="체크포인트 경로 (여러 개 가능)")
    parser.add_argument("--tokenizer", choices=["byte", "jamo", "keyboard"], default="byte")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[4.0, 6.0, 8.0, 12.0])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)

    # 모델 전부 로드
    models = []
    for ckpt_path in args.checkpoints:
        model, desc = load_model(ckpt_path, tokenizer, device)
        label = os.path.basename(ckpt_path).replace("entropy_lm_", "").replace("_final.pt", "")
        models.append((model, label, desc))

    print(f"모델: {', '.join(label for _, label, _ in models)}")
    print()

    # 문장별 → threshold별 → 모델별 비교
    for text in SAMPLE_TEXTS:
        print(f"{'='*70}")
        print(f"  원문: {text}")
        print(f"{'='*70}")

        for thr in args.thresholds:
            print(f"  thr={thr:.0f}:")
            for model, label, _ in models:
                slashed, n_patches = get_slashed(model, tokenizer, device, text, thr)
                print(f"    {label:>12s} ({n_patches:>2d}p): {slashed}")
            print()
        print()


if __name__ == "__main__":
    main()
