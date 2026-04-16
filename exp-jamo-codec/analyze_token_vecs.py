"""analyze_token_vecs.py — CompositionCodec 인코더의 token_vec 공간 분석.

체크포인트의 실제 정보 용량이 얼마나 쓰이고 있는지 진단:
- Effective rank (spectral entropy 기반)
- Participation ratio (스펙트럼 집중도)
- 95% variance rank
- Singular value 분포 요약
- 랜덤 페어 cos similarity 분포

d_model=256 인데 실효 랭크가 ~256 근처면 이미 용량 한계.
~200 이하면 여유 있음 → d_model 확장 효과 제한적일 수 있음.
150 이하면 더 작게 가도 됨.

사용:
    python exp-jamo-codec/analyze_token_vecs.py \\
        --checkpoint exp-jamo-codec/checkpoints_ft_variable/composition_5L_step2000.pt \\
        --corpus corpus/k-exaone_coverage_5_len1000.parquet \\
        --max_samples 5000 --max_seq_len 4096
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.composition_codec import CompositionCodec


def _load_model(checkpoint_path: str, device):
    """체크포인트 로드 + 가변 모드 자동 복원."""
    from tok.jamo_tokenizer import JamoTokenizer
    jamo = JamoTokenizer()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    d = saved_args.get("d_model", 256)
    nl = saved_args.get("n_layers", 5)
    k = saved_args.get("kernel_size", 7)
    seg_masked = saved_args.get("segment_masked", False)
    parallel_decoder = saved_args.get("parallel_decoder", False)
    decoder_layers = saved_args.get("decoder_layers", 2)
    decoder_heads = saved_args.get("decoder_heads", 4)
    max_jpt = saved_args.get("max_jamo_per_token", 32)
    fixed_slot = saved_args.get("fixed_slot", False)
    fixed_output_len = None
    if fixed_slot:
        trained_max_seq = saved_args.get("max_seq_len", 4096)
        fixed_output_len = trained_max_seq // max_jpt

    codec = CompositionCodec(
        jamo_vocab=jamo.vocab_size, d_model=d, n_layers=nl, kernel_size=k,
        segment_masked=seg_masked, parallel_decoder=parallel_decoder,
        decoder_layers=decoder_layers, decoder_heads=decoder_heads,
        max_jamo_per_token=max_jpt, fixed_output_len=fixed_output_len,
    ).to(device)

    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in sd):
        sd = {key[len(prefix):] if key.startswith(prefix) else key: v
              for key, v in sd.items()}
    codec.load_state_dict(sd)
    codec.eval()

    info = {
        "d_model": d, "n_layers": nl, "kernel_size": k,
        "fixed_slot": fixed_slot, "max_jamo_per_token": max_jpt,
        "step": ckpt.get("step", "?"),
        "params_M": sum(p.numel() for p in codec.parameters()) / 1e6,
    }
    return codec, info, jamo


def _tokenize_corpus(corpus_paths, text_key, max_samples, max_seq_len, max_jpt,
                    fixed_slot, append_pad_slot, bbpe_pad_id, n_workers=8):
    """eval_composition.py 의 토큰화 유틸 재사용."""
    from eval_composition import _read_texts, _pre_tokenize_mp
    texts = _read_texts(corpus_paths, text_key)
    texts = texts[:max_samples]
    model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"
    base_path = "/workspace/base-model-2"
    ds = _pre_tokenize_mp(
        texts, max_seq_len, max_jamo_per_token=max_jpt,
        n_workers=n_workers, chunk_size=2048,
        model_id=model_id, base_path=base_path,
        fixed_slot=fixed_slot, append_pad_slot=append_pad_slot,
        jamo_bos=2, jamo_eos=3, jamo_pad=0,
        bbpe_pad_id=bbpe_pad_id,
    )
    return ds


def collect_token_vecs(codec, loader, device, max_vecs=200000):
    """Encoder forward → token_vecs 수집 (유효 세그먼트만).

    Returns:
        [N, D] float32 numpy array
    """
    vecs = []
    total = 0
    raw = codec
    # torch.compile 래핑되어 있으면 원본 모듈로 접근
    encoder = raw.encoder if hasattr(raw, "encoder") else raw._orig_mod.encoder

    with torch.no_grad():
        for batch in loader:
            jamo_ids = batch[0].to(device, non_blocking=True)
            jamo_mask = batch[1].to(device, non_blocking=True)
            segment_ids = batch[2].to(device, non_blocking=True)
            n_segments = batch[3].to(device, non_blocking=True)

            # encoder forward: [B, P, D]
            z = encoder(jamo_ids, jamo_mask, segment_ids, n_segments)
            B, P, D = z.shape

            # 유효 세그먼트만 마스킹: seg_idx < n_segments
            seg_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)
            valid = seg_idx < n_segments.unsqueeze(1)  # [B, P]

            z_valid = z[valid].float().cpu().numpy()  # [N_valid, D]
            vecs.append(z_valid)
            total += len(z_valid)

            if total >= max_vecs:
                break

    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(vecs, axis=0)[:max_vecs]


def analyze(vecs: np.ndarray) -> dict:
    """token_vec 행렬 [N, D] → 랭크/스펙트럼 분석."""
    if vecs.size == 0:
        return {}
    N, D = vecs.shape

    # 평균 제거 (중심화)
    mean_vec = vecs.mean(axis=0, keepdims=True)
    centered = vecs - mean_vec

    # SVD (경제 모드): centered = U S V^T
    t0 = time.time()
    # D 가 작으니 full SVD 가능. N 이 크면 covariance matrix 경유가 더 빠름
    # cov = centered.T @ centered / (N-1) → [D, D]
    cov = centered.T @ centered / max(N - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)  # 오름차순, 음수는 수치 오차
    eigvals = np.clip(eigvals, 0, None)[::-1]  # 내림차순, 음수 클리핑
    svd_time = time.time() - t0

    # 특이값 = sqrt(eigenvalue of cov × (N-1)). 비율 계산이 목적이라 eigval 로 충분.
    total_var = eigvals.sum()
    if total_var <= 0:
        return {"N": N, "D": D, "error": "zero variance"}

    norm_eigvals = eigvals / total_var

    # 1) 95% variance rank
    cumsum = np.cumsum(norm_eigvals)
    rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)

    # 2) Effective rank (Roy & Vetterli 2007): exp(H) where H = -sum p log p
    # p = normalized eigenvalues
    nz = norm_eigvals[norm_eigvals > 1e-12]
    entropy = -(nz * np.log(nz)).sum()
    eff_rank = float(np.exp(entropy))

    # 3) Participation ratio: (sum λ)² / sum λ²
    part_ratio = float((eigvals.sum() ** 2) / (eigvals ** 2).sum())

    # 4) Condition number & top/bottom singular values
    s_max = float(np.sqrt(eigvals[0]))
    # 상위 10% 와 하위 10% 의 eigenvalue 비교
    k10 = max(1, D // 10)
    top10_var = norm_eigvals[:k10].sum()
    bot10_var = norm_eigvals[-k10:].sum()

    # 5) 랜덤 페어 cos similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / np.clip(norms, 1e-8, None)
    n_pairs = min(10000, N * (N - 1) // 2)
    rng = np.random.default_rng(42)
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    cos = (normalized[i] * normalized[j]).sum(axis=1)

    return {
        "N": N,
        "D": D,
        "total_variance": float(total_var),
        "rank_95_var": rank_95,
        "rank_99_var": rank_99,
        "effective_rank": eff_rank,
        "participation_ratio": part_ratio,
        "s_max": s_max,
        "top10_var_ratio": float(top10_var),
        "bot10_var_ratio": float(bot10_var),
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std()),
        "cos_abs_mean": float(np.abs(cos).mean()),
        "svd_time": svd_time,
        "eigvals_top10": [float(x) for x in eigvals[:10]],
        "eigvals_norm_top10": [float(x) for x in norm_eigvals[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description="token_vec 공간 분석")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--max_vecs", type=int, default=200000)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    codec, info, jamo = _load_model(args.checkpoint, device)
    print(f"모델: d={info['d_model']}, L={info['n_layers']}, k={info['kernel_size']}, "
          f"params={info['params_M']:.2f}M, step={info['step']}")
    print(f"  fixed_slot={info['fixed_slot']}, max_jamo_per_token={info['max_jamo_per_token']}")

    # 체크포인트 메타에서 bbpe_pad 및 슬롯 모드 가져오기
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    fixed_slot = saved.get("fixed_slot", False)
    append_pad_slot = saved.get("append_pad_slot", False)
    max_jpt = saved.get("max_jamo_per_token", 32)

    # BBPE pad id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    bbpe_tok = AutoTokenizer.from_pretrained(
        "LGAI-EXAONE/K-EXAONE-236B-A23B", trust_remote_code=True)
    bbpe_pad_id = bbpe_tok.pad_token_id if bbpe_tok.pad_token_id is not None else 0

    # 토큰화
    print(f"토큰화: {args.corpus}")
    t0 = time.time()
    ds = _tokenize_corpus(
        args.corpus, args.text_key, args.max_samples, args.max_seq_len,
        max_jpt, fixed_slot, append_pad_slot, bbpe_pad_id, args.n_workers,
    )
    print(f"  {len(ds):,} 샘플 ({time.time()-t0:.1f}s)")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # token_vecs 수집
    print(f"Encoder forward + token_vec 수집 (최대 {args.max_vecs:,}개)...")
    t0 = time.time()
    vecs = collect_token_vecs(codec, loader, device, args.max_vecs)
    print(f"  수집 완료: {len(vecs):,}개 벡터, D={vecs.shape[1]}, {time.time()-t0:.1f}s")

    # 분석
    print(f"SVD 분석 중...")
    metrics = analyze(vecs)

    # 출력
    print()
    print("=" * 70)
    print(f"=== Token vector 공간 분석 (N={metrics['N']:,}, D={metrics['D']}) ===")
    print("=" * 70)
    print(f"총 분산:              {metrics['total_variance']:.4f}")
    print(f"95% variance rank:     {metrics['rank_95_var']:>5d} / {metrics['D']} "
          f"({metrics['rank_95_var']/metrics['D']*100:.1f}%)")
    print(f"99% variance rank:     {metrics['rank_99_var']:>5d} / {metrics['D']} "
          f"({metrics['rank_99_var']/metrics['D']*100:.1f}%)")
    print(f"Effective rank (exp H):{metrics['effective_rank']:>8.2f} / {metrics['D']} "
          f"({metrics['effective_rank']/metrics['D']*100:.1f}%)")
    print(f"Participation ratio:   {metrics['participation_ratio']:>8.2f} / {metrics['D']} "
          f"({metrics['participation_ratio']/metrics['D']*100:.1f}%)")
    print(f"상위 10% eigenval 분산: {metrics['top10_var_ratio']*100:>6.2f}%  "
          f"(d=25.6 차원이 {metrics['top10_var_ratio']*100:.1f}% 설명)")
    print(f"하위 10% eigenval 분산: {metrics['bot10_var_ratio']*100:>6.2f}%")
    print()
    print(f"랜덤 페어 cos similarity:")
    print(f"  mean = {metrics['cos_mean']:+.4f}  (0 에 가까울수록 uniform 분포)")
    print(f"  |cos| mean = {metrics['cos_abs_mean']:.4f}  (작을수록 방향 다양)")
    print(f"  std = {metrics['cos_std']:.4f}")
    print()
    print(f"상위 10 eigenvalue (정규화):")
    for i, (ev, nev) in enumerate(zip(metrics['eigvals_top10'],
                                       metrics['eigvals_norm_top10'])):
        print(f"  [{i+1:2d}] {ev:>10.4f} ({nev*100:>5.2f}% of total)")
    print()

    # 해석
    print("=" * 70)
    print("=== 해석 ===")
    print("=" * 70)
    eff = metrics['effective_rank']
    D = metrics['D']
    ratio = eff / D
    if ratio > 0.9:
        print(f"⚠️  Effective rank = {eff:.1f}/{D} ({ratio*100:.0f}%): "
              f"정보 용량 포화 상태. d_model 확장이 가장 직접적 해법.")
    elif ratio > 0.7:
        print(f"🔶 Effective rank = {eff:.1f}/{D} ({ratio*100:.0f}%): "
              f"용량 활용도 높음. d_model 확장이 효과적.")
    elif ratio > 0.5:
        print(f"✅ Effective rank = {eff:.1f}/{D} ({ratio*100:.0f}%): "
              f"중간 활용도. 학습 더 하면 여유 차원 채워질 수 있음.")
    else:
        print(f"🟢 Effective rank = {eff:.1f}/{D} ({ratio*100:.0f}%): "
              f"여유 많음. d_model 줄여도 될 수 있음.")

    p95 = metrics['rank_95_var'] / D
    if p95 > 0.9:
        print(f"⚠️  95% var 를 {metrics['rank_95_var']}/{D} 차원이 설명 — "
              f"정보가 전 차원에 고르게 퍼져 있음. bottleneck.")
    elif p95 < 0.5:
        print(f"🟢 95% var 를 {metrics['rank_95_var']}/{D} 차원만으로 설명 — "
              f"저차원 부분 공간에 집중.")


if __name__ == "__main__":
    main()
