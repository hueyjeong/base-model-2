"""SimpleCodec 평가 — per-token 복원 정확도 + BBPE 토큰별 오류 분포.

각 BBPE 토큰을 독립 샘플로 취급:
- jamo_acc_valid: 실 자모 위치 정확도
- jamo_acc_all: 전 슬롯 (PAD 포함) 정확도
- per-BBPE-id ok/fail (토큰 전체 슬롯이 모두 맞아야 success)
- 카테고리별 (hangul/ascii/byte/special/other)

사용 예:
    python exp-jamo-codec/eval_simple.py \\
        --checkpoint exp-jamo-codec/checkpoints_simple/simple_codec_final.pt \\
        --corpus corpus/k-exaone_coverage_5_len1000.parquet \\
        --max_samples 50000 --batch_size 1024
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

from codec.simple_codec import SimpleCodec
from data.simple_dataset import SimpleJamoDataset, load_bbpe_tokenizer
from tok.jamo_tokenizer import JamoTokenizer


def _build_bbpe_token_names(bbpe_tok):
    """BBPE 토큰 ID → 표시 이름 + 카테고리. eval_composition.py 와 동일 로직."""
    import re
    vocab_size = len(bbpe_tok)
    names, cats = {}, {}
    special_ids = set(bbpe_tok.all_special_ids)

    # batch_decode (Rust parallel)
    decoded_list = bbpe_tok.batch_decode([[tid] for tid in range(vocab_size)])

    for tid, dec in enumerate(decoded_list):
        tok_str = bbpe_tok.convert_ids_to_tokens(tid)
        if tok_str is None:
            tok_str = f"<id:{tid}>"
        names[tid] = tok_str

        if tid in special_ids:
            cats[tid] = "special"
        elif tok_str.startswith("<0x") and tok_str.endswith(">"):
            cats[tid] = "byte"
        elif any('\uAC00' <= c <= '\uD7A3' or
                 '\u1100' <= c <= '\u11FF' or
                 '\u3130' <= c <= '\u318F' for c in dec):
            cats[tid] = "hangul"
            names[tid] = dec
        elif dec.strip() and all(c.isascii() for c in dec):
            cats[tid] = "ascii"
            names[tid] = dec
        else:
            cats[tid] = "other"
            names[tid] = dec

    return names, cats, vocab_size


def _print_stats(token_ok: np.ndarray, token_fail: np.ndarray,
                 names: dict, cats: dict, vocab_size: int):
    """BBPE 토큰별 집계 — 카테고리 + 실패율 내림차순."""
    total = token_ok + token_fail
    fail_rate = np.zeros(vocab_size, dtype=np.float64)
    nz = total > 0
    fail_rate[nz] = token_fail[nz] / total[nz]
    order = np.lexsort((-total, -fail_rate))

    # 카테고리별 집계
    cat_ok, cat_fail = {}, {}
    for i in range(vocab_size):
        c = cats.get(i, "unknown")
        cat_ok[c] = cat_ok.get(c, 0) + int(token_ok[i])
        cat_fail[c] = cat_fail.get(c, 0) + int(token_fail[i])
    n_appeared = int(np.sum(total > 0))

    print(f"\n{'='*80}")
    print(f"=== 카테고리별 복원 정확도 (BBPE 토큰 단위, 토큰 전체 슬롯 = success) ===")
    print(f"{'카테고리':<12} {'성공':<12} {'실패':<12} {'정확도':>8}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for c in ["hangul", "ascii", "byte", "special", "other"]:
        ok, fl = cat_ok.get(c, 0), cat_fail.get(c, 0)
        t = ok + fl
        acc = ok / max(t, 1) * 100
        print(f"{c:<12} {ok:<12,} {fl:<12,} {acc:>7.3f}%")
    print(f"\n  출현 토큰: {n_appeared:,} / {vocab_size:,}  (미출현: {vocab_size - n_appeared:,})")

    print(f"\n{'='*80}")
    print(f"=== BBPE 토큰별 오류 분포 (실패율 내림차순) ===")
    print(f"{'ID':>7}  {'토큰':<24} {'카테고리':<10} {'성공':>10} {'실패':>10} {'합계':>10} {'실패율':>8}")
    print(f"{'-'*7}  {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for idx in order:
        ok, fl = int(token_ok[idx]), int(token_fail[idx])
        t = ok + fl
        if t == 0:
            continue
        fr = fail_rate[idx] * 100
        name = names.get(idx, f"?:{idx}")
        cat = cats.get(idx, "unknown")
        if len(name) > 22:
            name = name[:20] + ".."
        marker = " !!!" if fr > 10 and t > 0 else ""
        print(f"{idx:>7}  {name:<24} {cat:<10} {ok:>10,} {fl:>10,} {t:>10,} {fr:>7.3f}%{marker}")


def main():
    parser = argparse.ArgumentParser(description="SimpleCodec 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="처리할 텍스트 라인 수 상한 (각 라인은 여러 토큰 생성)")
    parser.add_argument("--max_tokens", type=int, default=0,
                        help="처리할 토큰 수 상한 (0=무제한). max_samples 보다 우선")
    parser.add_argument("--show_errors", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true",
                        help="BF16 autocast (학습과 동일)")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 토크나이저
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    print(f"BBPE vocab: {len(bbpe):,}", flush=True)

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    d = saved.get("d_model", 256)
    el = saved.get("n_enc_layers", 5)
    dl = saved.get("n_dec_layers", 5)
    k = saved.get("kernel_size", 5)
    max_jamo = saved.get("max_jamo", 32)

    codec = SimpleCodec(
        jamo_vocab=jamo.vocab_size,
        d_model=d, n_enc_layers=el, n_dec_layers=dl,
        kernel_size=k, max_jamo=max_jamo,
        dropout=0.0,
    ).to(device)
    codec.eval()

    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in sd):
        sd = {key[len(prefix):] if key.startswith(prefix) else key: v
              for key, v in sd.items()}
    codec.load_state_dict(sd)
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in codec.parameters())
    print(f"모델: d={d}, enc_L={el}, dec_L={dl}, k={k}, max_jamo={max_jamo}, "
          f"params={n_params/1e6:.2f}M (step {step})")

    if args.compile:
        print("torch.compile 적용...", flush=True)
        codec = torch.compile(codec)
        # 워밍업
        dummy = torch.zeros(4, max_jamo, dtype=torch.long, device=device)
        dmask = torch.ones(4, max_jamo, dtype=torch.bool, device=device)
        with torch.no_grad():
            codec(dummy, dmask)

    # Dataset (스트리밍, 토큰 단위 yield)
    # max_samples 제한을 위해 기존 parquet 파일에서 직접 읽고 bbpe 처리
    dataset = SimpleJamoDataset(
        file_paths=args.corpus,
        bbpe_tokenizer=bbpe,
        jamo_tokenizer=jamo,
        max_jamo=max_jamo,
        text_key=args.text_key,
    )
    dataset._prewarm_cache(verbose=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True)

    # 통계
    vocab_size = len(bbpe)
    bbpe_ok = np.zeros(vocab_size, dtype=np.int64)
    bbpe_fail = np.zeros(vocab_size, dtype=np.int64)
    total_correct_valid = 0
    total_valid = 0
    total_correct_all = 0
    total_slots = 0
    n_tokens = 0
    errors = []

    # 카테고리/이름
    print("BBPE 토큰 이름 빌드 중...", flush=True)
    names, cats, _ = _build_bbpe_token_names(bbpe)

    # 추론
    print(f"추론 시작 (batch={args.batch_size})...", flush=True)
    t0 = time.time()

    ac_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
              if (device.type == "cuda" and args.bf16)
              else torch.autocast("cuda", enabled=False))

    last_report = 0
    with torch.no_grad():
        for batch in loader:
            jamo_ids = batch["jamo_ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            bbpe_ids = batch["bbpe_id"]  # CPU tensor

            with ac_ctx:
                out = codec(jamo_ids, mask)
                pred = out["logits"].argmax(dim=-1)  # [T, S]

            target_all = jamo_ids.clone()
            target_all[~mask] = 0
            correct_all = (pred == target_all)
            correct_valid = correct_all & mask

            total_correct_all += correct_all.sum().item()
            total_slots += target_all.numel()
            total_correct_valid += correct_valid.sum().item()
            total_valid += mask.sum().item()

            # 토큰 단위 success: 전 슬롯 정답이면 OK
            token_ok_mask = correct_all.all(dim=-1)  # [T]
            bbpe_np = bbpe_ids.numpy()
            ok_np = token_ok_mask.cpu().numpy()

            # np.add.at 로 fast 누적
            np.add.at(bbpe_ok, bbpe_np[ok_np], 1)
            np.add.at(bbpe_fail, bbpe_np[~ok_np], 1)

            # 오류 예시
            if len(errors) < args.show_errors:
                fail_idx = np.where(~ok_np)[0]
                for fi in fail_idx:
                    if len(errors) >= args.show_errors:
                        break
                    bid = int(bbpe_np[fi])
                    valid_len = int(mask[fi].sum().item())
                    gt_full = jamo_ids[fi].tolist()
                    pr_full = pred[fi].tolist()
                    gt_cut = gt_full[:valid_len]
                    pr_cut = pr_full[:pr_full.index(0)] if 0 in pr_full else pr_full
                    gt_str = jamo.decode(gt_cut, skip_special=False)
                    pr_str = jamo.decode(pr_cut, skip_special=False)
                    if gt_str != pr_str:
                        errors.append((names.get(bid, f"id:{bid}"), gt_str, pr_str))

            n_tokens += jamo_ids.size(0)
            if args.max_tokens and n_tokens >= args.max_tokens:
                break

            elapsed = time.time() - t0
            if elapsed - last_report >= 2.0:
                rate = n_tokens / max(elapsed, 1e-6)
                print(f"\r추론: {n_tokens:,} 토큰 ({elapsed:.1f}s, {rate:.0f} 토큰/s)",
                      end="", flush=True)
                last_report = elapsed

    print()
    t_total = time.time() - t0

    # 출력
    acc_valid = total_correct_valid / max(total_valid, 1) * 100
    acc_all = total_correct_all / max(total_slots, 1) * 100
    total_fail_tokens = int(bbpe_fail.sum())
    total_ok_tokens = int(bbpe_ok.sum())
    token_acc = total_ok_tokens / max(total_ok_tokens + total_fail_tokens, 1) * 100

    print(f"\n{'='*80}")
    print(f"=== 복원 정확도 ===")
    print(f"{'='*80}")
    print(f"  총 토큰:        {n_tokens:,}")
    print(f"  총 실자모:      {total_valid:,}")
    print(f"  총 슬롯:        {total_slots:,}  (실자모 + PAD)")
    print(f"  실자모 정확도:  {acc_valid:.6f}%  (slot 기준, PAD 제외)")
    print(f"  전체 정확도:    {acc_all:.6f}%  (slot 기준, PAD 포함)")
    print(f"  토큰 정확도:    {token_acc:.6f}%  (전 슬롯 맞아야 success)")
    print(f"  실패 토큰:      {total_fail_tokens:,}")
    print(f"  소요: {t_total:.1f}s ({n_tokens/max(t_total,1e-6):.0f} 토큰/s)")

    if errors:
        print(f"\n=== 오류 샘플 (최대 {args.show_errors}) ===")
        for i, (tok_name, gt, pr) in enumerate(errors):
            print(f"  [{i+1}] bbpe={tok_name!r}: {gt!r} → {pr!r}")

    _print_stats(bbpe_ok, bbpe_fail, names, cats, vocab_size)


if __name__ == "__main__":
    main()
