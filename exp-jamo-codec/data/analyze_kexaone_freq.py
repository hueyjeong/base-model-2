"""k-exaone_coverage_100_freq.npz 분석 → 토큰 분포 보고서.

- 내림차순 정렬 (doc_freq 기준)
- zero-sample 토큰 확인
- 분포 구간별 분포
- 상/하위 N 토큰 미리보기
- 전체 정렬 테이블을 CSV로 저장
"""
import argparse
import csv

import numpy as np
from transformers import AutoTokenizer


def fmt(n):
    return f"{n:,}"


def decode_piece(tok, tid: int) -> str:
    """BBPE 토큰을 원문자로 복원.

    - special token: added_tokens_decoder 에서 그대로
    - 일반 토큰: decode([tid])로 utf-8 복원 (Ġ→공백 등)
    """
    # added / special 우선 처리 (decode가 공백처리할 수 있음)
    added = getattr(tok, "added_tokens_decoder", {}) or {}
    if tid in added:
        return str(added[tid].content if hasattr(added[tid], "content") else added[tid])
    try:
        s = tok.decode([tid], skip_special_tokens=False,
                       clean_up_tokenization_spaces=False)
        if s:
            return s
    except Exception:
        pass
    # fallback: raw piece
    return str(tok.convert_ids_to_tokens(tid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="corpus/k-exaone_coverage_100_freq.npz")
    ap.add_argument("--tokenizer_id", default="LGAI-EXAONE/K-EXAONE-236B-A23B")
    ap.add_argument("--out_csv", default="corpus/k-exaone_coverage_100_freq.csv")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--bottom", type=int, default=30)
    args = ap.parse_args()

    data = np.load(args.freq)
    doc_freq = data["doc_freq"]
    token_freq = data["token_freq"]
    V = int(data["vocab_size"])
    n_rows = int(data["n_rows_processed"])
    print(f"[load] vocab={V}, rows_processed={fmt(n_rows)}")
    print(f"[load] total tokens counted = {fmt(int(token_freq.sum()))}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    # ── 1. 기본 통계 ──
    zero = int((doc_freq == 0).sum())
    nz = V - zero
    print("\n=== 기본 통계 ===")
    print(f"vocab_size       : {fmt(V)}")
    print(f"nonzero tokens   : {fmt(nz)}  ({nz / V * 100:.2f}%)")
    print(f"ZERO-sample tok  : {fmt(zero)}  ({zero / V * 100:.2f}%)")
    print(f"doc_freq  min/mean/median/max = "
          f"{fmt(int(doc_freq.min()))} / "
          f"{doc_freq.mean():.1f} / "
          f"{fmt(int(np.median(doc_freq)))} / "
          f"{fmt(int(doc_freq.max()))}")
    print(f"token_freq min/mean/median/max = "
          f"{fmt(int(token_freq.min()))} / "
          f"{token_freq.mean():.1f} / "
          f"{fmt(int(np.median(token_freq)))} / "
          f"{fmt(int(token_freq.max()))}")

    # ── 2. coverage 분포 (doc_freq bin) ──
    print("\n=== doc_freq 구간별 토큰 수 ===")
    bins = [
        (0, 0, "   = 0"),
        (1, 1, "   = 1"),
        (2, 9, "   2 ~ 9"),
        (10, 49, "  10 ~ 49"),
        (50, 99, "  50 ~ 99"),
        (100, 100, "  = 100 (목표)"),
        (101, 499, " 101 ~ 499"),
        (500, 999, " 500 ~ 999"),
        (1000, 4999, " 1k ~ 5k"),
        (5000, 9999, " 5k ~ 10k"),
        (10000, 99999, " 10k ~ 100k"),
        (100000, 10**9, " 100k+"),
    ]
    for lo, hi, label in bins:
        mask = (doc_freq >= lo) & (doc_freq <= hi)
        cnt = int(mask.sum())
        pct = cnt / V * 100
        print(f"  {label:<18s}: {fmt(cnt):>10s}  ({pct:6.2f}%)")

    # ── 3. 상위 top ──
    order = np.argsort(-doc_freq, kind="stable")  # 내림차순
    print(f"\n=== 상위 {args.top}개 토큰 (doc_freq 내림차순) ===")
    print(f"{'rank':>5} {'token_id':>9} {'doc_freq':>12} {'token_freq':>14}  piece")
    for r in range(args.top):
        tid = int(order[r])
        piece = decode_piece(tok, tid)
        raw = str(tok.convert_ids_to_tokens(tid))
        print(f"{r+1:>5} {tid:>9} {fmt(int(doc_freq[tid])):>12} "
              f"{fmt(int(token_freq[tid])):>14}  {piece!r:<16s} (raw={raw!r})")

    # ── 4. 하위 / zero-sample 샘플 ──
    print(f"\n=== ZERO-sample 토큰 예시 (처음 {args.bottom}개) ===")
    zero_ids = np.where(doc_freq == 0)[0]
    print(f"총 zero-sample: {fmt(len(zero_ids))}개")
    print(f"{'token_id':>9}  piece")
    for tid in zero_ids[: args.bottom]:
        piece = decode_piece(tok, int(tid))
        print(f"{tid:>9}  {piece!r}")

    # special token / added token 여부 확인
    special_ids = set(tok.all_special_ids) if hasattr(tok, "all_special_ids") else set()
    zero_special = [t for t in zero_ids.tolist() if t in special_ids]
    print(f"\n  - 그 중 special token: {len(zero_special)}개  ids={zero_special[:20]}")

    # ── 5. 1~99 샘플 (coverage 미달) 분석 ──
    under_mask = (doc_freq >= 1) & (doc_freq < 100)
    under_n = int(under_mask.sum())
    print(f"\n=== coverage<100 토큰 (1~99 샘플) 예시 ===")
    print(f"총 {fmt(under_n)}개 (vocab의 {under_n / V * 100:.2f}%)")
    under_ids = np.where(under_mask)[0]
    # doc_freq 낮은 순 정렬
    under_order = under_ids[np.argsort(doc_freq[under_ids])]
    print(f"{'token_id':>9} {'doc_freq':>9}  piece  (가장 적게 커버된 30개)")
    for tid in under_order[:30]:
        piece = decode_piece(tok, int(tid))
        print(f"{tid:>9} {int(doc_freq[tid]):>9}  {piece!r}")

    # ── 6. 전체 CSV 저장 ──
    print(f"\n=== CSV 저장: {args.out_csv} ===")
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token_id", "doc_freq", "token_freq", "piece"])
        for r, tid in enumerate(order.tolist(), start=1):
            piece = decode_piece(tok, int(tid))
            w.writerow([r, tid, int(doc_freq[tid]), int(token_freq[tid]), piece])
    print(f"  -> {fmt(V)} rows written")


if __name__ == "__main__":
    main()
