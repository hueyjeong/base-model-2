"""K-EXAONE coverage_100 parquet의 토큰 분포 카운트.

각 토큰별로:
  - doc_freq: 해당 토큰을 포함한 문서(샘플) 수
  - token_freq: 해당 토큰의 총 출현 횟수

결과를 npz로 저장.
"""
import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="corpus/k-exaone_coverage_100.parquet")
    ap.add_argument("--tokenizer_id", default="LGAI-EXAONE/K-EXAONE-236B-A23B")
    ap.add_argument("--out", default="corpus/k-exaone_coverage_100_freq.npz")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_bytes", type=int, default=4096,
                    help="문서당 utf-8 바이트 상한 (이후 잘라냄)")
    ap.add_argument("--limit_rg", type=int, default=-1, help="처음 N개 row_group만 (속도 견적용)")
    args = ap.parse_args()

    print(f"[load] tokenizer: {args.tokenizer_id}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    V = len(tok)
    print(f"[load] vocab_size = {V}")

    pf = pq.ParquetFile(args.parquet)
    n_rg = pf.num_row_groups if args.limit_rg < 0 else min(args.limit_rg, pf.num_row_groups)
    total_rows = sum(pf.metadata.row_group(r).num_rows for r in range(n_rg))
    print(f"[parquet] row_groups={n_rg}/{pf.num_row_groups}, rows={total_rows}")

    token_freq = np.zeros(V, dtype=np.int64)
    doc_freq = np.zeros(V, dtype=np.int64)

    t0 = time.time()
    n_done_rows = 0
    n_done_tokens = 0
    pbar = tqdm(total=total_rows, desc="rows", unit="row")

    max_bytes = args.max_bytes
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=["text"])
        texts = tbl["text"].to_pylist()
        del tbl

        for i in range(0, len(texts), args.batch):
            batch = texts[i : i + args.batch]
            # 문서당 max_bytes(utf-8) 이내로 truncate.
            # encode → slice → decode(errors='ignore')로 멀티바이트 경계 안전.
            if max_bytes > 0:
                batch = [
                    s.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
                    for s in batch
                ]
            enc = tok(batch, add_special_tokens=False)
            ids_list = enc["input_ids"]

            # 빈 문서 제거 + numpy 변환
            arrs = [np.asarray(x, dtype=np.int32) for x in ids_list if x]
            if not arrs:
                pbar.update(len(batch))
                continue
            lens = np.array([a.size for a in arrs], dtype=np.int64)
            all_ids = np.concatenate(arrs)
            n_done_tokens += int(all_ids.size)

            # token frequency
            token_freq += np.bincount(all_ids, minlength=V).astype(np.int64)

            # document frequency: (doc_idx, token_id) unique
            doc_idx = np.repeat(np.arange(len(arrs), dtype=np.int64), lens)
            pairs = doc_idx * V + all_ids.astype(np.int64)
            unique_pairs = np.unique(pairs)
            uniq_tokens = (unique_pairs % V).astype(np.int32)
            doc_freq += np.bincount(uniq_tokens, minlength=V).astype(np.int64)

            pbar.update(len(batch))
            n_done_rows += len(batch)

        # row_group마다 중간 저장 (인터럽트 대비)
        if (rg + 1) % 5 == 0 or rg == n_rg - 1:
            np.savez(
                args.out,
                token_freq=token_freq,
                doc_freq=doc_freq,
                n_rows_processed=np.int64(n_done_rows),
                n_row_groups_processed=np.int64(rg + 1),
                vocab_size=np.int64(V),
            )

    pbar.close()
    elapsed = time.time() - t0
    print(f"[done] rows={n_done_rows} tokens={n_done_tokens} elapsed={elapsed:.1f}s "
          f"({n_done_tokens/elapsed/1e6:.2f} Mtok/s)")

    np.savez(
        args.out,
        token_freq=token_freq,
        doc_freq=doc_freq,
        n_rows_processed=np.int64(n_done_rows),
        n_row_groups_processed=np.int64(n_rg),
        vocab_size=np.int64(V),
    )
    print(f"[save] {args.out}")

    # 즉석 통계
    nz = (doc_freq > 0).sum()
    print(f"[stat] vocab={V}, nonzero={nz}, zero={V - nz}")
    print(f"[stat] doc_freq: min={doc_freq.min()}, max={doc_freq.max()}, "
          f"mean={doc_freq.mean():.1f}, median={int(np.median(doc_freq))}")


if __name__ == "__main__":
    main()
