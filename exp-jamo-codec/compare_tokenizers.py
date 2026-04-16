"""KoELECTRA WordPiece vs K-EXAONE BBPE(+32자모 분절) 토크나이저 효율 비교.

지표:
    - UNK 비율 (토큰 수 대비, 문장당 1회 이상 발생 비율)
    - 토큰/문자 비율 (압축률)
    - 바이트당 토큰 수 (UTF-8 바이트 기준)
    - 고유 vocab 커버리지
    - K-EXAONE: 32자모 초과로 재분절된 토큰 비율 / 통계
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer

from tok.jamo_tokenizer import JamoTokenizer


def iter_texts(path: str, text_key: str = "text", limit: int = None):
    """JSONL 또는 Parquet 파일에서 텍스트 스트리밍."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        n = 0
        for batch in pf.iter_batches(batch_size=65536, columns=[text_key]):
            for txt in batch[text_key].to_pylist():
                if txt:
                    yield txt
                    n += 1
                    if limit is not None and n >= limit:
                        return
        return

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
                txt = obj.get(text_key, "")
                if txt:
                    yield txt
            except json.JSONDecodeError:
                continue


def measure_koelectra(tok, texts):
    """WordPiece 측정."""
    unk_id = tok.unk_token_id
    total_tokens = 0
    total_unk = 0
    total_chars = 0
    total_bytes = 0
    sents_with_unk = 0
    used_ids = set()
    n_sents = 0

    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)
        total_tokens += len(ids)
        unks = sum(1 for i in ids if i == unk_id)
        total_unk += unks
        sents_with_unk += int(unks > 0)
        total_chars += len(txt)
        total_bytes += len(txt.encode("utf-8"))
        used_ids.update(ids)
        n_sents += 1

    return {
        "n_sents": n_sents,
        "total_tokens": total_tokens,
        "total_unk": total_unk,
        "unk_ratio": total_unk / max(total_tokens, 1),
        "sents_with_unk": sents_with_unk,
        "sent_unk_ratio": sents_with_unk / max(n_sents, 1),
        "tokens_per_char": total_tokens / max(total_chars, 1),
        "tokens_per_byte": total_bytes and total_tokens / total_bytes,
        "chars_per_token": total_chars / max(total_tokens, 1),
        "vocab_used": len(used_ids),
        "vocab_size": tok.vocab_size,
    }


def measure_kexaone(bbpe, jamo, texts, max_jamo_per_token: int = 32):
    """K-EXAONE BBPE + 32자모 분절 측정.

    - 원본 BBPE 토큰 수
    - 32분절 적용 후 최종 패치 수 (concat 경로)
    - 32초과 토큰 수/비율
    - UNK 토큰 비율 (BBPE 차원)
    - 평균 자모/토큰
    """
    unk_id = getattr(bbpe, "unk_token_id", None)

    total_bbpe_tokens = 0
    total_final_patches = 0
    total_oversize_tokens = 0       # 32자모 초과 토큰
    total_oversize_split_patches = 0  # 초과 토큰을 분절했을 때 나오는 패치 수
    total_jamo_len = 0
    total_unk = 0
    total_chars = 0
    total_bytes = 0
    sents_with_unk = 0
    sents_with_oversize = 0
    used_ids = set()
    n_sents = 0

    oversize_samples = []  # (tok_str, jamo_len) 몇 개 수집

    for txt in texts:
        ids = bbpe.encode(txt, add_special_tokens=False)
        total_bbpe_tokens += len(ids)
        total_chars += len(txt)
        total_bytes += len(txt.encode("utf-8"))
        used_ids.update(ids)

        unks = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
        total_unk += unks
        sents_with_unk += int(unks > 0)

        has_oversize = False
        for tid in ids:
            tok_str = bbpe.decode([tid])
            jamo_ids = jamo.encode(tok_str, add_special=False)
            jlen = len(jamo_ids)
            total_jamo_len += jlen
            if jlen <= max_jamo_per_token:
                total_final_patches += 1
            else:
                has_oversize = True
                total_oversize_tokens += 1
                if len(oversize_samples) < 20:
                    oversize_samples.append((tok_str, jlen))
                # 공백 분절
                parts = re.split(r"( )", tok_str)
                n_patches = 0
                for part in parts:
                    if not part:
                        continue
                    pj = jamo.encode(part, add_special=False)
                    if len(pj) <= max_jamo_per_token:
                        n_patches += 1
                    else:
                        for ch in part:
                            cj = jamo.encode(ch, add_special=False)
                            if cj:
                                n_patches += 1
                total_oversize_split_patches += n_patches
                total_final_patches += n_patches

        if has_oversize:
            sents_with_oversize += 1
        n_sents += 1

    return {
        "n_sents": n_sents,
        "total_bbpe_tokens": total_bbpe_tokens,
        "total_final_patches": total_final_patches,
        "patches_per_char": total_final_patches / max(total_chars, 1),
        "tokens_per_char": total_bbpe_tokens / max(total_chars, 1),
        "tokens_per_byte": total_bbpe_tokens / max(total_bytes, 1),
        "chars_per_token": total_chars / max(total_bbpe_tokens, 1),
        "chars_per_patch": total_chars / max(total_final_patches, 1),
        "avg_jamo_per_token": total_jamo_len / max(total_bbpe_tokens, 1),
        "total_unk": total_unk,
        "unk_ratio": total_unk / max(total_bbpe_tokens, 1),
        "sents_with_unk": sents_with_unk,
        "sent_unk_ratio": sents_with_unk / max(n_sents, 1),
        "oversize_tokens": total_oversize_tokens,
        "oversize_ratio": total_oversize_tokens / max(total_bbpe_tokens, 1),
        "sents_with_oversize": sents_with_oversize,
        "sent_oversize_ratio": sents_with_oversize / max(n_sents, 1),
        "oversize_split_patches": total_oversize_split_patches,
        "oversize_split_expansion": (
            total_oversize_split_patches / max(total_oversize_tokens, 1)
        ),
        "vocab_used": len(used_ids),
        "vocab_size": bbpe.vocab_size,
        "oversize_samples": oversize_samples,
    }


def print_row(label, value, unit=""):
    if isinstance(value, float):
        print(f"  {label:40s} {value:>14.4f}{unit}")
    else:
        print(f"  {label:40s} {value:>14,}{unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/val_ko_50k.jsonl")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--koelectra_id",
        default="monologg/koelectra-base-v3-discriminator",
    )
    ap.add_argument(
        "--kexaone_id",
        default="LGAI-EXAONE/K-EXAONE-236B-A23B",
    )
    ap.add_argument("--max_jamo_per_token", type=int, default=32)
    args = ap.parse_args()

    print(f"[load] KoELECTRA: {args.koelectra_id}")
    t0 = time()
    koe = AutoTokenizer.from_pretrained(args.koelectra_id)
    print(f"  vocab={koe.vocab_size:,}  load={time()-t0:.1f}s")

    print(f"[load] K-EXAONE: {args.kexaone_id}")
    t0 = time()
    bbpe = AutoTokenizer.from_pretrained(args.kexaone_id, trust_remote_code=True)
    print(f"  vocab={bbpe.vocab_size:,}  load={time()-t0:.1f}s")

    print(f"[load] JamoTokenizer")
    jamo = JamoTokenizer()
    print(f"  vocab={jamo.vocab_size}")

    print(f"\n[corpus] {args.corpus} (limit={args.limit})")
    texts = list(iter_texts(args.corpus, args.text_key, args.limit))
    n_chars = sum(len(t) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    print(f"  n_sents={len(texts):,}  chars={n_chars:,}  bytes={n_bytes:,}")
    print(f"  avg_chars/sent={n_chars/len(texts):.1f}")

    print(f"\n[measure] KoELECTRA WordPiece")
    t0 = time()
    koe_stats = measure_koelectra(koe, texts)
    print(f"  elapsed={time()-t0:.1f}s")

    print(f"\n[measure] K-EXAONE BBPE (+ {args.max_jamo_per_token}-jamo split)")
    t0 = time()
    kex_stats = measure_kexaone(bbpe, jamo, texts, args.max_jamo_per_token)
    print(f"  elapsed={time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("KoELECTRA WordPiece")
    print("=" * 70)
    print_row("vocab size", koe_stats["vocab_size"])
    print_row("vocab used", koe_stats["vocab_used"])
    print_row("vocab coverage", koe_stats["vocab_used"]/koe_stats["vocab_size"])
    print_row("total tokens", koe_stats["total_tokens"])
    print_row("UNK tokens", koe_stats["total_unk"])
    print_row("UNK ratio (token)", koe_stats["unk_ratio"])
    print_row("sentences with UNK", koe_stats["sents_with_unk"])
    print_row("UNK ratio (sentence)", koe_stats["sent_unk_ratio"])
    print_row("tokens / char", koe_stats["tokens_per_char"])
    print_row("tokens / byte", koe_stats["tokens_per_byte"])
    print_row("chars / token", koe_stats["chars_per_token"])

    print("\n" + "=" * 70)
    print(f"K-EXAONE BBPE (+ {args.max_jamo_per_token}-jamo split)")
    print("=" * 70)
    print_row("vocab size", kex_stats["vocab_size"])
    print_row("vocab used", kex_stats["vocab_used"])
    print_row("vocab coverage", kex_stats["vocab_used"]/kex_stats["vocab_size"])
    print_row("total BBPE tokens", kex_stats["total_bbpe_tokens"])
    print_row("total final patches (post-split)", kex_stats["total_final_patches"])
    print_row("UNK tokens", kex_stats["total_unk"])
    print_row("UNK ratio (token)", kex_stats["unk_ratio"])
    print_row("sentences with UNK", kex_stats["sents_with_unk"])
    print_row("UNK ratio (sentence)", kex_stats["sent_unk_ratio"])
    print_row("tokens / char (BBPE)", kex_stats["tokens_per_char"])
    print_row("tokens / byte (BBPE)", kex_stats["tokens_per_byte"])
    print_row("chars / token (BBPE)", kex_stats["chars_per_token"])
    print_row("patches / char (post-split)", kex_stats["patches_per_char"])
    print_row("chars / patch (post-split)", kex_stats["chars_per_patch"])
    print_row("avg jamo / token", kex_stats["avg_jamo_per_token"])
    print_row("oversize tokens (>32 jamo)", kex_stats["oversize_tokens"])
    print_row("oversize token ratio", kex_stats["oversize_ratio"])
    print_row("sentences with oversize", kex_stats["sents_with_oversize"])
    print_row("oversize sentence ratio", kex_stats["sent_oversize_ratio"])
    print_row("oversize → patches expansion", kex_stats["oversize_split_expansion"])

    print("\n[oversize samples]")
    for tok_str, jlen in kex_stats["oversize_samples"]:
        disp = tok_str.replace("\n", "\\n")
        if len(disp) > 60:
            disp = disp[:60] + "..."
        print(f"  jamo={jlen:3d}  \"{disp}\"")

    print("\n" + "=" * 70)
    print("비교")
    print("=" * 70)
    koe_tpc = koe_stats["tokens_per_char"]
    kex_tpc = kex_stats["tokens_per_char"]
    kex_ppc = kex_stats["patches_per_char"]
    print(f"  tokens/char   KoELECTRA {koe_tpc:.4f}  vs  K-EXAONE(BBPE) {kex_tpc:.4f}   "
          f"K-EXAONE {koe_tpc/kex_tpc:.2f}x 더 적음")
    print(f"  patches/char  (post-split) K-EXAONE {kex_ppc:.4f}   "
          f"KoELECTRA 대비 {koe_tpc/kex_ppc:.2f}x")
    print(f"  UNK ratio     KoELECTRA {koe_stats['unk_ratio']*100:.3f}%  vs  "
          f"K-EXAONE {kex_stats['unk_ratio']*100:.3f}%")
    print(f"  sent UNK      KoELECTRA {koe_stats['sent_unk_ratio']*100:.2f}%  vs  "
          f"K-EXAONE {kex_stats['sent_unk_ratio']*100:.2f}%")


if __name__ == "__main__":
    main()
