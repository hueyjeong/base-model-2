"""CompositionCodec 평가 — 복원 정확도 + 오류 분석

멀티프로세스 토큰화: 각 워커가 독립 프로세스에서 토크나이저를 초기화해서 CPU 병렬 활용.
CUDA 오류 수정: 실제 JamoTokenizer를 워커에서도 정확히 사용.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.composition_codec import CompositionCodec

# ── BBPE 토큰별 통계 유틸 ──

def _build_bbpe_token_names(bbpe_tok):
    """BBPE 토큰 ID → 표시 이름 + 카테고리 매핑

    K-EXAONE 등 GPT-2 스타일 BPE는 UTF-8 바이트를 자체 문자로 매핑하므로,
    실제 디코딩 결과로 카테고리를 판정한다.
    """
    vocab_size = len(bbpe_tok)
    names = {}
    cats = {}
    special_ids = set(bbpe_tok.all_special_ids)

    # 배치로 디코딩 (개별 decode보다 빠름)
    decoded = {}
    for tid in range(vocab_size):
        try:
            decoded[tid] = bbpe_tok.decode([tid])
        except Exception:
            decoded[tid] = ""

    for tid in range(vocab_size):
        tok_str = bbpe_tok.convert_ids_to_tokens(tid)
        if tok_str is None:
            tok_str = f"<id:{tid}>"
        names[tid] = tok_str
        dec = decoded[tid]

        if tid in special_ids:
            cats[tid] = "special"
            # 특수 토큰은 내부 표현 그대로 표시
        elif tok_str.startswith("<0x") and tok_str.endswith(">"):
            cats[tid] = "byte"
            # 바이트 토큰은 내부 표현 그대로 표시
        elif any('\uAC00' <= c <= '\uD7A3' or
                 '\u1100' <= c <= '\u11FF' or
                 '\u3130' <= c <= '\u318F' for c in dec):
            cats[tid] = "hangul"
            names[tid] = dec  # 한글 등 non-ASCII는 디코딩 결과로 표시
        elif dec.strip() and all(c.isascii() for c in dec):
            cats[tid] = "ascii"
            names[tid] = dec  # ASCII도 디코딩 결과로 표시 (앞 공백 등 포함)
        else:
            cats[tid] = "other"
            names[tid] = dec  # 기타도 디코딩 결과로 표시

    return names, cats, vocab_size


def _print_token_stats(token_ok: np.ndarray, token_fail: np.ndarray,
                       names: dict, cats: dict, vocab_size: int):
    """BBPE 토큰별 오류 분포를 실패율 내림차순으로 출력"""
    total = token_ok + token_fail
    fail_rate = np.zeros(vocab_size, dtype=np.float64)
    nonzero = total > 0
    fail_rate[nonzero] = token_fail[nonzero] / total[nonzero]

    # 정렬: 실패율 내림차순, 동률이면 출현 횟수 내림차순
    order = np.lexsort((-total, -fail_rate))

    # 카테고리별 집계
    cat_ok = {}
    cat_fail = {}
    for i in range(vocab_size):
        c = cats.get(i, "unknown")
        cat_ok[c] = cat_ok.get(c, 0) + int(token_ok[i])
        cat_fail[c] = cat_fail.get(c, 0) + int(token_fail[i])

    n_appeared = int(np.sum(total > 0))
    n_zero = vocab_size - n_appeared

    print(f"\n{'='*80}")
    print(f"=== 카테고리별 복원 정확도 (BBPE 토큰 단위) ===")
    print(f"{'카테고리':<12} {'성공':<12} {'실패':<12} {'정확도':>8}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for c in ["hangul", "ascii", "byte", "special", "other"]:
        ok = cat_ok.get(c, 0)
        fl = cat_fail.get(c, 0)
        t = ok + fl
        acc = ok / max(t, 1) * 100
        print(f"{c:<12} {ok:<12,} {fl:<12,} {acc:>7.3f}%")

    print(f"\n  출현 토큰: {n_appeared:,} / {vocab_size:,}  (미출현: {n_zero:,})")

    print(f"\n{'='*80}")
    print(f"=== BBPE 토큰별 오류 분포 (실패율 내림차순, 전체 {vocab_size:,}개) ===")
    print(f"{'ID':>7}  {'토큰':<24} {'카테고리':<10} {'성공':>10} {'실패':>10} {'합계':>10} {'실패율':>8}")
    print(f"{'-'*7}  {'-'*24} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for idx in order:
        ok = int(token_ok[idx])
        fl = int(token_fail[idx])
        t = ok + fl
        fr = fail_rate[idx] * 100
        name = names.get(idx, f"?:{idx}")
        cat = cats.get(idx, "unknown")
        # 표시 이름 truncate (터미널 레이아웃용)
        if len(name) > 22:
            name = name[:20] + ".."
        marker = " !!!" if fr > 10 and t > 0 else ""
        print(f"{idx:>7}  {name:<24} {cat:<10} {ok:>10,} {fl:>10,} {t:>10,} {fr:>7.3f}%{marker}")


# ── 워커 함수 (pickle 가능해야 함, 별도 정의) ──

def _worker_init(model_id: str, base_path: str):
    """프로세스 시작 시 한 번만 실행 — 토크나이저 로딩"""
    global _bbpe_tok, _jamo_encode, _jamo_decode
    import os as _os
    import sys as _sys
    _os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 경로 설정
    _sys.path.insert(0, base_path)
    _sys.path.insert(0, os.path.join(base_path, "exp-jamo-codec"))
    _sys.path.insert(0, os.path.join(base_path, "exp-jamo-codec", "tok"))
    from transformers import AutoTokenizer
    from tok.jamo_tokenizer import JamoTokenizer
    _bbpe_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _jamo = JamoTokenizer()
    _jamo_encode = _jamo.encode
    _jamo_decode = _jamo.decode


def _worker_tokenize_batch(args):
    """배치 단위 토큰화 — 각 프로세스에서 실행.

    학습 포맷과 동일:
    - fixed_slot=True 면 각 segment 를 max_jamo_per_token 슬롯으로 padding
    - BOS/EOS 를 segment 로 감싸기 (특수 bbpe_id 는 bbpe_pad_id)
    - jamo_mask 는 유효 segment 가 차지하는 전체 범위(intra-PAD 포함)에서 True

    fixed_slot=False and append_pad_slot=True → 각 segment 끝에 JAMO_PAD 1슬롯 추가.
    fixed_slot=False and append_pad_slot=False → 기존(가변 길이) 동작.
    """
    (texts, max_seq_len, max_jamo_per_token, model_id, base_path,
     fixed_slot, append_pad_slot, jamo_bos, jamo_eos, jamo_pad,
     bbpe_pad_id) = args
    global _bbpe_tok, _jamo_encode
    if _bbpe_tok is None:
        _worker_init(model_id, base_path)

    # fixed_slot 와 append_pad_slot 동시 True 인 경우 학습 규칙(fixed_slot 우선) 따름
    if fixed_slot and append_pad_slot:
        append_pad_slot = False

    def _seg_cost(seq):
        if fixed_slot:
            return max_jamo_per_token
        return len(seq) + (1 if append_pad_slot else 0)

    def _extend_seg(all_jamo, seg_ids, all_bbpe, seq, seg_idx, bbpe_id):
        """한 segment 를 버퍼에 추가 (fixed_slot 이면 32 슬롯으로 padding)."""
        if fixed_slot:
            seq_t = list(seq[:max_jamo_per_token])
            pad_n = max_jamo_per_token - len(seq_t)
            all_jamo.extend(seq_t + [jamo_pad] * pad_n)
            seg_ids.extend([seg_idx] * max_jamo_per_token)
            all_bbpe.extend([bbpe_id] * max_jamo_per_token)
        else:
            all_jamo.extend(seq)
            seg_ids.extend([seg_idx] * len(seq))
            all_bbpe.extend([bbpe_id] * len(seq))
            if append_pad_slot:
                all_jamo.append(jamo_pad)
                seg_ids.append(seg_idx)
                all_bbpe.append(bbpe_id)

    results = []
    for text in texts:
        bbpe_ids = _bbpe_tok.encode(text, add_special_tokens=False)
        jamo_seqs = []
        bbpe_for_seq = []  # 각 jamo_seq에 대응하는 BBPE 토큰 ID
        for tid in bbpe_ids:
            tok_str = _bbpe_tok.decode([tid])
            jids = _jamo_encode(tok_str, add_special=False)
            if len(jids) <= max_jamo_per_token:
                jamo_seqs.append(jids)
                bbpe_for_seq.append(tid)
            else:
                parts = re.split(r'( )', tok_str)
                for part in parts:
                    if not part:
                        continue
                    pj = _jamo_encode(part, add_special=False)
                    if len(pj) <= max_jamo_per_token:
                        jamo_seqs.append(pj)
                        bbpe_for_seq.append(tid)
                    else:
                        for ch in part:
                            cj = _jamo_encode(ch, add_special=False)
                            if cj:
                                jamo_seqs.append(cj[:max_jamo_per_token])
                                bbpe_for_seq.append(tid)

        all_jamo = []
        seg_ids = []
        all_bbpe = []
        seg_idx = 0

        # 학습과 동일하게 [BOS] + 토큰 segments + [EOS] 로 문서 감싸기
        # EOS 자리를 먼저 reserve 후 중간 토큰 채우고, 마지막에 EOS 추가
        bos_cost = _seg_cost([jamo_bos])
        eos_cost = _seg_cost([jamo_eos])

        # BOS 추가 (예산 부족이면 아무것도 못 넣음)
        if bos_cost + eos_cost > max_seq_len:
            results.append(None)
            continue

        _extend_seg(all_jamo, seg_ids, all_bbpe, [jamo_bos], seg_idx, bbpe_pad_id)
        seg_idx += 1

        # 중간 토큰들: EOS 자리 남기고 채움
        for i, seq in enumerate(jamo_seqs):
            if len(all_jamo) + _seg_cost(seq) + eos_cost > max_seq_len:
                break
            _extend_seg(all_jamo, seg_ids, all_bbpe, seq, seg_idx, bbpe_for_seq[i])
            seg_idx += 1

        # EOS 추가
        _extend_seg(all_jamo, seg_ids, all_bbpe, [jamo_eos], seg_idx, bbpe_pad_id)
        seg_idx += 1

        L = len(all_jamo)
        if L == 0:
            results.append(None)
            continue

        pad_len = max_seq_len - L
        results.append({
            "jamo_ids": all_jamo + [jamo_pad] * pad_len,
            "mask": [True] * L + [False] * pad_len,
            "seg_ids": seg_ids + [0] * pad_len,
            "n_segments": seg_idx,
            "bbpe_ids": all_bbpe + [bbpe_pad_id] * pad_len,
        })
    return results


def _read_texts(file_paths, text_key: str = "text", min_length: int = 10):
    texts = []
    for fpath in file_paths:
        is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
        is_parquet = fpath.endswith(".parquet")

        if is_parquet:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(fpath)
            col = text_key or "text"
            for batch in pf.iter_batches(batch_size=65536, columns=[col]):
                for text in batch[col].to_pylist():
                    if text and len(text) >= min_length:
                        texts.append(text)
        elif is_jsonl:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < min_length:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get(text_key, line) if text_key else line
                    if len(text) >= min_length:
                        texts.append(text)
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= min_length:
                        texts.append(line)
    return texts


def _pre_tokenize_mp(texts: List[str], max_seq_len: int, max_jamo_per_token: int,
                     n_workers: int, chunk_size: int, model_id: str, base_path: str,
                     fixed_slot: bool = False, append_pad_slot: bool = False,
                     jamo_bos: int = 2, jamo_eos: int = 3, jamo_pad: int = 0,
                     bbpe_pad_id: int = 0):
    """멀티프로세스 토큰화"""
    batches = []
    for i in range(0, len(texts), chunk_size):
        batches.append((
            texts[i:i + chunk_size], max_seq_len, max_jamo_per_token,
            model_id, base_path,
            fixed_slot, append_pad_slot, jamo_bos, jamo_eos, jamo_pad,
            bbpe_pad_id,
        ))

    print(f"전처리: {len(batches)}배치 × {n_workers}워커, 청크={chunk_size}", flush=True)

    all_jamo_ids = []
    all_mask = []
    all_seg_ids = []
    all_n_segments = []
    all_bbpe_ids = []

    count = 0
    t0 = time.time()
    last_report = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init,
                             initargs=(model_id, base_path)) as pool:
        for batch_results in pool.map(_worker_tokenize_batch, batches, chunksize=4):
            for result in batch_results:
                if result is None:
                    continue
                all_jamo_ids.append(result["jamo_ids"])
                all_mask.append(result["mask"])
                all_seg_ids.append(result["seg_ids"])
                all_n_segments.append(result["n_segments"])
                all_bbpe_ids.append(result["bbpe_ids"])
                count += 1

            elapsed = time.time() - t0
            if elapsed - last_report >= 2.0:
                print(f"\r전처리: {count:,}/{len(texts):,} ({elapsed:.1f}s, {count/elapsed:.0f}/s)", end="", flush=True)
                last_report = elapsed

    elapsed = time.time() - t0
    print(f"\n전처리 완료: {count:,} 샘플, {elapsed:.1f}s ({count/elapsed:.0f} 샘플/s)")

    return TensorDataset(
        torch.tensor(all_jamo_ids, dtype=torch.long),
        torch.tensor(all_mask, dtype=torch.bool),
        torch.tensor(all_seg_ids, dtype=torch.long),
        torch.tensor(all_n_segments, dtype=torch.long),
        torch.tensor(all_bbpe_ids, dtype=torch.long),
    )


def evaluate(codec, loader, device, jamo_tok, show_errors=20):
    codec.eval()
    total_jamo = 0
    total_correct = 0
    errors = []

    with torch.no_grad():
        for batch in loader:
            jamo_ids = batch[0].to(device, non_blocking=True)
            jamo_mask = batch[1].to(device, non_blocking=True)
            segment_ids = batch[2].to(device, non_blocking=True)
            n_segments = batch[3].to(device, non_blocking=True)

            out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
            pred = out["logits"].argmax(dim=-1)

            correct_mask = (pred == jamo_ids) & jamo_mask
            total_correct += correct_mask.sum().item()
            total_jamo += jamo_mask.sum().item()

            if len(errors) < show_errors:
                wrong = (~correct_mask) & jamo_mask
                for b in range(jamo_ids.shape[0]):
                    w = wrong[b]
                    if w.any():
                        g = jamo_ids[b][w].cpu().tolist()
                        p = pred[b][w].cpu().tolist()
                        gt_str = jamo_tok.decode(g, skip_special=False)
                        pr_str = jamo_tok.decode(p, skip_special=False)
                        if gt_str != pr_str and len(errors) < show_errors:
                            errors.append((gt_str, pr_str))

    jamo_acc = total_correct / max(total_jamo, 1) * 100

    print(f"\n=== 복원 정확도 ===")
    print(f"  자모 정확도:    {jamo_acc:.4f}%")
    print(f"  총 자모:        {total_jamo:,}")

    if errors:
        print(f"\n=== 오류 샘플 (최대 {show_errors}개) ===")
        for i, (gt, pr) in enumerate(errors):
            print(f"  [{i+1}] 정답: '{gt}' → 예측: '{pr}'")

    return {"jamo_acc": jamo_acc}


def main():
    parser = argparse.ArgumentParser(description="CompositionCodec 평가")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--show_errors", type=int, default=30)
    parser.add_argument("--n_workers", type=int, default=4, help="토큰화 워커 수")
    parser.add_argument("--chunk_size", type=int, default=100000,
                        help="청크 크기 (한 번에 토큰화+추론할 샘플 수, 메모리 절약용)")
    parser.add_argument("--compile", action="store_true", help="torch.compile 적용")
    parser.add_argument("--force_variable", action="store_true",
                        help="체크포인트의 fixed_slot 설정을 무시하고 가변 길이로 토큰화 "
                             "(BOS/EOS 래핑은 유지). fixed_slot=True 로 학습된 모델에는 "
                             "학습 분포 밖이므로 정확도 저하 예상 — 실험/진단용.")
    parser.add_argument("--force_append_pad", action="store_true",
                        help="가변 + 세그먼트 끝 PAD 1슬롯 모드로 강제 (append_pad_slot=True).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "/workspace/base-model-2"
    model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"

    # 메인 스레드 토크나이저 (decode용)
    from tok.jamo_tokenizer import JamoTokenizer
    jamo = JamoTokenizer()

    # BBPE 토크나이저 (토큰별 통계용)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    bbpe_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"BBPE vocab: {len(bbpe_tok):,}", flush=True)

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    append_pad_slot = saved_args.get("append_pad_slot", False)
    # --force_variable: 학습 설정 무시하고 가변 길이로 실험 (BOS/EOS 는 유지)
    if args.force_variable:
        print("[WARNING] --force_variable: fixed_slot/append_pad_slot 을 False 로 강제. "
              "학습 분포 밖이므로 정확도 저하 가능.")
        fixed_slot = False
        append_pad_slot = False
    if args.force_append_pad:
        print("[WARNING] --force_append_pad: fixed_slot=False, append_pad_slot=True 로 강제.")
        fixed_slot = False
        append_pad_slot = True
    # fixed_output_len: 학습과 동일하게 fixed_slot=True 에서는 고정값 사용
    fixed_output_len = None
    if fixed_slot:
        # 학습과 동일한 공식: max_seq_len // max_jamo_per_token
        # 학습 시 max_seq_len 을 saved_args 에서 읽어 결정 (eval args.max_seq_len 과 다를 수 있음)
        trained_max_seq = saved_args.get("max_seq_len", args.max_seq_len)
        fixed_output_len = trained_max_seq // max_jpt

    codec = CompositionCodec(
        jamo_vocab=jamo.vocab_size, d_model=d, n_layers=nl, kernel_size=k,
        segment_masked=seg_masked,
        parallel_decoder=parallel_decoder,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        max_jamo_per_token=max_jpt,
        fixed_output_len=fixed_output_len,
    ).to(device)
    print(f"fixed_slot={fixed_slot}, append_pad_slot={append_pad_slot}, "
          f"fixed_output_len={fixed_output_len}, max_jamo_per_token={max_jpt}")

    sd = ckpt["model"]
    prefix = "_orig_mod."
    if any(key.startswith(prefix) for key in sd):
        sd = {key[len(prefix):] if key.startswith(prefix) else key: v for key, v in sd.items()}
    codec.load_state_dict(sd)

    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in codec.parameters())
    print(f"모델: d={d}, L={nl}, k={k}, params={n_params/1e6:.2f}M (step {step})")

    # 텍스트 읽기
    t_read = time.time()
    texts = _read_texts(args.corpus, args.text_key)
    texts = texts[:args.max_samples]
    print(f"텍스트 로드: {len(texts):,}행 ({time.time()-t_read:.1f}s)")

    # torch.compile
    if args.compile:
        print("torch.compile 적용 중...", flush=True)
        codec = torch.compile(codec)
        with torch.no_grad():
            dummy_ids = torch.zeros(2, 512, dtype=torch.long, device=device)
            dummy_msk = torch.zeros(2, 512, dtype=torch.bool, device=device)
            dummy_seg = torch.zeros(2, 512, dtype=torch.long, device=device)
            dummy_ns = torch.tensor([10, 10], device=device)
            codec(dummy_ids, dummy_msk, dummy_seg, dummy_ns)
        torch.cuda.synchronize()
        print("컴파일 완료")

    # GPU 워밍업
    print("GPU 워밍업...", flush=True)
    with torch.no_grad():
        dummy = torch.zeros(2, args.max_seq_len, dtype=torch.long, device=device)
        dummy_m = torch.zeros(2, args.max_seq_len, dtype=torch.bool, device=device)
        dummy_s = torch.zeros(2, args.max_seq_len, dtype=torch.long, device=device)
        dummy_n = torch.tensor([10, 10], device=device)
        codec(dummy, dummy_m, dummy_s, dummy_n)
    torch.cuda.synchronize()

    # ── 청크 단위 평가 (메모리 절약, 토큰화/추론 파이프라인) ──
    import queue as _queue
    import threading

    # BBPE pad id (특수 segment 용 sentinel)
    bbpe_pad_id = bbpe_tok.pad_token_id if bbpe_tok.pad_token_id is not None else 0

    def _tok_worker(texts, chunk_size, tokenized_q):
        """백그라운드 토큰화 스레드 (청크 분할도 여기서)"""
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            t1 = time.time()
            ds = _pre_tokenize_mp(
                chunk, args.max_seq_len, max_jamo_per_token=max_jpt,
                n_workers=args.n_workers, chunk_size=256,
                model_id=model_id, base_path=base_path,
                fixed_slot=fixed_slot, append_pad_slot=append_pad_slot,
                jamo_bos=2, jamo_eos=3, jamo_pad=0,
                bbpe_pad_id=bbpe_pad_id,
            )
            tokenized_q.put((ds, len(chunk), time.time() - t1))

    tokenized_q = _queue.Queue(maxsize=1)

    tok_thread = threading.Thread(
        target=_tok_worker, args=(texts, args.chunk_size, tokenized_q), daemon=True)
    tok_thread.start()
    del texts  # 메인에서 해제 (스레드가 참조 중)

    # BBPE 토큰별 통계 배열 초기화
    bbpe_vocab_size = len(bbpe_tok)
    bbpe_ok = np.zeros(bbpe_vocab_size, dtype=np.int64)
    bbpe_fail = np.zeros(bbpe_vocab_size, dtype=np.int64)
    print("BBPE 토큰 이름 빌드 중...", flush=True)
    bbpe_names, bbpe_cats, _ = _build_bbpe_token_names(bbpe_tok)

    # GPU 추론 (토큰화와 파이프라인)
    total_jamo = 0
    total_correct = 0
    total_errors = 0      # 전체 오류 샘플 수
    error_examples = []   # 오류 예시 (show_errors 개까지)
    t0 = time.time()
    chunk_idx = 0

    while tok_thread.is_alive() or not tokenized_q.empty():
        try:
            tensor_ds, n_chunk, tok_time = tokenized_q.get(timeout=1.0)
        except _queue.Empty:
            continue

        chunk_idx += 1
        chunk_errs = 0
        chunk_examples = []
        loader = DataLoader(tensor_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

        with torch.no_grad():
            for batch in loader:
                jamo_ids = batch[0].to(device, non_blocking=True)
                jamo_mask = batch[1].to(device, non_blocking=True)
                segment_ids = batch[2].to(device, non_blocking=True)
                n_segments = batch[3].to(device, non_blocking=True)

                out = codec(jamo_ids, jamo_mask, segment_ids, n_segments)
                pred = out["logits"].argmax(dim=-1)

                correct_mask = (pred == jamo_ids) & jamo_mask
                total_correct += correct_mask.sum().item()
                total_jamo += jamo_mask.sum().item()

                # BBPE 토큰별 segment-level 성공/실패 집계
                seg_cpu = segment_ids.cpu().numpy()
                correct_cpu = correct_mask.cpu().numpy()
                mask_cpu = jamo_mask.cpu().numpy()
                bbpe_cpu = batch[4].numpy()  # pinned CPU 메모리

                for b in range(jamo_ids.shape[0]):
                    m = mask_cpu[b]
                    if not m.any():
                        continue
                    vs = seg_cpu[b][m]
                    vc = correct_cpu[b][m]
                    vb = bbpe_cpu[b][m]

                    # 세그먼트 경계 검출 (연속 구간)
                    breaks = np.where(np.diff(vs) != 0)[0] + 1
                    starts = np.concatenate([[0], breaks])
                    ends = np.concatenate([breaks, [len(vs)]])

                    for si, ei in zip(starts, ends):
                        bid = int(vb[si])
                        if vc[si:ei].all():
                            bbpe_ok[bid] += 1
                        else:
                            bbpe_fail[bid] += 1

                wrong = (~correct_mask) & jamo_mask
                for b in range(jamo_ids.shape[0]):
                    if wrong[b].any():
                        chunk_errs += 1
                        if len(error_examples) + len(chunk_examples) < args.show_errors:
                            g = jamo_ids[b][wrong[b]].cpu().tolist()
                            p = pred[b][wrong[b]].cpu().tolist()
                            gt_str = jamo.decode(g, skip_special=False)
                            pr_str = jamo.decode(p, skip_special=False)
                            if gt_str != pr_str:
                                chunk_examples.append((gt_str, pr_str))

        total_errors += chunk_errs
        error_examples.extend(chunk_examples)

        acc = total_correct / max(total_jamo, 1) * 100
        preview = ""
        if chunk_examples:
            preview = " | " + "; ".join(f"'{g}'→'{p}'" for g, p in chunk_examples[:3])
        print(f"  청크 {chunk_idx}: {n_chunk}샘플, "
              f"토큰화 {tok_time:.1f}s ({n_chunk/max(tok_time,0.01):.0f}/s), "
              f"정확도 {acc:.2f}%, 오류 {chunk_errs}/{n_chunk}{preview}", flush=True)

        del tensor_ds, loader

    jamo_acc = total_correct / max(total_jamo, 1) * 100
    print(f"\n=== 복원 정확도 ===")
    print(f"  자모 정확도:    {jamo_acc:.4f}%")
    print(f"  총 자모:        {total_jamo:,}")
    print(f"  오류 샘플:      {total_errors:,}")
    if error_examples:
        print(f"\n=== 오류 샘플 (최대 {args.show_errors}개) ===")
        for i, (gt, pr) in enumerate(error_examples):
            print(f"  [{i+1}] 정답: '{gt}' → 예측: '{pr}'")

    # BBPE 토큰별 오류 분포 출력
    _print_token_stats(bbpe_ok, bbpe_fail, bbpe_names, bbpe_cats, bbpe_vocab_size)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
