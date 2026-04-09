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

import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from codec.composition_codec import CompositionCodec

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
    """배치 단위 토큰화 — 각 프로세스에서 실행"""
    texts, max_seq_len, max_jamo_per_token, model_id, base_path = args
    global _bbpe_tok, _jamo_encode
    if _bbpe_tok is None:
        _worker_init(model_id, base_path)

    results = []
    for text in texts:
        bbpe_ids = _bbpe_tok.encode(text, add_special_tokens=False)
        jamo_seqs = []
        for tid in bbpe_ids:
            tok_str = _bbpe_tok.decode([tid])
            jids = _jamo_encode(tok_str, add_special=False)
            if len(jids) <= max_jamo_per_token:
                jamo_seqs.append(jids)
            else:
                parts = re.split(r'( )', tok_str)
                for part in parts:
                    if not part:
                        continue
                    pj = _jamo_encode(part, add_special=False)
                    if len(pj) <= max_jamo_per_token:
                        jamo_seqs.append(pj)
                    else:
                        for ch in part:
                            cj = _jamo_encode(ch, add_special=False)
                            if cj:
                                jamo_seqs.append(cj[:max_jamo_per_token])

        all_jamo = []
        seg_ids = []
        seg_idx = 0
        for seq in jamo_seqs:
            if len(all_jamo) + len(seq) > max_seq_len:
                break
            all_jamo.extend(seq)
            seg_ids.extend([seg_idx] * len(seq))
            seg_idx += 1

        L = len(all_jamo)
        if L == 0:
            results.append(None)
            continue

        pad_len = max_seq_len - L
        results.append({
            "jamo_ids": all_jamo + [0] * pad_len,
            "mask": [True] * L + [False] * pad_len,
            "seg_ids": seg_ids + [0] * pad_len,
            "n_segments": seg_idx,
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
                     n_workers: int, chunk_size: int, model_id: str, base_path: str):
    """멀티프로세스 토큰화"""
    batches = []
    for i in range(0, len(texts), chunk_size):
        batches.append((texts[i:i + chunk_size], max_seq_len, max_jamo_per_token, model_id, base_path))

    print(f"전처리: {len(batches)}배치 × {n_workers}워커, 청크={chunk_size}", flush=True)

    all_jamo_ids = []
    all_mask = []
    all_seg_ids = []
    all_n_segments = []

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
    parser.add_argument("--chunk_size", type=int, default=256, help="워커당 청크 크기")
    parser.add_argument("--compile", action="store_true", help="torch.compile 적용")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "/workspace/base-model-2"
    model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"

    # 메인 스레드 토크나이저 (decode용)
    from tok.jamo_tokenizer import JamoTokenizer
    jamo = JamoTokenizer()

    # 체크포인트 로드
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    d = saved_args.get("d_model", 256)
    nl = saved_args.get("n_layers", 5)
    k = saved_args.get("kernel_size", 7)

    codec = CompositionCodec(
        jamo_vocab=jamo.vocab_size, d_model=d, n_layers=nl, kernel_size=k,
    ).to(device)

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

    # 멀티프로세스 토큰화
    t_tok = time.time()
    tensor_ds = _pre_tokenize_mp(
        texts, args.max_seq_len, max_jamo_per_token=32,
        n_workers=args.n_workers, chunk_size=args.chunk_size,
        model_id=model_id, base_path=base_path
    )

    # torch.compile
    if args.compile:
        print("torch.compile 적용 중...", flush=True)
        codec = torch.compile(codec, mode="reduce-overhead")
        with torch.no_grad():
            dummy_ids = torch.zeros(2, 512, dtype=torch.long, device=device)
            dummy_msk = torch.zeros(2, 512, dtype=torch.bool, device=device)
            dummy_seg = torch.zeros(2, 512, dtype=torch.long, device=device)
            dummy_ns = torch.tensor([10, 10], device=device)
            codec(dummy_ids, dummy_msk, dummy_seg, dummy_ns)
        torch.cuda.synchronize()
        print("컴파일 완료")

    # DataLoader
    loader = DataLoader(
        tensor_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 이미 메모리에 있음
        pin_memory=True,
    )

    # GPU 워밍업
    print("GPU 워밍업...", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            codec(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))
    torch.cuda.synchronize()

    # 추론
    t_inf = time.time()
    result = evaluate(codec, loader, device, jamo, args.show_errors)
    elapsed = time.time() - t_inf
    n_samples = len(tensor_ds)
    print(f"\n토큰화: {time.time()-t_tok:.1f}s, 추론: {elapsed:.1f}s ({n_samples/elapsed:.0f} 샘플/s)")


if __name__ == "__main__":
    main()
