"""SimpleJamoDataset — per-token 포맷 데이터셋.

각 샘플은 **1개 BBPE 토큰** 의 자모 시퀀스. 토큰 간 문맥 없음.
- jamo_ids: [max_jamo] (실자모 + PAD=0 로 패딩)
- mask: [max_jamo] bool (실자모 위치)
- bbpe_id: int (진단/평가용)

배치는 여러 토큰을 stack → [T, max_jamo]. GPU 가 모든 토큰 병렬 처리.

자모 > max_jamo 인 긴 BBPE 토큰은 공백 분할 → 문자 분할 fallback 로 분리.
"""
import json
import os
import re
import sys
from typing import List

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


JAMO_PAD = 0


def load_bbpe_tokenizer(model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class SimpleJamoDataset(IterableDataset):
    """BBPE 토큰화 → per-token 자모 분해 → [max_jamo] 패딩 스트리밍.

    Multi-document packing 없음 — 각 row 는 딱 1 토큰.
    """

    def __init__(
        self,
        file_paths,
        bbpe_tokenizer,
        jamo_tokenizer,
        max_jamo: int = 32,
        text_key: str = "text",
        min_length: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.bbpe = bbpe_tokenizer
        self.jamo = jamo_tokenizer
        self.max_jamo = max_jamo
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0
        self._resume_line = 0

        # BBPE tid → [jamo_seq, ...] (fallback 시 여러 부분)
        self._tok_cache: dict = {}
        self.encode_batch_size = 64

    # ─────────────────────────────────────────────
    #  Prewarm: vocab 전체 BBPE decode + jamo 분해
    # ─────────────────────────────────────────────
    def _prewarm_cache(self, verbose: bool = True):
        """vocab 전체의 (tok_str, jamo_seqs) cache 를 시작 시 한 번에."""
        import time
        vocab_size = int(self.bbpe.vocab_size)
        t0 = time.time()
        if verbose:
            print(f"[SimpleCache] {vocab_size} tokens → jamo 분해 ...")

        all_strs = self.bbpe.batch_decode([[tid] for tid in range(vocab_size)])
        for tid, tok_str in enumerate(all_strs):
            if not tok_str:
                self._tok_cache[tid] = []
                continue
            base = self.jamo.encode(tok_str, add_special=False)
            if len(base) <= self.max_jamo:
                self._tok_cache[tid] = [base]
            else:
                # fallback: 공백 → 문자 분할
                parts: List[List[int]] = []
                for part in re.split(r"( )", tok_str):
                    if not part:
                        continue
                    pj = self.jamo.encode(part, add_special=False)
                    if len(pj) <= self.max_jamo:
                        parts.append(pj)
                    else:
                        for ch in part:
                            cj = self.jamo.encode(ch, add_special=False)
                            if cj:
                                parts.append(cj[:self.max_jamo])
                self._tok_cache[tid] = parts
        if verbose:
            print(f"[SimpleCache] done {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────
    #  Text streaming
    # ─────────────────────────────────────────────
    def _iter_texts(self, resume_row: int = 0,
                    worker_stride: int = 1, worker_offset: int = 0):
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")

            if is_parquet:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)
                text_col = self.text_key or "text"

                abs_line = 0
                for batch in pf.iter_batches(batch_size=4096, columns=[text_col]):
                    col = batch[text_col]
                    n = len(col)
                    for i in range(n):
                        abs_idx = abs_line + i
                        if abs_idx < resume_row:
                            continue
                        if abs_idx % worker_stride != worker_offset:
                            continue
                        text = col[i].as_py()
                        if text and len(text) >= self.min_length:
                            yield abs_idx, text
                    abs_line += n
                continue

            abs_line = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if abs_line < resume_row:
                        abs_line += 1
                        continue
                    if abs_line % worker_stride != worker_offset:
                        abs_line += 1
                        continue
                    s = line.strip()
                    if len(s) < self.min_length:
                        abs_line += 1
                        continue
                    if is_jsonl:
                        try:
                            obj = json.loads(s)
                        except json.JSONDecodeError:
                            abs_line += 1
                            continue
                        text = obj.get(self.text_key, s) if self.text_key else s
                    else:
                        text = s
                    if len(text) >= self.min_length:
                        yield abs_line, text
                    abs_line += 1

    # ─────────────────────────────────────────────
    #  state_dict for resume
    # ─────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {"line_counter": self._line_counter}

    def load_state_dict(self, state: dict) -> None:
        self._resume_line = state.get("line_counter", 0)
        self._line_counter = self._resume_line

    # ─────────────────────────────────────────────
    #  Iteration — 토큰 단위로 yield
    # ─────────────────────────────────────────────
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        resume_line = self._resume_line
        if resume_line > 0:
            self._resume_line = 0

        rust_tok = self.bbpe.backend_tokenizer
        encode_bs = self.encode_batch_size

        first_epoch = True
        while True:
            resume_row = resume_line if first_epoch and resume_line > 0 else 0

            # BBPE encode batch
            buf_abs: List[int] = []
            buf_text: List[str] = []

            def _flush():
                encodings = rust_tok.encode_batch(buf_text, add_special_tokens=False)
                for a, enc in zip(buf_abs, encodings):
                    yield a, enc.ids

            for abs_line, text in self._iter_texts(
                resume_row=resume_row,
                worker_stride=total_workers,
                worker_offset=global_worker_id,
            ):
                buf_abs.append(abs_line)
                buf_text.append(text)
                if len(buf_text) >= encode_bs:
                    encodings = rust_tok.encode_batch(buf_text, add_special_tokens=False)
                    for a, bbpe_ids in zip(buf_abs, encodings):
                        self._line_counter = a + 1
                        for tid in bbpe_ids.ids:
                            for jamo_seq in self._tok_cache.get(tid, []):
                                padded = list(jamo_seq) + [JAMO_PAD] * (self.max_jamo - len(jamo_seq))
                                mask = [True] * len(jamo_seq) + [False] * (self.max_jamo - len(jamo_seq))
                                yield {
                                    "jamo_ids": torch.tensor(padded, dtype=torch.long),
                                    "mask": torch.tensor(mask, dtype=torch.bool),
                                    "bbpe_id": tid,
                                }
                    buf_abs, buf_text = [], []

            if buf_text:
                encodings = rust_tok.encode_batch(buf_text, add_special_tokens=False)
                for a, bbpe_ids in zip(buf_abs, encodings):
                    self._line_counter = a + 1
                    for tid in bbpe_ids.ids:
                        for jamo_seq in self._tok_cache.get(tid, []):
                            padded = list(jamo_seq) + [JAMO_PAD] * (self.max_jamo - len(jamo_seq))
                            mask = [True] * len(jamo_seq) + [False] * (self.max_jamo - len(jamo_seq))
                            yield {
                                "jamo_ids": torch.tensor(padded, dtype=torch.long),
                                "mask": torch.tensor(mask, dtype=torch.bool),
                                "bbpe_id": tid,
                            }

            resume_line = 0
            first_epoch = False


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tok.jamo_tokenizer import JamoTokenizer

    print("=== SimpleJamoDataset Smoke Test ===\n")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    ds = SimpleJamoDataset(
        file_paths=["corpus/k-exaone_coverage_5_len1000.parquet"],
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_jamo=32,
    )
    ds._prewarm_cache()

    for i, sample in enumerate(ds):
        if i >= 5:
            break
        L = int(sample["mask"].sum().item())
        text = jamo.decode(sample["jamo_ids"][:L].tolist(), skip_special=False)
        print(f"[{i}] bbpe_id={sample['bbpe_id']:>6}  len={L:>2}  text={text!r}")

    print("\n--- batching (DataLoader) ---")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, num_workers=0)
    for batch in loader:
        print(f"jamo_ids: {batch['jamo_ids'].shape}")
        print(f"mask:     {batch['mask'].shape}")
        print(f"bbpe_ids: {batch['bbpe_id'].tolist()[:8]}")
        break
    print("\nOK")
