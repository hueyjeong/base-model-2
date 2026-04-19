"""BBPE-only 데이터셋 (자모 분해 없음).

Binary ELECTRA 용. BBPETokenDataset 의 jamo 분해 부분을 제거하고 BBPE 토큰 ID
시퀀스만 packing 한다.

샘플 형식:
  bbpe_ids [P]            — BBPE 토큰 ID
  token_pad_mask [P]      — 유효 토큰 위치
  special_token_mask [P]  — BOS/EOS (마스킹 대상 제외)
  n_tokens (int)          — 유효 토큰 수
"""
import json
import os
import sys
from typing import List, Tuple

import torch
from torch.utils.data import IterableDataset


def load_bbpe_tokenizer(model_id_or_path: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    """HuggingFace model id 또는 로컬 디렉토리 경로 모두 지원.
    로컬 경로면 AutoTokenizer 가 디렉토리 감지해 PreTrainedTokenizerFast 로 로드.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)


class BBPEDataset(IterableDataset):
    """per-token BBPE-only IterableDataset.

    Multi-document packing: `[BOS] doc1 [EOS] [BOS] doc2 [EOS] ...`
    한 샘플은 max_patches 토큰까지 채우며, 한 문서가 남은 슬롯에 안 들어가면 flush.
    단일 문서 > max_patches 는 [BOS .. truncate .. EOS] 단독 샘플.
    """

    def __init__(
        self,
        file_paths,
        bbpe_tokenizer,
        max_patches: int = 512,
        text_key: str = "text",
        min_length: int = 10,
        rank: int = 0,
        world_size: int = 1,
        bos_id: int | None = None,
        eos_id: int | None = None,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.bbpe = bbpe_tokenizer
        self.max_patches = max_patches
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size
        # K-EXAONE: bos=1, eos=53
        self.bos_id = bos_id if bos_id is not None else int(bbpe_tokenizer.bos_token_id)
        self.eos_id = eos_id if eos_id is not None else int(bbpe_tokenizer.eos_token_id)
        self._line_counter = 0
        self._resume_line = 0
        self.encode_batch_size = 64

    # ── iteration helpers ──
    def _iter_texts(self, resume_row: int = 0):
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")

            if is_parquet:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)
                text_col = self.text_key or "text"

                rows_skipped = 0
                rg_start = 0
                target_offset = 0
                for rg_idx in range(pf.num_row_groups):
                    rg_rows = pf.metadata.row_group(rg_idx).num_rows
                    if rows_skipped + rg_rows <= resume_row:
                        rows_skipped += rg_rows
                        continue
                    rg_start = rg_idx
                    target_offset = resume_row - rows_skipped
                    break

                abs_line = rows_skipped
                for batch in pf.iter_batches(
                    batch_size=65536, columns=[text_col],
                    row_groups=list(range(rg_start, pf.num_row_groups)),
                ):
                    col = batch[text_col]
                    n = len(col)
                    start = target_offset if target_offset > 0 else 0
                    for i in range(start, n):
                        abs_idx = abs_line + i
                        text = col[i].as_py()
                        if text and len(text) >= self.min_length:
                            yield abs_idx, text
                    abs_line += n
                    target_offset = 0
                continue

            abs_line = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < self.min_length:
                        abs_line += 1
                        continue
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            abs_line += 1
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                    else:
                        text = line
                    if len(text) >= self.min_length:
                        yield abs_line, text
                    abs_line += 1

    # ── sample 구축 ──
    def _build_sample(self, tokens: List[Tuple[int, bool]]):
        """tokens (List[(bbpe_id, is_special)]) → sample dict."""
        P = self.max_patches
        n_tokens = len(tokens)
        if n_tokens == 0:
            return None

        bbpe_ids = torch.zeros(P, dtype=torch.long)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        special_token_mask = torch.zeros(P, dtype=torch.bool)

        for p, (tid, is_sp) in enumerate(tokens):
            bbpe_ids[p] = tid
            token_pad_mask[p] = True
            special_token_mask[p] = is_sp

        return {
            "bbpe_ids": bbpe_ids,
            "token_pad_mask": token_pad_mask,
            "special_token_mask": special_token_mask,
            "n_tokens": n_tokens,
            "_line_counter": self._line_counter,
        }

    # ── state dict ──
    def state_dict(self) -> dict:
        return {"line_counter": self._line_counter}

    def load_state_dict(self, state: dict) -> None:
        self._resume_line = state.get("line_counter", 0)
        self._line_counter = self._resume_line

    # ── main iterator ──
    def _iter_encoded_texts(self, resume_line: int, total_workers: int, global_worker_id: int):
        """worker interleaving + batch BBPE encode. (abs_line, bbpe_ids) yield."""
        rust_tok = self.bbpe.backend_tokenizer
        encode_bs = self.encode_batch_size
        buf_abs: List[int] = []
        buf_text: List[str] = []
        for abs_line, text in self._iter_texts(resume_row=resume_line):
            if abs_line % total_workers != global_worker_id:
                continue
            if abs_line < resume_line:
                continue
            buf_abs.append(abs_line)
            buf_text.append(text)
            if len(buf_text) >= encode_bs:
                encodings = rust_tok.encode_batch(buf_text, add_special_tokens=False)
                for a, enc in zip(buf_abs, encodings):
                    yield a, enc.ids
                buf_abs = []
                buf_text = []
        if buf_text:
            encodings = rust_tok.encode_batch(buf_text, add_special_tokens=False)
            for a, enc in zip(buf_abs, encodings):
                yield a, enc.ids

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        resume_line = self._resume_line
        if resume_line > 0:
            self._resume_line = 0
        first_epoch = True

        P = self.max_patches
        buffer: List[Tuple[int, bool]] = []  # (bbpe_id, is_special)

        while True:
            for abs_line, bbpe_ids in self._iter_encoded_texts(
                resume_line if first_epoch else 0, total_workers, global_worker_id
            ):
                self._line_counter = abs_line + 1
                if not bbpe_ids:
                    continue

                doc_tokens: List[Tuple[int, bool]] = (
                    [(self.bos_id, True)]
                    + [(int(t), False) for t in bbpe_ids]
                    + [(self.eos_id, True)]
                )
                doc_n = len(doc_tokens)

                # 단일 문서가 max_patches 초과 → truncate + 단독 샘플
                if doc_n > P:
                    if buffer:
                        s = self._build_sample(buffer)
                        if s is not None:
                            yield s
                        buffer = []
                    kept = [doc_tokens[0]]  # BOS
                    for tok in doc_tokens[1:-1]:
                        if len(kept) + 1 >= P:  # EOS 자리 확보
                            break
                        kept.append(tok)
                    kept.append(doc_tokens[-1])  # EOS
                    s = self._build_sample(kept)
                    if s is not None:
                        yield s
                    continue

                if buffer and len(buffer) + doc_n > P:
                    s = self._build_sample(buffer)
                    if s is not None:
                        yield s
                    buffer = []

                buffer.extend(doc_tokens)

                if len(buffer) == P:
                    s = self._build_sample(buffer)
                    if s is not None:
                        yield s
                    buffer = []

            if buffer:
                s = self._build_sample(buffer)
                if s is not None:
                    yield s
                buffer = []
            resume_line = 0
            first_epoch = False


# ── smoke test ──
if __name__ == "__main__":
    print("=== BBPEDataset smoke test ===")
    bbpe = load_bbpe_tokenizer()
    print(f"BBPE vocab: {bbpe.vocab_size:,}")
    print(f"BOS={bbpe.bos_token_id}, EOS={bbpe.eos_token_id}")

    corpus = os.environ.get(
        "CORPUS", "corpus/k-exaone_coverage_5_len1000.parquet",
    )
    ds = BBPEDataset(
        file_paths=[corpus],
        bbpe_tokenizer=bbpe, max_patches=128,
        text_key="text",
    )

    for i, sample in enumerate(ds):
        if i >= 3:
            break
        n_tok = sample["n_tokens"]
        n_spec = sample["special_token_mask"].sum().item()
        print(f"\nSample {i}:")
        print(f"  bbpe_ids: {tuple(sample['bbpe_ids'].shape)}  "
              f"token_pad_mask: {tuple(sample['token_pad_mask'].shape)}")
        print(f"  n_tokens={n_tok} (specials={n_spec})  "
              f"first 5 ids: {sample['bbpe_ids'][:5].tolist()}")
        # 앞 3토큰 decode
        for p in range(min(n_tok, 5)):
            tid = sample["bbpe_ids"][p].item()
            is_sp = bool(sample["special_token_mask"][p])
            print(f"    tok{p}: id={tid} {bbpe.decode([tid])!r} special={is_sp}")
    print("\nOK")
