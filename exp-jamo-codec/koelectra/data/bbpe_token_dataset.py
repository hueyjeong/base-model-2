"""BBPE + 자모 분해 per-token 데이터셋.

SimpleCodec 철학에 맞춘 포맷: 각 BBPE 토큰이 독립 슬롯.
- jamo_ids [P, S]    — 토큰 P개, 각 토큰당 자모 S 슬롯 (PAD=0)
- jamo_mask [P, S]   — 토큰 내부의 실제 자모 위치
- token_pad_mask [P] — 유효 토큰 위치 (False=문서 끝 이후 빈 슬롯)
- special_token_mask [P] — BOS/EOS 플래그 (마스킹 대상 제외용)
- n_tokens (int)     — 유효 토큰 수 (= token_pad_mask.sum())

segment_ids 없음. max_seq_len 없음. 오직 max_patches × max_jamo_per_token 만 차원.
"""
import json
import os
import re
import sys
from typing import List, Tuple

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# JamoTokenizer specials
JAMO_PAD = 0
JAMO_BOS = 2
JAMO_EOS = 3


def load_bbpe_tokenizer(model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    """K-EXAONE BBPE 토크나이저 로드"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def decompose_token(tok_str: str, jamo_tokenizer) -> List[int]:
    """BBPE 토큰 문자열 → 자모 ID 리스트"""
    return jamo_tokenizer.encode(tok_str, add_special=False)


class BBPETokenDataset(IterableDataset):
    """per-token 포맷 BBPE + 자모 분해 IterableDataset.

    Multi-document packing: `[BOS] doc1 tokens [EOS] [BOS] doc2 tokens [EOS] ...`
    한 샘플은 `max_patches` 토큰까지 채우며, 한 문서가 남은 슬롯에 안 들어가면 flush.

    길이 > `max_jamo_per_token` 인 토큰은 공백 분절 후 문자 단위 fallback.
    """

    def __init__(
        self,
        file_paths,
        bbpe_tokenizer,
        jamo_tokenizer,
        max_patches: int = 512,
        max_jamo_per_token: int = 32,
        text_key: str = "text",
        min_length: int = 10,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.bbpe = bbpe_tokenizer
        self.jamo = jamo_tokenizer
        self.max_patches = max_patches
        self.max_jamo_per_token = max_jamo_per_token
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0
        self._resume_line = 0
        self._tok_cache: dict = {}
        self.encode_batch_size = 64

    # ── iteration helpers ──
    def _iter_texts(self, resume_row: int = 0):
        """parquet/jsonl 스트리밍. (abs_line, text) yield.
        worker filtering 은 호출부에서 처리.
        """
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

    def _prewarm_cache(self, verbose: bool = True):
        """vocab 전체의 (tok_id → jamo 분해) cache 채움.

        학습 중 cache 성장으로 인한 CPU RAM 증가 제거. worker_init_fn 에서 호출.
        """
        import time
        vocab_size = int(self.bbpe.vocab_size)
        t0 = time.time()
        if verbose:
            print(f"[Cache prewarm] {vocab_size} tokens decoding...")
        all_strs = self.bbpe.batch_decode([[tid] for tid in range(vocab_size)])
        for tid, tok_str in enumerate(all_strs):
            if tid in self._tok_cache:
                continue
            base_jamo = decompose_token(tok_str, self.jamo)
            if len(base_jamo) <= self.max_jamo_per_token:
                self._tok_cache[tid] = (base_jamo,)
            else:
                parts_seqs: List[List[int]] = []
                parts = re.split(r"( )", tok_str)
                for part in parts:
                    if not part:
                        continue
                    pj = decompose_token(part, self.jamo)
                    if len(pj) <= self.max_jamo_per_token:
                        parts_seqs.append(pj)
                    else:
                        for ch in part:
                            cj = decompose_token(ch, self.jamo)
                            if cj:
                                parts_seqs.append(cj[:self.max_jamo_per_token])
                self._tok_cache[tid] = tuple(parts_seqs)
        if verbose:
            print(f"[Cache prewarm] done in {time.time()-t0:.1f}s "
                  f"({len(self._tok_cache):,} entries)")

    def _decompose_ids(self, bbpe_ids: List[int]) -> List[List[int]]:
        """BBPE 토큰 ID 리스트 → 각 토큰의 자모 시퀀스 리스트 (cache 사용).
        길이 > max_jamo_per_token 인 토큰은 공백/문자 단위로 나뉘어 여러 seq 반환.
        """
        cache = self._tok_cache
        jamo_seqs = []
        for tid in bbpe_ids:
            entry = cache.get(tid)
            if entry is None:
                tok_str = self.bbpe.decode([tid])
                base_jamo = decompose_token(tok_str, self.jamo)
                if len(base_jamo) <= self.max_jamo_per_token:
                    entry = (base_jamo,)
                else:
                    parts_seqs: List[List[int]] = []
                    parts = re.split(r"( )", tok_str)
                    for part in parts:
                        if not part:
                            continue
                        pj = decompose_token(part, self.jamo)
                        if len(pj) <= self.max_jamo_per_token:
                            parts_seqs.append(pj)
                        else:
                            for ch in part:
                                cj = decompose_token(ch, self.jamo)
                                if cj:
                                    parts_seqs.append(cj[:self.max_jamo_per_token])
                    entry = tuple(parts_seqs)
                cache[tid] = entry
            jamo_seqs.extend(entry)
        return jamo_seqs

    # ── sample 구축 ──
    def _build_sample(self, tokens: List[Tuple[List[int], bool]]):
        """tokens (List[(jamo_seq, is_special)]) → sample dict.

        각 토큰을 max_jamo_per_token 슬롯에 저장하고, 전체를 max_patches 로 pad.
        """
        P = self.max_patches
        S = self.max_jamo_per_token
        n_tokens = len(tokens)
        if n_tokens == 0:
            return None

        jamo_ids = torch.zeros(P, S, dtype=torch.long)
        jamo_mask = torch.zeros(P, S, dtype=torch.bool)
        token_pad_mask = torch.zeros(P, dtype=torch.bool)
        special_token_mask = torch.zeros(P, dtype=torch.bool)

        for p, (seq, is_sp) in enumerate(tokens):
            if is_sp:
                # BBPE-level special 토큰 (BOS/EOS): 전 슬롯을 해당 id 로 채움
                # seq = [JAMO_BOS] 또는 [JAMO_EOS] — 첫 원소 id 를 S 슬롯에 broadcast
                jamo_ids[p, :] = seq[0]
                jamo_mask[p, :] = True
            else:
                L = min(len(seq), S)
                if L > 0:
                    jamo_ids[p, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    jamo_mask[p, :L] = True
            token_pad_mask[p] = True
            special_token_mask[p] = is_sp

        return {
            "jamo_ids": jamo_ids,
            "jamo_mask": jamo_mask,
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
        """worker interleaving + batch BBPE encode.
        (abs_line, bbpe_ids) yield.
        """
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
        """Multi-document packing per-token 포맷으로 yield."""
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
        buffer: List[Tuple[List[int], bool]] = []  # (jamo_seq, is_special)

        while True:
            for abs_line, bbpe_ids in self._iter_encoded_texts(
                resume_line if first_epoch else 0, total_workers, global_worker_id
            ):
                self._line_counter = abs_line + 1

                doc_jamo_seqs = self._decompose_ids(bbpe_ids)
                if not doc_jamo_seqs:
                    continue

                # 각 토큰을 별도 슬롯으로 구성
                doc_tokens: List[Tuple[List[int], bool]] = (
                    [([JAMO_BOS], True)]
                    + [(seq, False) for seq in doc_jamo_seqs]
                    + [([JAMO_EOS], True)]
                )
                doc_n = len(doc_tokens)

                # 단일 문서가 max_patches 초과 → truncate + 단독 샘플
                if doc_n > P:
                    # 기존 buffer flush
                    if buffer:
                        s = self._build_sample(buffer)
                        if s is not None:
                            yield s
                        buffer = []
                    # 앞에서 잘라 [BOS..P-1..EOS] 구성
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

                # 현재 buffer 에 안 들어가면 flush
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

            # epoch 끝
            if buffer:
                s = self._build_sample(buffer)
                if s is not None:
                    yield s
                buffer = []
            resume_line = 0
            first_epoch = False


def _worker_init_fn(worker_id: int):
    """DataLoader worker 시작 시 cache pre-warm."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if hasattr(ds, "_prewarm_cache"):
        ds._prewarm_cache(verbose=(worker_id == 0))


# ── smoke test ──
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from tok.jamo_tokenizer import JamoTokenizer

    print("=== BBPETokenDataset smoke test ===")
    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()
    print(f"BBPE vocab: {bbpe.vocab_size:,}")
    print(f"Jamo vocab: {jamo.vocab_size}")

    import os as _os
    corpus = _os.environ.get(
        "CORPUS",
        "corpus/k-exaone_coverage_5_len1000.parquet",
    )
    ds = BBPETokenDataset(
        file_paths=[corpus],
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_patches=128, max_jamo_per_token=32,
        text_key="text",
    )
    ds._prewarm_cache(verbose=True)

    for i, sample in enumerate(ds):
        if i >= 3:
            break
        P, S = sample["jamo_ids"].shape
        n_tok = sample["n_tokens"]
        n_spec = sample["special_token_mask"].sum().item()
        total_jamo = sample["jamo_mask"].sum().item()
        print(f"\nSample {i}:")
        print(f"  jamo_ids: {tuple(sample['jamo_ids'].shape)}  "
              f"jamo_mask: {tuple(sample['jamo_mask'].shape)}  "
              f"token_pad_mask: {tuple(sample['token_pad_mask'].shape)}")
        print(f"  n_tokens={n_tok} (specials={n_spec}, P={P})  "
              f"total_jamo={total_jamo}  avg_jamo/tok={total_jamo/max(n_tok,1):.1f}")
        # 앞 3토큰 decode
        for p in range(min(n_tok, 3)):
            seq = sample["jamo_ids"][p][sample["jamo_mask"][p]].tolist()
            print(f"    tok{p}: {jamo.decode(seq, skip_special=False)!r}  "
                  f"special={bool(sample['special_token_mask'][p])}")
    print("\nOK")
