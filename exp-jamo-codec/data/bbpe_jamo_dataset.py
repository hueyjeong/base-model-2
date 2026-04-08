"""BBPE + 자모 분해 데이터셋

K-EXAONE 153K BBPE로 토큰 경계 결정 → 각 토큰을 자모/byte 분해.
CompositionCodec 학습용 데이터 파이프라인.
"""
import json
import os
import sys
from typing import List, Tuple

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── 상수 ──
MAX_JAMO_LEN = 32  # 99.9%ile=26, 32면 충분
JAMO_PAD = 0       # JamoTokenizer의 PAD ID


def load_bbpe_tokenizer(model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    """K-EXAONE BBPE 토크나이저 로드"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def build_long_token_set(bbpe_tokenizer, jamo_tokenizer,
                         max_jamo_len: int = MAX_JAMO_LEN) -> set:
    """32자모 초과 토큰 ID 집합 생성 (초기화 시 1회)"""
    long_ids = set()
    for i in range(bbpe_tokenizer.vocab_size):
        try:
            tok_str = bbpe_tokenizer.decode([i])
            if not tok_str:
                continue
            jamo_ids = jamo_tokenizer.encode(tok_str, add_special=False)
            if len(jamo_ids) > max_jamo_len:
                long_ids.add(i)
        except Exception:
            pass
    return long_ids


def decompose_token(tok_str: str, jamo_tokenizer) -> List[int]:
    """BBPE 토큰 문자열 → 자모/byte ID 리스트 (special 토큰 없이)"""
    return jamo_tokenizer.encode(tok_str, add_special=False)


class BBPEJamoDataset(IterableDataset):
    """BBPE 토큰화 → 자모 분해 스트리밍 데이터셋

    각 샘플:
        jamo_ids: [max_tokens, max_jamo_len] — 토큰별 자모 ID
        jamo_mask: [max_tokens, max_jamo_len] — 유효 자모 위치
        token_mask: [max_tokens] — 유효 토큰 위치
    """

    def __init__(
        self,
        file_paths,
        bbpe_tokenizer,
        jamo_tokenizer,
        max_tokens: int = 128,
        max_jamo_len: int = MAX_JAMO_LEN,
        text_key: str = "text",
        min_length: int = 10,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.bbpe = bbpe_tokenizer
        self.jamo = jamo_tokenizer
        self.max_tokens = max_tokens
        self.max_jamo_len = max_jamo_len
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size
        # 32자모 초과 토큰 집합 (이 토큰은 문자열로 풀어서 재분절)
        self.long_token_ids = build_long_token_set(bbpe_tokenizer, jamo_tokenizer, max_jamo_len)

    def _iter_texts(self):
        """파일에서 텍스트 스트리밍"""
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")

            if is_parquet:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)
                text_col = self.text_key or "text"
                for batch in pf.iter_batches(batch_size=65536, columns=[text_col]):
                    for text in batch[text_col].to_pylist():
                        if text and len(text) >= self.min_length:
                            yield text
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < self.min_length:
                        continue
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                    else:
                        text = line
                    if len(text) >= self.min_length:
                        yield text

    def _tokenize_and_decompose(self, text: str) -> Tuple[List[List[int]], List[str]]:
        """텍스트 → BBPE 토큰화 → 각 토큰 자모 분해

        Returns:
            jamo_seqs: 토큰별 자모 ID 리스트
            tok_strs: 토큰별 원문 문자열 (디버깅용)
        """
        bbpe_ids = self.bbpe.encode(text, add_special_tokens=False)
        jamo_seqs = []
        for tid in bbpe_ids:
            if tid in self.long_token_ids:
                continue  # 32자모 초과 토큰 스킵 (BBPE가 더 작은 단위로 분절했을 텍스트)
            tok_str = self.bbpe.decode([tid])
            jamo_ids = decompose_token(tok_str, self.jamo)
            jamo_seqs.append(jamo_ids[:self.max_jamo_len])  # 안전하게 truncate
        return jamo_seqs

    def _make_sample(self, jamo_seqs: List[List[int]]):
        """자모 시퀀스 리스트 → 패딩된 텐서"""
        n_tokens = min(len(jamo_seqs), self.max_tokens)
        jamo_seqs = jamo_seqs[:n_tokens]

        jamo_ids = torch.full(
            (self.max_tokens, self.max_jamo_len), JAMO_PAD, dtype=torch.long,
        )
        jamo_mask = torch.zeros(self.max_tokens, self.max_jamo_len, dtype=torch.bool)
        token_mask = torch.zeros(self.max_tokens, dtype=torch.bool)

        for i, seq in enumerate(jamo_seqs):
            L = len(seq)
            jamo_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
            jamo_mask[i, :L] = True
            token_mask[i] = True

        return {
            "jamo_ids": jamo_ids,       # [max_tokens, max_jamo_len]
            "jamo_mask": jamo_mask,      # [max_tokens, max_jamo_len]
            "token_mask": token_mask,    # [max_tokens]
            "n_tokens": n_tokens,
        }

    def __iter__(self):
        """스트리밍 + DDP 샤딩 + 무한 순환"""
        while True:
            for i, text in enumerate(self._iter_texts()):
                if self.world_size > 1 and i % self.world_size != self.rank:
                    continue
                jamo_seqs = self._tokenize_and_decompose(text)
                if not jamo_seqs:
                    continue
                yield self._make_sample(jamo_seqs)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tok.jamo_tokenizer import JamoTokenizer

    print("=== BBPEJamoDataset Smoke Test ===\n")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    print(f"BBPE vocab: {bbpe.vocab_size:,}")
    print(f"Jamo vocab: {jamo.vocab_size}")
    print(f"Max jamo len: {MAX_JAMO_LEN}")
    print()

    # 단일 텍스트 분해 예시
    tests = [
        "맞춤법을 확인해 주세요.",
        "김철수 씨가 프로그래밍을 배우기 시작했습니다.",
        "맞춤뻡을 확인해 주세요.",
    ]

    for text in tests:
        bbpe_ids = bbpe.encode(text, add_special_tokens=False)
        print(f"원문: {text}")
        print(f"BBPE ({len(bbpe_ids)}tok): ", end="")
        for tid in bbpe_ids:
            tok_str = bbpe.decode([tid])
            jamo_ids = decompose_token(tok_str, jamo)
            decoded = jamo.decode(jamo_ids, skip_special=False)
            print(f"[{tok_str.strip()}→{len(jamo_ids)}자모]", end=" ")
        print("\n")

    # 데이터셋 테스트
    print("--- 데이터셋 테스트 (val.parquet) ---")
    ds = BBPEJamoDataset(
        file_paths=["corpus/val.parquet"],
        bbpe_tokenizer=bbpe,
        jamo_tokenizer=jamo,
        max_tokens=64,
        text_key="text",
    )

    for i, sample in enumerate(ds):
        if i >= 3:
            break
        print(f"Sample {i}: jamo_ids={sample['jamo_ids'].shape}, "
              f"n_tokens={sample['n_tokens']}, "
              f"유효 자모 비율={sample['jamo_mask'].float().mean():.2f}")

    print("\n전체 테스트 통과!")
