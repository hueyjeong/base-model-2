"""BBPE + 자모 분해 데이터셋 (concat 방식)

K-EXAONE 153K BBPE로 토큰 경계 결정 → 각 토큰을 자모/byte 분해 → 1열 concat.
segment_ids로 토큰 경계를 표시.
"""
import json
import os
import re
import sys
from typing import List

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ── 상수 ──
JAMO_PAD = 0  # JamoTokenizer의 PAD ID


def load_bbpe_tokenizer(model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B"):
    """K-EXAONE BBPE 토크나이저 로드"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def decompose_token(tok_str: str, jamo_tokenizer) -> List[int]:
    """BBPE 토큰 문자열 → 자모/byte ID 리스트 (special 토큰 없이)"""
    return jamo_tokenizer.encode(tok_str, add_special=False)


class BBPEJamoDataset(IterableDataset):
    """BBPE 토큰화 → 자모 분해 → concat 스트리밍 데이터셋

    각 샘플:
        jamo_ids: [max_seq_len] — concat된 자모 ID
        jamo_mask: [max_seq_len] — 유효 자모 위치
        segment_ids: [max_seq_len] — 각 자모가 속한 토큰 ID
        n_segments: int — 토큰 수
    """

    def __init__(
        self,
        file_paths,
        bbpe_tokenizer,
        jamo_tokenizer,
        max_seq_len: int = 512,
        max_jamo_per_token: int = 32,
        text_key: str = "text",
        min_length: int = 10,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.bbpe = bbpe_tokenizer
        self.jamo = jamo_tokenizer
        self.max_seq_len = max_seq_len
        self.max_jamo_per_token = max_jamo_per_token
        self.text_key = text_key
        self.min_length = min_length
        self.rank = rank
        self.world_size = world_size

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

    def _tokenize_and_decompose(self, text: str) -> List[List[int]]:
        """텍스트 → BBPE 토큰화 → 각 토큰 자모 분해"""
        bbpe_ids = self.bbpe.encode(text, add_special_tokens=False)
        jamo_seqs = []
        for tid in bbpe_ids:
            tok_str = self.bbpe.decode([tid])
            jamo_ids = decompose_token(tok_str, self.jamo)
            if len(jamo_ids) <= self.max_jamo_per_token:
                jamo_seqs.append(jamo_ids)
            else:
                # 32자모 초과 → 공백 기준 어절 분절
                parts = re.split(r'( )', tok_str)
                for part in parts:
                    if not part:
                        continue
                    pj = decompose_token(part, self.jamo)
                    if len(pj) <= self.max_jamo_per_token:
                        jamo_seqs.append(pj)
                    else:
                        for ch in part:
                            cj = decompose_token(ch, self.jamo)
                            if cj:
                                jamo_seqs.append(cj[:self.max_jamo_per_token])
        return jamo_seqs

    def _make_sample(self, jamo_seqs: List[List[int]]):
        """자모 시퀀스 리스트 → concat 텐서 + segment_ids"""
        # concat
        all_jamo = []
        seg_ids = []
        seg_idx = 0
        for seq in jamo_seqs:
            if len(all_jamo) + len(seq) > self.max_seq_len:
                break
            all_jamo.extend(seq)
            seg_ids.extend([seg_idx] * len(seq))
            seg_idx += 1

        L = len(all_jamo)
        n_segments = seg_idx

        if L == 0:
            return None

        # 패딩
        pad_len = self.max_seq_len - L
        jamo_ids = torch.tensor(all_jamo + [JAMO_PAD] * pad_len, dtype=torch.long)
        jamo_mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)
        # 패딩 영역의 segment_ids는 마지막 segment로 (scatter_add에서 무시됨 — mask로 처리)
        segment_ids = torch.tensor(seg_ids + [0] * pad_len, dtype=torch.long)

        return {
            "jamo_ids": jamo_ids,         # [max_seq_len]
            "jamo_mask": jamo_mask,       # [max_seq_len]
            "segment_ids": segment_ids,   # [max_seq_len]
            "n_segments": n_segments,
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
                sample = self._make_sample(jamo_seqs)
                if sample is not None:
                    yield sample


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tok.jamo_tokenizer import JamoTokenizer

    print("=== BBPEJamoDataset (concat) Smoke Test ===\n")

    bbpe = load_bbpe_tokenizer()
    jamo = JamoTokenizer()

    print(f"BBPE vocab: {bbpe.vocab_size:,}")
    print(f"Jamo vocab: {jamo.vocab_size}")
    print()

    # 단일 텍스트 예시
    texts = [
        "맞춤법을 확인해 주세요.",
        "김철수 씨가 프로그래밍을 배우기 시작했습니다.",
    ]

    ds = BBPEJamoDataset(
        file_paths=["corpus/val.parquet"],
        bbpe_tokenizer=bbpe, jamo_tokenizer=jamo,
        max_seq_len=512, text_key="text",
    )

    for text in texts:
        jamo_seqs = ds._tokenize_and_decompose(text)
        total_jamo = sum(len(s) for s in jamo_seqs)
        print(f"원문: {text}")
        print(f"  {len(jamo_seqs)}토큰, {total_jamo}자모 (concat)")
        for j, seq in enumerate(jamo_seqs):
            decoded = jamo.decode(seq, skip_special=False)
            print(f"    seg{j}: [{decoded}] ({len(seq)}자모)")
        print()

    # 데이터셋 테스트
    print("--- 데이터셋 테스트 ---")
    for i, sample in enumerate(ds):
        if i >= 3:
            break
        L = sample["jamo_mask"].sum().item()
        n_seg = sample["n_segments"]
        print(f"Sample {i}: jamo_ids={sample['jamo_ids'].shape}, "
              f"유효={L}/{sample['jamo_ids'].size(0)} ({L/sample['jamo_ids'].size(0)*100:.0f}%), "
              f"segments={n_seg}")

    print("\n전체 테스트 통과!")
