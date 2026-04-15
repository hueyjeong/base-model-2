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


# ── 상수 (JamoTokenizer specials) ──
JAMO_PAD = 0
JAMO_BOS = 2
JAMO_EOS = 3
JAMO_SEP = 5


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
        max_patches: int = None,
        append_pad_slot: bool = False,
        fixed_slot: bool = False,
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
        self.max_patches = max_patches  # BBPE 토큰(패치) 수 상한 — None이면 자모 길이만 제한
        # 각 세그먼트 끝에 JAMO_PAD 1개를 추가해 decoder가 "PAD 출력"을 학습하게 함
        # (codec 단독 추론에서 PAD 조기 종료 용도)
        # append_pad_slot=True:  seg 끝에 +1 PAD (가변 길이, decoder 일반화 약함)
        # fixed_slot=True:       모든 seg를 max_jamo_per_token으로 padding (decode_from_vec 완벽 대응)
        self.append_pad_slot = append_pad_slot
        self.fixed_slot = fixed_slot
        if fixed_slot and append_pad_slot:
            # fixed_slot이 우선 (각 seg를 max_jamo_per_token으로 pad → +1 PAD 무의미)
            self.append_pad_slot = False
        self._line_counter = 0
        self._resume_line = 0
        # token_id → (tok_str, jamo_ids) lazy cache
        # self.bbpe.decode([tid])가 Python↔Rust 왕복 overhead 크므로,
        # 같은 토큰 반복 등장 시 cache 활용. Coverage 코퍼스는 150K 전부 등장하니
        # 첫 epoch 후 cache hit 100% 근접. DataLoader worker마다 별도 cache
        # (fork 시 copy-on-write; Python ref count로 실제 복사될 수 있지만 ~15MB).
        self._tok_cache: dict = {}

    def _iter_texts(self, resume_row: int = 0):
        """파일에서 텍스트 스트리밍

        Args:
            resume_row: 이 줄 번호부터 시작 (0이면 처음부터).
                       Parquet: row group 단위 skip으로 빠른 이동.

        Yields:
            (abs_line, text) — 절대 줄 번호와 텍스트
        """
        for fpath in self.file_paths:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            is_parquet = fpath.endswith(".parquet")

            if is_parquet:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(fpath)
                text_col = self.text_key or "text"

                # row group 단위 skip: resume_row 이전 row group 전부 건너뜀
                rows_skipped = 0
                rg_start = 0
                target_offset = 0

                for rg_idx in range(pf.num_row_groups):
                    rg_rows = pf.metadata.row_group(rg_idx).num_rows
                    if rows_skipped + rg_rows <= resume_row:
                        rows_skipped += rg_rows
                        continue
                    # 이 row group부터 읽기 시작
                    rg_start = rg_idx
                    target_offset = resume_row - rows_skipped
                    break

                # 지정된 row group부터 읽기
                abs_line = rows_skipped
                for batch in pf.iter_batches(
                    batch_size=65536, columns=[text_col],
                    row_groups=list(range(rg_start, pf.num_row_groups)),
                ):
                    texts = batch[text_col].to_pylist()
                    start = target_offset if target_offset > 0 else 0
                    for i, text in enumerate(texts):
                        if i < start:
                            continue
                        if text and len(text) >= self.min_length:
                            yield abs_line + i, text
                    abs_line += len(texts)
                    target_offset = 0  # 첫 배치만 offset 적용
                continue

            # JSONL/텍스트: 순차 읽기
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

    def _tokenize_and_decompose(self, text: str) -> List[List[int]]:
        """텍스트 → BBPE 토큰화 → 각 토큰 자모 분해 (캐시 사용).

        - encode는 문서 단위 1회 호출 (Rust 구현 그대로 빠름)
        - decode + decompose는 토큰마다 반복되므로 token_id별 cache로 가속
        """
        bbpe_ids = self.bbpe.encode(text, add_special_tokens=False)
        cache = self._tok_cache
        jamo_seqs = []
        for tid in bbpe_ids:
            entry = cache.get(tid)
            if entry is None:
                tok_str = self.bbpe.decode([tid])
                base_jamo = decompose_token(tok_str, self.jamo)
                if len(base_jamo) <= self.max_jamo_per_token:
                    entry = (base_jamo,)  # single seq
                else:
                    # 32자모 초과 → 공백 기준 어절 분절 (이 path는 드묾)
                    parts_seqs: List[List[int]] = []
                    parts = re.split(r'( )', tok_str)
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

    def _build_sample(self, buffer_jamo, buffer_segs, buffer_special, n_segments):
        """buffer 내용으로 샘플 텐서 구축 (multi-document packing 경로).

        Args:
            buffer_jamo: 자모 ID 리스트 (연속 concat)
            buffer_segs: 각 자모의 segment id 리스트
            buffer_special: 각 segment가 special(BOS/EOS/SEP)인지 bool 리스트 (len=n_segments)
            n_segments: 세그먼트(패치) 수
        """
        L = len(buffer_jamo)
        if L == 0 or n_segments == 0:
            return None

        pad_len = self.max_seq_len - L
        jamo_ids = torch.tensor(buffer_jamo + [JAMO_PAD] * pad_len, dtype=torch.long)
        jamo_mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)
        segment_ids = torch.tensor(buffer_segs + [0] * pad_len, dtype=torch.long)

        # special_patch_mask: 고정 크기 필요 (배치 stack용).
        # max_patches 지정 시 그 값, 아니면 max_seq_len(자모 수는 항상 패치 수 ≥)을 상한으로 사용.
        P = self.max_patches if self.max_patches is not None else self.max_seq_len
        spec = buffer_special + [False] * (P - n_segments)
        special_patch_mask = torch.tensor(spec[:P], dtype=torch.bool)

        return {
            "jamo_ids": jamo_ids,
            "jamo_mask": jamo_mask,
            "segment_ids": segment_ids,
            "n_segments": n_segments,
            "special_patch_mask": special_patch_mask,
            "_line_counter": self._line_counter,
        }

    def state_dict(self) -> dict:
        """데이터셋 진행 상태 반환 (체크포인트용)"""
        return {
            "line_counter": self._line_counter,
        }

    def load_state_dict(self, state: dict) -> None:
        """데이터셋 진행 상태 복원 (resume용)"""
        self._resume_line = state.get("line_counter", 0)
        self._line_counter = self._resume_line

    def __iter__(self):
        """Multi-document packing: [BOS]문서1[EOS][BOS]문서2[EOS]...를 상한까지 이어붙여 yield.

        상한 도달 시점에 현재 buffer를 yield하고 새 buffer로 시작. 새 문서가
        buffer에 통째로 안 들어가면 남은 자리에 넣을 수 있는 만큼만 넣고 flush.

        BOS/EOS 패치는 special_patch_mask=True로 표시되어 Generator MLM 마스킹
        대상에서 제외된다.

        DDP×Worker 샤딩은 문서 단위(abs_line)로 interleaving. 한 샘플이 여러
        문서를 묶을 수 있으므로 엄밀한 비트 재현성은 없지만 데이터 분포는 동일.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        resume_line = self._resume_line
        if resume_line > 0:
            self._resume_line = 0

        first_epoch = True

        # 패킹 buffer
        buf_jamo: List[int] = []
        buf_segs: List[int] = []
        buf_special: List[bool] = []
        seg_idx = 0

        while True:
            resume_row = resume_line if first_epoch and resume_line > 0 else 0

            for abs_line, text in self._iter_texts(resume_row=resume_row):
                # DDP × Worker interleaving — 문서 단위
                if abs_line % total_workers != global_worker_id:
                    continue
                if abs_line < resume_line:
                    continue

                self._line_counter = abs_line + 1

                doc_jamo_seqs = self._tokenize_and_decompose(text)
                if not doc_jamo_seqs:
                    continue

                # 한 문서를 [BOS] + 토큰들 + [EOS] 로 감싸기 (atomic packing)
                doc_segments = (
                    [([JAMO_BOS], True)]
                    + [(seq, False) for seq in doc_jamo_seqs]
                    + [([JAMO_EOS], True)]
                )
                doc_n_segs = len(doc_segments)
                if self.fixed_slot:
                    # 모든 segment를 max_jamo_per_token 슬롯으로 고정
                    doc_total_jamo = doc_n_segs * self.max_jamo_per_token
                else:
                    doc_total_jamo = sum(len(s) for s, _ in doc_segments)
                    if self.append_pad_slot:
                        doc_total_jamo += doc_n_segs  # 각 segment 끝에 PAD 1개 추가

                # 단일 문서가 상한을 초과하면 truncate해서 단독 샘플로 처리
                # (EOS 포함 위해 뒤에서부터 자름)
                if (doc_total_jamo > self.max_seq_len or
                        (self.max_patches is not None and doc_n_segs > self.max_patches)):
                    trunc = self._truncate_doc(doc_segments)
                    if trunc is not None:
                        # 기존 buffer 먼저 flush
                        if seg_idx > 0:
                            s = self._build_sample(buf_jamo, buf_segs, buf_special, seg_idx)
                            if s is not None:
                                yield s
                            buf_jamo, buf_segs, buf_special, seg_idx = [], [], [], 0
                        yield trunc
                    continue

                # 문서 전체가 현재 buffer에 안 들어가면 buffer flush
                if seg_idx > 0:
                    would_jamo = len(buf_jamo) + doc_total_jamo
                    would_segs = seg_idx + doc_n_segs
                    doesnt_fit = (
                        would_jamo > self.max_seq_len or
                        (self.max_patches is not None and would_segs > self.max_patches)
                    )
                    if doesnt_fit:
                        s = self._build_sample(buf_jamo, buf_segs, buf_special, seg_idx)
                        if s is not None:
                            yield s
                        buf_jamo, buf_segs, buf_special, seg_idx = [], [], [], 0

                # 문서 통째 buffer에 추가 (BOS..EOS 정합성 유지)
                for seq, is_special in doc_segments:
                    if self.fixed_slot:
                        # 각 segment를 max_jamo_per_token 슬롯으로 padding
                        seq_t = list(seq[:self.max_jamo_per_token])
                        pad_n = self.max_jamo_per_token - len(seq_t)
                        seq_t = seq_t + [JAMO_PAD] * pad_n
                        buf_jamo.extend(seq_t)
                        buf_segs.extend([seg_idx] * self.max_jamo_per_token)
                    else:
                        buf_jamo.extend(seq)
                        buf_segs.extend([seg_idx] * len(seq))
                        if self.append_pad_slot:
                            buf_jamo.append(JAMO_PAD)
                            buf_segs.append(seg_idx)
                    buf_special.append(is_special)
                    seg_idx += 1

            # epoch 끝 — 남은 buffer도 yield
            if seg_idx > 0:
                sample = self._build_sample(buf_jamo, buf_segs, buf_special, seg_idx)
                if sample is not None:
                    yield sample
                buf_jamo, buf_segs, buf_special, seg_idx = [], [], [], 0

            resume_line = 0
            first_epoch = False

    def _truncate_doc(self, doc_segments):
        """단일 문서가 상한 초과 시 앞에서부터 잘라 BOS..일부..EOS 구조로 구성.

        [BOS] + 앞 토큰들 + [EOS]로 잘라 단독 샘플 반환.
        """
        buf_jamo = []
        buf_segs = []
        buf_special = []
        seg_idx = 0
        max_L = self.max_seq_len
        max_P = self.max_patches

        # 각 segment에 추가되는 자모 수
        if self.fixed_slot:
            slot_size = self.max_jamo_per_token  # 모든 seg가 이 크기
        elif self.append_pad_slot:
            slot_size = None  # 가변 (len(seq)+1)
        else:
            slot_size = None  # 가변 (len(seq))

        def _add_seg(seq, is_sp):
            nonlocal seg_idx
            if self.fixed_slot:
                seq_t = list(seq[:self.max_jamo_per_token])
                pad_n = self.max_jamo_per_token - len(seq_t)
                buf_jamo.extend(seq_t + [JAMO_PAD] * pad_n)
                buf_segs.extend([seg_idx] * self.max_jamo_per_token)
            else:
                buf_jamo.extend(seq)
                buf_segs.extend([seg_idx] * len(seq))
                if self.append_pad_slot:
                    buf_jamo.append(JAMO_PAD)
                    buf_segs.append(seg_idx)
            buf_special.append(is_sp)
            seg_idx += 1

        def _seg_jamo_cost(seq):
            if self.fixed_slot:
                return self.max_jamo_per_token
            return len(seq) + (1 if self.append_pad_slot else 0)

        # [BOS] 추가 (무조건)
        bos_seq, bos_sp = doc_segments[0]
        _add_seg(bos_seq, bos_sp)

        # 중간 토큰들: [EOS] 자리 남겨두고 채움 (마지막 segment가 [EOS])
        eos_seq, eos_sp = doc_segments[-1]
        reserve_jamo = _seg_jamo_cost(eos_seq)
        reserve_seg = 1
        for seq, is_sp in doc_segments[1:-1]:
            if len(buf_jamo) + _seg_jamo_cost(seq) + reserve_jamo > max_L:
                break
            if max_P is not None and seg_idx + reserve_seg >= max_P:
                break
            _add_seg(seq, is_sp)

        # [EOS] 추가
        if len(buf_jamo) + _seg_jamo_cost(eos_seq) <= max_L and \
                (max_P is None or seg_idx < max_P):
            _add_seg(eos_seq, eos_sp)

        return self._build_sample(buf_jamo, buf_segs, buf_special, seg_idx)


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
