"""편집 태깅 데이터셋

텍스트 레벨 노이즈만 적용하여 (noised → original) 쌍을 생성하고,
Levenshtein 정렬로 편집 태그를 계산한다.

토큰 레벨 노이즈(masking/deletion/infilling)는 사용하지 않음:
- [MASK] 토큰은 편집 태그에 무의미
- 삭제/인필링은 인위적 패턴 생성
- 텍스트 레벨 노이즈만으로 실제 오류 학습에 충분

패킹 모드 (기본):
- 여러 문장을 max_seq_len까지 연결하여 PAD 낭비 제거
- 인코더-only 모델이므로 문서 경계 무시해도 안전
- n_iterations > 1 시 자동으로 개별 패딩 모드로 전환
"""
import json
import os
import sys
import random

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer_base import BaseTokenizer
from training.noising import DenoisingNoiser
from model.edit_tags import compute_edit_tags as _compute_edit_tags_py, TAG_KEEP

# C++ Levenshtein 가속 (DataLoader 워커에서도 사용 가능)
_lev_ext = None
_compute_edit_tags_cpp = None
try:
    from torch.utils.cpp_extension import load as _cpp_load
    _ext_src = os.path.join(os.path.dirname(__file__), "..", "model", "levenshtein_ext.cpp")
    if os.path.exists(_ext_src):
        _lev_ext = _cpp_load(
            name="levenshtein_ext",
            sources=[_ext_src],
            extra_cflags=["-O3", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
        _compute_edit_tags_cpp = _lev_ext.compute_edit_tags
except Exception:
    pass


def compute_edit_tags(source_ids, target_ids, vocab_size):
    """C++ 가속 compute_edit_tags (폴백: Python)"""
    if _compute_edit_tags_cpp is not None:
        return list(_compute_edit_tags_cpp(source_ids, target_ids, vocab_size))
    return _compute_edit_tags_py(source_ids, target_ids, vocab_size)


class EditorDataset(IterableDataset):
    """편집 태깅 학습용 스트리밍 데이터셋

    JSONL/TXT → 텍스트 레벨 노이즈 → 토크나이징 → Levenshtein 편집 태그

    Args:
        file_paths: 코퍼스 파일 경로
        tokenizer: BaseTokenizer 구현체
        noiser: DenoisingNoiser (텍스트 레벨 노이즈만 사용)
        vocab_size: 어휘 크기 (edit tag 계산용)
        max_seq_len: 최대 시퀀스 길이
        text_key: JSONL 텍스트 필드명
        lang_key: JSONL 언어 필드명
        min_length: 최소 텍스트 길이
        shuffle_files: 에폭마다 파일 셔플
        seed: 랜덤 시드
        rank: DDP rank
        world_size: DDP 프로세스 수
        pack: 패킹 활성화 (여러 문장을 연결하여 PAD 제거)
    """

    def __init__(
        self,
        file_paths: str | list[str],
        tokenizer: BaseTokenizer,
        noiser: DenoisingNoiser,
        vocab_size: int = 303,
        max_seq_len: int = 2048,
        text_key: str | None = None,
        lang_key: str | None = None,
        min_length: int = 10,
        shuffle_files: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        pack: bool = True,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.tokenizer = tokenizer
        self.noiser = noiser
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.lang_key = lang_key
        self.min_length = min_length
        self.shuffle_files = shuffle_files
        self.rng = random.Random(seed)
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0
        self.pack = pack

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "line_counter": self._line_counter,
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self._line_counter = state.get("line_counter", 0)

    def _iter_lines(self, skip_worker_id=None, skip_total=None):
        """파일에서 (text, lang) 스트리밍"""
        files = list(self.file_paths)
        if self.shuffle_files:
            self.rng.shuffle(files)

        line_idx = 0
        for fpath in files:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < self.min_length:
                        continue

                    if skip_total is not None and line_idx % skip_total != skip_worker_id:
                        line_idx += 1
                        continue

                    lang = None
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            line_idx += 1
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                        if self.lang_key:
                            lang = obj.get(self.lang_key)
                    else:
                        text = line

                    if len(text) < self.min_length:
                        line_idx += 1
                        continue

                    yield text, lang
                    line_idx += 1

    def _tokenize_pair(self, text: str, lang: str | None):
        """텍스트 → (noised_ids, edit_tags, original_ids) 변환 (패딩 없음)"""
        if lang is None:
            lang = self.noiser._detect_lang(text)

        original_ids = self.tokenizer.encode(text, add_special=False)
        if not original_ids:
            return None

        noised_text = self.noiser._apply_text_noise(text, lang)
        noised_ids = self.tokenizer.encode(noised_text, add_special=False)
        if not noised_ids:
            return None

        # 최대 길이 제한 (개별 문장이 max_seq_len 초과 시)
        noised_ids = noised_ids[:self.max_seq_len]
        original_ids = original_ids[:self.max_seq_len]

        tags = compute_edit_tags(noised_ids, original_ids, self.vocab_size)
        return noised_ids, tags, original_ids

    def _make_padded_sample(self, noised_ids, tags, original_ids, text_len):
        """개별 시퀀스를 max_seq_len으로 패딩하여 dict 반환"""
        seq_len = len(noised_ids)
        pad_len = self.max_seq_len - seq_len
        pad_id = self.tokenizer.pad_id

        return {
            "input_ids": torch.tensor(noised_ids + [pad_id] * pad_len, dtype=torch.long),
            "edit_tags": torch.tensor(tags + [TAG_KEEP] * pad_len, dtype=torch.long),
            "pad_mask": torch.tensor([True] * seq_len + [False] * pad_len, dtype=torch.bool),
            "original_ids": torch.tensor(
                original_ids + [pad_id] * (self.max_seq_len - len(original_ids)),
                dtype=torch.long,
            ),
            "n_chars": text_len,
        }

    def _make_packed_sample(self, buf_input, buf_tags, n_chars):
        """패킹된 버퍼를 max_seq_len으로 패딩하여 dict 반환"""
        seq_len = len(buf_input)
        pad_len = self.max_seq_len - seq_len
        pad_id = self.tokenizer.pad_id

        return {
            "input_ids": torch.tensor(buf_input + [pad_id] * pad_len, dtype=torch.long),
            "edit_tags": torch.tensor(buf_tags + [TAG_KEEP] * pad_len, dtype=torch.long),
            "pad_mask": torch.tensor([True] * seq_len + [False] * pad_len, dtype=torch.bool),
            "original_ids": torch.zeros(self.max_seq_len, dtype=torch.long),  # 패킹 시 미사용
            "n_chars": n_chars,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        if self.pack:
            yield from self._iter_packed(global_worker_id, total_workers)
        else:
            yield from self._iter_padded(global_worker_id, total_workers)

    def _iter_padded(self, global_worker_id, total_workers):
        """개별 패딩 모드 (n_iterations > 1 용)"""
        for i, (text, lang) in enumerate(self._iter_lines(
                skip_worker_id=global_worker_id, skip_total=total_workers)):
            self._line_counter = i * total_workers + global_worker_id + 1
            result = self._tokenize_pair(text, lang)
            if result is not None:
                noised_ids, tags, original_ids = result
                yield self._make_padded_sample(noised_ids, tags, original_ids, len(text))

    def _iter_packed(self, global_worker_id, total_workers):
        """패킹 모드: 여러 문장을 연결하여 max_seq_len 채움"""
        buf_input = []
        buf_tags = []
        buf_chars = 0

        for i, (text, lang) in enumerate(self._iter_lines(
                skip_worker_id=global_worker_id, skip_total=total_workers)):
            self._line_counter = i * total_workers + global_worker_id + 1
            result = self._tokenize_pair(text, lang)
            if result is None:
                continue

            noised_ids, tags, _ = result
            remaining = self.max_seq_len - len(buf_input)

            if len(noised_ids) > remaining:
                # 버퍼 방출
                if buf_input:
                    yield self._make_packed_sample(buf_input, buf_tags, buf_chars)
                buf_input = []
                buf_tags = []
                buf_chars = 0

            # 버퍼에 추가 (max_seq_len 초과 시 truncate)
            remaining = self.max_seq_len - len(buf_input)
            buf_input.extend(noised_ids[:remaining])
            buf_tags.extend(tags[:remaining])
            buf_chars += len(text)

        # 잔여 버퍼 방출
        if buf_input:
            yield self._make_packed_sample(buf_input, buf_tags, buf_chars)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="편집 태깅 데이터셋 테스트")
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--text_key", "-k", default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    from training.noising import NoiseConfig

    tok_path = os.path.join(
        os.path.dirname(__file__), "..", "keyboard_tokenizer", "keyboard_tokenizer.json"
    )
    tok = KeyboardTokenizer(tok_path)

    # 토큰 레벨 노이즈 비활성화
    cfg = NoiseConfig(
        token_mask_ratio=0.0,
        token_delete_ratio=0.0,
        text_infill_ratio=0.0,
    )
    noiser = DenoisingNoiser(tok, cfg, seed=42)

    dataset = EditorDataset(
        args.file, tok, noiser,
        vocab_size=tok.vocab_size,
        max_seq_len=512,
        text_key=args.text_key,
    )

    from model.edit_tags import TAG_KEEP, TAG_DELETE, tag_to_op

    for i, sample in enumerate(dataset):
        if i >= args.n:
            break
        input_ids = sample["input_ids"]
        edit_tags = sample["edit_tags"]
        pad_mask = sample["pad_mask"]
        valid_len = pad_mask.sum().item()

        # 태그 분포
        tags_valid = edit_tags[:valid_len]
        n_keep = (tags_valid == TAG_KEEP).sum().item()
        n_delete = (tags_valid == TAG_DELETE).sum().item()
        n_other = valid_len - n_keep - n_delete

        noised_text = tok.decode(input_ids[:valid_len].tolist())
        print(f"[Sample {i}] len={valid_len}, "
              f"KEEP={n_keep}, DELETE={n_delete}, REPLACE/INSERT={n_other}")
        print(f"  노이즈: {noised_text[:80]}...")
        print()
