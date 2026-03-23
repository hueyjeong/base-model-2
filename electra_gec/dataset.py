"""WordPiece 기반 GEC 에디터 데이터셋

텍스트 → 노이즈 적용 → WordPiece 토크나이징 → Levenshtein 정렬 → (action, content) 태그.
패킹 없음 (KoELECTRA = absolute position).
"""
from __future__ import annotations

import json
import os
import random

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

# C++ 가속 Levenshtein 우선 사용 (51x faster), 없으면 Python fallback
from model.edit_tags import TAG_KEEP, TAG_DELETE
try:
    from training.editor_dataset import compute_edit_tags
except ImportError:
    from model.edit_tags import compute_edit_tags


# ── Two-head 태그 상수 ──

ACTION_KEEP = 0
ACTION_DELETE = 1
ACTION_REPLACE = 2
ACTION_INSERT = 3
IGNORE = -100


def single_to_two_head(tag: int, vocab_size: int) -> tuple[int, int]:
    """Single-head 편집 태그 → (action, content) 변환"""
    if tag == TAG_KEEP:
        return ACTION_KEEP, IGNORE
    elif tag == TAG_DELETE:
        return ACTION_DELETE, IGNORE
    elif tag < 2 + vocab_size:
        return ACTION_REPLACE, tag - 2
    else:
        return ACTION_INSERT, tag - 2 - vocab_size


class WordPieceGECDataset(IterableDataset):
    """WordPiece 기반 GEC 편집 태깅 데이터셋

    Args:
        file_paths: 코퍼스 파일 경로
        noiser: DenoisingNoiser (텍스트 레벨 노이즈)
        tokenizer_name: HuggingFace 토크나이저 이름
        max_seq_len: 최대 시퀀스 길이 (CLS+SEP 포함, 기본 512)
        text_key: JSONL 텍스트 필드명
        min_length: 최소 텍스트 길이
        seed: 랜덤 시드
        rank: DDP rank
        world_size: DDP 프로세스 수
    """

    def __init__(
        self,
        file_paths: str | list[str],
        noiser,
        tokenizer_name: str = "monologg/koelectra-base-v3-discriminator",
        max_seq_len: int = 512,
        text_key: str | None = None,
        min_length: int = 10,
        shuffle_files: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.noiser = noiser
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.max_content = max_seq_len - 2  # CLS + SEP
        self.text_key = text_key
        self.min_length = min_length
        self.shuffle_files = shuffle_files
        self.rng = random.Random(seed)
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0
        self._epoch = 0
        # 진행률 추적: 코퍼스 총 바이트
        self.total_bytes = sum(os.path.getsize(f) for f in self.file_paths)
        self._bytes_read = 0

    def set_epoch(self, epoch: int):
        """에포크 시작 시 호출하여 셔플 시드 변경

        resume 중이면 리셋하지 않음 (load_state_dict에서 설정된 상태 유지).
        """
        self._epoch = epoch
        if not getattr(self, "_resuming", False):
            self.rng = random.Random(42 + epoch)
            self._bytes_read = 0
            self._line_counter = 0
            self._resume_bytes = 0
        else:
            self._resuming = False  # resume은 1회만

    def state_dict(self) -> dict:
        """데이터셋 RNG + 진행 상태 직렬화"""
        return {
            "rng_state": self.rng.getstate(),
            "line_counter": self._line_counter,
            "epoch": self._epoch,
            "bytes_read": self._bytes_read,
        }

    def load_state_dict(self, state: dict) -> None:
        """저장된 상태 복원 — 바이트 위치 기반 fast-forward

        _resume_bytes: worker fork 이후에도 __init__ 시점에 설정되어 있으므로
        worker에서 _iter_lines 호출 시 자동으로 건너뛰기 적용됨.
        """
        self.rng.setstate(state["rng_state"])
        self._line_counter = state.get("line_counter", 0)
        self._resume_line = state.get("line_counter", 0)
        self._epoch = state.get("epoch", 0)
        self._bytes_read = state.get("bytes_read", 0)
        self._resume_bytes = state.get("bytes_read", 0)
        self._resuming = True

    def _iter_lines(self, skip_worker_id=None, skip_total=None):
        """JSONL/TXT 파일에서 텍스트 스트리밍

        resume_bytes > 0이면 해당 바이트까지 빠르게 seek/skip 후 이어서 읽기.
        """
        files = list(self.file_paths)
        if self.shuffle_files:
            self.rng.shuffle(files)

        # Resume: 이전에 읽은 바이트 위치까지 빠르게 건너뛰기 (1회만 적용)
        resume_bytes = getattr(self, "_resume_bytes", 0)
        self._resume_bytes = 0  # persistent_workers에서 다음 에포크 재스킵 방지
        skipped_bytes = 0

        line_idx = 0
        for fpath in files:
            is_jsonl = fpath.endswith((".jsonl", ".json"))
            with open(fpath, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line_bytes = len(raw_line.encode("utf-8"))
                    self._bytes_read += line_bytes

                    # Resume fast-forward: 이전 위치까지 건너뛰기
                    if skipped_bytes < resume_bytes:
                        skipped_bytes += line_bytes
                        line_idx += 1
                        continue
                    line = raw_line.strip()
                    if len(line) < self.min_length:
                        continue

                    if skip_total is not None and line_idx % skip_total != skip_worker_id:
                        line_idx += 1
                        continue

                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            line_idx += 1
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                    else:
                        text = line

                    if len(text) < self.min_length:
                        line_idx += 1
                        continue

                    yield text
                    line_idx += 1

    def _tokenize_pair(self, text: str):
        """텍스트 → (input_ids, action_tags, content_tags) 변환"""
        lang = self.noiser._detect_lang(text)
        noised_text = self.noiser._apply_text_noise(text, lang)

        orig_ids = self.tokenizer.encode(text, add_special_tokens=False)
        noised_ids = self.tokenizer.encode(noised_text, add_special_tokens=False)

        if not orig_ids or not noised_ids:
            return None

        # CLS + SEP 여유 확보
        orig_ids = orig_ids[:self.max_content]
        noised_ids = noised_ids[:self.max_content]

        # Levenshtein DP → single-head 태그
        single_tags = compute_edit_tags(noised_ids, orig_ids, self.vocab_size)

        # single → two-head 변환
        actions = []
        contents = []
        for tag in single_tags:
            a, c = single_to_two_head(tag, self.vocab_size)
            actions.append(a)
            contents.append(c)

        # CLS/SEP 추가 (항상 KEEP)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_ids = [cls_id] + noised_ids + [sep_id]
        action_tags = [ACTION_KEEP] + actions + [ACTION_KEEP]
        content_tags = [IGNORE] + contents + [IGNORE]

        return input_ids, action_tags, content_tags

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        pad_id = self.tokenizer.pad_token_id

        for text in self._iter_lines(
            skip_worker_id=global_worker_id, skip_total=total_workers
        ):
            self._line_counter += 1
            try:
                result = self._tokenize_pair(text)
            except Exception:
                continue
            if result is None:
                continue

            input_ids, action_tags, content_tags = result
            seq_len = len(input_ids)

            if seq_len > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                action_tags = action_tags[:self.max_seq_len]
                content_tags = content_tags[:self.max_seq_len]
                seq_len = self.max_seq_len

            pad_len = self.max_seq_len - seq_len

            yield {
                "input_ids": torch.tensor(
                    input_ids + [pad_id] * pad_len, dtype=torch.long
                ),
                "attention_mask": torch.tensor(
                    [1] * seq_len + [0] * pad_len, dtype=torch.long
                ),
                "action_tags": torch.tensor(
                    action_tags + [IGNORE] * pad_len, dtype=torch.long
                ),
                "content_tags": torch.tensor(
                    content_tags + [IGNORE] * pad_len, dtype=torch.long
                ),
                "_bytes_read": self._bytes_read,
                "_total_bytes": self.total_bytes,
            }


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="WordPiece GEC 데이터셋 테스트")
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--text_key", "-k", default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    from training.noising import DenoisingNoiser, NoiseConfig

    kb_tok_path = os.path.join(
        os.path.dirname(__file__), "..", "keyboard_tokenizer", "keyboard_tokenizer.json"
    )
    kb_tok = KeyboardTokenizer(kb_tok_path)
    cfg = NoiseConfig(
        token_mask_ratio=0.0, token_delete_ratio=0.0, text_infill_ratio=0.0,
        weight_preset="realistic",
    )
    noiser = DenoisingNoiser(kb_tok, cfg, seed=42)

    dataset = WordPieceGECDataset(
        args.file, noiser, max_seq_len=512, text_key=args.text_key,
    )
    wp_tok = dataset.tokenizer

    for i, sample in enumerate(dataset):
        if i >= args.n:
            break
        attn_mask = sample["attention_mask"]
        action_tags = sample["action_tags"]
        valid_len = attn_mask.sum().item()
        valid_actions = action_tags[:valid_len]

        n_keep = (valid_actions == ACTION_KEEP).sum().item()
        n_del = (valid_actions == ACTION_DELETE).sum().item()
        n_rep = (valid_actions == ACTION_REPLACE).sum().item()
        n_ins = (valid_actions == ACTION_INSERT).sum().item()

        noised_text = wp_tok.decode(
            sample["input_ids"][:valid_len].tolist(), skip_special_tokens=True
        )
        print(f"[Sample {i}] len={valid_len} tokens")
        print(f"  KEEP={n_keep}  DELETE={n_del}  REPLACE={n_rep}  INSERT={n_ins}")
        print(f"  노이즈: {noised_text[:80]}...")
        print()
