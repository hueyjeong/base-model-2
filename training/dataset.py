"""디노이징 사전학습 데이터셋 — 스트리밍 + 패킹

스트리밍: 파일을 한 줄씩 읽으며 즉시 처리 (메모리 절약)
패킹: 여러 문장을 [BOS]문장1[EOS][BOS]문장2[EOS] 형태로 이어붙여
      목표 토큰 수 (pack_size)까지 채움.
PAD 없음, max_seq_len 없음.
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


class StreamingPackedDataset(IterableDataset):
    """스트리밍 + 패킹 디노이징 데이터셋

    파일을 한 줄씩 읽으면서 noiser를 적용하고,
    여러 문장의 토큰을 pack_size까지 이어붙여 하나의 시퀀스로 만든다.

    패킹 형태:
        encoder: [BOS]노이즈1[EOS][BOS]노이즈2[EOS]...
        decoder: [BOS]원본1[EOS][BOS]원본2[EOS]...

    Args:
        file_paths: 코퍼스 파일 경로 리스트
        tokenizer: BaseTokenizer 구현체
        noiser: DenoisingNoiser 인스턴스
        pack_size: 패킹 목표 토큰 수 (default 2048)
        text_key: JSONL에서 텍스트 필드 이름 (TXT면 None)
        lang_key: JSONL에서 언어 필드 이름 (None이면 자동 감지)
        min_length: 최소 텍스트 길이 (짧은 줄 필터링)
        shuffle_files: 에폭마다 파일 순서 셔플
        seed: 랜덤 시드
    """

    def __init__(
        self,
        file_paths: str | list[str],
        tokenizer: BaseTokenizer,
        noiser: DenoisingNoiser,
        pack_size: int = 2048,
        d_conv: int = 4,
        text_key: str | None = None,
        lang_key: str | None = None,
        min_length: int = 10,
        shuffle_files: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.tokenizer = tokenizer
        self.noiser = noiser
        self.pack_size = pack_size
        self.d_conv = d_conv  # Document isolation: Conv1d 격리용 PAD gap 크기
        self.text_key = text_key
        self.lang_key = lang_key
        self.min_length = min_length
        self.shuffle_files = shuffle_files
        self.rng = random.Random(seed)
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0  # 현재 에폭에서 지금까지 읽은 절대 라인 수
        self._last_file_order: list[str] = []  # 마지막 에폭의 파일 순서 (state_dict 용)
        self._resume_file_order: list[str] | None = None  # 복원 시 사용할 파일 순서
        self._resume_line: int = 0  # 복원 후 fast-forward 목표 라인 수

    def state_dict(self) -> dict:
        """파일 순서 + 라인 카운터 + 다음 에폭 셔플 RNG 반환"""
        return {
            "rng_state": self.rng.getstate(),
            "line_counter": self._line_counter,
            "file_order": list(self._last_file_order),  # 현재 에폭 파일 순서 저장
        }

    def load_state_dict(self, state: dict) -> None:
        """저장된 파일 순서 + 라인 카운터로 복원 준비"""
        self.rng.setstate(state["rng_state"])
        self._resume_line = state.get("line_counter", 0)
        self._line_counter = self._resume_line
        # 저장된 파일 순서가 있으면 복원 시 재사용 (re-shuffle 방지)
        self._resume_file_order = state.get("file_order") or None

    def _iter_lines(self):
        """파일에서 (text, lang) 한 줄씩 스트리밍"""
        if self._resume_file_order is not None:
            # 복원 모드: 저장된 순서 사용 (re-shuffle 금지)
            files = self._resume_file_order
            self._resume_file_order = None  # 한 번만 사용
        else:
            files = list(self.file_paths)
            if self.shuffle_files:
                self.rng.shuffle(files)
        self._last_file_order = list(files)  # state_dict 용 기록

        for fpath in files:
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) < self.min_length:
                        continue

                    lang = None
                    if is_jsonl:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get(self.text_key, line) if self.text_key else line
                        if self.lang_key:
                            lang = obj.get(self.lang_key)
                    else:
                        text = line

                    if len(text) < self.min_length:
                        continue

                    yield text, lang

    def _pack_sequences(self, noised_pairs):
        """(noised_ids, target_ids, n_chars, src_weights) 스트림을 pack_size까지 이어붙이기

        noised_ids, target_ids, src_weights 모두 이미 [BOS]...[EOS] 등 특수토큰 포함 상태.
        패킹은 단순 연결: [BOS]s1[EOS] [PAD gap] [BOS]s2[EOS]...
        Document isolation: Conv1d 격리를 위해 문서 사이에 d_conv개의 PAD 삽입.
        """
        src_buf = []
        tgt_buf = []
        weight_buf = []
        char_buf = 0
        pad_id = self.tokenizer.pad_id
        gap = [pad_id] * self.d_conv  # Conv1d 커널 크기만큼 PAD 삽입
        gap_weights = [0.0] * self.d_conv

        for noised_ids, target_ids, n_chars, src_weights in noised_pairs:
            # 이전 문서가 버퍼에 있으면 PAD gap 삽입 (Conv1d 격리)
            if src_buf:
                src_buf.extend(gap)
                tgt_buf.extend(gap)
                weight_buf.extend(gap_weights)

            src_buf.extend(noised_ids)
            tgt_buf.extend(target_ids)
            weight_buf.extend(src_weights)
            char_buf += n_chars

            # pack_size 이상이면 yield
            if len(src_buf) >= self.pack_size or len(tgt_buf) >= self.pack_size:
                yield self._make_sample(src_buf, tgt_buf, weight_buf, char_buf)
                src_buf = []
                tgt_buf = []
                weight_buf = []
                char_buf = 0

        # 잔여 버퍼 (마지막 불완전 팩)
        if src_buf:
            yield self._make_sample(src_buf, tgt_buf, weight_buf, char_buf)

    def _make_sample(self, src_ids, tgt_ids, src_weights, n_chars=0):
        """버퍼 → 텐서 dict (pack_size 초과 시 truncate, 부족 시 tokenizer.pad_id 로 pad)"""
        src_ids = src_ids[:self.pack_size]
        tgt_ids = tgt_ids[:self.pack_size]
        src_weights = src_weights[:self.pack_size]
        
        # Pad if shorter than pack_size
        pad_id = self.tokenizer.pad_id
        if len(src_ids) < self.pack_size:
            src_ids.extend([pad_id] * (self.pack_size - len(src_ids)))
            src_weights.extend([0.0] * (self.pack_size - len(src_weights)))
        if len(tgt_ids) < self.pack_size:
            tgt_ids.extend([pad_id] * (self.pack_size - len(tgt_ids)))
            
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_weights": torch.tensor(src_weights, dtype=torch.float),
            "n_chars": n_chars,
        }

    def __iter__(self):
        """스트리밍 → 노이즈 적용 → 패킹 → yield
        
        DDP(world_size > 1)일 경우 전체 GPU가 데이터를 나눠서 처리하고,
        num_workers > 0일 때 각 worker가 또 다시 할당된 줄을 처리하도록 분할.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # 전체 분할 수 (GPUs * DataLoader Workers)
        total_workers = self.world_size * num_workers
        # 현재 프로세스+워커의 고유 ID
        global_worker_id = (self.rank * num_workers) + worker_id

        def noised_stream():
            # 복원 후 fast-forward: resume_line 미만의 라인은 noiser 호출 없이 건너뜀
            resume_line = self._resume_line
            if resume_line > 0:
                self._resume_line = 0  # 다음 에폭에서는 정상 동작
            for i, (text, lang) in enumerate(self._iter_lines()):
                self._line_counter = i + 1
                if i < resume_line:
                    continue  # fast-forward: 파일만 읽고 noiser/pack은 건너뜀
                if i % total_workers != global_worker_id:
                    continue
                noised_ids, target_ids, src_weights = self.noiser(text, lang)
                yield noised_ids, target_ids, len(text), src_weights

        yield from self._pack_sequences(noised_stream())


# ── 테스트 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="스트리밍 패킹 데이터셋 테스트")
    parser.add_argument("--file", "-f", required=True, help="코퍼스 파일 경로")
    parser.add_argument("--text_key", "-k", default=None, help="JSONL 텍스트 필드")
    parser.add_argument("--lang_key", default=None, help="JSONL 언어 필드")
    parser.add_argument("--pack_size", type=int, default=2048, help="패킹 목표 토큰 수")
    parser.add_argument("--n", type=int, default=3, help="테스트 샘플 수")
    args = parser.parse_args()

    from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer
    from training.noising import NoiseConfig

    bbpe_path = os.path.join(
        os.path.dirname(__file__), "..", "bbpe_tokenizer", "bbpe.json"
    )
    tok = BBPETokenizer(bbpe_path)
    noiser = DenoisingNoiser(tok, NoiseConfig(), seed=42)

    dataset = StreamingPackedDataset(
        args.file, tok, noiser,
        pack_size=args.pack_size,
        text_key=args.text_key, lang_key=args.lang_key,
    )

    print(f"pack_size: {args.pack_size}")
    print()

    for i, sample in enumerate(dataset):
        if i >= args.n:
            break
        src_len = sample["src_ids"].shape[0]
        tgt_len = sample["tgt_ids"].shape[0]

        # BOS/EOS 경계 수 = 패킹된 문장 수
        bos_count = int((sample["src_ids"] == tok.bos_id).sum().item())
        eos_count = int((sample["src_ids"] == tok.eos_id).sum().item())

        # 디코딩 미리보기
        preview = tok.decode(sample["tgt_ids"][:60].tolist(), skip_special=False)

        print(f"[Pack {i}] src={src_len} tgt={tgt_len} "
              f"문장수≈{bos_count} (BOS={bos_count}, EOS={eos_count})")
        print(f"  미리보기: {preview[:120]}...")
        print()
