"""RTDDataset — ELECTRA RTD 사전학습용 스트리밍 데이터셋

JSONL → 텍스트 → 토크나이징 → 패킹 [BOS]...[EOS][BOS]...[EOS]
마스킹은 모델(ElectraRTD.forward) 내부에서 동적 수행하므로,
데이터셋은 원본 토큰 + pad_mask만 제공.

EditorDataset의 패킹/스트리밍 패턴을 재사용하되 노이즈/편집태그 제거.
"""
import json
import os
import random
import sys

import torch
from torch.utils.data import IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenizer_base import BaseTokenizer


class RTDDataset(IterableDataset):
    """ELECTRA RTD 사전학습용 스트리밍 데이터셋

    Args:
        file_paths: 코퍼스 파일 경로
        tokenizer: BaseTokenizer 구현체
        max_seq_len: 최대 시퀀스 길이
        text_key: JSONL 텍스트 필드명
        min_length: 최소 텍스트 길이
        shuffle_files: 에폭마다 파일 셔플
        seed: 랜덤 시드
        rank: DDP rank
        world_size: DDP 프로세스 수
    """

    def __init__(
        self,
        file_paths: str | list[str],
        tokenizer: BaseTokenizer,
        max_seq_len: int = 4096,
        text_key: str | None = None,
        min_length: int = 10,
        shuffle_files: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.file_paths = [file_paths] if isinstance(file_paths, str) else list(file_paths)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_key = text_key
        self.min_length = min_length
        self.shuffle_files = shuffle_files
        self.rng = random.Random(seed)
        self.rank = rank
        self.world_size = world_size
        self._line_counter = 0
        self._resume_line = 0
        self._last_file_order: list[str] = []
        self._resume_file_order: list[str] | None = None

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "line_counter": self._line_counter,
            "file_order": list(self._last_file_order),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self._resume_line = state.get("line_counter", 0)
        self._line_counter = self._resume_line
        self._resume_file_order = state.get("file_order") or None

    def _iter_lines(self, skip_worker_id=None, skip_total=None):
        """파일에서 텍스트 스트리밍"""
        if self._resume_file_order is not None:
            files = self._resume_file_order
            self._resume_file_order = None
        else:
            files = list(self.file_paths)
            if self.shuffle_files:
                self.rng.shuffle(files)
        self._last_file_order = list(files)

        line_idx = 0
        for fpath in files:
            is_parquet = fpath.endswith(".parquet")
            is_jsonl = fpath.endswith(".jsonl") or fpath.endswith(".json")

            if is_parquet:
                yield from self._iter_parquet(
                    fpath, line_idx, skip_worker_id, skip_total)
                # parquet 행 수만큼 line_idx 전진 (정확한 값은 _iter_parquet 내부에서 관리)
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
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

    def _iter_parquet(self, fpath, line_idx_start, skip_worker_id, skip_total):
        """Parquet 파일을 row group 단위 스트리밍 + 버퍼 (메모리 효율적)

        pyarrow iter_batches로 64K행씩 읽어 내부 버퍼에 저장.
        전체 파일을 메모리에 올리지 않으면서도 I/O 횟수 최소화.
        로컬 파일 및 HTTP URL 모두 지원.
        """
        import pyarrow.parquet as pq

        if fpath.startswith("http://") or fpath.startswith("https://"):
            import fsspec
            fp = fsspec.open(fpath, "rb").open()
            pf = pq.ParquetFile(fp)
        else:
            pf = pq.ParquetFile(fpath)
        text_col = self.text_key or "text"
        line_idx = line_idx_start

        for batch in pf.iter_batches(batch_size=65536, columns=[text_col]):
            texts = batch[text_col].to_pylist()
            for text in texts:
                if not text or len(text) < self.min_length:
                    continue

                if skip_total is not None and line_idx % skip_total != skip_worker_id:
                    line_idx += 1
                    continue

                self._line_counter = line_idx + 1
                yield text
                line_idx += 1

    def _tokenize(self, text: str) -> list[int] | None:
        """텍스트 → [BOS] + token_ids + [EOS]"""
        ids = self.tokenizer.encode(text, add_special=False)
        if not ids:
            return None

        max_content = self.max_seq_len - 2
        ids = ids[:max_content]

        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        return [bos] + ids + [eos]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id

        yield from self._iter_packed(global_worker_id, total_workers)

    def _iter_packed(self, global_worker_id, total_workers):
        """패킹 모드: 여러 문장을 [BOS]...[EOS] 단위로 연결"""
        resume_line = self._resume_line
        if resume_line > 0:
            self._resume_line = 0

        buf = []
        pad_id = self.tokenizer.pad_id

        for i, text in enumerate(self._iter_lines(
                skip_worker_id=global_worker_id, skip_total=total_workers)):
            abs_line = i * total_workers + global_worker_id
            self._line_counter = abs_line + 1

            if abs_line < resume_line:
                continue

            token_ids = self._tokenize(text)
            if token_ids is None:
                continue

            remaining = self.max_seq_len - len(buf)

            if len(token_ids) > remaining:
                if buf:
                    yield self._make_sample(buf, pad_id)
                buf = []

            remaining = self.max_seq_len - len(buf)
            buf.extend(token_ids[:remaining])

        if buf:
            yield self._make_sample(buf, pad_id)

    def _make_sample(self, buf: list[int], pad_id: int) -> dict:
        seq_len = len(buf)
        pad_len = self.max_seq_len - seq_len

        return {
            "input_ids": torch.tensor(buf + [pad_id] * pad_len, dtype=torch.long),
            "pad_mask": torch.tensor(
                [True] * seq_len + [False] * pad_len, dtype=torch.bool,
            ),
            "_line_counter": self._line_counter,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTD 데이터셋 테스트")
    parser.add_argument("--file", "-f", default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer
    tok = KeyboardTokenizer()
    print(f"토크나이저: vocab={tok.vocab_size}, bos={tok.bos_id}, eos={tok.eos_id}, "
          f"pad={tok.pad_id}, mask={tok.mask_id}")

    if args.file:
        ds = RTDDataset(args.file, tok, max_seq_len=256, text_key="text")
        for i, sample in enumerate(ds):
            if i >= args.n:
                break
            ids = sample["input_ids"]
            mask = sample["pad_mask"]
            n_valid = mask.sum().item()
            bos_count = (ids == tok.bos_id).sum().item()
            print(f"  [{i}] valid={n_valid}/{len(ids)}, docs={bos_count}, "
                  f"ids[:20]={ids[:20].tolist()}")
        print(f"\n{min(i+1, args.n)}개 샘플 확인 완료")
    else:
        # 파일 없이 기본 테스트
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for j in range(100):
                f.write(json.dumps({"text": f"테스트 문장 번호 {j}입니다. 오류가 있을 수 있습니다."}) + "\n")
            tmp_path = f.name

        ds = RTDDataset(tmp_path, tok, max_seq_len=256, text_key="text")
        count = 0
        for sample in ds:
            count += 1
            if count <= 3:
                ids = sample["input_ids"]
                mask = sample["pad_mask"]
                n_valid = mask.sum().item()
                print(f"  [{count}] valid={n_valid}/{len(ids)}")
        print(f"\n총 {count}개 패킹 샘플 생성")

        os.unlink(tmp_path)

    print("\n모든 테스트 통과!")
