"""BBPE 토크나이즈 → packed sequence streaming dataset.

각 yield: token_ids[L+1] (input + target shift 용 한 칸 여유).
"""

from __future__ import annotations

import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class BBPEStreamDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer_path: str,
        text_key: str = "text",
        seq_len: int = 512,
        min_bytes: int = 16,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.path = Path(parquet_path)
        self.tokenizer_path = tokenizer_path
        self.text_key = text_key
        self.seq_len = seq_len
        self.min_bytes = min_bytes
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        # 워커별로 lazy 초기화
        self._tk = None
        self._bos = None
        self._eos = None

    def _ensure_tk(self):
        if self._tk is None:
            self._tk = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self._bos = self._tk.bos_token_id
            self._eos = self._tk.eos_token_id

    def __iter__(self):
        self._ensure_tk()
        pf = pq.ParquetFile(self.path)
        rng = random.Random(self.seed + self.rank)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        stride = self.world_size * num_workers
        offset = self.rank * num_workers + worker_id

        buf: list[int] = []
        epoch = 0
        while True:
            group_indices = list(range(pf.num_row_groups))
            if self.shuffle:
                rng_ep = random.Random(self.seed + self.rank + epoch * 977)
                rng_ep.shuffle(group_indices)
            for gi in group_indices:
                rg = pf.read_row_group(gi, columns=[self.text_key])
                texts = rg[self.text_key].to_pylist()
                if self.shuffle:
                    rng_ep = random.Random(
                        self.seed + self.rank + epoch * 977 + gi
                    )
                    idxs = list(range(len(texts)))
                    rng_ep.shuffle(idxs)
                else:
                    idxs = range(len(texts))
                # batched tokenize for speed
                batch_size = 256
                kept_local = [
                    (li, texts[li]) for li in idxs
                    if (li % stride) == offset and isinstance(texts[li], str)
                    and len(texts[li].encode("utf-8")) >= self.min_bytes
                ]
                for start in range(0, len(kept_local), batch_size):
                    chunk = kept_local[start : start + batch_size]
                    only_texts = [t for _, t in chunk]
                    encs = self._tk(only_texts, add_special_tokens=False, truncation=False)
                    ids_list = encs["input_ids"]
                    for ids in ids_list:
                        buf.append(self._bos)
                        buf.extend(ids)
                        buf.append(self._eos)
                        while len(buf) >= self.seq_len + 1:
                            chunk2 = buf[: self.seq_len + 1]
                            buf = buf[self.seq_len + 1 :]
                            yield torch.tensor(chunk2, dtype=torch.int32)
            epoch += 1


def collate_bbpe(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0).to(torch.long)


def precompute_byte_lengths(tokenizer_path: str) -> torch.Tensor:
    """각 token id 의 byte 길이 (decode 후 UTF-8). Special tokens = 0."""
    tk = AutoTokenizer.from_pretrained(tokenizer_path)
    n = tk.vocab_size
    specials = set(tk.all_special_ids)
    bl = torch.zeros(n, dtype=torch.int32)
    for i in range(n):
        if i in specials:
            continue
        try:
            s = tk.decode([i], skip_special_tokens=False)
            bl[i] = len(s.encode("utf-8"))
        except Exception:
            bl[i] = 0
    return bl


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="corpus/jamo-codec-v3/val.parquet")
    p.add_argument("--tokenizer", default="checkpoints/bbpe_35k")
    p.add_argument("--seq_len", type=int, default=290)
    p.add_argument("--n", type=int, default=2)
    args = p.parse_args()

    ds = BBPEStreamDataset(args.parquet, args.tokenizer, seq_len=args.seq_len)
    it = iter(ds)
    bl = precompute_byte_lengths(args.tokenizer)
    print(f"byte_lengths: total={bl.sum().item()} mean={bl.float().mean().item():.2f}")
    for i in range(args.n):
        x = next(it)
        bytes_total = bl[x.long()].sum().item()
        print(f"[{i}] tokens={tuple(x.shape)} dtype={x.dtype} bytes_in_seq={bytes_total}")
        print(f"    first 30 ids: {x[:30].tolist()}")
