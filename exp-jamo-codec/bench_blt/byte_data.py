"""Byte-level streaming dataset (UTF-8 packing).

vocab = 258
  0..255: raw UTF-8 bytes
  256   : BOS (doc start, patch reset)
  257   : EOS (doc end)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

BOS = 256
EOS = 257
VOCAB = 258


class ByteStreamDataset(IterableDataset):
    """Parquet 행을 byte 로 encode → byte_seq 길이로 packing.

    각 yield: int16 tensor [byte_seq], BOS/EOS 포함.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        text_key: str = "text",
        byte_seq: int = 1024,
        min_bytes: int = 16,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.path = Path(parquet_path)
        self.text_key = text_key
        self.byte_seq = byte_seq
        self.min_bytes = min_bytes
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def __iter__(self):
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
                for local_i in idxs:
                    if (local_i % stride) != offset:
                        continue
                    text = texts[local_i]
                    if not isinstance(text, str):
                        continue
                    b = text.encode("utf-8", errors="replace")
                    if len(b) < self.min_bytes:
                        continue
                    buf.append(BOS)
                    buf.extend(b)
                    buf.append(EOS)
                    while len(buf) >= self.byte_seq:
                        chunk = buf[: self.byte_seq]
                        buf = buf[self.byte_seq :]
                        yield torch.tensor(chunk, dtype=torch.int16)
            epoch += 1


def collate_bytes(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0).to(torch.long)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet",
        default="corpus/jamo-codec-v3/val.parquet",
    )
    p.add_argument("--byte_seq", type=int, default=1024)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    ds = ByteStreamDataset(args.parquet, byte_seq=args.byte_seq)
    it = iter(ds)
    for i in range(args.n):
        x = next(it)
        print(f"[{i}] shape={tuple(x.shape)} dtype={x.dtype}")
        print(
            f"    first 50 ids: {x[:50].tolist()}"
        )
        bos_count = (x == BOS).sum().item()
        eos_count = (x == EOS).sum().item()
        print(f"    BOS={bos_count}  EOS={eos_count}")
        try:
            txt = bytes(
                [v if v < 256 else ord(" ") for v in x[:200].tolist()]
            ).decode("utf-8", errors="replace")
            print(f"    decoded[:200]: {txt[:100]!r}")
        except Exception as e:
            print(f"    decode error: {e}")
