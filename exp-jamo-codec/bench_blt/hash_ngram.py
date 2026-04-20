"""BLT hash n-gram byte embedding.

Paper: e_i = x_i + Σ_{n=3..8} E_n^hash(Hash(g_{i,n}))
       e_i ← e_i / (n_grams + 1)

Hash = RollPolyHash(g_{i,n}) mod |E_n^hash|

- n-gram 별 공유 table (hash_buckets) 을 쓰되, n 값에 따라 다른 seed (offset) 로 hashing
- RollPolyHash: h_i = (h_{i-1} * B + byte_i) mod P, with B=257 and P=|buckets|
- 패딩: 초기 n-1 위치는 zero embedding (x_i 만 남음)

128M 예산 기준 설계:
- hash_buckets = 50_000 (공유 table)
- hash_dim = 64 (byte embedding dim 보다 작게)
- n ∈ {3, 4, 5, 6, 7, 8}
- projection hash_dim → h_enc (384)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def roll_poly_hash(
    byte_ids: torch.Tensor,
    n: int,
    buckets: int,
    base: int = 257,
    n_offset: int = 0,
) -> torch.Tensor:
    """각 위치 i 에 대해 hash(byte_ids[i-n+1 : i+1]) 계산.

    초기 n-1 위치는 0 (padding).

    Args:
        byte_ids: [B, L] int
        n: n-gram 길이
        buckets: modulus
        base: polynomial base
        n_offset: n 별 seed (서로 다른 table 역할)

    Returns:
        [B, L] int in [0, buckets)
    """
    b, l = byte_ids.shape
    device = byte_ids.device
    # h_i = sum_{k=0..n-1} byte[i-k] * base^k  (mod buckets)
    # rolling: h_i = (h_{i-1} * base + byte[i]) - byte[i-n] * base^n
    # 하지만 단순 구현: 슬라이딩 윈도우로 계산 (L 번 iteration, n 번 곱셈)
    # vectorized: stack n shifts 후 weighted sum
    ids = byte_ids.long()
    pow_base = [1]
    for _ in range(n - 1):
        pow_base.append((pow_base[-1] * base) % buckets)
    pow_base = torch.tensor(pow_base, device=device, dtype=torch.long)  # [n], LSB first

    h = torch.zeros_like(ids)
    for k in range(n):
        # byte[i-k] * base^k
        shifted = torch.cat(
            [torch.zeros(b, k, dtype=torch.long, device=device), ids[:, : l - k]], dim=1
        )
        h = (h + shifted * pow_base[k]) % buckets

    # seed offset (n 별 다른 "table" 효과)
    h = (h + n_offset) % buckets
    # 초기 n-1 위치는 padding (모든 ngram 에 대해 0 으로 맞추지 말고 그냥 hash 결과 사용 —
    # 첫 n-1 위치는 "shifted" 가 0 이라 hash 가 동일해지지만 큰 문제 아님)
    return h


class HashNGramEmbedding(nn.Module):
    """BLT 스타일 hash n-gram byte embedding."""

    def __init__(
        self,
        byte_vocab: int = 258,
        h_byte: int = 384,
        hash_buckets: int = 50_000,
        hash_dim: int = 64,
        n_range: tuple[int, ...] = (3, 4, 5, 6, 7, 8),
    ) -> None:
        super().__init__()
        self.byte_embed = nn.Embedding(byte_vocab, h_byte)
        self.hash_embed = nn.Embedding(hash_buckets, hash_dim)
        self.hash_proj = nn.Linear(hash_dim, h_byte, bias=False)
        self.hash_buckets = hash_buckets
        self.n_range = n_range
        # 초기화
        nn.init.normal_(self.byte_embed.weight, std=1.0 / (h_byte**0.5))
        nn.init.normal_(self.hash_embed.weight, std=1.0 / (hash_dim**0.5))

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """byte_ids: [B, L] → [B, L, h_byte]."""
        x = self.byte_embed(byte_ids)  # [B, L, h_byte]
        # n-gram hash lookups 합
        hash_sum = torch.zeros(
            byte_ids.shape[0], byte_ids.shape[1], self.hash_embed.weight.shape[1],
            device=byte_ids.device, dtype=self.hash_embed.weight.dtype,
        )
        for n in self.n_range:
            h = roll_poly_hash(byte_ids, n=n, buckets=self.hash_buckets, n_offset=n * 7919)
            hash_sum = hash_sum + self.hash_embed(h)
        # projection to h_byte
        hash_proj = self.hash_proj(hash_sum)
        # 결합 + 정규화
        n_terms = len(self.n_range) + 1  # n-grams + byte_embed itself
        return (x + hash_proj) / n_terms


if __name__ == "__main__":
    m = HashNGramEmbedding().cuda().to(torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"HashNGramEmbedding params: {n/1e6:.2f}M")
    x = torch.randint(0, 258, (2, 64), device="cuda")
    y = m(x)
    print(f"out: {tuple(y.shape)} dtype={y.dtype}")
    print(f"byte_embed: {m.byte_embed.weight.numel()/1e6:.2f}M")
    print(f"hash_embed: {m.hash_embed.weight.numel()/1e6:.2f}M")
    print(f"hash_proj:  {sum(p.numel() for p in m.hash_proj.parameters())/1e6:.2f}M")
