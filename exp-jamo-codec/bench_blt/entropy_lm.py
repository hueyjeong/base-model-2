"""1M byte-level causal Transformer LM (BLT entropy model).

논문 대비 100x 축소 (100M → 1M).
구성: 6L h=128 heads=4 d_ff=384, SWA window=256, RoPE.
byte_vocab=258 (BOS/EOS 포함).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH = True
except ImportError:
    HAVE_FLASH = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, L, H, D], cos/sin: [L, D]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)


class RoPECache:
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        self.head_dim = head_dim
        self.base = base
        self._cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, seq: int, device: torch.device, dtype: torch.dtype):
        key = (seq, device, dtype)
        if key in self._cache:
            return self._cache[key]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
        )
        t = torch.arange(seq, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


class SWABlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        d_ff: int,
        window: int,
        rope: RoPECache,
    ) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.h = hidden
        self.nh = n_heads
        self.hd = hidden // n_heads
        self.w = window
        self.rope = rope
        self.n1 = nn.RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.n2 = nn.RMSNorm(hidden)
        self.ff_gate = nn.Linear(hidden, d_ff, bias=False)
        self.ff_up = nn.Linear(hidden, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, hidden, bias=False)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, l, 3, self.nh, self.hd).unbind(dim=2)
        cos, sin = self.rope.get(l, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if HAVE_FLASH:
            o = flash_attn_func(q, k, v, causal=True, window_size=(self.w, 0))
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            idx = torch.arange(l, device=x.device)
            dist = idx.unsqueeze(0) - idx.unsqueeze(1)
            mask = (dist >= 0) & (dist < self.w)
            amask = torch.zeros(l, l, dtype=x.dtype, device=x.device)
            amask.masked_fill_(~mask, float("-inf"))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=amask).transpose(1, 2)
        return self.out(o.reshape(b, l, self.h))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_down(F.silu(self.ff_gate(x)) * self.ff_up(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.n1(x))
        x = x + self._ffn(self.n2(x))
        return x


class EntropyByteLM(nn.Module):
    def __init__(
        self,
        vocab: int = 258,
        hidden: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 384,
        window: int = 256,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.rope = RoPECache(hidden // n_heads)
        self.blocks = nn.ModuleList(
            [SWABlock(hidden, n_heads, d_ff, window, self.rope) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(hidden)
        self.head_bias = nn.Parameter(torch.zeros(vocab))
        nn.init.normal_(self.embed.weight, std=1.0 / (hidden**0.5))

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(byte_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x @ self.embed.weight.t() + self.head_bias

    @torch.no_grad()
    def entropy(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """각 위치의 H(x_t) 계산. [B, L] → [B, L]."""
        logits = self.forward(byte_ids)
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        H = -(p * logp).sum(dim=-1)  # [B, L]
        return H


if __name__ == "__main__":
    m = EntropyByteLM()
    n = sum(p.numel() for p in m.parameters())
    print(f"EntropyByteLM params: {n/1e6:.2f}M")
    print(m)
    x = torch.randint(0, 258, (2, 128))
    y = m(x)
    print(f"logits: {tuple(y.shape)}")
    H = m.entropy(x)
    print(f"entropy: {tuple(H.shape)} mean={H.mean():.3f} max={H.max():.3f}")
