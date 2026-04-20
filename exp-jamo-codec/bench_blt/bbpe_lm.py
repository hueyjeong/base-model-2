"""BBPE Base 급 causal Transformer LM (~133M).

BLT 와 동일한 SwiGLU/RMSNorm/RoPE/flash_attn 기반.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from entropy_lm import RoPECache, apply_rope

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH = True
except ImportError:
    HAVE_FLASH = False


class CausalAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int, rope: RoPECache) -> None:
        super().__init__()
        self.h = hidden
        self.nh = n_heads
        self.hd = hidden // n_heads
        self.rope = rope
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, l, 3, self.nh, self.hd).unbind(dim=2)
        cos, sin = self.rope.get(l, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if HAVE_FLASH:
            o = flash_attn_func(q, k, v, causal=True)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return self.out(o.reshape(b, l, self.h))


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, d_ff, bias=False)
        self.up = nn.Linear(hidden, d_ff, bias=False)
        self.down = nn.Linear(d_ff, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, hidden: int, n_heads: int, d_ff: int, rope: RoPECache) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(hidden)
        self.attn = CausalAttn(hidden, n_heads, rope)
        self.n2 = nn.RMSNorm(hidden)
        self.ff = SwiGLU(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class BBPELM(nn.Module):
    def __init__(
        self,
        vocab: int = 35000,
        hidden: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        tie_embed: bool = True,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.rope = RoPECache(hidden // n_heads)
        self.blocks = nn.ModuleList(
            [Block(hidden, n_heads, d_ff, self.rope) for _ in range(n_layers)]
        )
        self.norm = nn.RMSNorm(hidden)
        if tie_embed:
            self.head = None
        else:
            self.head = nn.Linear(hidden, vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=1.0 / (hidden**0.5))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.head is None:
            return x @ self.embed.weight.t()
        return self.head(x)


if __name__ == "__main__":
    m = BBPELM().cuda().to(torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"BBPELM params: {n/1e6:.2f}M")
    x = torch.randint(0, 35000, (2, 256), device="cuda")
    y = m(x)
    print(f"logits: {tuple(y.shape)}")
    loss = F.cross_entropy(y.reshape(-1, 35000).float(), x.reshape(-1).long())
    loss.backward()
    print(f"loss={loss.item():.3f}")
