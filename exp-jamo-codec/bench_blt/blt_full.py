"""BLT 128M — Local Encoder + Latent Transformer + Local Decoder.

논문 구조 그대로:
- Encoder: hash-ngram byte emb → 2L SWA self-attn → CrossAttn pool (patch queries init by max-pool)
- Latent: 12L full-attn (patch-level block-causal) with RoPE
- Decoder: CrossAttn unpool (byte queries from encoder last layer residual, KV=patch)
          → 6L SWA self-attn → byte head (vocab=258)

Cross-attn 마스크: within-patch (pool) 와 byte→own-patch (unpool), 둘 다 dense mask 로 구현.
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

from hash_ngram import HashNGramEmbedding

BYTE_VOCAB = 258


# ---------- Self-attention blocks ----------


class SWASelfAttn(nn.Module):
    """Sliding-window causal self-attn with RoPE."""

    def __init__(self, hidden: int, n_heads: int, window: int, rope: RoPECache) -> None:
        super().__init__()
        self.h = hidden
        self.nh = n_heads
        self.hd = hidden // n_heads
        self.w = window
        self.rope = rope
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, l, 3, self.nh, self.hd).unbind(dim=2)
        cos, sin = self.rope.get(l, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if not HAVE_FLASH:
            raise RuntimeError("flash_attn 필요")
        o = flash_attn_func(q, k, v, causal=True, window_size=(self.w, 0))
        return self.out(o.reshape(b, l, self.h))


class CausalFullAttn(nn.Module):
    """Full causal self-attn with RoPE — latent transformer 용."""

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


class EncBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, d_ff: int, window: int, rope: RoPECache) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(hidden)
        self.attn = SWASelfAttn(hidden, n_heads, window, rope)
        self.n2 = nn.RMSNorm(hidden)
        self.ff = SwiGLU(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class LatentBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, d_ff: int, rope: RoPECache) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(hidden)
        self.attn = CausalFullAttn(hidden, n_heads, rope)
        self.n2 = nn.RMSNorm(hidden)
        self.ff = SwiGLU(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class DecBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, d_ff: int, window: int, rope: RoPECache) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(hidden)
        self.attn = SWASelfAttn(hidden, n_heads, window, rope)
        self.n2 = nn.RMSNorm(hidden)
        self.ff = SwiGLU(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


# ---------- Cross-attention (within-patch) ----------


class CrossAttnPool(nn.Module):
    """Encoder cross-attn: query=patch(max-pool init), KV=byte_feat, within-patch mask."""

    def __init__(self, h_byte: int, h_patch: int, n_heads: int) -> None:
        super().__init__()
        self.nh = n_heads
        self.hd = h_patch // n_heads
        self.h_byte = h_byte
        self.h_patch = h_patch
        self.q_proj = nn.Linear(h_byte, h_patch, bias=False)
        self.kv_proj = nn.Linear(h_byte, 2 * h_patch, bias=False)
        self.out = nn.Linear(h_patch, h_patch, bias=False)

    def forward(
        self,
        byte_feat: torch.Tensor,  # [B, L, h_byte]
        patch_query_init: torch.Tensor,  # [B, n_patch, h_byte]
        within_mask: torch.Tensor,  # [B, n_patch, L] bool (True = allow attend)
    ) -> torch.Tensor:
        b, l, _ = byte_feat.shape
        np = patch_query_init.shape[1]
        q = self.q_proj(patch_query_init).reshape(b, np, self.nh, self.hd).transpose(1, 2)  # [B, nh, np, hd]
        kv = self.kv_proj(byte_feat)
        k, v = kv.chunk(2, dim=-1)  # [B, L, h_patch]
        k = k.reshape(b, l, self.nh, self.hd).transpose(1, 2)  # [B, nh, L, hd]
        v = v.reshape(b, l, self.nh, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hd**0.5)  # [B, nh, np, L]
        mask = within_mask.unsqueeze(1)  # [B, 1, np, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # 빈 patch (모든 위치 mask) 는 NaN → 0 처리
        attn = torch.nan_to_num(attn, nan=0.0)
        o = torch.matmul(attn, v)  # [B, nh, np, hd]
        o = o.transpose(1, 2).reshape(b, np, self.h_patch)
        return self.out(o)


class CrossAttnUnpool(nn.Module):
    """Decoder cross-attn: query=byte_feat, KV=patch_rep, mask=own-patch only."""

    def __init__(self, h_byte: int, h_patch: int, n_heads: int) -> None:
        super().__init__()
        self.nh = n_heads
        self.hd = h_byte // n_heads
        self.h_byte = h_byte
        self.h_patch = h_patch
        self.q_proj = nn.Linear(h_byte, h_byte, bias=False)
        self.kv_proj = nn.Linear(h_patch, 2 * h_byte, bias=False)
        self.out = nn.Linear(h_byte, h_byte, bias=False)

    def forward(
        self,
        byte_q: torch.Tensor,  # [B, L, h_byte]
        patch_kv: torch.Tensor,  # [B, np, h_patch]
        own_mask: torch.Tensor,  # [B, L, np] bool
    ) -> torch.Tensor:
        b, l, _ = byte_q.shape
        np = patch_kv.shape[1]
        q = self.q_proj(byte_q).reshape(b, l, self.nh, self.hd).transpose(1, 2)  # [B, nh, L, hd]
        kv = self.kv_proj(patch_kv)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(b, np, self.nh, self.hd).transpose(1, 2)
        v = v.reshape(b, np, self.nh, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hd**0.5)  # [B, nh, L, np]
        mask = own_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        o = torch.matmul(attn, v)  # [B, nh, L, hd]
        o = o.transpose(1, 2).reshape(b, l, self.h_byte)
        return self.out(o)


# ---------- Patching helpers ----------


def patch_ids_from_boundaries(boundaries: torch.Tensor) -> torch.Tensor:
    """boundaries: [B, L] bool (True = patch 시작점).
    Returns patch_ids [B, L] int (cumsum - 1 기반, 각 byte 가 속한 patch index).
    """
    # 각 row 의 첫 위치는 반드시 True (BOS 처럼 취급)
    pids = (boundaries.long().cumsum(dim=1) - 1).clamp(min=0)
    return pids


def max_pool_patches(
    byte_feat: torch.Tensor,  # [B, L, H]
    patch_ids: torch.Tensor,  # [B, L]
    n_patches: int,
) -> torch.Tensor:
    """각 patch 에 속한 bytes 의 max-pool 반환 [B, n_patches, H]."""
    b, l, h = byte_feat.shape
    out = byte_feat.new_full((b, n_patches, h), float("-inf"))
    # scatter_reduce_(amax) 사용
    idx = patch_ids.unsqueeze(-1).expand(b, l, h).clamp(min=0, max=n_patches - 1)
    out.scatter_reduce_(1, idx, byte_feat, reduce="amax", include_self=True)
    out = torch.nan_to_num(out, neginf=0.0)
    return out


def build_within_patch_mask(patch_ids: torch.Tensor, n_patches: int) -> torch.Tensor:
    """[B, L] → [B, n_patches, L] bool. mask[b,j,i] = (patch_ids[b,i] == j)."""
    # [B, 1, L] == [1, n_patches, 1] → [B, n_patches, L]
    jrange = torch.arange(n_patches, device=patch_ids.device).view(1, -1, 1)
    return patch_ids.unsqueeze(1) == jrange


def build_own_patch_mask(patch_ids: torch.Tensor, n_patches: int) -> torch.Tensor:
    """[B, L] → [B, L, n_patches] bool. mask[b,i,j] = (patch_ids[b,i] == j)."""
    jrange = torch.arange(n_patches, device=patch_ids.device).view(1, 1, -1)
    return patch_ids.unsqueeze(-1) == jrange


# ---------- BLT 128M ----------


class BLT128M(nn.Module):
    def __init__(
        self,
        byte_vocab: int = BYTE_VOCAB,
        hash_buckets: int = 50_000,
        hash_dim: int = 64,
        h_enc: int = 384,
        h_lat: int = 768,
        h_dec: int = 384,
        enc_layers: int = 2,
        lat_layers: int = 12,
        dec_layers: int = 6,
        enc_heads: int = 6,
        lat_heads: int = 12,
        dec_heads: int = 6,
        xattn_heads: int = 8,
        enc_ff: int = 1024,
        lat_ff: int = 3072,
        dec_ff: int = 1024,
        swa_window: int = 512,
    ) -> None:
        super().__init__()
        # Hash n-gram byte embedding
        self.emb = HashNGramEmbedding(
            byte_vocab=byte_vocab,
            h_byte=h_enc,
            hash_buckets=hash_buckets,
            hash_dim=hash_dim,
        )
        # Encoder
        self.rope_enc = RoPECache(h_enc // enc_heads)
        self.encoder = nn.ModuleList(
            [EncBlock(h_enc, enc_heads, enc_ff, swa_window, self.rope_enc) for _ in range(enc_layers)]
        )
        self.enc_norm = nn.RMSNorm(h_enc)
        self.pool = CrossAttnPool(h_byte=h_enc, h_patch=h_lat, n_heads=xattn_heads)
        # Latent
        self.rope_lat = RoPECache(h_lat // lat_heads)
        self.latent = nn.ModuleList(
            [LatentBlock(h_lat, lat_heads, lat_ff, self.rope_lat) for _ in range(lat_layers)]
        )
        self.lat_norm = nn.RMSNorm(h_lat)
        # Decoder
        self.unpool = CrossAttnUnpool(h_byte=h_dec, h_patch=h_lat, n_heads=xattn_heads)
        self.enc_to_dec = nn.Linear(h_enc, h_dec, bias=False)
        self.rope_dec = RoPECache(h_dec // dec_heads)
        self.decoder = nn.ModuleList(
            [DecBlock(h_dec, dec_heads, dec_ff, swa_window, self.rope_dec) for _ in range(dec_layers)]
        )
        self.dec_norm = nn.RMSNorm(h_dec)
        self.byte_head = nn.Linear(h_dec, byte_vocab, bias=False)
        nn.init.normal_(self.byte_head.weight, std=1.0 / (h_dec**0.5))

    def forward(
        self,
        byte_ids: torch.Tensor,  # [B, L]
        boundaries: torch.Tensor,  # [B, L] bool, True = patch 시작
    ) -> torch.Tensor:
        """반환: byte logits [B, L, byte_vocab]."""
        # 1. byte embedding
        e = self.emb(byte_ids)  # [B, L, h_enc]
        h = e
        for blk in self.encoder:
            h = blk(h)
        h = self.enc_norm(h)  # [B, L, h_enc]

        # 2. patching
        patch_ids = patch_ids_from_boundaries(boundaries)
        n_patches = int(patch_ids.max().item()) + 1
        within_mask = build_within_patch_mask(patch_ids, n_patches)  # [B, np, L]
        own_mask = build_own_patch_mask(patch_ids, n_patches)  # [B, L, np]

        # 3. pool (query init = max-pool)
        patch_init = max_pool_patches(h, patch_ids, n_patches)  # [B, np, h_enc]
        patch_feat = self.pool(h, patch_init, within_mask)  # [B, np, h_lat]

        # 4. latent
        for blk in self.latent:
            patch_feat = blk(patch_feat)
        patch_feat = self.lat_norm(patch_feat)

        # 5. decoder: byte residual + cross-attn unpool → decoder blocks
        byte_q = self.enc_to_dec(h)  # [B, L, h_dec]
        byte_q = byte_q + self.unpool(byte_q, patch_feat, own_mask)
        d = byte_q
        for blk in self.decoder:
            d = blk(d)
        d = self.dec_norm(d)
        return self.byte_head(d)


if __name__ == "__main__":
    m = BLT128M().cuda().to(torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"BLT128M params: {n/1e6:.2f}M")
    # 각 컴포넌트 별
    print(f"  emb:      {sum(p.numel() for p in m.emb.parameters())/1e6:.2f}M")
    print(f"  encoder:  {sum(p.numel() for p in m.encoder.parameters())/1e6:.2f}M")
    print(f"  pool:     {sum(p.numel() for p in m.pool.parameters())/1e6:.2f}M")
    print(f"  latent:   {sum(p.numel() for p in m.latent.parameters())/1e6:.2f}M")
    print(f"  unpool:   {sum(p.numel() for p in m.unpool.parameters())/1e6:.2f}M")
    print(f"  decoder:  {sum(p.numel() for p in m.decoder.parameters())/1e6:.2f}M")
    print(f"  head:     {sum(p.numel() for p in m.byte_head.parameters())/1e6:.2f}M")

    # forward sanity
    B, L = 2, 256
    byte_ids = torch.randint(0, 258, (B, L), device="cuda")
    # 랜덤 boundary (매 ~6 byte)
    boundaries = torch.rand(B, L, device="cuda") < (1 / 6)
    boundaries[:, 0] = True
    y = m(byte_ids, boundaries)
    print(f"logits: {tuple(y.shape)}")

    # backward
    loss = F.cross_entropy(y.reshape(-1, 258).float(), byte_ids.reshape(-1).long())
    loss.backward()
    print(f"loss={loss.item():.3f}  backward OK")
