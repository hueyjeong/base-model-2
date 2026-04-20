"""BBPE Base (KoELECTRA 급, 128M) vs BLT 128M end-to-end 속도/메모리 비교.

공정 비교:
- 같은 batch, 같은 token 수 기준 텍스트 (BBPE: 512 token, BLT: 2048 byte ≈ 600 token)
- 둘 다 flash_attn SWA + Transformer, BF16, torch.compile

BBPE Base (≈128M 타겟, KoELECTRA Base 골격):
  Embedding(35K, 768) — 26.9M (tied)
  12L Transformer full-attn h=768 heads=12 d_ff=3072 — 85M
  Output tied with embedding
  Total ≈ 112M

BLT 128M (현 설계):
  Byte emb(256, 384) + Hash n-gram table 100K × 96 = 9.6M
  Encoder 2L SWA h=384 heads=6 d_ff=1024 — 3.2M
  Latent 15L full-attn h=768 heads=12 d_ff=3072 — 106M
  Decoder 6L SWA h=384 heads=6 d_ff=1024 — 9.6M
  Cross-attn pool/unpool — 1M
  Total ≈ 130M

사용:
    source .venv/bin/activate
    python exp-jamo-codec/bench_blt/bbpe_vs_blt.py
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH = True
except ImportError:
    HAVE_FLASH = False


# ---------- 공통 블록 ----------


class FullAttn(nn.Module):
    """Non-windowed causal/non-causal self-attn (latent + BBPE 용)."""

    def __init__(self, hidden: int, n_heads: int, causal: bool = False) -> None:
        super().__init__()
        self.h = hidden
        self.nh = n_heads
        self.hd = hidden // n_heads
        self.causal = causal
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, l, 3, self.nh, self.hd).unbind(dim=2)
        if HAVE_FLASH:
            o = flash_attn_func(q, k, v, causal=self.causal)
        else:
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal).transpose(1, 2)
        return self.out(o.reshape(b, l, self.h))


class SWAAttn(nn.Module):
    """BLT encoder/decoder 용 sliding-window causal attention."""

    def __init__(self, hidden: int, n_heads: int, window: int = 512) -> None:
        super().__init__()
        self.h = hidden
        self.nh = n_heads
        self.hd = hidden // n_heads
        self.w = window
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, l, 3, self.nh, self.hd).unbind(dim=2)
        if HAVE_FLASH:
            o = flash_attn_func(q, k, v, causal=True, window_size=(self.w, 0))
        else:
            raise RuntimeError("flash_attn 필요")
        return self.out(o.reshape(b, l, self.h))


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, d_ff, bias=False)
        self.up = nn.Linear(hidden, d_ff, bias=False)
        self.down = nn.Linear(d_ff, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block — full or SWA attn."""

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        d_ff: int,
        attn_kind: str = "full",
        window: int = 512,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(hidden)
        if attn_kind == "full":
            self.attn = FullAttn(hidden, n_heads, causal=causal)
        elif attn_kind == "swa":
            self.attn = SWAAttn(hidden, n_heads, window=window)
        else:
            raise ValueError(attn_kind)
        self.n2 = nn.RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class CrossAttnPool(nn.Module):
    """BLT encoder cross-attn: query=patch, KV=byte, within-patch mask.

    단순화: 여기서는 실제 mask 대신 average pooling 으로 근사 (FLOP 등가).
    실제 구현은 block-diagonal mask 로 within-patch 제한.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.kv_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self, byte_feat: torch.Tensor, patch_size: int
    ) -> torch.Tensor:
        # byte_feat: [B, L_b, H], L_b = n_patch * patch_size
        b, lb, h = byte_feat.shape
        np = lb // patch_size
        # 단순 pool — FLOP equivalent proxy
        patch_init = byte_feat[:, : np * patch_size].reshape(b, np, patch_size, h).amax(dim=2)
        q = self.q_proj(patch_init)  # [B, np, H]
        kv = self.kv_proj(byte_feat[:, : np * patch_size])  # [B, np*ps, 2H]
        k, v = kv.chunk(2, dim=-1)
        # block-diagonal attention: 각 patch 는 자기 ps bytes 만
        k = k.reshape(b, np, patch_size, h)
        v = v.reshape(b, np, patch_size, h)
        q = q.unsqueeze(2)  # [B, np, 1, H]
        scores = (q * k).sum(dim=-1, keepdim=True) / (h**0.5)  # [B, np, ps, 1]
        attn = F.softmax(scores, dim=2)
        o = (attn * v).sum(dim=2)  # [B, np, H]
        return self.out(o)


class CrossAttnUnpool(nn.Module):
    """BLT decoder cross-attn: query=byte, KV=patch rep."""

    def __init__(self, hidden_byte: int, hidden_patch: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_byte, hidden_byte, bias=False)
        self.kv_proj = nn.Linear(hidden_patch, 2 * hidden_byte, bias=False)
        self.out = nn.Linear(hidden_byte, hidden_byte, bias=False)

    def forward(
        self,
        byte_q: torch.Tensor,  # [B, L_b, Hb]
        patch_kv: torch.Tensor,  # [B, np, Hp]
        patch_size: int,
    ) -> torch.Tensor:
        b, lb, hb = byte_q.shape
        np = patch_kv.shape[1]
        q = self.q_proj(byte_q[:, : np * patch_size]).reshape(b, np, patch_size, hb)
        kv = self.kv_proj(patch_kv)  # [B, np, 2 Hb]
        k, v = kv.chunk(2, dim=-1)  # [B, np, Hb]
        scores = (q * k.unsqueeze(2)).sum(dim=-1, keepdim=True) / (hb**0.5)
        attn = F.softmax(scores, dim=2)
        o = attn * v.unsqueeze(2)
        return self.out(o.reshape(b, np * patch_size, hb))


# ---------- BBPE Base (KoELECTRA 급) ----------


class BBPEBase(nn.Module):
    def __init__(
        self,
        vocab: int = 35000,
        hidden: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, n_heads, d_ff, attn_kind="full", causal=False)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(hidden)
        # output head — tied with embedding
        self.head_bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = x @ self.embed.weight.t() + self.head_bias
        return logits


# ---------- BLT 128M ----------


class BLT128M(nn.Module):
    def __init__(
        self,
        byte_vocab: int = 256,
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
        enc_ff: int = 1024,
        lat_ff: int = 3072,
        dec_ff: int = 1024,
        swa_window: int = 512,
        n_hash_grams: int = 6,  # n = 3..8
    ) -> None:
        super().__init__()
        self.byte_embed = nn.Embedding(byte_vocab, h_enc)
        # 공유 hash table (작은 dim) + projection (param 절감)
        self.hash_embed = nn.Embedding(hash_buckets, hash_dim)
        self.hash_proj = nn.Linear(hash_dim, h_enc, bias=False)
        self.n_hash_grams = n_hash_grams
        self.hash_buckets = hash_buckets

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(h_enc, enc_heads, enc_ff, attn_kind="swa", window=swa_window)
                for _ in range(enc_layers)
            ]
        )
        self.pool = CrossAttnPool(h_enc)
        self.enc_to_lat = nn.Linear(h_enc, h_lat, bias=False)

        self.latent = nn.ModuleList(
            [
                TransformerBlock(h_lat, lat_heads, lat_ff, attn_kind="full", causal=True)
                for _ in range(lat_layers)
            ]
        )
        self.lat_norm = nn.RMSNorm(h_lat)

        self.unpool = CrossAttnUnpool(h_dec, h_lat)
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(h_dec, dec_heads, dec_ff, attn_kind="swa", window=swa_window)
                for _ in range(dec_layers)
            ]
        )
        self.dec_norm = nn.RMSNorm(h_dec)
        self.byte_head = nn.Linear(h_dec, byte_vocab, bias=False)

    def _hash_ngram_lookup(self, byte_ids: torch.Tensor) -> torch.Tensor:
        # 단순 proxy — n-gram 별 shift 후 lookup 합. hash_dim -> h_enc projection
        b, l = byte_ids.shape
        base = byte_ids.long()
        hash_dim = self.hash_embed.weight.shape[1]
        emb = torch.zeros(b, l, hash_dim, device=byte_ids.device, dtype=self.hash_embed.weight.dtype)
        for n in range(self.n_hash_grams):
            ids = (base * (257 ** (n + 1)) + n * 17) % self.hash_buckets
            emb = emb + self.hash_embed(ids)
        emb = emb / (self.n_hash_grams + 1)
        return self.hash_proj(emb)

    def forward(self, byte_ids: torch.Tensor, patch_size: int = 6) -> torch.Tensor:
        # byte_ids: [B, L_b]
        e = self.byte_embed(byte_ids)
        e = e + self._hash_ngram_lookup(byte_ids).to(e.dtype)
        e = e / (self.n_hash_grams + 1)
        # encoder
        h = e
        for blk in self.encoder:
            h = blk(h)
        patch = self.pool(h, patch_size=patch_size)  # [B, np, h_enc]
        patch = self.enc_to_lat(patch)  # [B, np, h_lat]
        # latent
        for blk in self.latent:
            patch = blk(patch)
        patch = self.lat_norm(patch)
        # decoder
        byte_q = e  # residual 직결 (BLT 설계)
        np = patch.shape[1]
        byte_q_trim = byte_q[:, : np * patch_size]
        dec_in = self.unpool(byte_q_trim, patch, patch_size=patch_size)
        for blk in self.decoder:
            dec_in = blk(dec_in)
        dec_in = self.dec_norm(dec_in)
        return self.byte_head(dec_in)


# ---------- 벤치마크 ----------


def count_params_m(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


def bench(
    name: str,
    model: nn.Module,
    make_input,
    n_iter: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    model = model.to(device=device, dtype=dtype).train()
    x = make_input()

    # warmup: fwd-only + fwd+bwd 각각 (compile 이 모드별 다르게 잡힘)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x) if not isinstance(x, tuple) else model(*x)
    torch.cuda.synchronize()
    for _ in range(5):
        y = model(x) if not isinstance(x, tuple) else model(*x)
        g = torch.randn_like(y)
        y.backward(g)
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # forward only (train 모드 유지, grad 꺼두고 측정)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(x) if not isinstance(x, tuple) else model(*x)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    # forward + backward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = model(x) if not isinstance(x, tuple) else model(*x)
        g = torch.randn_like(y)
        y.backward(g)
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
    bwd_ms = max(0.0, total_ms - fwd_ms)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    params_m = count_params_m(model)
    return {
        "name": name,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "total_ms": total_ms,
        "peak_mb": peak_mb,
        "params_m": params_m,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--bbpe_seq", type=int, default=512)
    p.add_argument("--blt_byte_seq", type=int, default=2048)
    p.add_argument("--patch_size", type=int, default=6)
    p.add_argument("--iter", type=int, default=20)
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}, flash_attn={HAVE_FLASH}")
    print(
        f"batch={args.batch}  bbpe_seq={args.bbpe_seq}  blt_byte_seq={args.blt_byte_seq}  "
        f"patch_size={args.patch_size}  iter={args.iter}  compile={not args.no_compile}"
    )
    print()

    results = []

    # BBPE Base
    print("--- BBPE Base ---")
    bbpe = BBPEBase()
    if not args.no_compile:
        bbpe = torch.compile(bbpe)

    def _mk_bbpe():
        return torch.randint(0, 35000, (args.batch, args.bbpe_seq), device=device)

    r = bench("BBPE-Base", bbpe, _mk_bbpe, args.iter, device, dtype)
    results.append(r)
    print(
        f"  params={r['params_m']:.1f}M  fwd={r['fwd_ms']:.2f}ms  bwd={r['bwd_ms']:.2f}ms  "
        f"peak={r['peak_mb']:.0f}MB"
    )
    del bbpe
    gc.collect()
    torch.cuda.empty_cache()

    # BLT 128M
    print("\n--- BLT 128M ---")
    blt = BLT128M()
    if not args.no_compile:
        blt = torch.compile(blt)

    def _mk_blt():
        return torch.randint(0, 256, (args.batch, args.blt_byte_seq), device=device)

    r = bench("BLT-128M", blt, _mk_blt, args.iter, device, dtype)
    results.append(r)
    print(
        f"  params={r['params_m']:.1f}M  fwd={r['fwd_ms']:.2f}ms  bwd={r['bwd_ms']:.2f}ms  "
        f"peak={r['peak_mb']:.0f}MB"
    )
    del blt
    gc.collect()
    torch.cuda.empty_cache()

    # summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'name':<12} {'params':>8} {'fwd':>10} {'bwd':>10} {'fwd+bwd':>10} {'peak':>10}")
    for r in results:
        print(
            f"{r['name']:<12} {r['params_m']:>7.1f}M "
            f"{r['fwd_ms']:>8.2f}ms {r['bwd_ms']:>8.2f}ms {r['total_ms']:>8.2f}ms "
            f"{r['peak_mb']:>8.0f}MB"
        )
    print()
    # throughput normalized to "byte equivalent"
    bbpe_bytes = args.bbpe_seq * 3.5
    blt_bytes = args.blt_byte_seq
    bbpe = results[0]
    blt = results[1]
    bbpe_Mbs = bbpe_bytes * args.batch / (bbpe["total_ms"] / 1000.0) / 1e6
    blt_Mbs = blt_bytes * args.batch / (blt["total_ms"] / 1000.0) / 1e6
    print(f"BBPE Base throughput (train)  ≈ {bbpe_Mbs:.2f} Mbyte/s  ({bbpe_bytes:.0f} B/sample)")
    print(f"BLT 128M  throughput (train)  ≈ {blt_Mbs:.2f} Mbyte/s  ({blt_bytes} B/sample)")
    print(f"BLT / BBPE  = {blt_Mbs / bbpe_Mbs:.2f}x")
    bbpe_inf = bbpe_bytes * args.batch / (bbpe["fwd_ms"] / 1000.0) / 1e6
    blt_inf = blt_bytes * args.batch / (blt["fwd_ms"] / 1000.0) / 1e6
    print(f"BBPE Base throughput (fwd)    ≈ {bbpe_inf:.2f} Mbyte/s")
    print(f"BLT 128M  throughput (fwd)    ≈ {blt_inf:.2f} Mbyte/s")
    print(f"BLT / BBPE  (fwd)  = {blt_inf / bbpe_inf:.2f}x")


if __name__ == "__main__":
    main()
