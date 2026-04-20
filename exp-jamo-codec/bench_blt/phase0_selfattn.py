"""Phase 0 — self-attention 단독 micro-benchmark.

BLT encoder/decoder 의 byte-level self-attention 비용을
Transformer SWA(512) vs Mamba-2(ds=64) 로 직접 비교.

합격선: seq=2048 에서 forward speedup ≥ 3x.

사용:
    source .venv/bin/activate
    python exp-jamo-codec/bench_blt/phase0_selfattn.py
    python exp-jamo-codec/bench_blt/phase0_selfattn.py --seqs 256 512 1024 2048 4096 8192
    python exp-jamo-codec/bench_blt/phase0_selfattn.py --batch 4 --hidden 384
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func  # type: ignore

    HAVE_FLASH = True
except ImportError:
    HAVE_FLASH = False

try:
    from mamba_ssm import Mamba2  # type: ignore

    HAVE_MAMBA = True
except ImportError:
    HAVE_MAMBA = False

try:
    from fla.layers import GatedLinearAttention, LinearAttention  # type: ignore

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


class TransformerSWA(nn.Module):
    """Sliding-window causal self-attention (BLT local encoder 설정).

    flash-attn 이 있으면 window_size 로 bounded causal, 없으면 SDPA + mask fallback.
    """

    def __init__(self, hidden: int, n_heads: int = 6, window: int = 512) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.window = window
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, D]

        if HAVE_FLASH:
            # flash-attn 은 (B, L, H, D) 를 받음. window_size=(w, 0) = (left, right)
            o = flash_attn_func(
                q, k, v, causal=True, window_size=(self.window, 0)
            )  # [B, L, H, D]
        else:
            # SDPA fallback — SWA mask 를 명시 구성 (O(L^2) memory)
            q = q.transpose(1, 2)  # [B, H, L, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            idx = torch.arange(l, device=x.device)
            dist = idx.unsqueeze(0) - idx.unsqueeze(1)  # [L, L]
            mask = (dist >= 0) & (dist < self.window)
            attn_mask = torch.zeros(l, l, dtype=x.dtype, device=x.device)
            attn_mask.masked_fill_(~mask, float("-inf"))
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )
            o = o.transpose(1, 2)  # [B, L, H, D]

        o = o.reshape(b, l, h)
        return self.out(o)


class MambaBlock(nn.Module):
    """Mamba-2 self-mixing (single direction, BLT 용)."""

    def __init__(self, hidden: int, d_state: int = 64, headdim: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.mamba = Mamba2(
            d_model=hidden,
            d_state=d_state,
            d_conv=4,
            expand=2,
            headdim=headdim,
            chunk_size=256,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class LinearAttnBlock(nn.Module):
    """FLA LinearAttention (chunk mode) — causal linear attention, fused Triton kernel."""

    def __init__(self, hidden: int, n_heads: int = 6) -> None:
        super().__init__()
        self.attn = LinearAttention(
            mode="chunk",
            hidden_size=hidden,
            num_heads=n_heads,
            feature_map="elementwise_product",
            output_norm="rmsnorm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LinearAttention(hidden_states, ...) → (hidden, None, None)
        out = self.attn(x)
        if isinstance(out, tuple):
            return out[0]
        return out


class GLABlock(nn.Module):
    """FLA Gated Linear Attention (chunk mode) — linear attention + gating."""

    def __init__(self, hidden: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = GatedLinearAttention(
            mode="chunk",
            hidden_size=hidden,
            num_heads=n_heads,
            expand_k=0.5,
            expand_v=1.0,
            use_short_conv=False,
            fuse_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(x)
        if isinstance(out, tuple):
            return out[0]
        return out


@dataclass
class Result:
    name: str
    seq: int
    fwd_ms: float
    bwd_ms: float
    peak_mb: float
    params_m: float


def count_params_m(m: nn.Module) -> float:
    return sum(p.numel() for p in m.parameters()) / 1e6


@torch.no_grad()
def _warmup_fwd(model: nn.Module, x: torch.Tensor, n: int = 10) -> None:
    for _ in range(n):
        _ = model(x)
    torch.cuda.synchronize()


def _warmup_fwd_bwd(
    model: nn.Module, x: torch.Tensor, grad: torch.Tensor, n: int = 10
) -> None:
    for _ in range(n):
        x.grad = None
        for p in model.parameters():
            p.grad = None
        y = model(x)
        y.backward(grad)
    torch.cuda.synchronize()


def bench_one(
    name: str,
    model: nn.Module,
    seq: int,
    batch: int,
    hidden: int,
    n_iter: int,
    dtype: torch.dtype,
    device: torch.device,
    do_backward: bool,
) -> Result:
    model = model.to(device=device, dtype=dtype).train()
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)

    # warmup + compile
    _warmup_fwd(model.eval(), x)
    if do_backward:
        model.train()
        _warmup_fwd_bwd(model, x, grad)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # forward only
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000.0 / n_iter

    bwd_ms = 0.0
    if do_backward:
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            x.grad = None
            for p in model.parameters():
                p.grad = None
            y = model(x)
            y.backward(grad)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0 / n_iter
        bwd_ms = max(0.0, total_ms - fwd_ms)

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    params_m = count_params_m(model)
    return Result(name, seq, fwd_ms, bwd_ms, peak_mb, params_m)


def run(
    seqs: list[int],
    batch: int,
    hidden: int,
    n_heads: int,
    window: int,
    d_state: int,
    headdim: int,
    n_iter: int,
    use_compile: bool,
    skip_backward: bool,
) -> list[Result]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    results: list[Result] = []

    if not HAVE_MAMBA:
        raise RuntimeError("mamba_ssm 설치 필요")

    def _try(name: str, make_model) -> None:
        mdl = make_model()
        if use_compile:
            mdl = torch.compile(mdl)
        try:
            r = bench_one(
                name,
                mdl,
                seq,
                batch,
                hidden,
                n_iter,
                dtype,
                device,
                do_backward=not skip_backward,
            )
            results.append(r)
            print(
                f"  {name:<8} fwd={r.fwd_ms:7.2f}ms bwd={r.bwd_ms:7.2f}ms "
                f"peak={r.peak_mb:7.0f}MB params={r.params_m:.2f}M"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {name:<8} OOM")
            results.append(Result(name, seq, float("nan"), float("nan"), 0, 0))
        except Exception as e:
            print(f"  {name:<8} ERR {type(e).__name__}: {e}")
            results.append(Result(name, seq, float("nan"), float("nan"), 0, 0))
        finally:
            del mdl
            gc.collect()
            torch.cuda.empty_cache()

    for seq in seqs:
        print(f"\n=== seq={seq} ===")
        _try("TF-SWA", lambda: TransformerSWA(hidden=hidden, n_heads=n_heads, window=window))
        _try("Mamba-2", lambda: MambaBlock(hidden=hidden, d_state=d_state, headdim=headdim))
        if HAVE_FLA:
            _try("LinAttn", lambda: LinearAttnBlock(hidden=hidden, n_heads=n_heads))
            _try("GLA", lambda: GLABlock(hidden=hidden, n_heads=4))

    return results


def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 96)
    print("SUMMARY — forward ms (lower better), mem MB, speedup vs TF-SWA")
    print("=" * 96)
    by_seq: dict[int, dict[str, Result]] = {}
    for r in results:
        by_seq.setdefault(r.seq, {})[r.name] = r
    names = ["TF-SWA", "Mamba-2", "LinAttn", "GLA"]

    # header
    header = f"{'seq':>6}  "
    for n in names:
        header += f"{n + ' fwd':>11}  {n + ' ×':>6}  "
    print(header)
    for seq in sorted(by_seq):
        d = by_seq[seq]
        tf = d.get("TF-SWA")
        row = f"{seq:>6}  "
        for n in names:
            r = d.get(n)
            if r is None or r.fwd_ms != r.fwd_ms:
                row += f"{'--':>10}ms  {'--':>6}  "
            else:
                if tf is None or tf.fwd_ms != tf.fwd_ms or r.fwd_ms == 0:
                    ratio = "   nan"
                else:
                    ratio = f"{tf.fwd_ms / r.fwd_ms:5.2f}x"
                row += f"{r.fwd_ms:>9.2f}ms  {ratio:>6}  "
        print(row)
    print()
    # backward
    header = f"{'seq':>6}  "
    for n in names:
        header += f"{n + ' bwd':>11}  {n + ' ×':>6}  "
    print(header)
    for seq in sorted(by_seq):
        d = by_seq[seq]
        tf = d.get("TF-SWA")
        row = f"{seq:>6}  "
        for n in names:
            r = d.get(n)
            if r is None or r.bwd_ms != r.bwd_ms:
                row += f"{'--':>10}ms  {'--':>6}  "
            else:
                if tf is None or tf.bwd_ms != tf.bwd_ms or r.bwd_ms == 0:
                    ratio = "   nan"
                else:
                    ratio = f"{tf.bwd_ms / r.bwd_ms:5.2f}x"
                row += f"{r.bwd_ms:>9.2f}ms  {ratio:>6}  "
        print(row)
    print("=" * 96)
    print("합격선: seq=2048 forward × (TF-SWA / X) ≥ 3.0 — X 가 TF-SWA 대비 3배 빨라야 합격")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--iter", type=int, default=30)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--skip_backward", action="store_true")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}, mamba_ssm={HAVE_MAMBA}, flash_attn={HAVE_FLASH}")
    print(
        f"batch={args.batch} hidden={args.hidden} heads={args.n_heads} window={args.window} "
        f"ds={args.d_state} headdim={args.headdim}"
    )
    print(f"iter={args.iter} compile={not args.no_compile} backward={not args.skip_backward}")

    results = run(
        seqs=args.seqs,
        batch=args.batch,
        hidden=args.hidden,
        n_heads=args.n_heads,
        window=args.window,
        d_state=args.d_state,
        headdim=args.headdim,
        n_iter=args.iter,
        use_compile=not args.no_compile,
        skip_backward=args.skip_backward,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
