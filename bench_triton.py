#!/usr/bin/env python3
"""BitLinear vs BitLinearTriton 벤치마크"""
import torch
import time
import sys
sys.path.insert(0, ".")

from model.bitlinear import BitLinear
from model.triton_bitlinear import BitLinearTriton

def benchmark(name, layer, x, warmup=20, repeat=100):
    """Forward + backward 벤치마크"""
    # Warmup
    for _ in range(warmup):
        out = layer(x)
        loss = out.sum()
        loss.backward()
        layer.zero_grad()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(repeat):
        out = layer(x)
        loss = out.sum()
        loss.backward()
        layer.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_iter = elapsed / repeat * 1000
    print(f"  {name:25s}: {ms_per_iter:.3f} ms/iter")
    return ms_per_iter


def main():
    device = "cuda"
    dtype = torch.float32

    configs = [
        # (in, out, batch, seq_len)
        ("FFN gate/up (768→1280)", 768, 1280, 4, 512),
        ("FFN down (1280→768)",    1280, 768, 4, 512),
        ("Mamba in_proj (768→3072)", 768, 3072, 4, 512),
        ("Mamba out_proj (1536→768)", 1536, 768, 4, 512),
    ]

    for desc, in_f, out_f, B, T in configs:
        print(f"\n{desc} — input ({B}, {T}, {in_f})")

        orig = BitLinear(in_f, out_f).to(device, dtype)
        triton_layer = BitLinearTriton(in_f, out_f).to(device, dtype)

        # 같은 가중치 사용
        triton_layer.weight.data.copy_(orig.weight.data)
        triton_layer.norm.load_state_dict(orig.norm.state_dict())

        x = torch.randn(B, T, in_f, device=device, dtype=dtype, requires_grad=False)

        t_orig = benchmark("BitLinear (original)", orig, x)
        t_triton = benchmark("BitLinearTriton", triton_layer, x)

        speedup = t_orig / t_triton
        print(f"  → Speedup: {speedup:.2f}x")

    # 메모리 비교
    print("\n=== 메모리 사용량 비교 ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    orig = BitLinear(768, 1280).to(device, dtype)
    x = torch.randn(4, 512, 768, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    out = orig(x)
    out.sum().backward()
    orig_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  BitLinear (original): {orig_mem:.1f} MB peak")

    del orig, out, x
    torch.cuda.empty_cache()

    triton_layer = BitLinearTriton(768, 1280).to(device, dtype)
    x = torch.randn(4, 512, 768, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    out = triton_layer(x)
    out.sum().backward()
    triton_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  BitLinearTriton:      {triton_mem:.1f} MB peak")


if __name__ == "__main__":
    main()
