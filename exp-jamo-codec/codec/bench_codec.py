"""SimpleCodec vs SACodec 단독 timing (forward + backward).

각 codec 의 encode + decode 1회 + CE backward 만 측정. KoELECTRA 와 동등 조건:
- BF16 autocast, TF32 ON, eager mode
- 입력: [B*P, S=32] 토큰 batch
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from codec.simple_codec import SimpleCodec
from codec.sa_codec import SACodec
from codec.head_codec import HeadCodec


def bench(model, label, n_tokens, S, n_warmup=5, n_active=20):
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    j = torch.randint(1, 330, (n_tokens, S), device=device)
    mask = torch.zeros(n_tokens, S, dtype=torch.bool, device=device)
    # 각 토큰 평균 ~10 자모 (실측 분포에 가깝게)
    for t in range(n_tokens):
        L = int(torch.randint(3, 16, (1,)).item())
        mask[t, :L] = True

    def step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(j, mask)
            loss = out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # warmup
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()

    # active
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(n_active):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_active
    mem = torch.cuda.max_memory_allocated(device) / 1024**2
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{label:18s}  params={params:5.2f}M  {dt*1000:6.2f} ms/step  "
          f"{1/dt:6.1f} step/s  peak {mem:6.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=4096,
                    help="동시 처리 토큰 수 (B*P 비슷)")
    ap.add_argument("--S", type=int, default=32)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)

    print(f"=== Codec bench: T={args.n_tokens} tokens, S={args.S} slots ===\n")

    bench(SimpleCodec(d_model=256, n_enc_layers=5, n_dec_layers=5,
                       kernel_size=5, max_jamo=args.S),
          "SimpleCodec 5+5L", args.n_tokens, args.S)

    bench(SimpleCodec(d_model=256, n_enc_layers=2, n_dec_layers=2,
                       kernel_size=5, max_jamo=args.S),
          "SimpleCodec 2+2L", args.n_tokens, args.S)

    bench(SACodec(d_model=256, n_heads=8, d_ff=1024, max_jamo=args.S),
          "SACodec 1+1L (h8)", args.n_tokens, args.S)

    bench(SACodec(d_model=256, n_heads=4, d_ff=512, max_jamo=args.S),
          "SACodec 1+1L (h4 d_ff512)", args.n_tokens, args.S)

    bench(HeadCodec(d_model=256, max_jamo=args.S, dec_hidden=1024),
          "HeadCodec (no inter-slot)", args.n_tokens, args.S)

    bench(HeadCodec(d_model=256, max_jamo=args.S, dec_hidden=2048),
          "HeadCodec dec_h=2048", args.n_tokens, args.S)


if __name__ == "__main__":
    main()
