"""실제 학습 병목 프로파일링

torch.profiler로 어디에서 시간이 소비되는지 정확히 측정.
RTX 5060 Ti 16GB에서 실행 (싱글 GPU).
"""
import time
import os
import torch
import torch.nn as nn

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from model.cuda_bitlinear import replace_bitlinear_with_cuda

DEVICE = "cuda"
VOCAB = 303
B, L = 4, 2048  # 사용자 production config와 동일한 batch_size, pack_size

# 32M-ish config for 16GB GPU
config = BitMambaSeq2SeqConfig(
    vocab_size=VOCAB, d_model=448, d_inner=896, d_ff=768,
    n_encoder_layers=5, n_decoder_layers=9, n_heads=8, n_kv_heads=4, dt_rank=32,
    d_state=128, d_conv=4, mamba_version=2,
)


def make_data():
    src = torch.randint(1, VOCAB, (B, L), device=DEVICE)
    tgt = torch.randint(1, VOCAB, (B, L), device=DEVICE)
    mask = torch.ones_like(src, dtype=torch.bool)
    return src, tgt, mask


def profile_training():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Model config: d_model={config.d_model}, enc={config.n_encoder_layers}, dec={config.n_decoder_layers}")
    print(f"Batch: {B} x {L} = {B*L} tokens/step\n")

    model = BitMambaSeq2Seq(config).to(DEVICE)
    model = replace_bitlinear_with_cuda(model)
    model.encoder_embedding.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    src, tgt, mask = make_data()
    tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]

    # Warmup
    print("Warming up (3 steps)...")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(src, tgt_in, mask)
            loss = criterion(logits.view(-1, VOCAB), tgt_tgt.reshape(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Manual timing: forward vs backward vs optimizer
    print("\n--- Manual Timing (5 steps avg) ---")
    fwd_times = []
    bwd_times = []
    opt_times = []

    for step in range(5):
        optimizer.zero_grad()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(src, tgt_in, mask)
            loss = criterion(logits.view(-1, VOCAB), tgt_tgt.reshape(-1))
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        opt_times.append(t3 - t2)

    fwd_avg = sum(fwd_times) / len(fwd_times) * 1000
    bwd_avg = sum(bwd_times) / len(bwd_times) * 1000
    opt_avg = sum(opt_times) / len(opt_times) * 1000
    total_avg = fwd_avg + bwd_avg + opt_avg
    tok_per_sec = (B * (L-1)) / (total_avg / 1000)

    print(f"  Forward:   {fwd_avg:7.1f} ms ({fwd_avg/total_avg*100:.0f}%)")
    print(f"  Backward:  {bwd_avg:7.1f} ms ({bwd_avg/total_avg*100:.0f}%)")
    print(f"  Optimizer: {opt_avg:7.1f} ms ({opt_avg/total_avg*100:.0f}%)")
    print(f"  Total:     {total_avg:7.1f} ms/step")
    print(f"  tok/s:     {tok_per_sec:.0f}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

    # torch.profiler로 상세 분석
    print("\n--- torch.profiler 상세 분석 ---")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(src, tgt_in, mask)
            loss = criterion(logits.view(-1, VOCAB), tgt_tgt.reshape(-1))
        loss.backward()
        optimizer.step()

    # Top CUDA kernels by time
    print("\n[Top 15 CUDA kernels by GPU time]")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Top operations
    print("\n[Top 15 ops by total CPU+CUDA time]")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    profile_training()
