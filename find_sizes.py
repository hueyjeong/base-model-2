import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq

MODEL_CONFIGS = {
    "8M": dict(d_model=288, d_inner=576, d_ff=544, n_encoder_layers=3, n_decoder_layers=5, n_heads=8, n_kv_heads=4, dt_rank=24, d_state=16, d_conv=4),
    "16M": dict(d_model=352, d_inner=704, d_ff=640, n_encoder_layers=4, n_decoder_layers=7, n_heads=8, n_kv_heads=4, dt_rank=24, d_state=16, d_conv=4),
    "32M": dict(d_model=448, d_inner=896, d_ff=768, n_encoder_layers=5, n_decoder_layers=9, n_heads=8, n_kv_heads=4, dt_rank=32, d_state=16, d_conv=4),
    "64M": dict(d_model=576, d_inner=1152, d_ff=1088, n_encoder_layers=6, n_decoder_layers=10, n_heads=8, n_kv_heads=4, dt_rank=40, d_state=16, d_conv=4),
    "128M": dict(d_model=768, d_inner=1536, d_ff=1280, n_encoder_layers=7, n_decoder_layers=12, n_heads=12, n_kv_heads=4, dt_rank=48, d_state=16, d_conv=4),
}

def check(name, kwargs, target_m=None):
    kwargs["vocab_size"] = 64000
    config = BitMambaSeq2SeqConfig(**kwargs)
    model = BitMambaSeq2Seq(config)
    counts = model.count_parameters()
    total = counts["total_excl_embedding"]
    total_m = total / 1_000_000
    if target_m:
        print(f"{name:5s}: {total_m:6.2f}M (Ratio: {total_m/target_m:4.2f}x) | {kwargs}")
    else:
        print(f"{name:5s}: {total_m:6.2f}M | {kwargs}")

print("--- Existing Models ---")
for name, kwargs in MODEL_CONFIGS.items():
    target = int(name.replace("M", ""))
    check(name, kwargs, target)

print("\n--- New Candidates ---")

def make_candidate(d_model, enc, dec, heads, kv_heads):
    return dict(
        d_model=d_model,
        d_inner=d_model * 2,
        d_ff=int(max(d_model, int(d_model * 8 / 3 / 128) * 128)),
        n_encoder_layers=enc,
        n_decoder_layers=dec,
        n_heads=heads,
        n_kv_heads=kv_heads,
        dt_rank=d_model // 16,
        d_state=16,
        d_conv=4,
    )

check("256M", make_candidate(1024, 8, 16, 16, 4), 256)
check("256M", make_candidate(1024, 9, 14, 16, 4), 256)
check("512M", make_candidate(1536, 10, 18, 16, 2), 512)
check("512M", make_candidate(1408, 12, 18, 16, 4), 512)
check("1B", make_candidate(2048, 12, 20, 16, 4), 1000)
check("1B", make_candidate(1920, 14, 22, 16, 4), 1000)
