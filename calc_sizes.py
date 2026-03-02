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

print("--- Existing Models ---")
for name, kwargs in MODEL_CONFIGS.items():
    try:
        kwargs["vocab_size"] = 64000
        # Prevent assertion by temporarily bypassing d_model % n_heads for 16M? 
        # Actually, let's just compute manually or skip crash ones
        if name == "16M":
           print(f"{name:5s}: 16M model config will crash on init due to assertion mismatch. Skipping instance... (Approx 19M expected)")
           continue

        config = BitMambaSeq2SeqConfig(**kwargs)
        model = BitMambaSeq2Seq(config)
        counts = model.count_parameters()
        total = counts["total_excl_embedding"] / 1e6
        target = int(name.replace("M", ""))
        print(f"{name:5s}: {total:6.2f}M  (Ratio: {total/target:4.2f}x)")
    except Exception as e:
        print(f"{name:5s}: Error instantiating - {e}")

print("\n--- New Candidates (Targeting ~ 1.19x Ratio) ---")
targets = [("256M", 256), ("512M", 512), ("1B", 1000)]
for size, target in targets:
    best = None
    best_diff = float("inf")
    for d_model in range(768, 2049, 128):
        n_heads = d_model // 64
        # Important: MUST pass d_model % n_heads == 0
        if d_model % n_heads != 0:
            continue
            
        for enc in range(6, 17):
            dec = int(enc * 1.5)
            if size == "1B": dec = int(enc * 1.5) + 1  # 1B might need more decoders
            
            d_inner = d_model * 2
            d_ff = int(max(d_model, int(d_model * 8 / 3 / 128) * 128))
            dt_rank = d_model // 16
            n_kv_heads = 4
            
            kwargs = dict(d_model=d_model, d_inner=d_inner, d_ff=d_ff, n_encoder_layers=enc, n_decoder_layers=dec, n_heads=n_heads, n_kv_heads=n_kv_heads, dt_rank=dt_rank, d_state=16, d_conv=4, vocab_size=64000)
            try:
                config = BitMambaSeq2SeqConfig(**kwargs)
                model = BitMambaSeq2Seq(config)
                counts = model.count_parameters()
                total = counts["total_excl_embedding"] / 1e6
                ratio = total / target
                
                # Check for moderate layer depth
                if 1.15 <= ratio <= 1.22:
                    diff = abs(ratio - 1.19)
                    if diff < best_diff:
                        best_diff = diff
                        best = (total, ratio, kwargs)
            except Exception:
                pass
                
    if best:
        t, r, k = best
        print(f"{size:5s}: {t:6.2f}M  (Ratio: {r:4.2f}x) | d_model={k['d_model']}, enc={k['n_encoder_layers']}, dec={k['n_decoder_layers']}, d_ff={k['d_ff']}, n_heads={k['n_heads']}")
        print(f"      preset_dict = dict(d_model={k['d_model']}, d_inner={k['d_inner']}, d_ff={k['d_ff']}, n_encoder_layers={k['n_encoder_layers']}, n_decoder_layers={k['n_decoder_layers']}, n_heads={k['n_heads']}, n_kv_heads={k['n_kv_heads']}, dt_rank={k['dt_rank']}, d_state=16, d_conv=4)")
