import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig

# Test FP16 vs BF16 vs FP32 for NaN issue
device = "cuda"
config = BitMambaSeq2SeqConfig(
    vocab_size=303, d_model=288, d_inner=576, d_ff=544,
    n_encoder_layers=3, n_decoder_layers=5, n_heads=8, n_kv_heads=4, dt_rank=18
)

model = BitMambaSeq2Seq(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")

# Dummy data
src_ids = torch.randint(1, 303, (4, 1024), device=device)
src_mask = torch.ones_like(src_ids, dtype=torch.bool)
tgt_ids = torch.randint(1, 303, (4, 1024), device=device)
tgt_input = tgt_ids[:, :-1]
tgt_target = tgt_ids[:, 1:]

criterion = nn.CrossEntropyLoss(ignore_index=0)

print("Starting AMP (fp16) test loop...")
for i in range(10):
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits = model(src_ids, tgt_input, src_mask)
        loss = criterion(logits.view(-1, 303), tgt_target.reshape(-1))
    
    scaler.scale(loss).backward()
    
    # Check for NaN in gradients BEFORE step
    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"NaN grad found in {name} at step {i}")
            has_nan_grad = True
            break
            
    if has_nan_grad or torch.isnan(loss):
        print(f"FAILED: NaN loss at step {i}: {loss.item()}")
        break
        
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Step {i} loss: {loss.item():.4f}")
