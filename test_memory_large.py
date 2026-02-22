import torch
import torch.nn as nn
from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from model.triton_bitlinear import replace_bitlinear_with_triton
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

device = 'cuda'

config_128M = dict(
    d_model=768, d_inner=1536, d_ff=1280,
    n_encoder_layers=7, n_decoder_layers=12,
    n_heads=12, n_kv_heads=4, dt_rank=48,
    d_state=16, d_conv=4,
)
config = BitMambaSeq2SeqConfig(**config_128M, vocab_size=32000)
model = BitMambaSeq2Seq(config).to(device)
model = replace_bitlinear_with_triton(model)

fused_ce_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=0, reduction="mean")

B = 1
seq_len = 16384
src_ids = torch.randint(0, 32000, (B, seq_len), device=device)
tgt_ids = torch.randint(0, 32000, (B, seq_len), device=device)
src_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

def test_mem(grad_ckpt=False):
    model.train()
    model.encoder.gradient_checkpointing = grad_ckpt
    model.decoder.gradient_checkpointing = grad_ckpt

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1e9
    
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_out = model.encode(src_ids, src_mask)
            hidden = model.decode(tgt_ids[:, :-1], encoder_out, src_mask, return_hidden=True, src_ids=src_ids)
            loss = fused_ce_loss(
                model.lm_head.weight.float(),
                hidden.view(-1, hidden.size(-1)).float(),
                tgt_ids[:, 1:].reshape(-1),
            )
            
        fwd_mem = torch.cuda.max_memory_allocated() / 1e9
        loss.backward()
        bwd_mem = torch.cuda.max_memory_allocated() / 1e9
        
        # clear grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
                
        print(f"ckpt={grad_ckpt} | Fwd: {fwd_mem-start_mem:.2f} GB | Bwd: {bwd_mem-start_mem:.2f} GB")
    except Exception as e:
        print(f"ckpt={grad_ckpt} | Error: {e}")

print(f"Testing seq_len = {seq_len}")
# test_mem(False)  # We know this OOMs
test_mem(True)
