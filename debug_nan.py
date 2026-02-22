import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
import sys

def debug_nan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer("keyboard")
    pad_id = tokenizer.pad_id
    
    config = BitMambaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=288, n_encoder_layers=3, n_decoder_layers=5,
        use_copy_gate=True
    )
    model = BitMambaSeq2Seq(config).to(device)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()
        
    model.encoder.apply(
        lambda m: setattr(m, 'gradient_checkpointing', True) if hasattr(m, 'gradient_checkpointing') else None
    )
    model.decoder.apply(
        lambda m: setattr(m, 'gradient_checkpointing', True) if hasattr(m, 'gradient_checkpointing') else None
    )
        
    try:
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        fused_ce_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=pad_id)
    except ImportError:
        print("Liger Kernel CE not available.")
        sys.exit(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print("Loading test dataset...")
    from training.noising import DenoisingNoiser, NoiseConfig
    from training.dataset import StreamingPackedDataset
    from torch.utils.data import DataLoader

    noiser = DenoisingNoiser(tokenizer, NoiseConfig(), seed=42, use_korean_errors=True)
    dataset = StreamingPackedDataset(
        "corpus/sample_10g.jsonl", tokenizer, noiser,
        pack_size=2048, text_key="text", seed=42
    )
    
    def collate_packed(batch):
        src_list = [b["src_ids"] for b in batch]
        tgt_list = [b["tgt_ids"] for b in batch]
        # _make_sample guarantees truncating to shape (pack_size,), so we can just stack.
        src_ids = torch.stack(src_list, dim=0)
        tgt_ids = torch.stack(tgt_list, dim=0)
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_packed)
    data_iter = iter(loader)

    model.train()
    print("Testing forward pass + fused_ce_loss with REAL data and amp.autocast(bf16)....")

    for step in range(1, 15):
        batch = next(data_iter)
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        src_mask = (src_ids != pad_id).to(device)
        
        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]
        
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoder_out = model.encode(src_ids, src_mask)
            hidden = model.decode(tgt_input, encoder_out, src_mask, return_hidden=True, src_ids=src_ids, source_bias=0.8)
            
            # Fused CE Loss requires (input, weight, target)
            loss = fused_ce_loss(hidden.view(-1, hidden.size(-1)), model.lm_head.weight, tgt_target.reshape(-1))
            
        loss.backward()
        g_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"Step {step} | Loss: {loss.item():.4f} | Grad Norm: {g_norm.item():.4f}", flush=True)
        if torch.isnan(loss):
            print("NaN Loss Detected!", flush=True)
            # Find the bad parameter
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print(f"NaN Gradient in: {name}", flush=True)
            break

if __name__ == "__main__":
    debug_nan()
