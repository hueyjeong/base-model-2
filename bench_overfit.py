import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
import time
import json
from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer

def bench_model_overfit(name, source_bias=0.0, use_copy_gate=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer("keyboard")
    pad_id = tokenizer.pad_id
    
    config = BitMambaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=1024,
        use_copy_gate=use_copy_gate
    )
    
    model = BitMambaSeq2Seq(config).to(device)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 5 문장 로드
    texts = []
    with open("corpus/sample_10g.jsonl", "r", encoding="utf-8") as f:
        for _ in range(5):
            line = f.readline()
            if not line: break
            texts.append(json.loads(line)["text"])

    encoded = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in encoded)
    pad_id = config.pad_id
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
    
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    
    src_ids = input_ids
    tgt_input = input_ids[:, :-1]
    tgt_target = input_ids[:, 1:]
    src_mask = (src_ids != pad_id)
    
    print(f"\n--- {name} (Bias: {source_bias}, CopyGate: {use_copy_gate}) 1500-step Training ---")
    # 워밍업
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    torch.cuda.synchronize()
    start_time = time.time()
    total_tokens = 0
    
    for step in range(1, 1501):
        optimizer.zero_grad()
        
        logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        n_tok = (tgt_target != pad_id).sum().item()
        total_tokens += n_tok
        loss_val = loss.item()
        
        if step % 500 == 0 or step == 1 or loss_val < 0.01:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            tps = total_tokens / max(elapsed, 0.001)
            print(f"  Step {step:4d} | Loss: {loss_val:.5f} | Speed: {tps:.1f} tok/s")
            if loss_val < 0.01:
                break

    # 평가: 모델이 주어진 입력을 얼마나 잘 복원하는가? 
    model.eval()
    with torch.no_grad():
        logits = model(src_ids, tgt_input, src_mask, source_bias=source_bias)
        bpc = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1)).item()
        
        print(f"  Final Evaluation BPC: {bpc:.4f}")
        
if __name__ == "__main__":
    bench_model_overfit("Baseline", source_bias=0.0, use_copy_gate=False)
    bench_model_overfit("Trial A", source_bias=0.5, use_copy_gate=False)
    bench_model_overfit("Trial B", source_bias=0.0, use_copy_gate=True)
    bench_model_overfit("Trial A+B", source_bias=0.5, use_copy_gate=True)
