import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
import time
import json
from training.noising import DenoisingNoiser, NoiseConfig

def eval_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer("keyboard")
    pad_id = tokenizer.pad_id
    
    # 3 문장 로드
    texts = []
    with open("corpus/sample_10g.jsonl", "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline()
            if not line: break
            texts.append(json.loads(line)["text"])

    noiser = DenoisingNoiser(tokenizer, NoiseConfig(), use_korean_errors=True)
    
    src_list, tgt_list = [], []
    corrupted_texts = []
    for t in texts:
        s, t_out = noiser(t, "ko")
        src_list.append(torch.tensor(s, dtype=torch.long))
        tgt_list.append(torch.tensor(t_out, dtype=torch.long))
        corrupted_texts.append(tokenizer.decode(s))
        
    src_ids = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id).to(device)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id).to(device)
    tgt_input = tgt_padded[:, :-1]
    tgt_target = tgt_padded[:, 1:]
    src_mask = (src_ids != pad_id)
    
    configs = [
        ("Baseline", 0.0, False),
        ("Trial A (Bias 0.5)", 0.5, False),
        ("Trial B (Copy Gate)", 0.0, True),
        ("Trial A+B (Bias 0.5 + Gate)", 0.5, True)
    ]
    
    results = {i: {"Original": texts[i], "Corrupted": corrupted_texts[i]} for i in range(3)}
    
    for name, bias, gate in configs:
        print(f"\n--- Training {name} for 300 steps ---")
        config = BitMambaSeq2SeqConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=256, n_heads=8, n_encoder_layers=2, n_decoder_layers=2, d_ff=1024,
            use_copy_gate=gate
        )
        
        model = BitMambaSeq2Seq(config).to(device)
        model.encoder_embedding.float()
        if not config.tie_embeddings:
            model.decoder_embedding.float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        model.train()
        for step in range(1, 301):
            optimizer.zero_grad()
            logits = model(src_ids, tgt_input, src_mask, source_bias=bias)
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            for i in range(3):
                sample_src = src_ids[i:i+1]
                sample_mask = src_mask[i:i+1]
                encoder_out = model.encode(sample_src, sample_mask)
                
                # 프롬프트 없이 처음부터 추론
                cur_ids = torch.tensor([[tokenizer.bos_id]], device=device) if hasattr(tokenizer, 'bos_id') else torch.tensor([[1]], device=device) # 가정: 1이 특수 시작 토큰이나 임의 시작
                # 그냥 첫 번째 타겟 토큰 제공
                cur_ids = tgt_input[i:i+1, 0:1] 
                
                gen_tokens = cur_ids[0].tolist()
                
                for _ in range(60): # 60 토큰 추론
                    if gate:
                        # CopyGate에서 source_ids 내부 참조
                        logits = model.decode(cur_ids, encoder_out, sample_mask, src_ids=sample_src, source_bias=bias)
                    else:
                        logits = model.decode(cur_ids, encoder_out, sample_mask, src_ids=sample_src, source_bias=bias)
                        
                    next_token = logits[0, -1].argmax(dim=-1).item()
                    gen_tokens.append(next_token)
                    
                    if next_token == pad_id: 
                        break
                        
                    cur_ids = torch.cat([cur_ids, torch.tensor([[next_token]], device=device)], dim=1)
                    
                dec_text = tokenizer.decode(gen_tokens)
                results[i][name] = dec_text

    print("\n\n====== QUALITATIVE EVALUATION RESULTS ======")
    for i in range(3):
        print(f"\n[Sample {i+1}]")
        print(f"Original : {results[i]['Original']}")
        print(f"Corrupted: {results[i]['Corrupted']}")
        print("-" * 40)
        for name, _, _ in configs:
            print(f"{name:<25}: {results[i][name]}")

if __name__ == "__main__":
    eval_generation()
