import torch
import torch.nn as nn
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
from model.triton_bitlinear import replace_bitlinear_with_triton

def test_overfit(use_int8=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer("keyboard")
    
    config = BitMambaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=1024,
    )
    model = BitMambaSeq2Seq(config).to(device)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()
    
    if use_int8:
        model = replace_bitlinear_with_triton(model)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    import json
    
    # 실제 코퍼스에서 10줄 가져오기 (OOM 방지: 16GB GPU에서 AMP/GradCheck 없는 순수 루프)
    texts = []
    with open("corpus/sample_10g.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10: break
            texts.append(json.loads(line)["text"])
            
    encoded = [tokenizer.encode(t) for t in texts]
    
    # 패딩 및 텐서 변환
    max_len = max(len(seq) for seq in encoded)
    pad_id = config.pad_id
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
    
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    
    src_ids = input_ids
    tgt_input = input_ids[:, :-1]
    tgt_target = input_ids[:, 1:]
    src_mask = (src_ids != pad_id)
    
    print(f"\n[{'INT8 (BitLinearTriton)' if use_int8 else 'Original (BitLinear)'}] 실제 데이터 오버피팅 테스트 시작...")
    
    import time
    
    # 워밍업
    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask)
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
        logits = model(src_ids, tgt_input, src_mask)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        n_tok = (tgt_target != pad_id).sum().item()
        total_tokens += n_tok
        loss_val = loss.item()
        
        if step % 100 == 0 or step == 1 or loss_val < 0.01:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            tps = total_tokens / max(elapsed, 0.001)
            print(f"  Step {step:4d} | Loss: {loss_val:.5f} | Speed: {tps:.1f} tok/s")
            if loss_val < 0.01:
                break
                
    # ==== 생성 테스트 (추론 평가) ====
    print("\n--- 실제 텍스트 복원(생성) 테스트 ---")
    model.eval()
    with torch.no_grad():
        # 첫 3개 샘플만 테스트
        for i in range(3):
            # 1. 인코딩 구간
            sample_src = src_ids[i:i+1] # (1, seq_len)
            sample_mask = src_mask[i:i+1]
            encoder_out = model.encode(sample_src, sample_mask)
            
            # 2. 첫 6개 토큰을 프롬프트로 제공 후 순차 생성 (Greedy)
            ans = texts[i]
            # 원본 정답의 첫 6토큰을 시작 컨텍스트로 줌
            prompt_len = 6
            cur_ids = tgt_input[i:i+1, 0:prompt_len] 
            
            gen_tokens = cur_ids[0].tolist()
            
            start_gen_time = time.time()
            for _ in range(40): # 40 토큰 추론
                logits = model.decode(cur_ids, encoder_out, sample_mask)
                # 다음 토큰 (마지막 시점의 로짓)
                next_token = logits[0, -1].argmax(dim=-1).item()
                gen_tokens.append(next_token)
                
                if next_token == pad_id: 
                    break
                    
                # 입력 시퀀스 갱신 (전체 문맥 누적)
                cur_ids = torch.cat([cur_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
            elapsed_gen = time.time() - start_gen_time
            dec_text = tokenizer.decode(gen_tokens)
            
            print(f"[{i+1}] 원본: {ans[:40]}...")
            print(f"[{i+1}] 복원: {dec_text[:40]}... (추론속도: {(len(gen_tokens) - prompt_len) / elapsed_gen:.1f} tok/s)")
            print()

if __name__ == "__main__":
    test_overfit(use_int8=False) # Original Test
    test_overfit(use_int8=True)  # INT8 Test
