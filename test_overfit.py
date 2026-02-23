import torch
import torch.nn as nn
from difflib import SequenceMatcher
from model.seq2seq import BitMambaSeq2Seq
from model.config import BitMambaSeq2SeqConfig
from training.pretrain import load_tokenizer
from model.triton_bitlinear import replace_bitlinear_with_triton
from model.cuda_bitlinear import replace_bitlinear_with_cuda

def test_overfit(
    backend: str = "fp",
    steps: int = 800,
    target_loss: float = 0.01,
    gen_source_bias: float = 0.5,
):
    """소규모 데이터 과적합 테스트

    backend:
        - fp: 원본 BitLinear
        - triton: triton_bitlinear 교체
        - cuda: cuda_bitlinear 교체
    """
    if backend not in {"fp", "triton", "cuda"}:
        raise ValueError(f"지원하지 않는 backend: {backend}")

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
    
    if backend == "triton":
        model = replace_bitlinear_with_triton(model)
    elif backend == "cuda":
        model = replace_bitlinear_with_cuda(model)
        
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
    eos_id = tokenizer.eos_id
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
    
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    
    src_ids = input_ids
    tgt_input = input_ids[:, :-1]
    tgt_target = input_ids[:, 1:]
    src_mask = (src_ids != pad_id)
    
    backend_name = {
        "fp": "Original (BitLinear)",
        "triton": "INT8 (BitLinearTriton)",
        "cuda": "INT8 (BitLinearCuda)",
    }[backend]
    print(f"\n[{backend_name}] 실제 데이터 오버피팅 테스트 시작...")
    
    import time
    
    # 워밍업
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    total_tokens = 0
    reached_step = None
    final_loss = None
    
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask)
        loss = loss_fn(logits.reshape(-1, config.vocab_size), tgt_target.reshape(-1))
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        n_tok = (tgt_target != pad_id).sum().item()
        total_tokens += n_tok
        loss_val = loss.item()
        final_loss = loss_val
        
        if step % 100 == 0 or step == 1 or loss_val < target_loss:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start_time
            tps = total_tokens / max(elapsed, 0.001)
            print(f"  Step {step:4d} | Loss: {loss_val:.5f} | Speed: {tps:.1f} tok/s")
            if loss_val < target_loss:
                reached_step = step
                break

    if reached_step is None:
        reached_step = steps
    elapsed_total = max(time.time() - start_time, 1e-6)
    avg_tps = total_tokens / elapsed_total
                
    # ==== 생성 테스트 (추론 평가) ====
    print("\n--- 실제 텍스트 복원(생성) 테스트 ---")
    model.eval()
    quality_scores = []
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
                logits = model.decode(
                    cur_ids,
                    encoder_out,
                    sample_mask,
                    src_ids=sample_src,
                    source_bias=gen_source_bias,
                )
                # 다음 토큰 (마지막 시점의 로짓)
                next_token = logits[0, -1].argmax(dim=-1).item()
                gen_tokens.append(next_token)
                
                if next_token == pad_id or next_token == eos_id:
                    break
                    
                # 입력 시퀀스 갱신 (전체 문맥 누적)
                cur_ids = torch.cat([cur_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
            elapsed_gen = time.time() - start_gen_time
            dec_text = tokenizer.decode(gen_tokens)
            score = SequenceMatcher(None, ans[:120], dec_text[:120]).ratio()
            quality_scores.append(score)
            
            print(f"[{i+1}] 원본: {ans[:40]}...")
            print(
                f"[{i+1}] 복원: {dec_text[:40]}... "
                f"(추론속도: {(len(gen_tokens) - prompt_len) / elapsed_gen:.1f} tok/s, "
                f"유사도: {score:.3f})"
            )
            print()

    avg_quality = sum(quality_scores) / max(len(quality_scores), 1)
    result = {
        "backend": backend,
        "reached_step": reached_step,
        "final_loss": float(final_loss if final_loss is not None else float("nan")),
        "avg_tps": float(avg_tps),
        "avg_quality": float(avg_quality),
    }
    print(
        f"[{backend_name}] summary | reached_step={result['reached_step']} "
        f"| final_loss={result['final_loss']:.5f} | avg_tps={result['avg_tps']:.1f} "
        f"| avg_quality={result['avg_quality']:.3f}"
    )
    return result


def compare_until_target(
    steps: int = 2500,
    target_loss: float = 0.01,
    gen_source_bias: float = 0.5,
):
    fp = test_overfit("fp", steps=steps, target_loss=target_loss, gen_source_bias=gen_source_bias)
    cuda = test_overfit("cuda", steps=steps, target_loss=target_loss, gen_source_bias=gen_source_bias)

    print("\n=== FP vs CUDA 비교 요약 ===")
    print(
        f"FP   : step={fp['reached_step']}, loss={fp['final_loss']:.5f}, "
        f"speed={fp['avg_tps']:.1f} tok/s, quality={fp['avg_quality']:.3f}"
    )
    print(
        f"CUDA : step={cuda['reached_step']}, loss={cuda['final_loss']:.5f}, "
        f"speed={cuda['avg_tps']:.1f} tok/s, quality={cuda['avg_quality']:.3f}"
    )
    return {"fp": fp, "cuda": cuda}

if __name__ == "__main__":
    compare_until_target(steps=2500, target_loss=0.01, gen_source_bias=0.5)
