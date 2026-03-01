"""패킹된 다중 문장 과적합 테스트 — Document Isolation 검증

여러 문장을 한 팩에 [BOS]s1[EOS][PAD gap][BOS]s2[EOS] 형태로 넣고
과적합시켜 각 문장을 독립적으로 복원할 수 있는지 확인.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from training.pretrain import load_tokenizer


def pack_sentences(tokenizer, texts, d_conv=4):
    """여러 문장을 PAD gap 포함하여 하나의 팩으로 구성"""
    pad_id = tokenizer.pad_id
    gap = [pad_id] * d_conv

    src_ids = []
    tgt_ids = []
    boundaries = []  # (start, end) per sentence in the packed sequence

    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special=True)  # [BOS]...[EOS]
        if src_ids:
            src_ids.extend(gap)
            tgt_ids.extend(gap)
        start = len(tgt_ids)
        src_ids.extend(ids)
        tgt_ids.extend(ids)
        end = len(tgt_ids)
        boundaries.append((start, end))

    return (
        torch.tensor([src_ids], dtype=torch.long),
        torch.tensor([tgt_ids], dtype=torch.long),
        boundaries,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer("keyboard")

    # 코퍼스에서 5줄 가져오기
    texts = []
    with open("corpus/sample_1g.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            texts.append(json.loads(line)["text"][:100])  # 100자 제한 (메모리)

    print(f"문장 {len(texts)}개 로드")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:60]}...")

    # 패킹
    d_conv = 4
    src_ids, tgt_ids, boundaries = pack_sentences(tokenizer, texts, d_conv=d_conv)
    src_ids = src_ids.to(device)
    tgt_ids = tgt_ids.to(device)
    src_mask = (src_ids != tokenizer.pad_id)

    pack_len = src_ids.shape[1]
    print(f"\n패킹 길이: {pack_len} 토큰 (문장 {len(texts)}개 + PAD gap {d_conv}×{len(texts)-1})")

    # 모델 생성 (작은 크기)
    config = BitMambaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256, d_inner=512, d_ff=512,
        n_encoder_layers=2, n_decoder_layers=3,
        n_heads=8, n_kv_heads=4, dt_rank=16,
        d_state=16, d_conv=d_conv,
        bos_id=tokenizer.bos_id, pad_id=tokenizer.pad_id,
    )
    model = BitMambaSeq2Seq(config).to(device)
    model.encoder_embedding.float()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터: {n_params / 1e6:.2f}M")

    # 학습
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    tgt_input = tgt_ids[:, :-1]
    tgt_target = tgt_ids[:, 1:]

    steps = 500
    model.train()
    print(f"\n과적합 시작 ({steps} steps)...")
    t0 = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_input, src_mask)
        loss = loss_fn(logits.view(-1, config.vocab_size), tgt_target.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"  Step {step:4d} | Loss: {loss.item():.5f}")

    elapsed = time.time() - t0
    final_loss = loss.item()
    print(f"\n학습 완료 ({elapsed:.1f}s) | Final loss: {final_loss:.5f}")

    # 각 문장별 복원 테스트
    print("\n" + "=" * 60)
    print("문장별 복원 테스트 (Greedy decoding)")
    print("=" * 60)

    model.eval()
    eos_id = tokenizer.eos_id
    quality_scores = []

    with torch.no_grad():
        encoder_out = model.encode(src_ids, src_mask)

        for i, (text, (start, end)) in enumerate(zip(texts, boundaries)):
            # 각 문장의 첫 3 토큰을 프롬프트로 사용
            prompt_len = min(3, end - start - 1)
            prompt = tgt_ids[0, start:start + prompt_len].unsqueeze(0)  # (1, prompt_len)

            gen_tokens = prompt[0].tolist()
            cur_ids = prompt.clone()

            for _ in range(end - start + 10):  # 여유분
                logits = model.decode(
                    cur_ids, encoder_out, src_mask,
                    src_ids=src_ids, source_bias=0.5,
                )
                next_tok = logits[0, -1].argmax().item()
                gen_tokens.append(next_tok)
                if next_tok == eos_id or next_tok == tokenizer.pad_id:
                    break
                cur_ids = torch.cat([cur_ids, torch.tensor([[next_tok]], device=device)], dim=1)

            decoded = tokenizer.decode(gen_tokens, skip_special=True)
            original = text[:80]

            # 문자 단위 유사도
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, original[:80], decoded[:80]).ratio()
            quality_scores.append(score)

            print(f"\n[{i}] 원본: {original[:60]}...")
            print(f"[{i}] 복원: {decoded[:60]}...")
            print(f"[{i}] 유사도: {score:.3f}")

    avg_quality = sum(quality_scores) / len(quality_scores)
    print(f"\n{'=' * 60}")
    print(f"Final loss: {final_loss:.5f} | 평균 유사도: {avg_quality:.3f}")
    print(f"{'=' * 60}")

    if final_loss < 0.5:
        print("✅ 과적합 성공 (loss < 0.5)")
    else:
        print("⚠️  과적합 미완료 (loss >= 0.5, 스텝 수 증가 필요)")

    if avg_quality > 0.7:
        print("✅ 문장 복원 성공 (평균 유사도 > 0.7)")
    else:
        print(f"⚠️  문장 복원 미흡 (평균 유사도 {avg_quality:.3f})")


if __name__ == "__main__":
    main()
