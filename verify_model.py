"""모델 검증 스크립트

1. 모델 인스턴스화
2. 파라미터 수 카운팅 (임베딩 포함/제외)
3. Forward pass 테스트 (랜덤 입력)
4. Backward pass 테스트 (그래디언트 흐름)
"""
import sys
import torch

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def verify_model():
    print("=" * 60)
    print("BitNet-Mamba Seq2Seq 모델 검증")
    print("=" * 60)

    # 1. Config 생성
    config = BitMambaSeq2SeqConfig()
    print(f"\n[Config]")
    print(f"  d_model:           {config.d_model}")
    print(f"  n_encoder_layers:  {config.n_encoder_layers}")
    print(f"  n_decoder_layers:  {config.n_decoder_layers}")
    print(f"  d_inner:           {config.d_inner}")
    print(f"  d_state:           {config.d_state}")
    print(f"  d_ff:              {config.d_ff}")
    print(f"  n_heads:           {config.n_heads}")
    print(f"  vocab_size:        {config.vocab_size}")
    print(f"  tie_embeddings:    {config.tie_embeddings}")
    print(f"  tie_lm_head:       {config.tie_lm_head}")

    # 2. 모델 인스턴스화
    print("\n[모델 생성 중...]")
    model = BitMambaSeq2Seq(config)
    print("  ✓ 모델 생성 완료")

    # 3. 파라미터 수 카운팅
    counts = model.count_parameters()
    print(f"\n[파라미터 수]")
    for key, val in counts.items():
        print(f"  {key:30s}: {format_params(val):>10s} ({val:>12,d})")

    # 4. 임베딩 dtype 확인
    print(f"\n[임베딩 dtype]")
    for name, p in model.named_parameters():
        if "embedding" in name:
            print(f"  {name}: {p.dtype}")

    # 5. Forward pass 테스트
    print(f"\n[Forward Pass 테스트]")
    batch_size = 2
    src_len = 32
    tgt_len = 24

    src_ids = torch.randint(1, config.vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(1, config.vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        logits = model(src_ids, tgt_ids, src_mask)

    expected_shape = (batch_size, tgt_len, config.vocab_size)
    actual_shape = tuple(logits.shape)
    assert actual_shape == expected_shape, \
        f"Shape mismatch: {actual_shape} != {expected_shape}"
    print(f"  ✓ 출력 shape: {actual_shape} (expected: {expected_shape})")
    print(f"  ✓ 출력 dtype: {logits.dtype}")
    print(f"  ✓ 출력 범위: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # 6. Backward pass 테스트
    print(f"\n[Backward Pass 테스트]")
    model.train()
    logits = model(src_ids, tgt_ids, src_mask)
    loss = logits.sum()
    loss.backward()

    # 그래디언트가 있는 파라미터 확인
    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  ✓ 그래디언트 존재: {grad_params}/{total_params} 파라미터")

    # nan/inf 확인
    has_nan = any(p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
    has_inf = any(p.grad.isinf().any() for p in model.parameters() if p.grad is not None)
    print(f"  ✓ NaN 그래디언트: {'❌ 있음' if has_nan else '없음'}")
    print(f"  ✓ Inf 그래디언트: {'❌ 있음' if has_inf else '없음'}")

    # 7. Config 직렬화 테스트
    print(f"\n[Config 직렬화 테스트]")
    import tempfile, os
    tmp = os.path.join(tempfile.gettempdir(), "test_config.json")
    config.save(tmp)
    loaded = BitMambaSeq2SeqConfig.load(tmp)
    assert config == loaded, "Config 직렬화/역직렬화 불일치"
    os.unlink(tmp)
    print(f"  ✓ JSON 직렬화/역직렬화 일치")

    # 결과 요약
    target_params = 128_000_000
    actual_excl = counts["total_excl_embedding"]
    ratio = actual_excl / target_params * 100

    print(f"\n{'=' * 60}")
    print(f"검증 완료!")
    print(f"  임베딩 제외 파라미터: {format_params(actual_excl)} ({ratio:.1f}% of 128M target)")
    if 100_000_000 <= actual_excl <= 140_000_000:
        print(f"  ✓ 목표 범위 (100M ~ 140M) 내")
    else:
        print(f"  ⚠ 목표 범위 (100M ~ 140M) 벗어남 — d_ff 조정 필요")
    print(f"{'=' * 60}")

    return 0 if not has_nan and not has_inf else 1


if __name__ == "__main__":
    sys.exit(verify_model())
