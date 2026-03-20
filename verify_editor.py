"""BitEditor 검증 스크립트

Phase 1~4 구현 검증:
1. Config JSON roundtrip
2. RWKV-6 TimeMix 형상/기울기
3. BiRWKV 양방향 출력
4. MoE aux_loss 및 expert 기울기
5. SharedLinearSelfAttention LoRA zero-init 검증
6. BitEditor 전체 forward/backward
7. 파라미터 수 확인
8. 편집 태그 시스템
"""
import os
import sys
import tempfile

import torch


def test_config():
    """Config JSON roundtrip"""
    from model.editor_config import BitEditorConfig

    cfg = BitEditorConfig()
    print(f"[Config] d_model={cfg.d_model}, n_rwkv_layers={cfg.n_rwkv_layers}, "
          f"n_experts={cfg.n_experts}, n_tags={cfg.n_tags}")

    # JSON roundtrip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cfg.save(path)
        cfg2 = BitEditorConfig.load(path)
        assert cfg.d_model == cfg2.d_model
        assert cfg.attn_insertion_points == cfg2.attn_insertion_points
        assert cfg.n_tags == cfg2.n_tags
        print("[PASS] Config JSON roundtrip")
    finally:
        os.unlink(path)


def test_rwkv6():
    """RWKV-6 TimeMix 형상/기울기"""
    from model.rwkv_block import RWKV6TimeMix

    B, T, D = 2, 32, 384
    n_heads, headdim = 12, 32
    rwkv = RWKV6TimeMix(D, n_heads, headdim)
    rwkv._init_weights()

    x = torch.randn(B, T, D, requires_grad=True)
    out = rwkv(x)
    assert out.shape == (B, T, D), f"형상 불일치: {out.shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "입력 기울기 없음"
    assert not torch.isnan(x.grad).any(), "NaN 기울기"
    print(f"[PASS] RWKV-6 TimeMix: shape={out.shape}, grad OK")


def test_bi_rwkv():
    """BiRWKV 양방향 출력"""
    from model.bi_rwkv import BiRWKV

    B, T, D = 2, 32, 384
    bi_rwkv = BiRWKV(D, 12, 32)
    bi_rwkv._init_weights()

    x = torch.randn(B, T, D)
    out = bi_rwkv(x)
    assert out.shape == (B, T, D), f"형상 불일치: {out.shape}"

    # 양방향 vs 단방향 출력 차이 확인
    fwd_only = bi_rwkv.forward_rwkv(x)
    assert not torch.allclose(out, fwd_only, atol=1e-3), "BiRWKV가 단방향과 동일"
    print(f"[PASS] BiRWKV: shape={out.shape}, bidirectional OK")


def test_moe():
    """MoE aux_loss 및 expert 기울기"""
    from model.moe import MoEBitNetFFN

    B, T, D = 2, 32, 384
    moe = MoEBitNetFFN(D, d_ff=512, n_experts=16, top_k=1)

    x = torch.randn(B, T, D, requires_grad=True)
    out, aux_loss = moe(x)
    assert out.shape == (B, T, D), f"형상 불일치: {out.shape}"
    assert aux_loss.item() > 0, f"aux_loss가 0: {aux_loss.item()}"

    loss = out.sum() + aux_loss
    loss.backward()

    # 모든 expert에 기울기 흐름 확인 (top_k=1이므로 일부만)
    experts_with_grad = sum(
        1 for e in moe.experts
        if e.gate_proj.weight.grad is not None and e.gate_proj.weight.grad.abs().sum() > 0
    )
    print(f"[PASS] MoE: shape={out.shape}, aux_loss={aux_loss.item():.4f}, "
          f"experts_with_grad={experts_with_grad}/{len(moe.experts)}")


def test_shared_attention():
    """SharedLinearSelfAttention LoRA zero-init 검증"""
    from model.shared_attention import SharedLinearSelfAttention

    B, T, D = 2, 32, 384
    attn = SharedLinearSelfAttention(D, n_heads=24, n_insertion_points=3, lora_rank=16)
    attn.eval()  # dropout 비활성화

    x = torch.randn(B, T, D)

    # LoRA zero-init 시 insertion_idx 0과 1의 출력 차이 확인
    with torch.no_grad():
        out0 = attn(x, insertion_idx=0)
        out1 = attn(x, insertion_idx=1)
    # LoRA up이 zero-init이므로 초기에는 동일해야 함
    assert torch.allclose(out0, out1, atol=1e-5), \
        f"LoRA zero-init 실패: max diff={torch.abs(out0 - out1).max().item()}"
    assert out0.shape == (B, T, D)
    print(f"[PASS] SharedAttention: shape={out0.shape}, LoRA zero-init OK")


def test_editor_forward():
    """BitEditor 전체 forward/backward"""
    from model.editor_config import BitEditorConfig
    from model.editor import BitEditor

    cfg = BitEditorConfig(
        d_model=384,
        n_rwkv_layers=10,
        n_heads=12,
        headdim=32,
        d_ff=512,
        n_experts=16,
        top_k=1,
        n_attn_heads=24,
        attn_insertion_points=(3, 7, 9),
        lora_rank=16,
        vocab_size=303,
        n_tags=608,
        max_seq_len=2048,
    )

    model = BitEditor(cfg)

    # 파라미터 수
    params = model.count_parameters()
    active = model.estimate_active_params()
    print(f"[Info] 총 파라미터: {params['total']:,}")
    print(f"[Info] 활성 파라미터: {active:,}")
    print(f"[Info] 카테고리별: ", end="")
    for k, v in params.items():
        if k not in ("total", "trainable"):
            print(f"{k}={v:,} ", end="")
    print()

    # Forward
    B, T = 2, 64
    input_ids = torch.randint(1, cfg.vocab_size, (B, T))
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    pad_mask[:, -8:] = False  # 마지막 8 토큰은 PAD

    tag_logits, aux_loss = model(input_ids, pad_mask)
    assert tag_logits.shape == (B, T, cfg.n_tags), f"logits 형상 불일치: {tag_logits.shape}"
    print(f"[PASS] Forward: logits={tag_logits.shape}, aux_loss={aux_loss.item():.4f}")

    # Backward
    loss = torch.nn.functional.cross_entropy(
        tag_logits.view(-1, cfg.n_tags),
        torch.zeros(B * T, dtype=torch.long),
    ) + aux_loss
    loss.backward()

    # NaN/Inf 체크
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"
    print(f"[PASS] Backward: loss={loss.item():.4f}, no NaN/Inf gradients")


def test_edit_tags():
    """편집 태그 시스템"""
    from model.edit_tags import (
        compute_edit_tags, apply_edit_tags,
        compute_edit_tags_batch, TAG_KEEP,
    )

    vocab_size = 303

    # 동일 시퀀스
    src = [1, 5, 10, 20]
    tags = compute_edit_tags(src, src, vocab_size)
    assert all(t == TAG_KEEP for t in tags)
    assert apply_edit_tags(src, tags, vocab_size) == src
    print("[PASS] Edit tags: 동일 시퀀스")

    # 치환
    tgt = [1, 99, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    assert apply_edit_tags(src, tags, vocab_size) == tgt
    print("[PASS] Edit tags: 치환")

    # 삭제
    tgt = [1, 10, 20]
    tags = compute_edit_tags(src, tgt, vocab_size)
    assert apply_edit_tags(src, tags, vocab_size) == tgt
    print("[PASS] Edit tags: 삭제")

    # 삽입
    src2 = [1, 10, 20]
    tgt2 = [1, 5, 10, 20]
    tags = compute_edit_tags(src2, tgt2, vocab_size)
    result = apply_edit_tags(src2, tags, vocab_size)
    assert result == tgt2, f"삽입 roundtrip 실패: {result} != {tgt2}"
    print("[PASS] Edit tags: 삽입")

    # 배치
    source = torch.tensor([[1, 5, 10, 0], [1, 10, 20, 0]])
    target = torch.tensor([[1, 99, 10, 0], [1, 10, 0, 0]])
    tags_b = compute_edit_tags_batch(source, target, vocab_size, pad_id=0)
    assert tags_b.shape == (2, 4)
    print(f"[PASS] Edit tags: 배치 shape={tags_b.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("BitEditor 검증")
    print("=" * 60)

    test_config()
    print()

    test_edit_tags()
    print()

    test_rwkv6()
    print()

    test_bi_rwkv()
    print()

    test_moe()
    print()

    test_shared_attention()
    print()

    test_editor_forward()
    print()

    print("=" * 60)
    print("모든 검증 통과!")
    print("=" * 60)
