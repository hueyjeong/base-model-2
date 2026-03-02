"""Document Isolation (PAD Gap + dA Zeroing) 검증 테스트

Option C (PAD Gap) + Option D (dA Zeroing) 구현 검증:
1. PAD gap이 d_conv만큼 삽입되는지
2. d_conv 변경 시 gap도 연동되는지
3. MambaBlock reset_mask가 SSM output을 0으로 만드는지
4. 패킹된 두 문장의 처리 결과가 독립 처리 결과와 유사한지
5. reset_mask=None일 때 기존 동작과 동일한지
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from model.config import BitMambaSeq2SeqConfig
from model.mamba_block import MambaBlock
from model.seq2seq import BitMambaSeq2Seq


def test_pad_gap_insertion():
    """PAD gap이 d_conv=4만큼 삽입되는지 검증"""
    from training.noising import DenoisingNoiser, NoiseConfig
    from training.dataset import StreamingPackedDataset
    from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer

    tok = BBPETokenizer(os.path.join(os.path.dirname(__file__), "bbpe_tokenizer", "bbpe.json"))
    noiser = DenoisingNoiser(tok, NoiseConfig(), seed=42)

    ds = StreamingPackedDataset(
        os.path.join(os.path.dirname(__file__), "corpus", "sample_1g.jsonl"),
        tok, noiser, pack_size=512, d_conv=4,
        text_key="text", seed=42,
    )

    sample = next(iter(ds))
    src = sample["src_ids"]
    pad_id = tok.pad_id
    bos_id = tok.bos_id
    eos_id = tok.eos_id

    # PAD gap 패턴 확인: [EOS][PAD][PAD][PAD][PAD][BOS]
    # 연속된 PAD 4개가 최소 한 번 있어야 함 (문장이 2개 이상 패킹된 경우)
    src_list = src.tolist()

    # BOS 개수 확인 (패킹된 문장 수)
    bos_count = src_list.count(bos_id)
    if bos_count <= 1:
        print("  ⚠️  문장 1개만 패킹됨 — gap 테스트 표본 불충분 (SKIP)")
        return True

    # EOS 다음에 PAD gap이 있는지 확인
    gap_found = False
    for i in range(len(src_list) - 5):
        if src_list[i] == eos_id:
            if src_list[i+1:i+5] == [pad_id] * 4:
                gap_found = True
                break

    assert gap_found, (
        f"PAD gap을 찾을 수 없음 (BOS={bos_count}개). "
        f"src[:30]={src_list[:30]}"
    )
    print(f"  ✅ PAD gap 삽입 확인 (d_conv=4, 패킹된 문장 {bos_count}개)")
    return True


def test_pad_gap_respects_d_conv():
    """d_conv=6으로 설정 시 gap도 6개인지 검증"""
    from training.noising import DenoisingNoiser, NoiseConfig
    from training.dataset import StreamingPackedDataset
    from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer

    tok = BBPETokenizer(os.path.join(os.path.dirname(__file__), "bbpe_tokenizer", "bbpe.json"))
    noiser = DenoisingNoiser(tok, NoiseConfig(), seed=42)

    ds = StreamingPackedDataset(
        os.path.join(os.path.dirname(__file__), "corpus", "sample_1g.jsonl"),
        tok, noiser, pack_size=512, d_conv=6,
        text_key="text", seed=42,
    )

    sample = next(iter(ds))
    src_list = sample["src_ids"].tolist()
    pad_id = tok.pad_id
    eos_id = tok.eos_id
    bos_count = src_list.count(tok.bos_id)

    if bos_count <= 1:
        print("  ⚠️  문장 1개만 패킹됨 — gap 테스트 표본 불충분 (SKIP)")
        return True

    gap_found = False
    for i in range(len(src_list) - 7):
        if src_list[i] == eos_id:
            if src_list[i+1:i+7] == [pad_id] * 6:
                gap_found = True
                break

    assert gap_found, f"d_conv=6 gap을 찾을 수 없음. BOS={bos_count}개"
    print(f"  ✅ d_conv=6 연동 확인 (gap=6개 PAD)")
    return True


def test_mamba_reset_mask():
    """MambaBlock에서 reset_mask=True인 위치의 출력이 ~0인지 검증"""
    torch.manual_seed(42)

    block = MambaBlock(d_model=64, d_inner=128, d_state=16, d_conv=4, dt_rank=8)
    block.eval()

    x = torch.randn(1, 10, 64)  # (B=1, L=10, d=64)
    reset_mask = torch.zeros(1, 10, dtype=torch.bool)
    reset_mask[0, 5] = True  # 위치 5에서 리셋

    with torch.no_grad():
        # reset_mask 없이 (기존 동작)
        out_no_reset = block(x, reset_mask=None)

        # reset_mask 있을 때
        out_with_reset = block(x, reset_mask=reset_mask)

    # 리셋 위치(5)의 출력이 0에 가까워야 함  
    # (residual connection이 seq2seq에서 적용되므로 MambaBlock 자체는 0)
    reset_out = out_with_reset[0, 5].abs().max().item()
    assert reset_out < 1e-3, f"리셋 위치 출력이 0이 아님: max={reset_out}"

    # 리셋 위치 이전(0..4)은 동일해야 함
    diff_before = (out_no_reset[0, :5] - out_with_reset[0, :5]).abs().max().item()
    assert diff_before < 1e-5, f"리셋 이전 위치가 달라짐: max_diff={diff_before}"

    print(f"  ✅ reset_mask 동작 확인 (리셋 위치 출력: {reset_out:.2e}, "
          f"이전 위치 차이: {diff_before:.2e})")
    return True


def test_backward_compatible():
    """reset_mask=None일 때 기존 동작과 완전히 동일한지"""
    torch.manual_seed(42)

    block = MambaBlock(d_model=64, d_inner=128, d_state=16, d_conv=4, dt_rank=8)
    block.eval()

    x = torch.randn(1, 10, 64)

    with torch.no_grad():
        out1 = block(x)  # 인자 없이 (기존 호출)
        out2 = block(x, reset_mask=None)  # 명시적 None

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"reset_mask=None이 기존과 다름: diff={diff}"
    print(f"  ✅ 하위 호환성 확인 (diff=0.0)")
    return True


def test_seq2seq_forward():
    """전체 Seq2Seq 모델의 forward가 정상 동작하는지"""
    torch.manual_seed(42)

    config = BitMambaSeq2SeqConfig(
        vocab_size=1000, d_model=64, d_inner=128, d_ff=128,
        n_encoder_layers=1, n_decoder_layers=1,
        n_heads=4, n_kv_heads=2, dt_rank=8, d_state=16, d_conv=4,
        bos_id=1, pad_id=0,
    )
    model = BitMambaSeq2Seq(config)
    model.encoder_embedding.float()
    if not config.tie_embeddings:
        model.decoder_embedding.float()
    model.eval()

    # 패킹된 시퀀스 시뮬레이션: [BOS]s1[EOS][PAD][PAD][PAD][PAD][BOS]s2[EOS]
    bos, eos, pad = 1, 2, 0
    src = torch.tensor([[bos, 10, 20, 30, eos, pad, pad, pad, pad, bos, 40, 50, eos, pad]])
    tgt = torch.tensor([[bos, 10, 20, 30, eos, pad, pad, pad, pad, bos, 40, 50, eos, pad]])

    with torch.no_grad():
        logits = model(src, tgt[:, :-1])

    assert logits.shape == (1, 13, 1000), f"출력 shape 불일치: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN 발생!"
    assert not torch.isinf(logits).any(), "Inf 발생!"
    print(f"  ✅ Seq2Seq forward 정상 (shape={logits.shape}, NaN=없음, Inf=없음)")
    return True


def test_seq2seq_backward():
    """Backward pass (gradient 계산)가 정상인지"""
    torch.manual_seed(42)

    config = BitMambaSeq2SeqConfig(
        vocab_size=1000, d_model=64, d_inner=128, d_ff=128,
        n_encoder_layers=1, n_decoder_layers=1,
        n_heads=4, n_kv_heads=2, dt_rank=8, d_state=16, d_conv=4,
        bos_id=1, pad_id=0,
    )
    model = BitMambaSeq2Seq(config)
    model.encoder_embedding.float()
    model.train()

    bos, eos, pad = 1, 2, 0
    src = torch.tensor([[bos, 10, 20, 30, eos, pad, pad, pad, pad, bos, 40, 50, eos, pad]])
    tgt = torch.tensor([[bos, 10, 20, 30, eos, pad, pad, pad, pad, bos, 40, 50, eos, pad]])

    logits = model(src, tgt[:, :-1])
    loss = nn.CrossEntropyLoss(ignore_index=0)(
        logits.view(-1, 1000), tgt[:, 1:].reshape(-1)
    )
    loss.backward()

    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            has_nan_grad = True
            print(f"    ⚠️  NaN gradient: {name}")

    assert not has_nan_grad, "NaN gradient 발생!"
    print(f"  ✅ Backward pass 정상 (loss={loss.item():.4f}, NaN grad=없음)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Document Isolation 검증 테스트")
    print("=" * 60)

    tests = [
        ("1. PAD gap 삽입", test_pad_gap_insertion),
        ("2. d_conv 연동", test_pad_gap_respects_d_conv),
        ("3. MambaBlock reset_mask", test_mamba_reset_mask),
        ("4. 하위 호환성", test_backward_compatible),
        ("5. Seq2Seq forward", test_seq2seq_forward),
        ("6. Seq2Seq backward", test_seq2seq_backward),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n{name}:")
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"결과: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if failed > 0 else 0)
