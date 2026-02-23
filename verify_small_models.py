"""소형 모델 검증 스크립트

bbpe_tokenizer 기반으로 BitNet-Mamba Seq2Seq가
8M, 16M, 32M, 64M 사이즈에서 정상 작동하는지 검증한다.

검증 항목:
    1. Config 생성 및 제약 조건 통과
    2. 모델 인스턴스화 + 파라미터 수 확인
    3. bbpe_tokenizer 로드 + 인코딩/디코딩 라운드트립
    4. Forward pass (랜덤 입력)
    5. Backward pass + 그래디언트 NaN/Inf 확인
    6. Config JSON 직렬화 라운드트립
"""
import os
import sys
import tempfile
import traceback

import torch

from model.config import BitMambaSeq2SeqConfig
from model.seq2seq import BitMambaSeq2Seq
from bbpe_tokenizer.bbpe_wrapper import BBPETokenizer


# ── 소형 모델 Config 프리셋 ──────────────────────────────────
SMALL_CONFIGS = {
    "8M": dict(
        d_model=256, d_inner=512, d_ff=448,
        n_encoder_layers=3, n_decoder_layers=5,
        n_heads=8, dt_rank=16,
        d_state=16, d_conv=4,
    ),
    "16M": dict(
        d_model=320, d_inner=640, d_ff=640,
        n_encoder_layers=4, n_decoder_layers=6,
        n_heads=8, dt_rank=24,
        d_state=16, d_conv=4,
    ),
    "32M": dict(
        d_model=448, d_inner=896, d_ff=768,
        n_encoder_layers=4, n_decoder_layers=7,
        n_heads=8, dt_rank=32,
        d_state=16, d_conv=4,
    ),
    "64M": dict(
        d_model=512, d_inner=1024, d_ff=896,
        n_encoder_layers=6, n_decoder_layers=10,
        n_heads=8, dt_rank=32,
        d_state=16, d_conv=4,
    ),
}


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def verify_one(label: str, config_kwargs: dict, tokenizer: BBPETokenizer) -> bool:
    """단일 모델 사이즈 검증. 성공 시 True 반환."""
    print(f"\n{'━' * 60}")
    print(f"  {label} 모델 검증")
    print(f"{'━' * 60}")

    # -- vocab_size를 bbpe 토크나이저에 맞춤 --
    config_kwargs["vocab_size"] = tokenizer.vocab_size
    config = BitMambaSeq2SeqConfig(**config_kwargs)

    print(f"  d_model={config.d_model}, d_inner={config.d_inner}, d_ff={config.d_ff}")
    print(f"  encoder_layers={config.n_encoder_layers}, decoder_layers={config.n_decoder_layers}")
    print(f"  n_heads={config.n_heads}, dt_rank={config.dt_rank}")
    print(f"  vocab_size={config.vocab_size}")

    # 1. 모델 인스턴스화
    print(f"\n  [1] 모델 인스턴스화…")
    model = BitMambaSeq2Seq(config)
    print(f"      ✓ 성공")

    # 2. 파라미터 수 확인
    counts = model.count_parameters()
    target = int(label.replace("M", "")) * 1_000_000
    excl = counts["total_excl_embedding"]
    ratio = excl / target * 100
    print(f"\n  [2] 파라미터 수")
    print(f"      임베딩:               {format_params(counts['embedding']):>10s}")
    print(f"      인코더:               {format_params(counts['encoder']):>10s}")
    print(f"      디코더:               {format_params(counts['decoder']):>10s}")
    print(f"      LM Head:              {format_params(counts['lm_head']):>10s}")
    print(f"      전체 (임베딩 제외): {format_params(excl):>10s}  ({ratio:.1f}% of {label})")

    # 3. 토크나이저 연동 테스트
    print(f"\n  [3] bbpe_tokenizer 연동")
    test_texts = [
        "맞춤법을 확인해 주세요.",
        "한글과 English 혼합 테스트입니다.",
        "띄어쓰기 오류가 있습니다.",
    ]
    for txt in test_texts:
        ids = tokenizer.encode(txt, add_special=True)
        decoded = tokenizer.decode(ids, skip_special=True)
        ok = txt.strip() == decoded.strip()
        print(f"      원문: {txt}")
        print(f"      토큰수: {len(ids)}, 복원: {'✓' if ok else '✗ → ' + repr(decoded)}")

    # 4. Forward pass
    print(f"\n  [4] Forward pass")
    batch_size = 2
    src_len = 32
    tgt_len = 24

    src_ids = torch.randint(1, config.vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(1, config.vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        logits = model(src_ids, tgt_ids, src_mask)

    expected = (batch_size, tgt_len, config.vocab_size)
    actual = tuple(logits.shape)
    assert actual == expected, f"Shape mismatch: {actual} != {expected}"
    print(f"      ✓ 출력 shape: {actual}")
    print(f"      ✓ 출력 dtype: {logits.dtype}")
    print(f"      ✓ 범위: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # 5. Backward pass
    print(f"\n  [5] Backward pass")
    model.train()
    logits = model(src_ids, tgt_ids, src_mask)
    loss = logits.sum()
    loss.backward()

    grad_params = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    has_nan = any(p.grad.isnan().any() for p in model.parameters() if p.grad is not None)
    has_inf = any(p.grad.isinf().any() for p in model.parameters() if p.grad is not None)
    print(f"      ✓ 그래디언트: {grad_params}/{total_params} 파라미터")
    print(f"      {'❌ NaN 있음' if has_nan else '✓ NaN 없음'}")
    print(f"      {'❌ Inf 있음' if has_inf else '✓ Inf 없음'}")

    # 6. Config 직렬화 라운드트립
    print(f"\n  [6] Config 직렬화")
    tmp = os.path.join(tempfile.gettempdir(), f"test_config_{label}.json")
    config.save(tmp)
    loaded = BitMambaSeq2SeqConfig.load(tmp)
    assert config == loaded, "Config 직렬화/역직렬화 불일치"
    os.unlink(tmp)
    print(f"      ✓ JSON 라운드트립 일치")

    # 결과
    passed = not has_nan and not has_inf
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {status} — {label} ({format_params(excl)} excl-embed, {ratio:.1f}% of target)")

    # 메모리 정리
    del model, logits, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return passed


def main():
    print("=" * 60)
    print("  BitNet-Mamba Seq2Seq — 소형 모델 사이즈 검증")
    print("  (bbpe_tokenizer 기반)")
    print("=" * 60)

    # bbpe 토크나이저 로드
    bbpe_path = os.path.join(os.path.dirname(__file__), "bbpe_tokenizer", "bbpe.json")
    if not os.path.exists(bbpe_path):
        print(f"\n❌ bbpe 토크나이저 파일이 없습니다: {bbpe_path}")
        print("  먼저 train_tokenizer.py로 토크나이저를 생성해 주세요.")
        return 1

    tokenizer = BBPETokenizer(bbpe_path)
    print(f"\n  bbpe_tokenizer 로드 완료 (vocab_size={tokenizer.vocab_size})")

    results = {}
    for label, kwargs in SMALL_CONFIGS.items():
        try:
            passed = verify_one(label, dict(kwargs), tokenizer)
            results[label] = passed
        except Exception:
            traceback.print_exc()
            results[label] = False

    # ── 최종 요약 ─────────────────────────────────────────────
    print(f"\n\n{'=' * 60}")
    print("  최종 결과 요약")
    print(f"{'=' * 60}")
    all_pass = True
    for label, passed in results.items():
        target = int(label.replace("M", "")) * 1_000_000
        status = "✅ PASS" if passed else "❌ FAIL"
        cfg = SMALL_CONFIGS[label]
        print(f"  {label:>4s}  {status}  (d_model={cfg['d_model']}, "
              f"enc={cfg['n_encoder_layers']}, dec={cfg['n_decoder_layers']})")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  🎉 모든 소형 모델 사이즈에서 정상 작동 확인!")
    else:
        print(f"\n  ⚠ 일부 사이즈에서 문제 발견 — 위 로그 확인")

    print(f"{'=' * 60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
