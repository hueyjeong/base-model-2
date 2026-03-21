# Phase 2: Keyboard 토크나이저 + ELECTRA RTD Pretrain

## 목표

keyboard 토크나이저(303 vocab)로 ELECTRA 방식 사전학습 → GEC fine-tune.
자체 토크나이저로도 pretrained attention이 가능한지 검증.

## ELECTRA RTD 개요

```
원본 토큰:    ㅎ ㅏ ㄴ ㄱ ㅜ ㄱ ㅓ
               ↓ (15% 마스킹)
마스크 입력:  ㅎ ㅏ [M] ㄱ ㅜ [M] ㅓ
               ↓ Generator
대체 토큰:    ㅎ ㅏ ㅁ ㄱ ㅜ ㄹ ㅓ
               ↓ Discriminator
판별:         O  O  X  O  O  X  O   (real/replaced)
```

- Generator: 작은 모델, MLM으로 마스크 위치 예측 → 샘플링
- Discriminator: 본 모델, 모든 토큰이 real인지 replaced인지 이진 분류
- 학습 효율: MLM(15% 토큰만 학습) vs RTD(100% 토큰 학습) → **~4배 효율적**

## keyboard 토크나이저 + RTD의 시너지

자모 단위 토큰 대체 = **자모 수준의 오류 감지 학습**
- Generator가 "ㄴ→ㅁ", "ㄱ→ㄹ" 같은 대체를 생성
- Discriminator가 이를 감지하는 법을 학습
- 이것은 곧 GEC에서 필요한 **자모 단위 오류 탐지 능력**과 정확히 일치

## 모델 구성

### Generator (~2M params)
```
Embedding(303, d=128)
├── Transformer Encoder × 4 layers
│   ├── Self-Attention (2 heads, d=128)
│   └── FFN (d_ff=512)
└── MLM Head: Linear(128 → 303)
```

### Discriminator (~8M params, Phase 1과 비교 가능한 규모)
```
Embedding(303, d=256)
├── Transformer Encoder × 12 layers
│   ├── Self-Attention (4 heads, d=256)
│   └── FFN (d_ff=1024, SwiGLU)
└── RTD Head: Linear(256 → 2)   [pretrain]
    Action Head: Linear(256 → 4)  [fine-tune]
    Content Head: Linear(256 → 303) [fine-tune]
```

### 128M Discriminator (BiMamba-2와 직접 비교용)
```
Embedding(303, d=640)
├── Transformer Encoder × 15 layers
│   ├── Self-Attention (GQA-4, d=640)
│   └── FFN (d_ff=1707, SwiGLU)
└── RTD Head → Action Head(4) + Content Head(303)
```

## 구현 단계

### Step 1: ELECTRA Pretrain 구현
- Generator + Discriminator 모델 정의
- RTD 학습 루프 (joint training)
- 기존 corpus/sample_full.jsonl 사용
- 목표: ~100K steps (corpus 크기에 따라 조정)

### Step 2: KLUE 벤치마크 평가 (선택)
- 사전학습된 discriminator의 언어 이해력 측정
- NLI, STS 등으로 KoELECTRA와 비교
- keyboard 토크나이저의 한계 파악

### Step 3: GEC Fine-tune
- RTD head → Action Head(4) + Content Head(303)으로 교체
- Two-head 태그 체계 (Phase 1과 동일)
- 확신도 threshold 튜닝, iterative refinement
- 기존 노이즈 엔진 + 편집 태그 파이프라인 재사용
- Phase 1과 동일 조건에서 평가

### Step 4: 비교 분석
| 모델 | pretrain | fine-tune | edit_P | edit_R | F0.5 |
|------|----------|-----------|--------|--------|------|
| BiMamba-2 | 없음 | GEC | 93.1% | 71.0% | ? |
| KoELECTRA (Phase 1) | 범용 한국어 | GEC | ? | ? | ? |
| Keyboard ELECTRA | 자체 RTD | GEC | ? | ? | ? |

## 핵심 질문

1. **keyboard 303 vocab으로 충분한 언어 모델링이 가능한가?**
   - 자모 단위라 정보 손실 없음, 다만 시퀀스가 길어짐
   - 긴 시퀀스 → attention O(n²) 비용 증가 → window attention 또는 제한된 seq_len

2. **RTD pretrain이 GEC 성능을 실제로 올려주는가?**
   - exp-attention에서 from-scratch attention은 edit_R ~2%로 실패
   - RTD pretrain 후 같은 attention이 동작하면 → pretrain이 핵심 요인 확인

3. **BiMamba-2 from-scratch와 비교해 비용 대비 이점이 있는가?**
   - pretrain 비용 (100K+ steps) + fine-tune 비용 vs BiMamba-2 직행
   - 최종 성능이 월등히 높아야 pretrain 비용 정당화

## 리스크

- keyboard 토크나이저의 긴 시퀀스(~3x) → pretrain 비용 증가
- 303 vocab으로 RTD가 의미 있는 학습 신호를 줄 수 있는지 불확실
- 비교 공정성: KoELECTRA는 대규모 코퍼스, 자체 모델은 제한된 코퍼스
