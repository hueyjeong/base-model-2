# Phase 2: Keyboard 토크나이저 + BiMamba2 ELECTRA RTD Pretrain

## 목표

keyboard 토크나이저(303 vocab) + BiMamba2 인코더로 ELECTRA RTD 사전학습 → GEC fine-tune.
DenseEditor와 동일한 BiMamba2 mixing + SwiGLU FFN 구조, 단 BitLinear(ternary) 대신 Int8Linear(INT8 QAT).
INT8 QAT + BF16 AMP — 기존 학습 인프라 재사용.

## 배경: 왜 BiMamba2인가?

### DenseEditor 검증 결과
- **BiMamba2 ds=64**: 10k DDP에서 loss 0.107, R 71.0% — 다른 mixing 대비 압도적 1위
- Mamba-1 대비: loss 37%↓, recall +19.5pp, CPU 22% 빠름
- 이미 검증된 학습 파이프라인(DDP, INT8 QAT, 패킹, document isolation) 전부 재사용 가능

### 설계 원칙
1. **검증된 구조 재사용**: DenseEditor BiMamba2 + SwiGLU FFN 구조, Int8Linear로 양자화
2. **INT8 QAT**: 전 학습 과정을 INT8 QAT로 진행 → 같은 모델 용량이면 파라미터 수 극대화
3. **기존 인프라 활용**: pretrain_dense_editor.py의 DDP/패킹/체크포인트 코드 최대한 재사용
4. **128M 유지**: 기존 DenseEditor와 동일 규모 → 직접 비교 가능

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
- Discriminator: 본 모델 (BiMamba2 인코더), 모든 토큰이 real인지 replaced인지 이진 분류
- 학습 효율: MLM(15% 토큰만 학습) vs RTD(100% 토큰 학습) → **~4배 효율적**

## keyboard 토크나이저 + RTD의 시너지

자모 단위 토큰 대체 = **자모 수준의 오류 감지 학습**
- Generator가 "ㄴ→ㅁ", "ㄱ→ㄹ" 같은 대체를 생성
- Discriminator가 이를 감지하는 법을 학습
- 이것은 곧 GEC에서 필요한 **자모 단위 오류 탐지 능력**과 정확히 일치

## INT8 QAT 전략

- **Discriminator**: INT8 QAT (Int8Linear). pretrain(RTD) + fine-tune(GEC) 모두 적용
- **동기**: 같은 모델 파일 크기(바이트)에서 FP32 대비 4× 많은 파라미터 수용 가능
- **BitNetFFN 아님**: 기존 DenseEditor의 BitLinear(ternary) 대신 표준 Linear + INT8 양자화 사용
- **Generator도 INT8 QAT**: Discriminator와 동일하게 Int8Linear + Int8FFN 적용

## 모델 구성

### Generator (~32M params, BiMamba2, INT8 QAT)
```
Embedding(303, d=384)
├── BiMamba2Layer × 10
│   ├── RMSNorm → BiMamba2Mixing (d_state=64, headdim=64, expand=2, ngroups=1, Int8Linear)
│   └── RMSNorm → Int8FFN SwiGLU (d_ff=1024)  # 384 × 8/3
├── Final RMSNorm
└── MLM Head: Linear(384 → 303)
```
Discriminator의 1/4 크기 (ELECTRA 논문 기본 비율).

### Discriminator — 128M (INT8 QAT)
```
Embedding(303, d=640)
├── EncoderLayer × 15
│   ├── RMSNorm → BiMamba2Mixing
│   │   (d_state=64, headdim=64, expand=2, ngroups=1, chunk_size=256)
│   └── RMSNorm → SwiGLU FFN (d_ff=1707, Int8Linear)  # 640 × 8/3
├── Final RMSNorm
└── RTD Head: Linear(640 → 2)   [pretrain]
    Tag Head: Linear(640 → n_tags=608) [fine-tune]
```

**기존 DenseEditor와의 차이**:
- BitLinear(ternary) → Int8Linear(INT8 QAT)
- Tag Head → RTD Head(2-class)로 pretrain, 이후 Tag Head로 교체

## 구현 단계

### Step 1: ELECTRA RTD 학습 루프 구현
- pretrain_dense_editor.py를 기반으로 ELECTRA 학습 스크립트 작성
- Generator (소형 BiMamba2) + Discriminator (DenseEditor 128M) joint training
- Generator MLM loss + Discriminator RTD loss 동시 최적화
- 마스킹 전략: 15% 토큰 [MASK] → Generator가 대체 → Discriminator가 판별
- 기존 패킹/document isolation/DDP 코드 재사용
- 기존 corpus/sample_full.jsonl 사용
- 목표: ~100K steps (corpus 크기에 따라 조정)

### Step 2: KLUE 벤치마크 평가
- 사전학습된 discriminator의 언어 이해력 측정
- **KLUE-TC (YNAT)**: 뉴스 토픽 7-class 분류 (가장 단순, KoELECTRA-base ~87%)
  - 랜덤 14%, 70%+ 나오면 한국어 이해 확인
- KLUE-NLI: 문장쌍 → 3-class (KoELECTRA-base ~81%)
- KLUE-STS: 문장쌍 유사도 (KoELECTRA-base ~93 Pearson)
- `datasets` 라이브러리: `load_dataset("klue", "ynat")`
- keyboard 토크나이저의 한계 파악 (seq 길이 ~3x 불이익)

### Step 3: GEC Fine-tune
- RTD head → Tag Head(n_tags=608)로 교체 (single-head, DenseEditor와 동일)
- 확신도 threshold 튜닝, iterative refinement
- 기존 노이즈 엔진 + 편집 태그 파이프라인 재사용
- pretrain_dense_editor.py의 GEC 학습 코드 거의 그대로 사용 가능

### Step 4: 비교 분석
| 모델 | pretrain | 구조 | edit_P | edit_R | F0.5 |
|------|----------|------|--------|--------|------|
| DenseEditor 128M | 없음 (from scratch) | BiMamba2 15L | 93.1% | 71.0% | ? |
| Keyboard ELECTRA 128M | 자체 RTD | BiMamba2 15L | ? | ? | ? |

**핵심 비교**: 동일 구조에서 RTD pretrain 유무만 다름 → pretrain 효과 순수 측정

## 핵심 질문

1. **RTD pretrain이 GEC 성능을 실제로 올려주는가?**
   - 기존 DenseEditor(from-scratch)는 edit_R 71.0%
   - RTD pretrain 후 동일 구조로 fine-tune하면 이를 넘는가?
   - 특히 edit_R (재현율) 개선이 핵심 — 현재 71%가 병목

2. **303 vocab으로 RTD가 의미 있는 학습 신호를 줄 수 있는가?**
   - 자모 토큰 303개 중 대부분이 자모/특수문자 → 대체 시 쉽게 구별 가능?
   - Generator가 너무 쉬우면 Discriminator가 충분히 학습 못함
   - Generator 크기/학습 밸런스 조정 필요할 수 있음

3. **Pretrain 비용 대비 효용이 있는가?**
   - keyboard 토크나이저 seq ~3x → pretrain 비용 증가
   - 100K steps RTD pretrain + GEC fine-tune vs 그냥 GEC 300K steps
   - 총 학습 비용 대비 성능 향상이 유의미한지

## 리스크

- keyboard 토크나이저의 긴 시퀀스(~3x) → pretrain 비용 증가
- 303 vocab으로 RTD가 의미 있는 학습 신호를 줄 수 있는지 불확실
  - 자모 수준에서 "real vs replaced" 구분이 너무 쉬울 수 있음
  - → Generator를 키우거나 마스킹 비율 조정으로 대응
- Generator-Discriminator 학습 밸런스 맞추기 (loss weight 비율)
- Pretrain → Fine-tune 전이 시 catastrophic forgetting 가능성
