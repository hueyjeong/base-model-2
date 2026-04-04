# Jamo Codec 실험 계획

## 목표

자모 토크나이저의 긴 시퀀스 문제를 해결하기 위한 Neural Codec.
자모/바이트 입력을 연속 잠재 공간으로 압축하여 backbone 처리 속도를 개선하되,
복원 정확도를 거의 무손실(>99.99%) 수준으로 유지.

핵심 질문:
1. 어떤 입력 표현이 codec에 가장 유리한가? (byte / 자모 / 키보드)
2. 어떤 codec 구조가 가장 효율적인가? (conv / cross-attn / 가변 패칭)
3. 압축률과 복원 품질의 트레이드오프는?

## 배경: BLT (Byte Latent Transformer, Meta 2024)

- Entropy Model (50M, ctx=512B) → 동적 패치 경계 결정
- Local Encoder: lightweight transformer, cross-attention으로 bytes→patches
- Global Transformer: 압축된 patch 단위로 처리
- Local Decoder: cross-attention으로 patches→bytes
- 결과: 동급 BPE 모델 대비 FLOP 50% 절감, 더 나은 스케일링
- 엔트로피 모델 키울수록 bpb 감소, 50M/512B가 최적점 (수확 체감)

## GEC 태스크와의 적합성

- 오타도 패턴: "습니다"→"스빈다" 같은 오류는 반복적이고 학습 가능
- 연속 공간의 이점: 이산 토큰에서는 다수의 편집(ㅡ→ㅂ, ㅂ→ㄴ, ㄴ→ㄷ)이
  연속 공간에서는 패치 벡터의 미세한 차이로 표현될 수 있음
  → global transformer가 작은 perturbation 교정으로 GEC 수행 가능
- BLT의 노이즈 강건성: 입력이 깨져도 패치 표현이 안정적
  → GEC에서 원하는 성질 그 자체 (노이즈 입력 → 올바른 출력)

## 실험 축

### Axis 1: Codec 구조 (단순→복잡 순서로 ablation)

| 단계 | 구조 | BLT 여부 | 설명 |
|------|------|---------|------|
| **A. Conv stride** | Conv1d encoder + ConvTranspose1d decoder | 아님 (baseline) | 고정 stride, 가장 단순 |
| **B. Cross-attention** | Local transformer encoder/decoder | BLT 부분 적용 | BLT의 local model만 차용, 고정 stride |
| **C. 엔트로피 기반 가변 패칭** | B + Entropy model + 동적 경계 | BLT 풀 구현 | B에 가변 패칭 추가 |

B와 C를 분리하는 이유: cross-attention 자체의 이득과 가변 패칭의 이득을 독립 측정.

고정 stride의 한계:
- 한국어 음절은 2~3 자모(종성 유무), 키보드 표현은 SHIFT 때문에 가변 길이
- stride=3은 종성 있는 순수 자모(CVC)에서만 음절 경계와 일치
- 키보드 입력이나 종성 없는 음절이 섞이면 경계 어긋남
- 음절 경계가 맞더라도 단어 경계(2~5음절)는 별개 문제
- 예: "뛰어쓰기" 키보드 11토큰 → stride=3 패치 4개, 음절/단어 경계 불일치
- 이것이 BLT가 고정 stride 대신 엔트로피 기반 가변 패칭을 채택한 이유

그럼에도 Phase 1에서 고정 stride로 시작하는 이유:
- codec 자체의 가능성(압축→복원)을 최소 구현으로 빠르게 검증
- 고정 stride의 한계를 정량적으로 확인해야 가변 패칭의 이득을 비교 가능

각 단계에서 이전 대비 유의미한 개선이 없으면 멈추고 그 단계 채택.

### Axis 2: 입력 표현

| 조건 | 입력 단위 | vocab size | "까마귀" 길이 |
|------|----------|------------|-------------|
| **Byte** | UTF-8 바이트 | 256 (+특수) | 9 bytes |
| **자모** | 초/중/종성 분해 | ~40 (+ASCII+특수) | 7 자모 |
| **키보드** | 2벌식 키스트로크 | 현행 303 | 8 (SHIFT+자모) |

자모는 byte 대비 한글 구조가 명시적, 키보드는 SHIFT/BLANK로 입력 오류 표현 가능.

### Axis 3: 압축률

- 2x, 4x, 8x 실험 (로그 스케일 3점)
  - 2x: 음절 미만 (~자모 2개/패치)
  - 4x: 음절~1.3음절 (~자모 4개/패치)
  - 8x: 단어 수준 (~자모 8개/패치, "맞춤법"=9자모 ≈ 1패치)
- 압축률에 따른 두 가지 상반된 힘:
  - 복원 관점: 압축↑ → 정보 손실↑ → 복원 어려움
  - GEC 관점: 압축↑ → 표현이 더 추상적/의미적 → 오타와 정답이 연속 공간에서
    더 가까운 점으로 매핑 → 교정이 더 작은 perturbation으로 가능
- 고정 압축률의 위험: 인명, 고유어, 신조어, 작중어 등 저빈도 토큰은
  학습 데이터 부족으로 고압축 시 구분 불가능해질 수 있음
  → 가변 패칭의 핵심 존재 이유: 고엔트로피(저빈도/예측 어려운) 구간은
    자동으로 적게 압축하여 정보 보존
- 최적점: 복원 정확도가 충분히 유지되는 선에서 가장 높은 압축률
- 복원 정확도 vs 압축률, downstream GEC 품질 vs 압축률 커브 측정
- 특히 인명/신조어/작중어 등 저빈도어에 대한 복원 정확도를 별도 측정 필요

BLT 용어 참고:
- θ_e: Local Encoder (bytes/자모 → patches)
- θ_g: Global Transformer (backbone, 압축된 patch 처리)
- θ_d: Local Decoder (patches → bytes/자모)

## 실험 순서

### Phase 1: Conv Codec 프로토타입 + 입력 표현 비교

**목표**: codec 가능성 검증 + 입력 표현 간 차이 측정

1. 3종 토크나이저 준비 (byte / 자모 / 키보드)
2. Conv codec (stride=3) 구현
3. 동일 코퍼스에서 reconstruction accuracy 비교
4. 학습 속도, 수렴 속도, 복원 정확도 비교

**성공 기준**: 자모 입력 stride=3에서 복원 정확도 >99.9%

### Phase 2: Cross-Attention Local Encoder/Decoder

**목표**: Conv 대비 cross-attention의 품질/속도 트레이드오프 측정

1. BLT식 local transformer encoder/decoder 구현
2. Phase 1 최적 입력 표현 위에서 비교
3. 압축률 2x/3x/4x sweep

**진행 조건**: Phase 1에서 codec 자체가 가능하다는 검증 후

### Phase 3: 엔트로피 기반 가변 패칭

**목표**: 가변 압축의 실제 이득 측정

1. 소형 엔트로피 모델 (n-gram 또는 작은 transformer)
2. 정보 밀도 기반 패치 경계 결정
3. 고정 stride 대비 복원 품질 / 평균 압축률 비교

**진행 조건**: Phase 2에서 cross-attention이 conv 대비 유의미한 이득일 때

### Phase 4: Backbone 통합

**목표**: codec + backbone end-to-end 학습, 실제 GEC 태스크 성능

1. Phase 1~3 중 최적 codec 선택
2. ELECTRA RTD pretrain 또는 편집 태깅 학습에 통합
3. codec 없는 baseline (현행 4096 seq) 대비 속도/품질 비교

## 평가 지표

- **복원 정확도**: character-level accuracy, sequence-level exact match
- **압축률**: 입력 토큰 수 / 출력 패치 수
- **codec 속도**: encode + decode latency (GPU, CPU)
- **backbone 속도**: 압축된 seq에서 tok/s
- **총 처리량**: codec overhead 포함 end-to-end tok/s
- **downstream 품질**: GEC F0.5 (Phase 4)

## 파일 구조 (예상)

```
exp-jamo-codec/
├── PLAN.md              # 이 문서
├── tokenizers/          # 3종 토크나이저 (byte, jamo, keyboard)
├── codec/               # codec 모델 구현
│   ├── conv_codec.py    # Phase 1: Conv 기반
│   ├── xattn_codec.py   # Phase 2: Cross-attention 기반
│   └── entropy_patch.py # Phase 3: 엔트로피 패칭
├── train_codec.py       # codec 단독 학습 스크립트
├── eval_codec.py        # 복원 정확도 / 속도 벤치마크
└── results/             # 실험 결과 로그
```

## 제약 조건

- 학습: RTX 5060 Ti 16GB (개발), 5090 x4~8 DDP (본학습)
- codec 자체는 가벼워야 함 — backbone 속도 이득을 상쇄하지 않을 것
- CPU 추론 시에도 codec overhead가 무시 가능해야 함
- 기존 keyboard tokenizer의 SHIFT/BLANK 정보가 가치 있는지 검증 필요
