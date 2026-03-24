# Phase 2: Keyboard 토크나이저 + 커스텀 ELECTRA RTD Pretrain

## 목표

keyboard 토크나이저(303 vocab) + 커스텀 인코더 구조로 ELECTRA RTD 사전학습 → GEC fine-tune.
표준 Transformer 대신 Conv1d + ChunkFFT + Sliding Window/Full Attention 샌드위치 구조 사용.
ONNX RT 배포를 전제로 설계 (GPU/CPU 모두 지원).
전 학습 과정 INT8 QAT — 같은 모델 용량(바이트)이면 파라미터 수를 극대화.

## 배경: 왜 커스텀 구조인가?

### 추론 성능 제약
- **ONNX RT CPU** (KoELECTRA-base): seq=512에서 156ms → seq=4096은 수 초 (불가)
- **CUDA PyTorch** (DenseEditor 12L): seq=4096에서 62ms (현행 베이스라인)
- **wgpu Vulkan**: Tensor Core 접근 불가 → CUDA 대비 4x+ 느림
- keyboard 토크나이저는 seq=4096이 일반 WordPiece seq=512급 → **seq=4096이 실질 운용 범위**
- Full Attention 12L@seq=4096: attention만 387G FLOPs → 대부분의 연산이 attention에 소모

### 설계 원칙
1. **연산 절약**: Full Attention은 전역 동기화용 2~3개만, 나머지는 Sliding Window
2. **Conv1d 전처리**: 자모 토큰은 개별로 의미 없음 → conv1d로 음절 단위 feature 합성
3. **ChunkFFT**: conv1d 결과를 청크 단위 FFT → 슬라이스별 "분위기" 요약 → attention이 참조
4. **ONNX 호환**: 모든 부품이 표준 연산 (conv1d, FFT, attention, linear)
5. **INT8 QAT**: 전 학습 과정을 INT8 QAT로 진행 → 같은 모델 용량이면 파라미터 수 극대화
6. **FFN 축소**: d_ff 축소 가능 (넓게 줬던 건 BitLinear 보상 목적이었음)

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
- Discriminator: 본 모델 (커스텀 인코더), 모든 토큰이 real인지 replaced인지 이진 분류
- 학습 효율: MLM(15% 토큰만 학습) vs RTD(100% 토큰 학습) → **~4배 효율적**

## keyboard 토크나이저 + RTD의 시너지

자모 단위 토큰 대체 = **자모 수준의 오류 감지 학습**
- Generator가 "ㄴ→ㅁ", "ㄱ→ㄹ" 같은 대체를 생성
- Discriminator가 이를 감지하는 법을 학습
- 이것은 곧 GEC에서 필요한 **자모 단위 오류 탐지 능력**과 정확히 일치

## 커스텀 인코더 구조

### 핵심 아이디어: Conv1d → ChunkFFT → Attention 샌드위치

```
입력 토큰 (seq=4096, d=768)
  │
  ├── Conv1d (k=4~8): 자모 → 음절 패턴 합성
  │
  ├── ChunkFFT:
  │     chunk_size=256으로 분할 → 16 슬라이스
  │     각 슬라이스 FFT → learned projection → mood vector (d=768)
  │     16개 mood vector에 Full Attention → 슬라이스 간 교차 참조
  │     결과를 해당 슬라이스 토큰에 주입 (cross-attention 또는 add)
  │
  └── Attention + FFN 레이어 스택
```

### 레이어 구성 (11L, 다이아몬드 윈도우)

```
Conv1d + ChunkFFT          ← 자모 합성 + 전역 분위기 주입
FA                         ← 전역 초기화
WA(w=64)                   ← 좁은 로컬
WA(w=32)                   ← 최소 단위 편집 판단
WA(w=64)                   ← 넓어지기 시작
WA(w=128)
WA(w=256)                  ← 최대 수용 범위
WA(w=128)
WA(w=64)
WA(w=32)                   ← 최종 편집 결정
WA(w=64)
FA                         ← 전역 마무리 정합
```

**다이아몬드 구조 의도**:
- 초반 FA로 전체 문장 구조 파악
- 좁은 WA에서 세밀한 편집 판단 (어절 단위)
- 점진적으로 넓혀 주변 어절 확인
- 다시 좁혀 최종 편집 결정
- 마지막 FA로 전체 정합성 확인

### 비용 비교 (seq=4096, d=768)

| 컴포넌트 | FLOPs/layer | 비고 |
|----------|-------------|------|
| ChunkFFT (16슬라이스 × 256) | ~125M | 사실상 무료 |
| FA (Full Attention) | 25.8G | 2~3개만 사용 |
| WA(w=256) | 1.6G | |
| WA(w=128) | 0.8G | |
| WA(w=64) | 0.4G | |
| WA(w=32) | 0.2G | |
| FFN (d_ff=2048) | 12.9G | 매 레이어 |

| 구성 | Attention 합계 | FFN 합계 | 전체 |
|------|---------------|----------|------|
| FA × 15 (표준) | 387G | 290G | **677G** |
| 다이아몬드 11L (위 구성) | 55.4G | 142G | **197G** |
| **절감** | **7배** | **2배** | **3.4배** |

FFN이 d_ff=2048 기준. d_ff를 더 줄이면 추가 절감.

### Attention 설정
- **RoPE**: 순수 sin/cos 연산, ONNX 완벽 호환, 위치 인코딩 표준
- **GQA** (Grouped Query Attention): KV head 수 줄여 메모리 절감, ONNX 지원
- **SwiGLU FFN**: d_ff = d_model × 2~2.67
- **INT8 QAT**: pretrain부터 INT8 양자화 적용 학습, 배포 시 INT8 그대로 사용

### ChunkFFT 상세

```python
# 1. conv1d 출력을 chunk_size=256으로 분할
chunks = x.reshape(B, n_chunks, chunk_size, D)  # [B, 16, 256, 768]

# 2. 각 청크에 FFT (seq축)
freq = torch.fft.rfft(chunks, dim=2)  # [B, 16, 129, 768]

# 3. learned projection → 청크당 1 mood vector
mood = self.mood_proj(freq.flatten(-2, -1))  # [B, 16, 768]

# 4. mood vectors끼리 Full Attention (16×16, 사실상 무료)
mood = self.mood_attn(mood)  # [B, 16, 768]

# 5. broadcast: 각 토큰에 해당 청크의 mood 더함
x = x + mood[:, chunk_indices, :]
```

## INT8 QAT 전략

- **전 과정 INT8 QAT**: pretrain(RTD) + fine-tune(GEC) 모두 INT8 양자화 적용 학습
- **동기**: 같은 모델 파일 크기(바이트)에서 FP32 대비 4× 많은 파라미터 수용 가능
  - 예: 128MB 용량 → FP32 32M params vs INT8 128M params
- **INT8 matmul**: weight INT8 × activation INT8 → INT32 accumulate → 정수 경로, FP 변환 불필요
- **ONNX RT 호환**: INT8 QDQ 노드로 export → 모든 EP(CUDA/CPU/DirectML)에서 INT8 추론
- **Generator는 FP16/BF16**: 작은 모델(~2M)이므로 양자화 불필요, 안정적 MLM 학습 우선

## 모델 구성

### Generator (~2M params, 표준 Transformer, FP16)
```
Embedding(303, d=128)
├── Transformer Encoder × 4 layers
│   ├── Self-Attention (2 heads, d=128)
│   └── FFN (d_ff=512)
└── MLM Head: Linear(128 → 303)
```

### Discriminator — Small (~8M params, INT8 QAT)
```
Embedding(303, d=256)
├── Conv1d(k=4) + ChunkFFT(chunk=256)
├── FA (4 heads, GQA-2, RoPE)
├── WA(64) × 1 + WA(32) × 1 + WA(64) × 1
│   (4 heads, GQA-2, RoPE, SwiGLU d_ff=512)
├── FA
└── RTD Head: Linear(256 → 2)   [pretrain]
    Tag Head: Linear(256 → n_tags) [fine-tune]
```

### Discriminator — 128M (INT8 QAT, BiMamba-2 직접 비교용)
```
Embedding(303, d=768)
├── Conv1d(k=4) + ChunkFFT(chunk=256)
├── 다이아몬드 11L:
│   FA → WA(64) → WA(32) → WA(64) → WA(128)
│   → WA(256) → WA(128) → WA(64) → WA(32) → WA(64) → FA
│   (12 heads, GQA-4, RoPE, SwiGLU d_ff=2048)
└── RTD Head: Linear(768 → 2)   [pretrain]
    Tag Head: Linear(768 → n_tags) [fine-tune]
```

## 구현 단계

### Step 1: 커스텀 인코더 구현
- Conv1d 전처리 레이어
- ChunkFFT 모듈 (FFT + learned projection + mood attention + broadcast)
- Sliding Window Attention (attention mask 기반, RoPE, GQA)
- 다이아몬드 레이어 스택 조립
- ONNX export 가능 여부 확인 (torch.fft.rfft, attention mask 등)

### Step 2: ELECTRA Pretrain
- Generator (표준 소형 Transformer) + Discriminator (커스텀 인코더)
- RTD 학습 루프 (joint training)
- 기존 corpus/sample_full.jsonl 사용
- 목표: ~100K steps (corpus 크기에 따라 조정)

### Step 3: KLUE 벤치마크 평가
- 사전학습된 discriminator의 언어 이해력 측정
- **KLUE-TC (YNAT)**: 뉴스 토픽 7-class 분류 (가장 단순, KoELECTRA-base ~87%)
  - 랜덤 14%, 70%+ 나오면 한국어 이해 확인
- KLUE-NLI: 문장쌍 → 3-class (KoELECTRA-base ~81%)
- KLUE-STS: 문장쌍 유사도 (KoELECTRA-base ~93 Pearson)
- `datasets` 라이브러리: `load_dataset("klue", "ynat")`
- keyboard 토크나이저의 한계 파악 (seq 길이 ~3x 불이익)

### Step 4: GEC Fine-tune
- RTD head → Tag Head(n_tags)로 교체 (single-head, DenseEditor와 동일)
- vocab 303이면 n_tags = KEEP + DELETE + INSERT_x × vocab ≈ 608, 충분히 작음
- 확신도 threshold 튜닝, iterative refinement
- 기존 노이즈 엔진 + 편집 태그 파이프라인 재사용

### Step 5: ONNX RT 배포
- Discriminator ONNX export
- Execution Provider 테스트: CUDA, DirectML, CPU (MLAS)
- seq=4096 추론 벤치마크

### Step 6: 비교 분석
| 모델 | pretrain | 구조 | edit_P | edit_R | F0.5 | seq=4096 추론 |
|------|----------|------|--------|--------|------|--------------|
| BiMamba-2 128M | 없음 | BiMamba2 15L | 93.1% | 71.0% | ? | CUDA 62ms |
| KoELECTRA (Phase 1) | 범용 한국어 | Transformer 12L | ? | ? | ? | ONNX ? |
| Keyboard ELECTRA Small | 자체 RTD | 커스텀 8L | ? | ? | ? | ONNX ? |
| Keyboard ELECTRA 128M | 자체 RTD | 커스텀 다이아몬드 11L | ? | ? | ? | ONNX ? |

## 핵심 질문

1. **Conv1d + ChunkFFT 전처리가 자모 토큰의 한계를 보완하는가?**
   - 자모 개별 토큰은 의미 없음 → conv1d가 음절 feature를 만들어야
   - ChunkFFT가 실제로 유용한 "분위기" 신호를 제공하는가?

2. **다이아몬드 윈도우 구조가 Full Attention 대비 품질 손실 없이 동작하는가?**
   - FA 2개 + WA 다이아몬드가 FA 11개와 비슷한 품질을 내는지
   - 윈도우 크기별 최적 조합은 실험으로 결정

3. **RTD pretrain이 GEC 성능을 실제로 올려주는가?**
   - exp-attention에서 from-scratch attention은 edit_R ~2%로 실패
   - RTD pretrain 후 같은 attention이 동작하면 → pretrain이 핵심 요인 확인

4. **ONNX RT 배포 시 seq=4096에서 실용적 속도가 나오는가?**
   - Sliding Window가 ONNX에서 실제로 sparse 최적화를 받는지
   - 안 받으면 dense attention mask로 fallback → 속도 이점 제한적

## 리스크

- keyboard 토크나이저의 긴 시퀀스(~3x) → pretrain 비용 증가
- 303 vocab으로 RTD가 의미 있는 학습 신호를 줄 수 있는지 불확실
- ChunkFFT가 실제로 학습 가능한 유용한 신호인지 (ablation 필요)
- Sliding Window Attention의 ONNX 최적화 수준이 EP마다 다름
- 커스텀 구조 → 기존 pretrained 모델 활용 불가, 처음부터 학습 필수
