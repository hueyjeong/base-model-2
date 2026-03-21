# ELECTRA 기반 GEC 편집 태깅 실험

## 배경

Attention from-scratch 실험 결과 (exp-attention):
- Full Attention + GQA + SwiGLU + INT8 QAT → **128M/10k step에서 edit_R ~2.4%**
- Hybrid (Conv1d + Window Attention + FFT) → 마찬가지로 편집 불능
- **결론**: pretrained encoder 없이 attention 기반 GEC 편집 태깅은 불가

GECToR 논문에서도 확인:
- BERT-base pretrained → F0.5 56.8
- LSTM random init → F0.5 35.0 (pretrained 없이도 SSM 계열은 어느 정도 작동)
- **핵심**: transformer가 GEC를 하려면 사전학습된 언어 이해가 필수

현재 BiMamba-2 모델:
- val_loss 0.107, P 93.1%, R 71.0% (10k DDP step)
- sequential inductive bias 덕에 from-scratch로도 작동
- 단, CPU 추론에서 scan 순차 의존성, 양방향 2x 비용 등 한계

## 핵심 설계 결정

### 1. Two-head 태그 체계 (Action + Content 분리)

기존 DenseEditor: 통합 태그 (KEEP/DELETE/INSERT_ㅎ/INSERT_ㅏ/... → 608개)
→ 토크나이저 vocab에 비례하여 태그 폭증, WordPiece(35K)에서 사용 불가

**새 접근: 동작/내용 분리**
```
Encoder hidden states
  ├── Action Head → KEEP / DELETE / INSERT / REPLACE  (4-class)
  └── Content Head → vocab 중 어떤 토큰?  (INSERT/REPLACE 위치만 사용)
```
- 토크나이저 무관: action head는 항상 4-class
- 다중 토큰 편집: iterative refinement로 해결 (1 pass = 1 토큰)

### 2. 확신도 기반 편집 결정 (GECToR 스타일)

GECToR의 두 가지 confidence 메커니즘을 도입:

**Keep probability inflation**
```
action_logits에 KEEP bias 추가 → "확실할 때만 편집"
bias 값을 dev set에서 튜닝 → Precision/Recall 트레이드오프 조절
```

**편집 확신도 = action_prob × content_prob**
```
REPLACE(0.92) × "잘한다"(0.85) = 0.782  → 적용
INSERT(0.51) × "를"(0.40)     = 0.204  → 스킵 (threshold 미달)
```

### 3. 확신도 가중 Gumbel Consensus

기존 consensus (단순 다수결) → 확신도 가중 투표:
```
Pass 1: REPLACE "잘한다" (conf=0.78)
Pass 2: REPLACE "잘한다" (conf=0.82)
Pass 3: KEEP             (conf=0.55)  ← 확신 낮은 KEEP
→ 가중합: REPLACE 1.60 vs KEEP 0.55 → REPLACE 채택
```

### 4. Iterative Refinement + 확신도 조기 종료

```
Pass 1: 5개 편집 적용 (conf > threshold)
Pass 2: 2개 추가 편집
Pass 3: 모든 위치 conf < threshold → 조기 종료
```
encoder-only 1 pass ≈ <1ms (GPU) / ~5ms (CPU) → 3-5 pass도 실용적.

## 실험 목적

1. pretrained Korean encoder + GEC 편집 태깅의 **실제 성능 상한선** 확인
2. Two-head 태그 체계 + 확신도 파이프라인 검증
3. 자체 토크나이저(keyboard, 303 vocab)로 ELECTRA RTD pretrain → GEC 파이프라인 검증
4. BiMamba-2 from-scratch vs pretrained attention의 실질적 비교

## 실험 구조

### Phase 0: 노이즈 엔진 개선 (`phase0-noise-improvement/`)

실제 오류 분포 기반으로 노이즈 엔진을 체계화. Phase 1/2에 선행.

- G2PK 기반 발음→철자 노이즈 추가 ("밥을"→"바블")
- 이형문자(Heterograph) 노이즈 추가 (유사 발음 음절 치환)
- 가중치 리밸런싱: 실제 오류 분포 반영 (`realistic` 프리셋)
- KAGAS 11-type 매핑 테이블 (표준 평가 체계)

### Phase 1: KoELECTRA-Small-v3 + GECToR (`phase1-koelectra-gector/`)

기존 pretrained 모델을 가져다가 GEC fine-tune.
빠르게 상한선을 측정하고 평가 파이프라인을 구축한다.

- 모델: `monologg/koelectra-small-v3-discriminator` (~14M) + Two-head
- 토크나이저: WordPiece (vocab ~35K)
- 태스크: Action(4-class) + Content(vocab-class) 예측
- 확신도 threshold 튜닝 → P/R 트레이드오프 최적화
- 학습: 기존 노이즈 엔진으로 오류 생성 → Levenshtein 편집 태그
- 평가: edit_P, edit_R, F0.5

### Phase 2: Keyboard 토크나이저 + ELECTRA RTD Pretrain (`phase2-keyboard-electra/`)

자체 토크나이저로 ELECTRA 방식 사전학습 후 GEC fine-tune.

- Generator: 작은 모델 (본 모델의 1/4~1/3)
- Discriminator: 본 모델 (attention encoder) + Two-head
- RTD: generator가 대체한 토큰을 discriminator가 real/fake 판별
- keyboard 토크나이저(303 vocab)의 RTD = 자모 단위 "오류 감지" ≈ GEC 노이즈와 유사
- pretrain 후 KLUE 벤치마크로 언어 이해력 측정
- GEC fine-tune 후 Phase 1 / BiMamba-2와 비교

## 성공 기준

| 모델 | edit_P | edit_R | F0.5 | 비고 |
|------|--------|--------|------|------|
| BiMamba-2 (현행) | 93.1% | 71.0% | - | 10k DDP, from-scratch, 통합태그 |
| Phase 1 KoELECTRA | ? | ? | ? | pretrained, Two-head |
| Phase 2 Keyboard ELECTRA | ? | ? | ? | self-pretrain, Two-head |

## 의존성

- `transformers` (KoELECTRA 로드)
- 기존 코드 재사용: 노이즈 엔진, Levenshtein 태깅, 평가 메트릭
