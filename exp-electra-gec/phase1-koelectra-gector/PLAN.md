# Phase 1: KoELECTRA-Small-v3 + GECToR (Two-head)

## 목표

pretrained Korean encoder + Two-head 태그 체계로 GEC fine-tune.
BiMamba-2와 비교할 수 있는 기준점 확보 + 확신도 파이프라인 구축.

## 모델 구성

```
KoELECTRA-Small-v3 (discriminator)
├── Embedding: WordPiece (vocab=35000, d=256)
├── ELECTRA Encoder × 12 layers
│   ├── Self-Attention (4 heads, d=256)
│   └── FFN (d_ff=1024)
├── Action Head: Linear(256 → 4)      ← KEEP/DELETE/INSERT/REPLACE
├── Content Head: Linear(256 → 35000)  ← 어떤 토큰? (INSERT/REPLACE 시)
└── 출력: (action, content, confidence) per token
```

- 파라미터: ~14M (encoder) + action head + content head
- 비교 대상: BiMamba-2 128M (d=640, 15L)

## Two-head 태그 체계

### Action Head (4-class)
| 태그 | 의미 |
|------|------|
| KEEP | 토큰 유지 |
| DELETE | 토큰 삭제 |
| INSERT | 현재 토큰 뒤에 새 토큰 삽입 |
| REPLACE | 현재 토큰을 다른 토큰으로 교체 |

### Content Head (vocab-class)
- INSERT/REPLACE로 판별된 위치에서만 사용
- 전체 vocab에서 대상 토큰 예측
- KEEP/DELETE 위치의 content 출력은 무시 (loss 계산에서 제외)

### 확신도 (Confidence)
```python
action_prob = softmax(action_logits)  # (B, T, 4)
content_prob = softmax(content_logits)  # (B, T, V)

# 편집 확신도
confidence = action_prob.max(dim=-1) * content_prob.max(dim=-1)
# KEEP/DELETE는 action_prob만 사용 (content 무관)

# Keep bias (inference 시 적용, 학습 시 미적용)
action_logits[:, :, KEEP] += keep_bias  # dev set에서 튜닝
```

### Loss 함수
```python
# Action loss: 모든 위치
action_loss = CE(action_logits, action_targets)

# Content loss: INSERT/REPLACE 위치만
edit_mask = (action_targets == INSERT) | (action_targets == REPLACE)
content_loss = CE(content_logits[edit_mask], content_targets[edit_mask])

# 총 loss
loss = action_loss + α * content_loss  # α는 하이퍼파라미터
```

## 핵심 과제

### 1. WordPiece 기준 편집 태그 생성

```
원본:   "나는 한국어를 잘한다"
오류:   "나는 한국어를 잘하다"
         ↓ WordPiece 토크나이징
원본 토큰: [나는, 한국어를, 잘, ##한다]
오류 토큰: [나는, 한국어를, 잘, ##하다]
         ↓ Levenshtein 정렬
action:  [KEEP, KEEP, KEEP, REPLACE]
content: [  -,    -,   -,  ##한다 ]
```

### 2. 평가 파이프라인

```
모델 출력: action + content per token
  → 확신도 계산
  → threshold 이상인 편집만 적용
  → 교정 텍스트 복원 (WordPiece detokenize)
  → 원본 텍스트와 비교 → edit_P, edit_R, F0.5
```

### 3. Iterative Refinement

```
Pass 1: 오류 입력 → (action, content) 예측 → 확신도 높은 편집 적용
Pass 2: Pass 1 결과 → 다시 예측 → 추가 편집 적용
...
Pass N: 모든 위치 conf < threshold → 종료
```

## 구현 단계

### Step 1: 환경 준비
- `transformers` 설치 확인
- KoELECTRA-Small-v3 로드 테스트
- WordPiece 토크나이저 동작 확인

### Step 2: Two-head 데이터 파이프라인
- WordPiece 기준 Levenshtein → (action, content) 태그 쌍 생성
- 노이즈 엔진 재사용 (텍스트 레벨)
- DataLoader 구현

### Step 3: 모델 구현
- KoELECTRA encoder + Action Head + Content Head wrapper
- 확신도 계산 로직
- Keep bias 적용 (inference)
- 학습 루프 (action_loss + content_loss)

### Step 4: 학습 및 평가
- Fine-tune (예상: 5-10 epoch)
- threshold / keep_bias 튜닝 (dev set)
- iterative refinement 횟수 실험
- BiMamba-2와 비교

### Step 5: Consensus 통합
- Gumbel sampling 기반 multi-pass
- 확신도 가중 투표
- consensus + iterative refinement 조합 실험

## 예상 결과

GECToR 논문 (영어, BEA-2019):
- BERT-base: F0.5 = 56.8 → 65.3 (3-stage)
- RoBERTa: F0.5 = 59.5 → 66.5 (3-stage)

한국어에서는 데이터/태그 체계 차이로 직접 비교 불가하지만,
pretrained encoder의 효과를 Two-head 체계에서 확인하는 것이 목표.
