# ELECTRA GEC 문자 레벨 평가 결과 (2026-03-24)

## 평가 조건

- **코퍼스**: `corpus/val_ko_50k_shuffled.jsonl` 500문장 (len >= 10)
- **노이즈**: `DenoisingNoiser` seed=42, realistic preset, error_prob=0.5, error_count=3
- **메트릭**: 문자 레벨 diff (`char_edits`) 기반 P/R/F0.5
  - 노이즈 텍스트 → 모델 교정 → 원본과 비교
  - TP = 정답 편집 ∩ 예측 편집, FP = 예측에만 있는 편집, FN = 정답에만 있는 편집
- **동일 조건**: 모든 모델에 같은 500문장, 같은 noiser 사용

## 1. KoELECTRA Two-head 체크포인트별 추이

모델: `KoELECTRAGECToR` (koelectra-base-v3, 110M) + Action(4) + Content(35K tied)

| step | stage | kb=0.0 P/R/F0.5 | kb=2.0 P/R/F0.5 |
|------|-------|------------------|------------------|
| 375k | full_finetune | 0.641 / 0.822 / 0.671 | — |
| 425k | full_finetune | 0.649 / 0.829 / 0.678 | 0.676 / 0.786 / **0.696** |
| 450k | full_finetune | 0.648 / 0.826 / 0.677 | 0.635 / 0.766 / 0.658 |
| **475k** | full_finetune | 0.654 / 0.818 / **0.681** | 0.682 / 0.795 / **0.702** |
| 500k | full_finetune | 0.648 / 0.829 / 0.677 | 0.632 / 0.764 / 0.655 |
| 525k | full_finetune | 0.649 / 0.828 / 0.678 | 0.633 / 0.766 / 0.656 |

**결론**: 475k가 피크 (F0.5=0.702, kb=2.0). 이후 포화/미세 하락. 150k step 학습 추가로 +3pp 수준.

## 2. DenseEditor vs ELECTRA 비교

| 모델 | params | vocab | head | P | R | F0.5 | sent/s |
|------|--------|-------|------|---|---|------|--------|
| **DenseEditor Mamba2 90k** | 128M | 303 | single (608 tags) | **0.852** | 0.762 | **0.833** | 17.6 |
| ELECTRA 475k (best) | 110M | 35K | two-head tied | 0.682 | **0.795** | 0.702 | ~98 |
| ELECTRA 475k (기본) | 110M | 35K | two-head tied | 0.654 | 0.818 | 0.681 | ~98 |

DenseEditor가 F0.5에서 **+13pp**, Precision에서 **+17pp** 우위.

## 3. Iterative Refinement / Threshold 튜닝 (ELECTRA)

### conf_threshold 실험 (step_375000)

| pass | kb | ct | P | R | F0.5 |
|------|----|----|---|---|------|
| 1 | 0.0 | 0.0 | 0.641 | 0.822 | **0.671** |
| 5 | 1.0 | 0.7 | 0.653 | 0.730 | 0.667 |
| 5 | 0.0 | 0.8 | 0.571 | 0.655 | 0.586 |
| 5 | 0.0 | 0.9 | 0.261 | 0.116 | 0.209 |
| 20 | 0.0 | 0.9 | 0.261 | 0.116 | 0.209 |
| 20 | 2.0 | 0.9 | 0.000 | 0.000 | 0.000 |

**결론**: 높은 threshold + 다수 pass는 역효과. 1-pass가 최적.
- 매 pass마다 정상 텍스트를 오교정 → FP 누적
- ct=0.9에서 P도 하락 → DELETE가 threshold 미적용이었던 버그 발견 후 수정해도 개선 안 됨

### 그리드 서치 (step_375000, 200샘플 튜닝 → 500 전체 평가)

튜닝 최적: n_passes=3, kb=1.0, ct=0.0 → 튜닝셋 F0.5=0.734
전체 평가: P=0.640, R=0.792, F0.5=0.665 — baseline(0.671)보다 오히려 낮음.

## 4. 성능 격차 분석

### ELECTRA P가 낮은 원인: content accuracy 한계

학습 시 토큰 레벨 메트릭:
```
act_acc=0.974  cont_acc=0.868  P=0.895  R=0.939
```

- action head (97.4%)는 우수하나, content head (86.8%)가 병목
- content 오류 1건 = 문자 레벨에서 FP + FN 동시 발생 (WordPiece 토큰 → 여러 글자)
- iterative pass에서 오교정 텍스트가 다음 입력 → 눈덩이 효과

### 구조적 원인

1. **Vocab 크기**: 303 vs 35,000 — content 예측 난이도 100배 차이
2. **Head 분리**: DenseEditor는 "여기를 ㅎ으로 바꿔"가 하나의 결정, ELECTRA는 "바꿔" + "ㅎ으로" 독립 결정 → 오류 곱 누적
3. **Tied embedding**: KoELECTRA 임베딩은 RTD(판별)용으로 학습됨, 교정 토큰 생성과 목적 불일치

## 5. 버그 수정

- `model.py:137`: `is_edit`에 `ACTION_DELETE` 누락 → DELETE가 conf_threshold 필터 미적용
- 수정: `is_edit = (actions == ACTION_DELETE) | (actions == ACTION_REPLACE) | (actions == ACTION_INSERT)`

## 6. 향후 실험 방향

| 실험 | 목적 |
|------|------|
| ELECTRA 1-head (70K tags) | head 분리 vs 통합 효과 검증 |
| ELECTRA 2-head untied | content head 독립 학습 효과 |
| DenseEditor 2-head | 303 vocab에서 head 분리 영향 |
| DenseEditor + attention | Mamba 대체 → 속도 개선 (17→98+ sent/s) |

핵심 질문: **vocab 크기 vs head 구조** 중 어느 것이 더 결정적인가?
