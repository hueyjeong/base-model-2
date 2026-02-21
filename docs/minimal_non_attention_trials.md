# 최소 변경 2안 실험 체크리스트 (No-Attention 유지)

## 목적

Cross-Attention/Transformer를 도입하지 않고, 현재 BitMamba Seq2Seq 구조에서
원문 참조력 개선 효과를 **낮은 구현 비용**으로 빠르게 검증한다.

- 기준 구조: [model/seq2seq.py](../model/seq2seq.py)
- 관련 학습 진입점: [training/pretrain.py](../training/pretrain.py)
- 상세 기술 검토: [docs/non_attention_source_reference_review.md](non_attention_source_reference_review.md)

---

## Trial A — Source-Aware Logit Bias

### 핵심 아이디어
디코딩 시 source에 등장한 token id에 작은 logit 보너스를 부여하여
원문 보존/복사 성향을 강화한다.

### 구현 범위 (최소)
- `decode()`의 최종 logits 직후 bias 추가
- 위치: [model/seq2seq.py](../model/seq2seq.py#L169-L177)

### 예상 비용
- 파라미터 증가: 0
- 추론 지연: 거의 없음 ~ 소폭 증가
- 학습 처리량: 소폭 감소 가능

### 체크리스트
- [ ] source token set 생성 방식 고정 (중복 제거 여부 포함)
- [ ] bias 강도 3점만 탐색 (예: 0.2 / 0.5 / 0.8)
- [ ] 길이 변화율(출력길이/입력길이) 로그 추가
- [ ] 과교정 비율(변경 토큰 비율) 로그 추가
- [ ] `val_bpc` + 샘플 정성평가 100문장 비교

### 합격 기준 (권장)
- `val_bpc` 악화 없이
- 과교정/환각 감소
- 추론 속도 저하가 5% 이내

### 중단 기준
- bias 증가에 따라 recall이 급격히 하락하거나
- 속도 저하가 10% 이상이면 Trial A는 보류

---

## Trial B — Copy Gate (Linear d_model→1)

### 핵심 아이디어
생성 분포와 source 기반 복사 분포를 게이트로 혼합:
`p = g * p_gen + (1-g) * p_copy`

### 구현 범위 (최소)
- `decode()` hidden에서 gate 산출 후 최종 분포 혼합
- 위치: [model/seq2seq.py](../model/seq2seq.py#L164-L177)

### 예상 비용
- 파라미터 증가: `d_model + 1`
  - 8M: 289
  - 16M: 353
  - 32M: 449
  - 64M: 577
  - 128M: 769
- 추론 지연: 소폭 증가
- 학습 처리량: 소폭 감소

### 체크리스트
- [ ] `p_copy` 정의 고정 (source unigram 분포부터 시작)
- [ ] gate 초기값 보수적(생성 쏠림 방지)
- [ ] 혼합 후 수치 안정성(log-space or epsilon) 점검
- [ ] `val_bpc` + 변경율 + 예문 100개 비교
- [ ] Trial A 대비 추가 이득 확인

### 합격 기준 (권장)
- Trial A보다 정성 품질(복사 정확도/자연스러움) 우위
- 처리량 저하가 8% 이내

### 중단 기준
- 학습 초반 불안정(손실 진동) 지속
- 속도 저하가 12% 이상이면 보류

---

## 공통 실행 프로토콜

### 1) 비교 축 고정
- 토크나이저 고정
- 모델 크기 고정 (권장: 16M 또는 32M 1개만 우선)
- 데이터/seed/스텝 동일

### 2) 1차 스모크
- 500~1,000 step
- NaN/Inf, 속도, loss 하강 여부만 확인

### 3) 2차 단기 비교
- 5k~10k step
- `val_bpc` + 정성 100문장

### 4) 최종 선택
- 품질 이득이 명확하고 속도 손해가 허용 범위면 채택

---

## 권장 순서

1. Trial A 먼저 (가장 싸고 안전)
2. Trial B 추가 (A 대비 순증 확인)

둘 다 통과하면 그 다음 단계로 Edit Gate를 검토한다.
