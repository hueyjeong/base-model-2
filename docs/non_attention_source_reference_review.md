# Cross-Attention 없이 원문 참조력 강화안 검토 (BitNet + Mamba + Seq2Seq)

## 문서 목적

이 문서는 현재 프로젝트 구조에서 **Transformer/Cross-Attention을 추가하지 않고** 원문 참조력을 높일 수 있는 방법을 검토한다.
요청 범위에 맞춰 다음을 모두 포함한다.

- 실제 적용 가능 여부
- 어디를 어떻게 바꿔야 하는지(구현 포인트)
- 느려지는지/빨라지는지
- 모델이 무거워지는지(파라미터/메모리)
- 리스크와 우선순위

---

## 현재 구조 요약 (근거)

- 현재 디코더는 Cross-Attention 없이 encoder 출력을 prefix-concat 방식으로 사용한다.
  - `decode()`에서 `torch.cat([encoder_out, tgt_emb], dim=1)` 후 디코더를 통과시키고 target 부분만 slice한다.
  - 근거: [model/seq2seq.py](../model/seq2seq.py#L124-L161)
- `encoder_mask`는 인자만 있고 decode 내부에서 실사용하지 않는다.
  - 근거: [model/seq2seq.py](../model/seq2seq.py#L139)
- 디코더 레이어는 `MambaBlock + RMSNorm + BitNetFFN` 구성이다.
  - 근거: [model/decoder.py](../model/decoder.py#L19-L73)
- Mamba CUDA 커널은 환경 조건 충족 시 사용하고, 아니면 Python fallback scan으로 간다.
  - 근거: [model/mamba_block.py](../model/mamba_block.py#L21-L23), [model/mamba_block.py](../model/mamba_block.py#L119-L137)
- 모델 크기 실험 프리셋은 8M~128M이다.
  - 근거: [training/pretrain.py](../training/pretrain.py#L80-L113)

---

## 결론 요약

- **가능성**: 제안한 대안들은 모두 현재 구조에 적용 가능하다.
- **효율성**: Cross-Attention을 넣는 것보다 대부분 가볍고, 특히 디코딩 지연시간 증가가 작다.
- **우선순위**:
  1. Source-aware logit bias + Copy gate (가성비 최상)
  2. Layer-wise source conditioning (FiLM/AdaNorm)
  3. Edit gate + 반복 정제(Iterative refinement)
- **주의점**: 방법을 많이 동시에 넣으면 학습 안정성과 하이퍼파라미터 탐색 비용이 커진다.

---

## 방법별 상세 검토

## 1) Source-aware Logit Bias (원문 토큰 보너스)

### 아이디어
디코더 마지막 logits에 원문에 등장한 토큰 ID에 대해 보너스 bias를 더한다.

- 간단형: source 토큰 집합에 고정 +b
- 고급형: source 토큰 빈도/위치 기반 가중치

### 적용 위치
- `BitMambaSeq2Seq.decode()`에서 `logits` 계산 직후
- 근거 위치: [model/seq2seq.py](../model/seq2seq.py#L169-L177)

### 적용 가능성
- **매우 높음**. 기존 아키텍처 변경 거의 없음.

### 성능 영향
- 추론 속도: **소폭 저하 또는 거의 동일** (토큰 집합 scatter/add 연산 추가)
- 학습 속도: **소폭 저하**

### 모델 무게 영향
- 파라미터 증가: **0** (고정 보너스면)
- 활성 메모리: **미미한 증가**

### 리스크
- 과도한 bias는 수정이 필요한 토큰도 복사하려는 성향을 강화할 수 있음.

### 총평
- 가장 저비용으로 원문 참조 성향을 올리는 1순위.

---

## 2) Copy Gate (생성 분포 + 복사 분포 혼합)

### 아이디어
최종 분포를
- 생성 분포 `p_gen` (기존 LM head)
- 복사 분포 `p_copy` (source 기반)
로 나누고, 게이트 `g`로 혼합한다.

`p = g * p_gen + (1-g) * p_copy`

Cross-Attention 없이도 `p_copy`를 source BOW/빈도 기반으로 만들 수 있다.

### 적용 위치
- `BitMambaSeq2Seq.decode()`의 logits 단계
- 학습 loss 계산부는 기존 CE 그대로 사용 가능(혼합 결과 logits/prob로 치환)
- 근거 위치: [model/seq2seq.py](../model/seq2seq.py#L124-L177)

### 적용 가능성
- **높음**. 디코더 블록 자체 변경 없이 head 쪽에서 구현 가능.

### 성능 영향
- 추론 속도: **소폭 저하** (혼합/정규화 추가)
- 학습 속도: **소폭 저하**

### 모델 무게 영향
- 최소형 게이트(`Linear(d_model,1)`) 기준 증가량: `d_model+1`
  - 8M(288): 289
  - 16M(352): 353
  - 32M(448): 449
  - 64M(576): 577
  - 128M(768): 769
- 사실상 무시 가능한 수준.

### 리스크
- 복사 분포 설계가 너무 단순하면 rare token 교정에서 이득이 제한될 수 있음.

### 총평
- GEC의 높은 복사 비율과 잘 맞고, 파라미터 대비 효과가 큼.

---

## 3) Layer-wise Source Conditioning (FiLM/AdaNorm)

### 아이디어
현재는 source 정보를 prefix로 한 번 넣고 끝난다. 이를 보완하기 위해 각 디코더 레이어마다 source summary를 재주입한다.

- 예: source pooled vector → `(gamma, beta)` 생성 → `norm` 출력에 modulation
- attention 없이도 “원문 신호 희석”을 줄이는 방식

### 적용 위치
- 디코더 레이어의 norm 지점
- 근거 위치: [model/decoder.py](../model/decoder.py#L42-L46), [model/decoder.py](../model/decoder.py#L59-L66)

### 적용 가능성
- **높음~중간**. 디코더 레이어 인터페이스 변경 필요(레이어에 source condition 전달).

### 성능 영향
- 추론 속도: **소폭~중간 저하** (레이어별 projection 추가)
- 학습 속도: **소폭~중간 저하**

### 모델 무게 영향 (대략)
- 레이어별 `2 * d_model * d_model` (gamma/beta projection) 가정 시 총 증가:
  - 8M: 829,440
  - 16M: 1,734,656
  - 32M: 3,612,672
  - 64M: 6,635,520
  - 128M: 14,155,776
- 즉, **효과는 크지만 다른 옵션보다 확실히 무거움**.

### 리스크
- 과주입 시 학습이 불안정하거나 over-conditioning이 생길 수 있음.

### 총평
- 참조력 개선 잠재력은 크지만, “경량” 관점에서는 2차 도입 권장.

---

## 4) Edit Gate (수정량 제어: KEEP vs EDIT)

### 아이디어
각 타깃 위치에서 수정 강도를 조절하는 게이트를 둔다.

- 게이트가 낮으면 원문 보존(복사 편향)
- 게이트가 높으면 적극 수정

### 적용 위치
- `decode()` 출력 직전 hidden/logit 스케일링
- 노이징 강도와 함께 학습 전략 조정 필요

### 적용 가능성
- **높음**. 헤드 근처 변경으로 가능.

### 성능 영향
- 추론/학습 속도: **거의 동일~소폭 저하**

### 모델 무게 영향
- 최소형: `Linear(d_model, 1)` 수준 (Copy gate와 동일급, 매우 작음)

### 리스크
- 게이트가 보수적으로 수렴하면 교정 recall이 떨어질 수 있음.

### 총평
- 과교정 억제에 유리, 경량 유지에 적합.

---

## 5) Segment/Boundary 강화 (encoder 구간 vs target 구간 신호 강화)

### 아이디어
prefix-concat 구조에서 구간 경계를 더 명확히 인코딩한다.

- segment embedding (source=0, target=1)
- boundary token/positional reset 등

### 적용 위치
- concat 직전 임베딩 합산 단계
- 근거 위치: [model/seq2seq.py](../model/seq2seq.py#L155)

### 적용 가능성
- **매우 높음**.

### 성능 영향
- 추론/학습 속도: **거의 동일**

### 모델 무게 영향
- segment embedding 2개면 `2*d_model`
  - 8M: 576, 128M: 1,536
- 사실상 무시 가능.

### 리스크
- 단독 효과는 제한적일 수 있음(보통 다른 방법과 조합 시 유의미).

### 총평
- 싸고 안전한 보조 수단. 1번/2번과 함께 넣기 좋음.

---

## 6) Iterative Refinement (반복 정제)

### 아이디어
한 번에 전체를 생성하지 않고, 짧은 패스를 2~3번 돌며 고친다.

- 패스별 변경량 제한
- 불확실 구간 우선 수정

### 적용 위치
- 모델 내부보다 **추론 루프/서빙 로직**

### 적용 가능성
- **높음**. 모델 재설계 없이 inference policy로 적용 가능.

### 성능 영향
- 지연시간: **패스 수에 비례해 증가**
  - 2패스면 대략 1.7~1.9x
  - 3패스면 대략 2.4~2.8x
- 학습 속도: 영향 없음(추론 전용이면)

### 모델 무게 영향
- 파라미터 증가: 0

### 리스크
- 실시간 서비스 SLA가 빡빡하면 부적합할 수 있음.

### 총평
- 품질 이득은 크지만 latency 예산이 있을 때만 채택.

---

## 7) Monotonic/Copy 보조 Loss

### 아이디어
주 손실(CE) 외에 “입력과 크게 벗어나지 않도록” 보조 손실을 추가한다.

- soft monotonicity
- edit distance surrogate
- copy-consistency regularizer

### 적용 위치
- 학습 루프 손실 결합 부분
- 근거 위치: [training/pretrain.py](../training/pretrain.py#L411-L456)

### 적용 가능성
- **중간**. loss 구현/튜닝 비용 필요.

### 성능 영향
- 학습 속도: **중간 저하** (보조 항 계산)
- 추론 속도: 영향 없음

### 모델 무게 영향
- 보통 파라미터 증가 0 (손실 항만 추가)

### 리스크
- 보조항 가중치 튜닝 실패 시 주 손실 최적화를 방해.

### 총평
- 실험 여력이 있을 때 장기적으로 가치가 큼.

---

## 방법별 비교 표

| 방법 | 적용 가능성 | 추론 속도 영향 | 학습 속도 영향 | 파라미터 영향 | 구현 난이도 | 권장도 |
|---|---|---:|---:|---:|---:|---:|
| Source-aware logit bias | 매우 높음 | 매우 작음 | 매우 작음 | 0 | 하 | 매우 높음 |
| Copy gate | 높음 | 작음 | 작음 | 매우 작음 | 하~중 | 매우 높음 |
| Layer-wise source conditioning | 높음~중간 | 작음~중간 | 작음~중간 | 중간~큼 | 중 | 높음 |
| Edit gate | 높음 | 매우 작음 | 매우 작음 | 매우 작음 | 하 | 높음 |
| Segment/boundary 강화 | 매우 높음 | 거의 없음 | 거의 없음 | 거의 없음 | 하 | 중~높음 |
| Iterative refinement | 높음 | 큼(패스 수 비례) | 없음 | 0 | 하(서빙) | 조건부 |
| Monotonic/copy 보조 loss | 중간 | 없음 | 중간 | 0 | 중~상 | 중 |

---

## 경량 GEC 기준 추천 로드맵 (Transformer 배제 전제)

### Phase 1 (최소 변경, 고가성비)

1. Source-aware logit bias
2. Copy gate
3. Segment/boundary 강화

예상 효과:
- 원문 보존성 개선
- 과교정/환각 감소
- 속도 저하 최소

### Phase 2 (중간 변경)

4. Edit gate
5. Layer-wise source conditioning (필요 시 일부 레이어만)

예상 효과:
- 수정량 제어 + 참조력 추가 강화
- 다만 파라미터/튜닝 비용 증가

### Phase 3 (고급 실험)

6. Monotonic/copy 보조 loss
7. Iterative refinement(서비스 latency 허용 시)

---

## 현재 프로젝트에 대한 현실적 판단

- 현재는 prefix-concat 구조가 이미 동작하므로 “원문 참조가 완전히 없는” 상태는 아니다.
- 다만 GEC의 핵심인 복사-교정 균형에서, cross-attention 없이도 보완이 필요하다.
- 경량 목표를 유지하려면 **헤드 중심 보강(1,2,4) + 경계 신호(5)**가 가장 안전하다.
- 레이어 전체 조건주입(3)은 효과 잠재력이 크지만, 64M 이상에서 파라미터/튜닝 부담이 눈에 띄게 증가한다.

---

## 부록: 파라미터 증가량 산출 기준

아래는 대략 비교를 위한 근사치다.

- Copy gate / Edit gate: `Linear(d_model,1)` → `d_model+1`
- Segment embedding: `2*d_model`
- Layer-wise FiLM: decoder layer마다 `2*d_model*d_model`

사용한 d_model/decoder_layer 값은 `MODEL_CONFIGS` 기준.
근거: [training/pretrain.py](../training/pretrain.py#L80-L113)
