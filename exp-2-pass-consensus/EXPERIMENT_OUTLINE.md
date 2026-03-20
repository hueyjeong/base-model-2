# BitNet Mamba-2 Encoder-Only GECToR: 합의 기반 2단계 반복 교정 실험 계획

## 1) 배경과 현재 상황

우리는 **CPU 추론 지향의 128M BitNet Mamba-2 Encoder-Only GECToR 모델**을 다루고 있다.

현재 학습은 계속 진행 중이지만, **현재까지는 50k 체크포인트가 가장 좋은 지표를 보이는 기준 체크포인트**다.  
따라서 아래 모든 실험은 **50k 체크포인트를 기준 모델**로 사용한다.

중요한 전제:
- 이 모델의 추론/eval은 완전 결정론적이지 않다.
- 같은 입력 문장 `x`에 대해 `gec(x)`를 여러 번 실행하면 결과가 달라질 수 있다.
- 이 stochasticity를 이용하면, 여러 번의 추론 결과를 합의(consensus)나 투표(voting) 방식으로 결합해 품질을 조정할 수 있다.
- 특히 이 태스크는 iterative editing 성격이 있으므로, false positive(잘못된 수정)를 줄이는 방향이 중요할 수 있다.

또한 시스템 측면에서:
- 이 모델은 CPU 추론을 목표로 한다.
- 모델 크기가 CPU cache보다 클 가능성이 높아 같은 문장을 여러 번 병렬 처리한다고 해서 무조건 효율이 좋다고 볼 수는 없다.
- 다만 Mamba-2 / scan 계열의 순차적 특성 때문에 CPU의 계산 자원이 완전히 포화되지 않는 구간이 있을 수 있으며, 이 경우 **같은 입력의 여러 stochastic sample을 batch로 처리했을 때 wall-clock cost가 pass 수만큼 선형 증가하지 않을 가능성**이 있다.

---

## 2) 이번 실험에서 검증하려는 핵심 아이디어

이번 실험의 핵심 아이디어는 **단순 multi-sample ensemble이 아니라, 합의 기반 2단계 반복 교정**이다.

원문 입력을 `x`, 교정 함수를 `gec(·)`라고 하자.

### 1단계
같은 원문 `x`에 대해 stochastic하게 두 번 추론한다.

- `a = gec(x)`
- `b = gec(x)`

그리고 두 결과가 **공통으로 동의한 수정만** 채택하여 1단계 합의 결과 `y`를 만든다.

- `y = consensus(a, b)`
- 직관적으로는 `y ≈ a ∩ b`

여기서 `y`는 “원문 `x`에 대해 두 번의 stochastic 추론이 모두 동의한, 비교적 확실한 수정만 반영한 결과”이다.

### 2단계
이제 `y`를 다시 교정기의 입력으로 넣는다.

- `a' = gec(y)`
- `b' = gec(y)`

그리고 다시 두 결과의 합의를 구한다.

- `z = consensus(a', b')`
- 직관적으로는 `z ≈ a' ∩ b'`

### 최종 결과
최종 출력은 1단계 합의 결과 `y`와, 그 위에서 2단계가 찾아낸 추가 수정 `z`를 결합한 결과다.

직관적 표기는:

- `final ≈ (a ∩ b) ∪ (a' ∩ b')`

하지만 실제 구현에서는 이것을 단순 집합 연산으로 보기보다, 다음처럼 해석하는 것이 더 정확하다.

- `y`는 1단계 교정이 반영된 **중간 텍스트**
- `z`는 `y`를 입력으로 한 2단계에서 새로 얻은 **추가 수정 결과**
- 따라서 최종 출력은 실질적으로  
  **“1단계 합의 교정 결과 `y`를 만든 뒤, 그 위에 2단계 합의 교정을 한 번 더 적용한 결과”** 로 이해하는 편이 맞다

즉 수학 표기는 `(a ∩ b) ∪ (a' ∩ b')`에 가깝지만, 구현/평가 시에는 **텍스트 수준의 2단계 적용 결과**로 보는 것이 자연스럽다.

---

## 3) 이 아이디어가 왜 의미가 있는가

이 구조의 핵심 논리는 다음과 같다.

### 1) 1단계 합의는 precision을 방어한다
같은 원문 `x`에 대해 두 번 추론한 뒤 공통으로 동의한 수정만 반영하면:
- 애매한 수정이나 우연한 false positive는 탈락할 가능성이 높다
- 따라서 **precision이 높아질 가능성**이 있다
- 반면 일부 true positive는 한 번만 잡히고 한 번은 놓칠 수 있으므로 **recall은 떨어질 수 있다**

즉 1단계 합의는 “보수적이고 신뢰도 높은 수정만 먼저 반영하는 필터” 역할을 한다.

### 2) 2단계는 1단계에서 놓친 recall을 회복할 수 있다
1단계 결과 `y`는 원문 `x`보다 더 정리된 상태일 수 있다.  
예를 들어:
- 한두 군데 확실한 오류가 먼저 수정되면
- 그에 따라 문맥이나 구조가 더 선명해져서
- 처음에는 애매해 보였던 다른 오류가 2단계에서는 더 쉽게 드러날 수 있다

즉 2단계 `gec(y)`는:
- 1단계에서 놓친 수정 중 일부를
- 더 좋은 입력 상태에서 다시 발견할 가능성이 있다

### 3) iterative contamination을 줄이려는 설계다
일반적인 iterative editing의 문제는:
- 1단계에서 잘못 고친 내용이
- 2단계 입력을 오염시키는 것

하지만 여기서는 1단계 출력으로 **아무 결과나 쓰지 않고, 합의된 수정만 반영한 `y`** 를 사용한다.  
즉:
- 단순 2-pass보다
- **더 보수적으로 정제된 중간 결과를 다음 단계 입력으로 사용**한다는 장점이 있다

이 때문에 이 구조는 단순 반복 교정보다 더 안정적일 가능성이 있다.

---

## 4) 실험의 핵심 목표

이번 실험의 목표는 두 가지다.

### 1차 목표: 품질 가설 검증
다음 질문에 답하는 것:

> 합의 기반 2단계 반복 교정이 single-pass나 단순 2-pass보다 더 좋은 precision-recall tradeoff를 만드는가?

특히 우리가 중요하게 보는 것은:
- false positive를 줄이면서
- recall을 지나치게 희생하지 않고
- 최종적으로 **F0.5** 기준 더 좋은 결과를 얻는지

### 2차 목표: CPU 실용성 검증
다음 질문에 답하는 것:

> 이 구조가 CPU에서 wall-clock cost 측면에서도 실용적인가?

즉:
- 2-sample consensus
- 2-stage consensus iterative inference

같은 방식이 실제 CPU 환경에서 감당 가능한 비용인지,  
그리고 batch 처리 시 cost가 pass 수에 비례해 단순 선형 증가하는지 확인한다.

---

## 5) 최적화 우선순위

이 태스크는 **precision 민감한 태스크**로 간주한다.

이유:
- iterative correction에서는 잘못된 수정이 후속 처리와 사용자 신뢰를 해칠 수 있다
- 반면 놓친 수정은 추가 pass로 일부 회복 가능할 수 있다
- 따라서 false negative보다 false positive를 더 비싸게 보는 관점이 합리적이다

따라서 지표 우선순위는 다음과 같다.

### 주 지표
- **F0.5**

### 보조 지표
- Precision
- Recall
- F1

Recall은 완전히 무시하지 않고, 최소 수준을 유지하는지 함께 본다.

---

## 6) 검증할 가설

### H1. 1단계 합의(consensus-2)는 precision을 높일 것이다
같은 원문 `x`에 대해 두 번 추론 후 합의만 반영하면:
- Precision은 single-pass보다 높아질 가능성이 크다
- Recall은 single-pass보다 낮아질 가능성이 크다

### H2. 2단계 반복 교정은 1단계 합의의 recall 손실을 일부 회복할 수 있다
`y = consensus(gec(x), gec(x))`를 다시 입력으로 넣으면:
- 1단계에서는 잡히지 않았던 일부 수정이
- 2단계에서 더 잘 드러날 수 있다

### H3. 합의 기반 2단계 반복 교정은 single-pass보다 더 좋은 precision 중심 tradeoff를 만들 수 있다
즉 다음 구조:

- `a = gec(x)`
- `b = gec(x)`
- `y = consensus(a, b)`
- `a' = gec(y)`
- `b' = gec(y)`
- `z = consensus(a', b')`
- `final = apply(y, z)` 또는 이에 상응하는 최종 결과

가:
- single-pass보다 precision을 유지 또는 향상시키고
- consensus-2보다 recall을 일부 회복하여
- 최종적으로 **F0.5에서 우수한 결과**를 낼 수 있다는 가설

### H4. CPU에서는 비용이 pass 수만큼 선형 증가하지 않을 수 있다
멀티 샘플 + batch 처리 시:
- 2회 stochastic inference가 정확히 2배 cost가 아닐 수 있고
- 2-stage 구조도 naive한 4배 cost보다 나을 수 있다

이는 반드시 측정으로 확인해야 한다.

---

## 7) 실험할 variation

모든 실험은 **50k 체크포인트**로 수행한다.

아래 variation은 단순히 “최종 성능 누가 더 높나”만 보려는 것이 아니라,  
**어느 메커니즘에서 이득이 발생하는지 분해해서 이해하기 위한 통제 실험**이다.

### V1. Single-pass baseline
- `out = gec(x)`
- 가장 기본 기준선

목적:
- 현재 모델의 기본 precision / recall / F0.5를 확인

---

### V2. Iterative 2-pass baseline (합의 없음)
- `y = gec(x)`
- `final = gec(y)`

목적:
- 단순 반복 교정 자체가 어떤 효과를 내는지 확인
- 이후 합의 기반 구조와 비교할 때 “반복”의 기여와 “합의”의 기여를 분리하기 위한 기준선

---

### V3. Consensus-2 on x (1단계 합의만)
- `a = gec(x)`
- `b = gec(x)`
- `final = consensus(a, b)`

즉:
- `final ≈ a ∩ b`

목적:
- 합의만 적용했을 때 precision이 얼마나 오르고 recall이 얼마나 줄어드는지 확인

---

### V4. 2-stage consensus iterative (핵심 실험)
- `a = gec(x)`
- `b = gec(x)`
- `y = consensus(a, b)`
- `a' = gec(y)`
- `b' = gec(y)`
- `final = combine(y, consensus(a', b'))`

여기서 `combine`은 구현에 따라:
- `y`를 텍스트로 두고, 2단계 합의 결과를 추가 적용하는 방식일 수 있고
- edit-level merge일 수도 있다

핵심은:
- **2단계 입력이 반드시 `y`여야 한다**
- `a'`, `b'`는 `x`가 아니라 **`y`에 종속된 stochastic run**이어야 한다

목적:
- 당신이 원래 제안한 핵심 아이디어 검증

---

### 선택 variation (시간이 되면)
#### V5. Union-2 on x
- `a = gec(x)`
- `b = gec(x)`
- `final = union(a, b)`

목적:
- consensus의 반대편 극단을 확인
- 모델이 recall-limited인지 precision-limited인지 감을 잡기 위한 비교군

#### V6. Consensus iterative without stochastic pairing control variants
필요하면 다음도 생각할 수 있다.
- `y = consensus(gec(x), gec(x))`
- `final = gec(y)`  
즉 2단계에서는 다시 consensus하지 않고 한 번만 적용

목적:
- “2단계에서도 합의가 필요한가?”를 확인하는 보조 실험

우선순위는 V1~V4가 가장 높다.

---

## 8) 구현 시 중요한 정의

### 8.1 `consensus(a, b)`의 의미
여기서 `a`, `b`는 단순히 확률 분포가 아니라 **교정 결과**다.  
따라서 `consensus(a, b)`를 정의할 때 다음을 명확히 해야 한다.

#### 가능한 정의
1. **edit-level consensus**
   - 두 결과가 동일한 edit를 예측한 경우만 유지
   - 가장 직접적이고 추천되는 방식

2. **token/span-level consensus**
   - 위치는 같지만 replacement가 다를 때 어떻게 처리할지 정의 필요

권장:
- 현재 evaluator가 사용하는 edit representation과 최대한 일치하게
- **edit 위치/span + replacement/action 내용이 동일한 경우만 같은 edit로 간주**

즉 `consensus(a, b)`는  
“두 run이 동일한 edit를 예측한 경우만 채택한 결과”로 구현하는 것이 가장 명확하다.

---

### 8.2 `combine(y, consensus(a', b'))`의 의미
이 부분은 매우 중요하다.  
`(a∩b) ∪ (a'∩b')`를 문자 그대로 집합 연산처럼 구현하면, 실제 텍스트 레벨에서는 충돌이나 중복 문제가 생길 수 있다.

따라서 구현상 더 정확한 해석은:

- `y`는 이미 1단계 교정이 반영된 텍스트
- `consensus(a', b')`는 **`y`를 기준으로 한 2단계 수정 결과**
- 최종 출력은 **`y`에 2단계 합의 수정을 적용한 텍스트**

즉 V4의 최종 결과는 사실상:

- `final = apply_consensus_pass(y)`

에 가깝고,  
직관적 설명으로만 `(a∩b) ∪ (a'∩b')`라고 이해하면 된다.

클로드 코드 구현 시에는 이 점을 혼동하지 않도록 해석해야 한다.

---

## 9) 데이터 및 평가 제약조건

현재 평가에는 stochasticity가 섞여 있고, eval split 자체도 full validation이 아닐 수 있다.  
따라서 **한 번의 측정값만으로 판단하면 위험하다.**

### 필수 조건
- 모든 variation은 **동일한 evaluation dataset / split** 사용
- 데이터 순서 고정
- 체크포인트 고정: **50k**
- variation 외의 설정은 최대한 고정
- 각 variation은 반드시 **반복 측정**

### 반복 횟수
- 최소 3회
- 가능하면 5회

### randomness 기록
- 각 repeat의 seed 또는 randomness 설정을 기록
- variation 간 비교가 가능하도록 randomness 처리 방식을 일관되게 유지

중요한 점:
- “운 좋은 한 번”이 아니라
- **방법 자체의 평균적 성능 차이**를 보고 싶다

---

## 10) 보고할 지표

### 주 지표
- **F0.5**

### 보조 지표
- Precision
- Recall
- F1

### 있으면 함께 보고할 것
- Tag accuracy
- Val loss

### 출력 행동 관련 통계
- 문장당 평균 edit 수
- 총 predicted edit 수
- 변경된 문장 비율

이 값들은 어떤 variation이 지나치게 보수적이거나 공격적으로 변했는지 이해하는 데 중요하다.

---

## 11) 해석 기준

### V3 (1단계 합의만)가 예상대로라면
보통 다음 패턴을 기대한다.
- Precision > V1
- Recall < V1

이 결과가 나오면, 합의가 실제로 false positive를 줄이는 방향으로 작동하고 있다고 볼 수 있다.

### V4 (2단계 합의 반복)가 예상대로라면
다음 패턴을 기대한다.
- Precision은 V1 대비 유지 또는 개선
- Recall은 V3보다 회복
- F0.5는 V1, V3보다 높을 가능성이 있음

즉 V4는:
- V3의 보수성을 유지하면서
- 일부 recall을 되찾는 구조로 작동해야 이상적이다

### V2 (단순 2-pass baseline)가 중요하긴 한 이유
만약 V4가 좋아도, 그것이:
- 진짜 “합의 기반 구조” 덕분인지
- 아니면 단순히 “2-pass”라서 좋아진 것인지
분리해서 봐야 한다

그래서 V2는 반드시 비교 기준으로 필요하다.

### V5 (Union-2)가 크게 좋다면
이는 현재의 직관과 달리, 모델이 precision-limited보다 recall-limited일 가능성을 시사한다.  
예상과 다르더라도 결과는 정직하게 보고해야 한다.

---

## 12) 선택 기준

variation 선택 기준은 다음 순서로 한다.

1. **평균 F0.5가 가장 높은 것**
2. 비슷하면 **precision이 더 높은 것**
3. 단, recall이 지나치게 무너지면 탈락

또한 반드시 다음을 함께 본다.
- mean ± std
- raw per-run 결과

차이가 실제 구조적 개선인지, 단순 noise인지 구분해야 하기 때문이다.

---

## 13) CPU 성능 실험 (2단계)

품질 실험에서 상위 variation만 CPU 성능 측정을 한다.

### 최소 측정 대상
- V1 Single-pass
- V3 Consensus-2
- V4 2-stage consensus iterative

### 보고할 항목
- Wall-clock latency
- Throughput (sent/sec, tok/sec 가능하면 둘 다)
- Memory / RSS
- CPU utilization (가능하면)

### 보고 시 중요 포인트
우리는 단순히 속도만 보고 싶은 것이 아니라:
- 품질 개선이 얼마나 있었는지
- 그 품질 개선이 CPU cost를 정당화하는지
- cost가 샘플 수/패스 수만큼 단순 선형 증가하는지 아닌지

를 함께 보고 싶다.

---

## 14) 권장 실행 순서

### Step 1
먼저 품질 실험:
- V1 Single-pass
- V2 Iterative 2-pass baseline
- V3 Consensus-2
- V4 2-stage consensus iterative

### Step 2
시간이 되면 추가:
- V5 Union-2

### Step 3
Phase 1에서 유망한 variation만 골라 CPU 성능 측정

---

## 15) 원하는 산출물 형식

### A. Raw per-run 결과표
각 variation, 각 repeat에 대해:
- Variant
- Repeat ID
- Seed(s)
- Precision
- Recall
- F0.5
- F1
- Tag Acc (있으면)
- Val Loss (있으면)
- Avg edits/sentence
- Notes

### B. Summary table
각 variation에 대해:
- Precision mean ± std
- Recall mean ± std
- F0.5 mean ± std
- F1 mean ± std
- Avg edits/sentence mean ± std

### C. 해석 요약
짧게 다음을 써줄 것:
- F0.5 기준 최고 variation은 무엇인지
- precision은 예상대로 올라갔는지
- recall 하락은 감수 가능한 수준인지
- V4가 실제로 유망한지
- 차이가 noise 범위인지 아닌지

### D. CPU benchmark table
선택된 variation에 대해:
- Variant
- Batch size
- Samples per input
- Latency
- Throughput
- Memory
- Notes

### E. 최종 추천
다음 중 하나를 고르고 이유를 설명:
- single-pass 유지
- consensus-2 사용
- 2-stage consensus iterative 사용
- 아직 추가 실험 필요

---

## 16) 하지 말아야 할 것

- 체크포인트를 바꾸지 말 것: **50k 사용**
- 학습 재설정 / 재학습하지 말 것
- variation마다 다른 eval split을 쓰지 말 것
- `a'`, `b'`를 원문 `x`에서 다시 뽑지 말 것  
  - **반드시 `y = consensus(a, b)`를 입력으로 사용해야 함**
- 한 번의 lucky run만 보고 결론 내리지 말 것

---

## 17) 한 줄 요약

> **50k 체크포인트**를 사용하여, 원문 `x`에서의 2-sample consensus로 보수적인 1단계 교정 결과 `y`를 만든 뒤, 다시 `y`에 대해 2-sample consensus를 수행하는 **합의 기반 2단계 반복 교정 구조**가 single-pass 및 단순 2-pass보다 더 나은 **precision 중심 품질(F0.5)** 을 제공하는지, 그리고 그 비용이 CPU에서 실용적인지 검증한다.