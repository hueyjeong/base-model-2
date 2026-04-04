# Jamo Codec 실험 일지

## 2026-04-05 — 프로젝트 시작

### 동기

- ELECTRA RTD pretrain에서 자모 토크나이저(vocab=303)의 긴 시퀀스가 학습 속도 병목
- Full Attention (d=768, 20L): 8x 5090 DDP에서 184K tok/s (seq=4096)
- BiMamba2: 600K tok/s였으나 BF16 backward NaN 문제로 사용 불가
- Neural Codec으로 시퀀스 압축 → backbone 부담 경감이 근본 해결책

### 핵심 아이디어

- BLT (Byte Latent Transformer, Meta 2024) 방식 참고
- 자모 시퀀스 → 연속 잠재 공간으로 압축 → backbone 처리 → 복원
- 연속 공간에서 오타와 정답이 가까운 점으로 매핑 → GEC에 유리할 수 있음
- 가변 패칭: 저빈도어(인명, 신조어 등)는 적게 압축하여 정보 보존

### 논의 사항

- 입력 표현: byte vs 자모 분해 vs 키보드 시퀀스 — 어떤 게 codec에 유리한지 실험 필요
- 키보드 토크나이저의 SHIFT/BLANK는 유지 — 키보드 입력 오류를 편집 연산으로 표현
- 고정 stride의 한계: 자모 길이가 가변적이라 음절/단어 경계와 불일치
- 압축률 트레이드오프: 높을수록 GEC에서 유리할 수 있으나 저빈도어 구분 어려움

### 실험 계획

Phase 1(Conv) → 2(Cross-Attention) → 3(가변 패칭) → 4(Backbone 통합)
상세: [PLAN.md](PLAN.md)

---

## 2026-04-05 — Phase 1 결과: Conv Codec 9조합 sweep

### 실험 설정

- 학습: val.parquet (525만 rows), 평가: test.parquet (525만 rows)
- 모델: ConvCodec d=256, 3 layers, kernel=5 (2.6~4.2M params)
- 10K steps, batch=32, seq=512, lr=3e-4, BF16
- 9조합: 3 토크나이저(byte/jamo/keyboard) × 3 stride(2/4/8)

### 결과

| 조합 | 500step | 1000step | 2000step | test 토큰acc | test seq EM |
|------|---------|----------|----------|-------------|-------------|
| byte_s2 | 83.8% | 99.98% | 100% | 100% | 100% |
| byte_s4 | 81.8% | 99.95% | 100% | 100% | 100% |
| byte_s8 | 78.8% | 99.86% | 100% | 100% | 100% |
| jamo_s2 | 84.0% | 99.95% | 100% | 100% | 100% |
| jamo_s4 | 81.5% | 99.89% | 100% | 100% | 100% |
| jamo_s8 | 79.2% | 99.74% | 99.99% | 100% | **99.95%** |
| kbd_s2 | 85.0% | 99.96% | 100% | 100% | 100% |
| kbd_s4 | 82.7% | 99.89% | 100% | 100% | 100% |
| kbd_s8 | 80.7% | 99.76% | 99.99% | 100% | 100% |

레이턴시 (GPU, stride=4 기준):
- seq=512: encode 0.37ms, decode 0.36ms, total 0.73ms
- seq=2048: encode 0.47ms, decode 0.55ms, total 1.02ms

### 관찰

1. **Conv 3M params로 모든 조합에서 거의 완벽 복원** — 2K steps면 수렴
2. **입력 표현 간 차이 거의 없음** — byte/jamo/keyboard 모두 동일 수준
3. **stride=8에서도 복원 가능** — jamo_s8만 seq EM 99.95%, 나머지 100%
4. **codec 레이턴시 무시 가능** — seq=2048에서 총 1ms
5. **복원은 쉬운 태스크** — Conv baseline만으로 충분, cross-attention 불필요

### 시사점

- 복원 정확도만으로는 codec 구조/입력 표현 간 차별화 불가
- 진짜 차이는 **backbone 결합 후 downstream(GEC) 성능**에서 나올 것
- Phase 2(cross-attention) 건너뛰고 **Phase 4(backbone 통합) 직행** 고려
- 또는: 더 높은 압축률(16x, 32x) 실험하여 Conv의 한계점 탐색

---

## 2026-04-05 — 극한 압축률 테스트 (stride 8/16/32)

### 설정

- jamo 토크나이저 대표 (Phase 1에서 입력 표현 간 차이 없었으므로)
- ConvCodec d=256, 3 layers, 10K steps, 나머지 동일

### 결과

| stride | 압축 비율 | 1K step acc | 2K step acc | test 토큰acc | test 문자acc | test seq EM |
|--------|----------|-------------|-------------|-------------|-------------|-------------|
| 8x | 512→64 | 99.75% | 99.99% | 100% | 100% | 100% |
| 16x | 512→32 | 99.29% | 99.96% | 100% | 99.98% | 99.85% |
| 32x | 512→16 | 97.18% | 99.65% | 100% | 99.73% | 98.68% |

복원 예시: 5개 샘플 모두 stride=32에서도 전부 MATCH.

### 관찰

1. **stride=32 (512→16 벡터)에서도 토큰 정확도 100%** — Conv가 예상보다 훨씬 강력
2. 문자 정확도 99.73%, seq EM 98.68% → 긴 시퀀스에서 미세한 차이만 존재
3. 수렴 속도는 압축률에 비례해서 느려짐 (1K step: 8x 99.75% vs 32x 97.18%)
4. **복원은 Conv만으로 충분 — cross-attention(Phase 2) 불필요 확정**

### 시사점

- Conv codec의 복원 능력 한계가 사실상 없음 (seq=512 범위 내)
- 차별화는 **z 공간의 구조적 품질**에서 나올 것
  - 같은 100% 복원이라도 z가 의미적으로 잘 조직되어 있는지
  - 오타 시퀀스와 정답 시퀀스의 z 거리가 가까운지
  - backbone이 z 위에서 효과적으로 학습할 수 있는지
- → z 공간 분석 필요

---

