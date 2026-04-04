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

