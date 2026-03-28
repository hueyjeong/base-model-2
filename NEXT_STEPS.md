# DenseEditor 다음 단계 계획

## 현재 상태 (2026-03)
- step 100k, F0.5 93.93% (내부 벤치마크, 자모 단위)
- **학습 코퍼스 오염 확인**: Lang-8 교정문, NIKL 교정쌍 target_text가 포함됨
- 내부 벤치마크 수치는 과대평가 가능성 있음
- 외부 벤치마크(KAGAS) 미실시

## Phase 1: 현재 학습 — 한계까지 뽑기
- [ ] step 100k → 125k WSD decay (1e-4 → 1e-5, min_lr_ratio=0.1)
- [ ] 정체 시 125k → ? WSD decay (1e-5 → 1e-6) 추가 가능
- [ ] 모델이 뽑아낼 수 있는 성능 한도까지 학습 계속
- [ ] 최종 체크포인트 평가 (threshold 스윕 + 글자/어절/형태소)

## Phase 2: KAGAS 외부 벤치마크
- [ ] KAGAS 데이터 신청 (Kor-Native, Kor-Learner) — 신청 완료, 승인 대기
  - 신청 폼: https://forms.gle/kF9pvJbLGvnh8ZnQ6
  - Lang-8 원본 (별도): https://docs.google.com/forms/d/17gZZsC_rnaACMXmPiab3kjqBEtRHPMz0UG9Dk-x_F0k/
  - KAGAS GitHub: https://github.com/soyoung97/Standard_Korean_GEC
- [ ] Kor-Lang8 테스트셋 구축 (Lang-8 원본으로 KAGAS 스크립트 실행)
- [ ] 평가 파이프라인 구축: 오류문 입력 → 모델 교정 → 디코딩 → M2/GLEU 평가
  - M2 생성: `parallel_to_m2_korean.py` (KAGAS, 형태소 단위 Kkma POS 태깅)
  - M2 scorer: https://github.com/ayaka14732/m2scorer.git (Python3 호환)
- [ ] 현재 모델(오염 코퍼스)로 일단 돌려서 **상한선** 확인
- [ ] 오염 범위 측정: 학습 코퍼스 vs KAGAS 테스트셋 텍스트 겹침 비율
- [ ] KAGAS 베이스라인 참고: Hanspell F0.5=25.85%, KoBART F0.5=31.70% (Kor-Union)

## Phase 3: 깨끗한 코퍼스 재구축

### 오염 소스 제거
- [x] `export_corpus_ko.py`에서 주석 처리 완료 (`/workspace/WorkSpace/IDK/`)
  - 제거: gemini 교정쌍, NIKL 교정쌍 target, Lang-8 교정문, NIKL 원문
  - 유지: 위키백과 (`train_kowiki.jsonl`), 나무위키 (`train_namu.jsonl`)

### 추가 코퍼스 후보
| 데이터셋 | 규모 | 문체 | 라이선스 | 오염 위험 | URL |
|----------|------|------|----------|-----------|-----|
| **FineWeb-2 ko** | 수십~수백GB | 웹 전반 | ODC-By (상업적 OK) | 낮음 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **CulturaX ko** | 수십GB | 웹 전반 | 연구용 | 낮음 | https://huggingface.co/datasets/uonlp/CulturaX |
| **CC-100 ko** | ~38.6GB | 웹 전반 | 연구용 | 낮음 | https://huggingface.co/datasets/lcw99/cc100-ko-only |
| **KOREAN-WEBTEXT** | 2.2B 토큰 | 웹 (고품질) | 확인 필요 | 낮음 | https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT |
| **OSCAR ko** | ~7GB | 웹 전반 | CC0 | 낮음 | https://huggingface.co/datasets/lcw99/oscar-ko-only |
| 모두의 말뭉치 (문어/구어/웹/메신저) | 수천만 어절 | 다양 | 연구용 (NIKL 신청) | **학습자 말뭉치 제외 필수** | corpus.korean.go.kr |

- 웹 크롤 데이터는 GEC 정답문으로 쓰려면 추가 필터링 필요 (비문, 광고, 스팸 등)
- 모두의 말뭉치: KAGAS 테스트셋 받은 후 텍스트 매칭으로 오염 체크, 안전한 하위셋만 사용
  - **반드시 제외**: 학습자 말뭉치 (NIKL Spelling Correction Corpus = KAGAS Kor-Learner 원천)
  - 안전할 가능성: 신문, 구어, 웹, 메신저 등 KAGAS와 무관한 도메인

### 셀프 정제 전략
- [ ] Phase 1 최종 모델 + threshold 0.8~0.9 (P=99%)로 웹 크롤 텍스트 교정
  - 건드린 곳은 거의 확실히 맞고, 안 건드린 곳은 원문 그대로 → 원문보다 나빠질 수 없는 구조
  - 교정된 텍스트를 정답문으로 사용
- [ ] 코퍼스 비중 구성:
  - **앵커 (높은 비중)**: 위키 + 나무위키 + 오염 제거된 모두의 말뭉치
  - **바리에이션 (낮은 비중)**: 셀프 교정된 웹 크롤 텍스트
- [ ] val/test set 분리 철저히 — 이번에는 처음부터!

## Phase 4: 재학습 (scratch)
- [ ] 깨끗한 코퍼스로 1e-3부터 scratch 학습
- [ ] 기존 하이퍼파라미터 재활용 (edit_loss_weight=1.0, WSD 단계적 decay 등)
- [ ] KAGAS 외부 벤치마크로 공정 평가
- [ ] threshold + consensus 최적 조합 탐색

## Phase 5: 서비스 배포
- [ ] confidence threshold로 P 극대화 (99%+)
- [ ] consensus + 사전 기반 후보정으로 R 회복
- [ ] ONNX RT 또는 Rust 추론 엔진으로 배포
- [ ] 무료 서비스 → 사용자 피드백/실제 오류 데이터 수집
- [ ] 수집 데이터로 finetune → 선순환

## 미래 실험
- 뉴럴 자모 어댑터 아키텍처 (자모세 제거)
- 256~512M INT8 QAT로 언어 이해까지 확장
- ELECTRA RTD pretrain + 편집 태깅 (exp-electra-gec)
- 16~32M 증류 모델 (모바일/WASM 배포)

## 비용 참고
- 현재까지 약 100만원+ 투자 (GPU 인스턴스, 이전 시행착오 포함)
- Phase 1 완료 + 재학습(Phase 4) 포함 총 4~500만원 예상
- 기적의 공포탄 프로젝트: primer(인프라/데이터) > cartridge(학습) > projectile(축포) ㅋㅋ
