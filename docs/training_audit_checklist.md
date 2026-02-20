# 학습 구현 점검 체크리스트 (BitNet-Mamba Seq2Seq)

기준 커맨드:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u -m training.pretrain \
  --tokenizer mecab_bbpe \
  --size 16M --corpus corpus/sample_1g.jsonl --text_key text \
  --pack_size 4096 --batch_size 12 --grad_accum_steps 1 --num_workers 32 \
  --grad_ckpt --amp --fused_ce --compile --lr 5e-4 \
  --warmup_steps 1000 --log_every 1 --save_every 5000 \
  --val_corpus corpus/val_50k.jsonl --val_every 200 --val_steps 20 \
  --save_dir checkpoints/16M_mecab_bbpe \
  > training_mecab_bbpe.log 2>&1 &
```

---

## 목적

- 구현 오류(학습이 틀린 방향으로 수렴하거나 지표가 왜곡되는 문제) 조기 발견
- 속도 병목(I/O, 커널 fallback, 설정 충돌) 분리
- 실험 간 공정한 비교 기준 정립

---

## P0 (즉시 확인: 결과 신뢰도/실행 안정성)

### 1) NFD 토크나이저 프리셋 클래스명 불일치

- [ ] 확인 파일: `training/pretrain.py`, `nfd_tokenizer/tokenizer_wrapper.py`
- [ ] 증상: `--tokenizer nfd` 실행 시 클래스 로드 실패 가능
- [ ] 기대 상태: 프리셋 class 이름과 실제 wrapper class가 정확히 일치
- [ ] 비고: 현재 실험은 `mecab_bbpe`라 직접 영향은 없지만, 비교 실험 재현성 관점에서 P0

### 2) JSONL 기본 파싱 경로 점검 (`text_key` 누락 시)

- [ ] 확인 파일: `training/dataset.py`
- [ ] 증상: `--text_key` 누락하면 JSON 한 줄 전체 문자열을 텍스트로 사용 가능
- [ ] 기대 상태: 기본 필드 추론 또는 명시적 에러로 잘못된 학습 입력 차단
- [ ] 현재 커맨드 상태: `--text_key text`를 줬으므로 당장 안전

### 3) `src_mask` 실사용 여부 (PAD 영향)

- [ ] 확인 파일: `model/seq2seq.py`, `model/encoder.py`, `model/decoder.py`
- [ ] 증상: 마스크를 만들고 전달하지만 내부 계산에 실질 반영이 없음
- [ ] 리스크: 짧은 샘플 PAD가 상태 전이에 섞여 품질 저하 가능
- [ ] 기대 상태: PAD 구간이 상태 업데이트/출력에 영향을 주지 않음

---

## P1 (지표 해석/실험 비교 신뢰도)

### 4) Train 로그 토큰 집계가 PAD 포함인지 확인

- [ ] 확인 파일: `training/pretrain.py` (`n_tokens = tgt_target.numel()`)
- [ ] 증상: PAD 포함 토큰수로 tok/s, loss 집계 시 지표 왜곡
- [ ] 기대 상태: non-pad 토큰 기준 집계

### 5) Train loss/BPC 집계 방식의 가중 평균 여부

- [ ] 확인 파일: `training/pretrain.py`, `validate()`
- [ ] 증상: train은 배치 평균 합산, val은 토큰 가중 평균에 가까워 지표 비교 불공정
- [ ] 기대 상태: train/val 모두 동일한 토큰 가중 기준으로 집계

### 6) `max_chars` 종료 기준의 과대 집계 여부

- [ ] 확인 파일: `training/dataset.py`, `training/pretrain.py`
- [ ] 증상: 토큰은 pack_size로 truncate되지만 문자수는 truncate 전 합을 누적 가능
- [ ] 결과: 문자 예산 조기 소진(실제 학습량 대비 과대 계산)
- [ ] 기대 상태: 실제 사용된 샘플 분량과 문자 예산 일치

---

## P1 (성능/처리량)

### 7) DataLoader 멀티워커 I/O 중복 스캔 확인

- [ ] 확인 파일: `training/dataset.py` (`i % num_workers` 분배)
- [ ] 증상: 각 worker가 파일 전체를 읽고 일부만 사용 → 디스크/CPU 낭비
- [ ] 현재 설정 영향: `--num_workers 32`에서 병목 가능성 매우 큼
- [ ] 기대 상태: worker별 파일/샤드 분리로 중복 읽기 제거

### 8) Mamba CUDA selective scan fallback 여부

- [ ] 확인 파일: `model/mamba_block.py`
- [ ] 증상: CUDA 커널 미탑재 시 Python fallback scan 사용(시퀀스 길수록 급격히 느림)
- [ ] 기대 상태: `mamba_ssm` CUDA 경로 사용
- [ ] 확인 방법: 시작 로그/환경에서 커널 import 성공 여부 점검

### 9) `--compile` + `--grad_ckpt` + `--fused_ce` 조합에서 실속도 검증

- [ ] 증상: 조합에 따라 기대 대비 속도 상승이 제한되거나 초기 그래프 컴파일 비용이 큼
- [ ] 기대 상태: 워밍업 구간 이후 안정적인 tok/s 상승
- [ ] 확인 방법: step 구간별 tok/s 이동평균(초기 200 step 제외)

---

## P2 (학습 안정성)

### 10) `--amp`(fp16) vs `--bf16` 안정성 비교

- [ ] 확인 파일: `training/pretrain.py`
- [ ] 증상: fp16은 스케일링/overflow 민감, BitLinear 조합에서 불안정 가능
- [ ] 기대 상태: 장기 학습에서 NaN/Inf 없이 loss 하강
- [ ] 권장 실험: 동일 설정으로 `--amp`와 `--bf16` A/B 2~5k step 비교

### 11) 로그 주기(`--log_every 1`) 오버헤드 점검

- [ ] 증상: 잦은 stdout flush로 처리량 저하 가능
- [ ] 기대 상태: 모니터링 품질 유지하면서 I/O 오버헤드 최소화
- [ ] 권장: 10~50 step 간격 비교 벤치

---

## 빠른 실행용 점검 순서 (권장)

1. [ ] **P0 1~3** 먼저 확인 (실행 실패/학습 타당성)
2. [ ] **P1 7~9** 확인 (지금 커맨드에서 실제 속도 병목)
3. [ ] **P1 4~6** 정리 (지표 해석 신뢰도 확보)
4. [ ] **P2 10~11**로 안정화/운영 최적화

---

## 실험 로그에 꼭 남길 항목 (비교 표준)

- [ ] git commit hash
- [ ] tokenizer 종류/버전 (예: `mecab_bbpe.json` 수정 여부)
- [ ] `mamba_ssm` CUDA 커널 사용 여부
- [ ] precision 모드 (`amp`/`bf16`)
- [ ] train/val loss 집계 기준 (pad 포함/제외)
- [ ] 실제 종료 기준 (`max_steps`/`max_chars`)과 총 처리 chars/tokens

---

## 현재 커맨드 기준 즉시 리스크 요약

- [ ] 높은 리스크: `num_workers=32`에서 데이터 읽기 중복 병목 가능성
- [ ] 중간 리스크: train 지표 집계(PAD 포함)로 체감 성능 판단 왜곡 가능
- [ ] 중간 리스크: `src_mask` 미반영으로 패딩 영향 누적 가능
- [ ] 낮은 리스크(현재 실험 직접 영향 적음): `nfd` preset 불일치
