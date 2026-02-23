# 실험 핸드오프 정리 (2026-02-23)

## 1) 현재 상황 요약

- 현재 환경에서는 `batch_size=4`가 메모리 한계로 불안정/중단되어, **신뢰 가능한 비교 범위는 batch 1~2**.
- 병목의 성격은 단순 VRAM 용량 부족보다도 **메모리 대역폭 + 커널 실행 효율** 영향이 큼.
- 따라서 현 머신에서는 batch 1~2 기준 최적화를 확정하고, 더 좋은 GPU에서 batch 확장 실험을 이어가는 것이 타당.

## 2) 코드 변경 사항(핵심)

### A. CUDA INT8 백엔드 도입 및 실험 분리
- `training/pretrain.py`
  - `--int8_backend {triton,cuda}` 지원
  - `--cuda_graph`는 실험 플래그(조건 미충족 시 자동 비활성)
  - `--int8` 사용 시 `--compile`은 현재 경로에서 실효 없음(건너뜀)

### B. CUDA 백엔드 최적화
- `model/cuda_bitlinear.py`
  - forward: `torch._int_mm` fast-path + 8배수 자동 패딩
  - fallback: C++/CUDA extension 경로
  - backward 모드: `fp32_tf32`, `fp16_tc`, `bf16_tc`, `int8`, `auto`
  - `gradw_lt`, fused activation/weight quant 게이트 지원

### C. 메모리 절감 패치(최근)
- `model/cuda_bitlinear.py`
  - autograd context에서 `w_deq_saved`(dequant 가중치 복제 저장) 제거
  - backward에서 `w_int8 * scale`로 재구성
  - 목적: activation/save tensor 메모리 감소

### D. 벤치 스크립트 정비
- `bench_cuda_ablation.py`
  - `--size`, `--max_steps`, `--batch_sizes` 파라미터화
  - 기본은 non-graph 케이스만 실행
  - 그래프는 `--include_graph_experimental`일 때만 수행

## 3) 주요 관측 결과

## 3-1. 64M, batch=1, max_steps=50 (CUDA 백엔드)

- baseline_no_graph: `9201.1 tok/s`
- gradw_lt_only: `9447.3 tok/s` (`+2.68%`)
- fused_quant_only: `10392.1 tok/s` (`+12.94%`)
- all_optimizations(non-graph): `10147.5 tok/s` (`+10.29%`)

해석:
- 64M에서는 fused quant 효과가 크고, gradw_lt는 단독 효과가 제한적.

## 3-2. "CUDA 적용 전 vs 현재" 비교(64M, batch=1, max_steps=50)

- before (triton int8):
  - `avg_tok_s=5913.5`
  - `peak_delta_mib=10765`
- current (cuda + gradw_lt+fused_quant):
  - `avg_tok_s=10295.0`
  - `peak_delta_mib=11454`

해석:
- 속도는 대폭 증가(약 +74%)
- 메모리는 증가(약 +689 MiB)

## 3-3. backward 모드 비교(64M, batch=1, max_steps=50)

- `fp32_tf32`: `10349.0 tok/s`, `peak_delta_mib=11459`
- `bf16_tc`: `10791.1 tok/s`, `peak_delta_mib=10055`

해석:
- `bf16_tc`가 **속도↑ + 메모리↓**를 동시에 보임(현 시점 권장)

## 3-4. 최근 메모리 절감 패치 적용 후(64M, batch=1, max_steps=50)

- 설정: `bf16_tc + gradw_lt + fused_quant`
- 결과: `avg_tok_s=10588.6`, `peak_delta_mib=9846`

비교 기준(`bf16_tc` 이전 수치 대비):
- 메모리: `10055 -> 9846` (약 `-209 MiB`)
- 속도: 소폭 변동(유사 수준)

## 3-5. fused_ce / grad_ckpt 영향 분리 (64M, batch=1, max_steps=30)

- fused=0, ckpt=0: `10199.4 tok/s`, `9856 MiB`
- fused=0, ckpt=1: `8117.5 tok/s`, `2179 MiB`
- fused=1, ckpt=0: `10263.0 tok/s`, `9930 MiB`
- fused=1, ckpt=1: `7961.6 tok/s`, `2175 MiB`

해석:
- `grad_ckpt` 영향이 매우 큼: 메모리 약 `-7.7GB`, 속도 약 `-20~22%`
- `fused_ce` 영향은 메모리/속도 모두 상대적으로 작음

## 3-6. 32M 재측정 (batch 1~2 신뢰, batch4 중단)

- batch=1
  - baseline: `15501.0 tok/s`
  - fused_only: `16069.3` (`+3.67%`)
  - all(non-graph): `16807.2` (`+8.43%`)
- batch=2
  - baseline: `13593.6 tok/s`
  - fused_only: `15499.1` (`+14.02%`)
  - all(non-graph): `15850.0` (`+16.60%`)
- batch=4
  - 현 환경에서 중단(메모리 한계/장시간 정체)

해석:
- 32M에서는 batch2에서 최적화 효과가 더 뚜렷함.

## 4) 운영 권장안(현 머신)

- CUDA Graph: 기본 비활성(실험 플래그로만 유지)
- 기본 권장 조합(속도/메모리 균형):
  - `BITLINEAR_CUDA_BACKWARD=bf16_tc`
  - `BITLINEAR_CUDA_GRADW_LT=1`
  - `BITLINEAR_CUDA_FUSED_ACT=1`
  - `BITLINEAR_CUDA_FUSED_WEIGHT=1`
- 학습 옵션:
  - `--int8 --int8_backend cuda`
  - `--compile`은 `--int8` 경로에서 현재 실효 없음
- VRAM 부족 시:
  - `--grad_ckpt`를 켜서 batch 확보(속도 희생 감수)

## 5) 더 좋은 GPU에서 이어갈 실험 계획

우선순위 순:

1. **batch 확장 가능 구간 탐색**
   - 32M, 64M 각각 batch 1/2/4/8 가능 여부(OOM 없이 50 step)
2. **처리량 최적점 탐색**
   - `effective batch` 고정 후
   - (micro-batch↑, grad_accum↓) vs (micro-batch↓, grad_accum↑) 비교
3. **메모리-속도 Pareto 작성**
   - 축: `tok/s` vs `peak_delta_mib`
   - 설정: `grad_ckpt on/off`, `bf16_tc/fp32_tf32`, `fused on/off`
4. **CUDA Graph 재평가(선택)**
   - 대형 GPU에서만 실험
   - batch 1~2 고정, non-graph 대비 순수 이득 검증

## 6) 재현용 커맨드 예시

### 6-1. 32M batch 1~2 벤치

```bash
source .venv/bin/activate
python bench_cuda_ablation.py --size 32M --max_steps 30 --batch_sizes 1 2
```

### 6-2. 64M 권장 조합 단일 실행

```bash
source .venv/bin/activate
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1

python -m training.pretrain \
  --size 64M \
  --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --bf16 \
  --int8 \
  --int8_backend cuda \
  --log_every 10
```

### 6-3. 메모리 절약 우선(속도 희생)

```bash
python -m training.pretrain \
  --size 64M \
  --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --bf16 \
  --fused_ce \
  --grad_ckpt \
  --int8 \
  --int8_backend cuda
```

## 7) 결론

- 현 머신 기준으로는 **batch 1~2에서 최적화 검증을 마치고**,
- CUDA Graph는 실험 플래그로만 유지,
- 실전 기본값은 `bf16_tc + gradw_lt + fused_quant`가 가장 합리적.
- batch 4 이상 확장 및 고활용(200%/400%)은 더 큰 GPU에서 micro-batch 확장 중심으로 재평가 권장.

---

## 8) 96GB 머신 추가 결과 (2026-02-23, 같은 코드베이스)

### 8-1. 배치 확장 가능 범위 (quick feasibility)

- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97887 MiB)`
- 64M: batch `1/2/4/8` 가능, `16` OOM
- 128M: batch `1/2/4` 가능, `8/16` OOM

### 8-2. non-graph 기준 배치별 실측 요약 (max_steps=50)

#### 64M
- batch=1: `fused_quant`가 baseline 대비 `+3.97%`
- batch=2: `fused_quant`/`all_opt` 모두 `+4.3%`
- batch=4: `all_opt`가 `+10.41%`
- batch=8: `fused_quant`가 `+12.58%`

#### 128M
- batch=1: `fused_quant`가 `+3.18%`
- batch=2: `fused_quant`가 `+7.72%`
- batch=4: `fused_quant`가 `+11.97%`

해석:
- `gradw_lt` 단독 이득은 작거나 불안정했고,
- 이번 머신에서는 **fused quant(activation+weight)**가 가장 일관된 이득을 제공.

### 8-3. micro-batch vs grad_accum (128M)

- `bs1_ga1`: `27400.1 tok/s`
- `bs1_ga2`: `27052.1 tok/s`
- `bs2_ga1`: `28058.9 tok/s` ← 최고
- `bs2_ga2`: `27919.1 tok/s`

해석:
- 같은 effective batch를 맞춰도 `batch_size=2`가 가장 유리.
- `batch_size>2`는 가능하더라도 처리량/효율이 떨어지는 구간이 많음.

### 8-4. pack_size sweep (fused quant + bf16_tc, batch=2)

#### 128M (max_steps=60)
- `pack=4096`: `28187.5 tok/s`, `29756 MiB`
- `pack=6144`: `26820.0 tok/s`, `43678 MiB`
- `pack=8192`: `25988.7 tok/s`, `57428 MiB`
- `pack=10240`: `25450.3 tok/s`, `71340 MiB`

#### 64M (max_steps=60)
- `pack=4096`: `43314.6 tok/s`, `19270 MiB`
- `pack=6144`: `41421.2 tok/s`, `28198 MiB`
- `pack=8192`: `40676.8 tok/s`, `36938 MiB`
- `pack=10240`: `39569.5 tok/s`, `45970 MiB`

해석:
- 64M/128M 모두 **`pack_size=4096`가 처리량 최고 + 메모리 최소**.

### 8-5. lr / warmup 탐색 (128M, batch=2, pack=4096, non-graph)

#### 1차 스윕 (max_steps=200)
- 후보 중 상위: `lr=1.6e-4`, `warmup=50 or 100`

#### 2차 재검증 (max_steps=400)
- `lr=1.6e-4, wu=50`: `tail_bpc=3.426`, `27642 tok/s`
- `lr=1.6e-4, wu=100`: `tail_bpc=3.355`, `27819 tok/s` ← best
- `lr=1.3e-4, wu=50`: `tail_bpc=3.395`, `27732 tok/s`

해석:
- 현재 데이터/세팅에서는 **`lr=1.6e-4`, `warmup_steps=100`**이 가장 안정적.

### 8-6. 옵션 on/off 분해 (128M, lr=1.6e-4, wu=50, max_steps=120)

- baseline (`fused=0, ckpt=0`): `27672 tok/s`, `peak 29756 MiB`
- `grad_ckpt=1`: 약 `-27% tok/s`, 메모리 약 `-24 GB`
- `fused_ce=1`: 속도 소폭 감소(약 `-0.9%`), 메모리 변화 미미

해석:
- **메모리 여유가 있으면 `grad_ckpt`는 끄는 게 유리**.
- `fused_ce`는 이번 세팅에서 필수 옵션이 아님(성능 이득 미확인).

### 8-7. 96GB 머신 운영 권장 프리셋

권장 우선순위 (128M 기준):
1) `--int8 --int8_backend cuda`
2) `--batch_size 2 --grad_accum_steps 1`
3) `--pack_size 4096`
4) `--lr 1.6e-4 --warmup_steps 100`
5) 기본적으로 `--grad_ckpt` OFF, `--fused_ce` OFF
6) `--cuda_graph`는 실험 플래그로만 유지

권장 환경변수:
```bash
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=0
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
```

권장 실행 예시:
```bash
python -m training.pretrain \
  --corpus corpus/sample_10g.jsonl \
  --text_key text \
  --size 128M \
  --pack_size 4096 \
  --tokenizer keyboard \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --bf16 \
  --int8 \
  --int8_backend cuda \
  --lr 1.6e-4 \
  --warmup_steps 100 \
  --val_corpus corpus/val_50k.jsonl \
  --val_every 200 \
  --save_dir checkpoints/run_v1 \
  --save_every 100 \
  --log_every 10 \
  --num_workers 0
```
