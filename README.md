# base-model-2

한국어 문법 오류 교정(GEC)을 위한 BitNet-Mamba Seq2Seq 프로젝트입니다.

- 모델: BitMamba encoder-decoder
- 토크나이저: keyboard / char / nfd / bbpe / mecab_bbpe
- 학습 스크립트: `training/pretrain.py`

## 현재 권장 실행 정책 (핸드오프 반영)

- CUDA Graph: **기본 비활성**, 실험 플래그로만 사용 (`--cuda_graph`)
- INT8 CUDA 백엔드 권장: `--int8 --int8_backend cuda`
- 권장 기본 조합: **non-graph + `gradw_lt` + `fused_quant`**
- `--int8` 사용 시 `--compile`은 현재 경로에서 실효 없음(스킵)
- 현재 환경에서 신뢰 비교 범위: batch `1~2`

## 권장 환경변수 (INT8 CUDA)

```bash
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
```

## 빠른 실행 예시

```bash
source .venv/bin/activate
python -m training.pretrain \
  --size 32M \
  --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --bf16 \
  --int8 \
  --int8_backend cuda \
  --log_every 10
```

## 노이즈 설정 파일 예시

- 기본 동작: `--noise_config` 미지정 시 `NoiseConfig` 기본값 사용
- 예시 파일: `training/noise_config.example.json`

```bash
python -m training.pretrain \
  --size 32M \
  --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --bf16 \
  --noise_config training/noise_config.example.json
```

스케일과 함께 쓰는 예시(기본값 대비 텍스트 노이즈 0.8배, 토큰 노이즈 1.2배):

```bash
python -m training.pretrain \
  --size 32M \
  --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --bf16 \
  --noise_config training/noise_config.example.json \
  --noise_scale_text 0.8 \
  --noise_scale_token 1.2
```

적용 순서:

1. `NoiseConfig` 기본값
2. `--noise_config` 파일 값으로 override
3. `--noise_scale`, `--noise_scale_text`, `--noise_scale_token` 배율 적용

## 노이즈 권장 범위 표

아래 범위는 **권장값**이며, 강제 제한은 아닙니다.

| 필드 | 기본값 | 권장 범위 |
|---|---:|---:|
| `korean_error_prob` | 0.30 | 0.05 ~ 0.50 |
| `korean_error_count` | 2 | 1 ~ 4 |
| `spacing_noise_prob` | 0.30 | 0.05 ~ 0.60 |
| `spacing_remove_ratio` | 0.30 | 0.05 ~ 0.70 |
| `spacing_insert_prob` | 0.05 | 0.00 ~ 0.20 |
| `spacing_full_remove_prob` | 0.10 | 0.00 ~ 0.30 |
| `keyboard_typo_prob` | 0.15 | 0.02 ~ 0.40 |
| `keyboard_typo_ratio` | 0.05 | 0.01 ~ 0.20 |
| `ngram_shuffle_prob` | 0.10 | 0.00 ~ 0.30 |
| `ngram_n` | 3 | 2 ~ 5 |
| `word_reorder_prob` | 0.10 | 0.00 ~ 0.30 |
| `word_reorder_swaps` | 2 | 0 ~ 4 |
| `token_mask_ratio` | 0.15 | 0.05 ~ 0.35 |
| `token_delete_ratio` | 0.05 | 0.00 ~ 0.20 |
| `text_infill_ratio` | 0.15 | 0.05 ~ 0.35 |
| `infill_poisson_lambda` | 3.0 | 1.0 ~ 6.0 |
| `token_noise_mask_weight` | 0.4 | 0.0 ~ 2.0 |
| `token_noise_delete_weight` | 0.2 | 0.0 ~ 2.0 |
| `token_noise_infill_weight` | 0.4 | 0.0 ~ 2.0 |
| `keyboard_shift_typo_prob_alpha` | 0.30 | 0.00 ~ 0.70 |
| `keyboard_shift_typo_prob_ko_alpha` | 0.30 | 0.00 ~ 0.70 |
| `keyboard_shift_typo_prob_ko_jamo` | 0.30 | 0.00 ~ 0.70 |

## 메모리/속도 트레이드오프 핵심

- `grad_ckpt`:
  - 장점: 메모리 크게 절감
  - 단점: 처리량(tok/s) 유의미 감소
- `fused_ce`:
  - 영향: 보통 `grad_ckpt`보다 작음
- 요약: 메모리 부족 시 `grad_ckpt`로 batch 확보, 여유 시 off로 속도 우선

## 벤치 도구

### A/B 벤치

```bash
python bench_cuda_ablation.py --size 32M --max_steps 30 --batch_sizes 1 2
```

- 기본: non-graph 케이스만 실행
- 그래프 실험 포함 시:

```bash
python bench_cuda_ablation.py --size 32M --max_steps 30 --batch_sizes 1 2 --include_graph_experimental
```

## 문서

- 실험 핸드오프: `docs/experiment_handoff_2026-02-23.md`
- INT8 CUDA 사용 가이드: `docs/int8_cuda_backend.md`
- Vast.ai 세팅: `README-vastai.md`
