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
