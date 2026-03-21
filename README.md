# base-model-2

한국어 문법 오류 교정(GEC) 프로젝트.

## 모델

### DenseEditor (현행 메인)

인코더-only 편집 태깅 모델. 128M params, BiMamba-2 SSD + BitNet 1.58-bit.

- Mixing: BiMamba2 (양방향 Mamba-2, d_state=64)
- FFN: BitNetFFN SwiGLU
- 토크나이저: keyboard (vocab=303, 자모 단위)
- 10k DDP 결과: val_loss=0.107, P=93.1%, R=71.0%

### Seq2Seq (레거시)

BitMamba encoder-decoder. `training/pretrain.py`로 학습.

## 실행 정책

- INT8 CUDA 백엔드 권장: `--int8 --int8_backend cuda`

## 권장 환경변수 (INT8 CUDA)

```bash
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
```

## 빠른 실행

```bash
source .venv/bin/activate

# DenseEditor 학습 (DDP 4GPU)
torchrun --nproc_per_node=4 -m training.pretrain_dense_editor \
    --mixing_type mamba2 --d_model 640 \
    --corpus corpus/sample_full.jsonl --text_key text \
    --bf16 --int8 --int8_backend cuda \
    --batch_size 8 --grad_accum_steps 2 \
    --lr 1e-3 --schedule wsd --max_steps 300000

# 품질 벤치마크 (오버핏 테스트)
python bench_quality.py --mixing_types mamba2 --d_model 640 \
    --corpus corpus/val_50k.jsonl --max_steps 2000

# 실제 오류 분포 기반 노이즈 프리셋
python bench_quality.py --mixing_types mamba2 --d_model 640 \
    --corpus corpus/val_50k.jsonl --max_steps 2000 --noise_preset realistic
```

## 노이즈 엔진

27개 한국어 오류 타입 (패턴 기반 + MeCab 동적 생성) + 텍스트/토큰 레벨 노이즈.

### 가중치 프리셋

- `default`: 기존 가중치 (하위호환)
- `realistic`: KoGEC 2025 실제 오류 분포 기반, hit rate 보정

```bash
# CLI
--noise_preset realistic

# 노이즈 설정 파일
training/noise_config.example.json
```

### realistic 프리셋 KAGAS 분포 (1000문장 검증)

| KAGAS 유형 | KoGEC 목표 | realistic |
|-----------|-----------|-----------|
| WS (띄어쓰기) | 25% | 24% |
| PUNCT (구두점) | 30% | 28% |
| DEL (삭제) | 11% | 10% |
| PRO_NOUN (체언) | 11% | 12% |
| VERB_ADJ (용언) | 11% | 8% |
| SPELL (철자) | ~10% | 6% |
| INS (삽입) | 6% | 5% |
| END (어미) | 4% | 2% |
| PART (조사) | 2% | 2% |
| MODIFIER (수식어) | 1% | 1% |
| SP_RELATION (구문) | 1% | 2% |

```bash
# 분포 검증
python error_generation/test_distribution.py \
    --corpus corpus/val_50k.jsonl --n_samples 1000
```

## 문서

- `CLAUDE.md`: Claude Code 가이드 (아키텍처, 명령어, 컨벤션)
- `inference_dense/BENCHMARK.md`: CPU 인퍼런스 벤치마크
- `exp-electra-gec/OVERVIEW.md`: ELECTRA GEC 실험 계획
- `docs/int8_cuda_backend.md`: INT8 CUDA 사용 가이드
- `README-vastai.md`: Vast.ai 세팅
