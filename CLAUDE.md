# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

한국어 문법 오류 교정(GEC)을 위한 **DenseEditor** (인코더-only 편집 태깅) + **BitNet-Mamba Seq2Seq** 프로젝트.
128M 파라미터, BiMamba-2 SSD + BitNet 1.58-bit, CPU/GPU 이중 추론 지원.

- 코드 주석/docstring은 **한국어**, 식별자는 영어
- 항상 한국어로 응답할 것

## Commands

```bash
source .venv/bin/activate

# ── DenseEditor (현행 메인 모델) ──

# 모델 검증 (forward/backward)
python model/dense_editor.py

# DenseEditor 학습 (DDP 4GPU)
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
torchrun --nproc_per_node=4 -m training.pretrain_dense_editor \
    --mixing_type mamba2 --d_model 640 \
    --corpus corpus/sample_full.jsonl --text_key text \
    --bf16 --int8 --int8_backend cuda \
    --batch_size 8 --grad_accum_steps 2 \
    --lr 1e-3 --schedule wsd --max_steps 300000

# 품질 벤치마크 (2000-step 오버핏)
python bench_quality.py --mixing_types mamba2 --d_model 640 \
    --corpus corpus/val_50k.jsonl --max_steps 2000

# CPU 인퍼런스 벤치마크 (Rust+C)
cd inference_dense
cargo build --release --features avx2-only
cargo run --release --features avx2-only -- --benchmark-full --seq-len 2048 --d-model 640
OMP_NUM_THREADS=8 cargo run --release --features avx2-only -- --benchmark-full --d-model 640

# ── Seq2Seq (레거시) ──

python -m training.pretrain \
  --size 256M --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --batch_size 1 --bf16 --int8 --int8_backend cuda

# ── 토크나이저 ──

python keyboard_tokenizer/ko_keyboard.py
python keyboard_tokenizer/keyboard_wrapper.py
python error_generation/test_errors.py

# ── 노이즈 엔진 검증 ──

# 1000문장 통계 검증 (오류 분포 + hit rate)
python error_generation/test_distribution.py \
    --corpus corpus/val_50k.jsonl --n_samples 1000
```

테스트 프레임워크(pytest 등) 없음. 모든 테스트는 `if __name__ == "__main__"` 블록으로 직접 실행.

## Architecture

### DenseEditor (`model/dense_editor.py`) — 현행 메인 모델

인코더-only 편집 태깅 모델. 입력 토큰마다 편집 태그(KEEP/DELETE/INSERT_x) 예측.

```
Embedding (vocab=303, d_model=640) × sqrt(d_model)
├── DenseEditorLayer × 15
│   ├── RMSNorm → BiMamba2Mixing → Dropout → (+residual)
│   └── RMSNorm → BitNetFFN(SwiGLU) → Dropout → (+residual)
├── Final RMSNorm
└── Tag Head (BitLinear: d_model → n_tags=608)
```

**핵심 컴포넌트:**

- **BiMamba2Mixing** (`model/mixing/bi_mamba2.py`): 양방향 Mamba-2 SSD
  - GPU: `mamba_ssm.Mamba2` fused CUDA kernel (chunk-parallel SSD)
  - CPU: Python sequential scan fallback
  - Document isolation: `reset_mask → cumsum → seq_idx` (BOS 위치에서 state 리셋)
  - 양방향: fwd + bwd(input flip) → element-wise addition
  - `bwd_reset[:, 0] = True` — flipped 시퀀스 시작의 seq_idx >= 0 보장
  - 설정: d_state=64, headdim=64, expand=2, ngroups=1, chunk_size=256
- **BitLinear** (`model/bitlinear.py`): 1.58-bit ternary weights + INT8 activation
- **BitNetFFN** (`model/encoder.py`): SwiGLU (gate_proj + up_proj → down_proj), d_ff = d_model × 8/3
- **DenseEditorConfig** (`model/dense_editor_config.py`): `make_config(mixing_type, d_model, target_params)`

**Mixing layer 레지스트리** (`model/mixing/__init__.py`):
mamba, mamba2, fnet, tcn, rwkv, retnet, xlstm, mlstm, attention, hybrid

**확정 아키텍처**: Mamba-2 ds=64 (loss 37%↓, recall +19.5pp vs Mamba-1, CPU 22% 빠름)

### DenseEditor 학습 (`training/pretrain_dense_editor.py`)

- LR 스케줄: `cosine` (기본) 또는 `wsd` (Warmup-Stable-Decay)
  - WSD: warmup → 80% stable at peak LR → 20% cosine decay
  - cosine: warmup → 전 구간 cosine decay
- Label smoothing: `--label_smoothing 0.1` (기본 활성)
- Edit loss weight: `--edit_loss_weight 2.0` (non-KEEP 태그 2배 가중치)
- 한국어 오류 증강: `--error_prob 0.5 --error_count 3 --noise_preset default|realistic`
- Min LR: `--min_lr_ratio 0.01` (max_lr의 1%)
- 패킹: `[BOS]문장1[EOS][BOS]문장2[EOS]...` → max_seq_len까지 연결, BOS에서 state 리셋
- Iterative refinement: `--n_iterations 1` (기본), fine-tuning 시 2-3
- 체크포인트: model + optimizer + data_state(noiser+dataset) + epoch_state 저장/복원

### EditorDataset (`training/editor_dataset.py`)

- JSONL 스트리밍 → 텍스트 노이즈 → 토크나이징 → Levenshtein 편집 태그
- 패킹 모드: 여러 문장을 `[BOS]...[EOS]` 단위로 연결, PAD 최소화
- DDP: rank별 line interleaving, state_dict로 학습 재개 지원
- C++ Levenshtein 확장: iterative refinement 배치 병렬 처리

### CPU 인퍼런스 (`inference_dense/`)

Rust + C 추론 엔진. BitNet 1.58-bit, AVX2 i8 sgemm.

- `src/mixing/mamba2.rs`: Mamba2Block + BiMamba2 (양방향)
- `c_kernels/mixing_kernels.c`: `mamba2_scan_avx2` (head-parallel, FMA), `causal_conv1d_avx2`
- 벤치마크: `inference_dense/BENCHMARK.md` (8종 아키텍처 비교, Mamba-2 ds=64 최종 확정)

### Seq2Seq 모델 (`model/seq2seq.py`) — 레거시

`BitMambaSeq2Seq` Encoder-Decoder. Mamba-1/2 + LinearCrossAttention + BitNetFFN.
Copy Gate (Trial B), Source-Aware Logit Bias (Trial A) 포함.

### Tokenizers

`BaseTokenizer` ABC → 5종 구현 (keyboard, nfd, char, bbpe, mecab_bbpe).
DenseEditor 기본: keyboard (vocab_size=303).

### INT8 CUDA 권장 설정

```bash
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
```

### Document Isolation (패킹 시 문서 간 정보 누출 방지)

- **Mamba-2 (DenseEditor)**: `reset_mask = (input_ids == bos_id)` → `cumsum - 1` → `seq_idx`로 네이티브 isolation
- **Mamba-1 (Seq2Seq)**: BOS 위치에서 dt=1e4로 SSM state 완전 리셋
- **Cross-Attention (Seq2Seq)**: per-document context matrix + CUDA scatter/gather

### 노이즈 엔진 (`error_generation/`, `training/noising.py`)

한국어 오류 생성 + 텍스트/토큰 레벨 노이즈. GEC 학습 데이터 증강.

- **error_generation**: 27개 오류 타입 (패턴 기반 + MeCab 동적 생성)
  - 패턴 기반: `common_misspellings`, `consonant_errors` 등 (고정 사전 매칭)
  - 동적 생성: `spacing_errors`, `conjugation_errors`, `particle_errors`, `suffix_errors`, `foreign_style` (MeCab 형태소 분석 기반 fallback)
  - G2PK 발음→철자: `g2pk_noise` (g2pk 패키지 필요, soft import)
  - 이형문자: `heterograph` (종성/모음 혼동)
  - KAGAS 매핑: `kagas_mapping` (11-type 표준 분류)
- **가중치 프리셋** (`training/noising.py: WEIGHT_PRESETS`):
  - `default`: 기존 가중치 (하위호환)
  - `realistic`: KoGEC 2025 실제 오류 분포 기반 (hit rate 보정 포함)
- **NoiseConfig**: `weight_preset` 필드로 프리셋 선택
- **CLI**: `--noise_preset default|realistic`

### ELECTRA GEC 실험 (`exp-electra-gec/`)

Pretrained encoder + Two-head GEC 편집 태깅 실험 (브랜치: `exp-electra-gec`).

- Phase 0: 노이즈 엔진 개선 (완료) — G2PK, heterograph, 가중치 리밸런싱, KAGAS 매핑
- Phase 1: KoELECTRA-Small-v3 + GECToR Two-head (계획)
- Phase 2: Keyboard 토크나이저 + ELECTRA RTD pretrain (계획)
- Two-head 태그 체계: Action(4-class) + Content(vocab-class) 분리
- 확신도 기반 편집 결정 + Gumbel consensus

## Key References

- `inference_dense/BENCHMARK.md`: CPU 인퍼런스 + GPU 품질 벤치마크 결과
- `AGENTS.md`: AI 어시스턴트용 상세 프로젝트 컨텍스트
- `training/noise_config.example.json`: 노이즈 설정 템플릿
- `exp-electra-gec/OVERVIEW.md`: ELECTRA GEC 실험 전체 개요
- Docker: `nvidia/cuda:12.8.0-devel-ubuntu24.04` 기반, Python 3.12, CUDA 12.8
