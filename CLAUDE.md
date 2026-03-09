# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

한국어 문법 오류 교정(GEC)을 위한 **BitNet-Mamba Encoder-Decoder (Seq2Seq)** 프로젝트. 8M~1B 파라미터 스케일 지원, 5종 한국어 토크나이저 탑재.

- 코드 주석/docstring은 **한국어**, 식별자는 영어
- 항상 한국어로 응답할 것

## Commands

```bash
source .venv/bin/activate

# 모델 검증 (forward/backward, param count, config JSON roundtrip)
python verify_model.py

# 토크나이저 테스트 (inline __main__ 블록 — pytest 없음)
python keyboard_tokenizer/ko_keyboard.py
python keyboard_tokenizer/keyboard_wrapper.py
python nfd_tokenizer/make_tokenizer.py

# 에러 생성 테스트
python error_generation/test_errors.py

# 학습
python -m training.pretrain \
  --size 256M --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --batch_size 1 --bf16 \
  --int8 --int8_backend cuda

# 벤치마크
python bench_cuda_ablation.py --size 32M --max_steps 30 --batch_sizes 1 2

# 토크나이저 빌드 (결정적)
python nfd_tokenizer/make_tokenizer.py
python keyboard_tokenizer/make_tokenizer.py
python char_tokenizer/make_tokenizer.py

# 토크나이저 학습 (코퍼스 기반)
python -m bbpe_tokenizer.train_tokenizer -i corpus/sample_10g.jsonl --text_key text
python -m mecab_bbpe_tokenizer.train_tokenizer -i corpus/sample_10g.jsonl --text_key text
```

테스트 프레임워크(pytest 등) 없음. 모든 테스트는 `if __name__ == "__main__"` 블록으로 직접 실행.

## Architecture

### Model (`model/`)

`BitMambaSeq2Seq` (seq2seq.py) — Encoder-Decoder 전체 모델:

- **Encoder** (encoder.py): `EncoderLayer` × N (기본 6). 레이어 구조: `Mamba → (+residual) → RMSNorm → BitNetFFN(SwiGLU) → (+residual) → RMSNorm`
- **Decoder** (decoder.py): `DecoderLayer` × N (기본 10). 레이어 구조: `Mamba → RMSNorm → LinearCrossAttention → RMSNorm → BitNetFFN → RMSNorm` (각각 residual connection)
- **LinearCrossAttention** (linear_attention.py): O(N) Grouped-Query Linear Attention. φ(x) = relu(x) + 1 feature map (≥1 보장으로 denominator 폭발 방지). Document isolation을 위한 CUDA scatter/gather 커널 탑재
- **BitLinear** (bitlinear.py): 1.58-bit ternary weights {-1, 0, +1} + 8-bit activation quantization. STE(Straight-Through Estimator)로 gradient 전파. INT8 backends: Triton (`triton_bitlinear.py`), CUDA (`cuda_bitlinear.py`)
- **MambaBlock** (mamba_block.py): Mamba-1 selective scan. `mamba_ssm` CUDA 커널 자동 감지. Document isolation: BOS 위치에서 dt=1e4로 SSM state 완전 리셋
- **Mamba2Block** (mamba2_block.py): Mamba-2 SSD chunk-parallel. `mamba_ssm.Mamba2` 래핑. reset_mask → `seq_idx` 변환으로 네이티브 document isolation
- **Config** (config.py): `BitMambaSeq2SeqConfig` dataclass. `mamba_version` (1 or 2), `use_copy_gate` (Trial B), `n_kv_heads` (GQA/MQA) 등
- **Copy Gate** (seq2seq.py): Trial B — decoder logit과 source unigram 분포를 gate로 혼합. Gate collapse 방지: `gate = 0.5 + 0.5 * sigmoid(...)` (생성 분포 최소 50% 보장)
- **Source-Aware Logit Bias** (seq2seq.py): Trial A — src_weights 기반으로 원문 토큰에 logit bias 가산

모델 사이즈 프리셋 (`training/pretrain.py`의 `MODEL_CONFIGS`): 8M, 16M, 32M, 64M, 128M, 256M, 512M, 1B

### Tokenizers

`BaseTokenizer` ABC (`tokenizer_base.py`) → 5종 구현. 모델 코드는 `BaseTokenizer`에만 의존.

| 토크나이저 | wrapper | 특징 |
|---|---|---|
| keyboard | `keyboard_tokenizer/keyboard_wrapper.py` | 한국어 2벌식 키스트로크 시퀀스 |
| nfd | `nfd_tokenizer/tokenizer_wrapper.py` | NFD 분해 + ByteLevel BPE + Hanja 전처리 |
| char | `char_tokenizer/char_wrapper.py` | 문자 단위 |
| bbpe | `bbpe_tokenizer/bbpe_wrapper.py` | ByteLevel BPE (HuggingFace tokenizers) |
| mecab_bbpe | `mecab_bbpe_tokenizer/mecab_bbpe_wrapper.py` | MeCab 형태소 분석 + BPE |

**불변 규칙:**
- `[PAD]` 토큰은 항상 ID 0 (`BitMambaSeq2SeqConfig.pad_id` 기본값)
- `decode(encode(text)) ≈ text` roundtrip 보존 필수
- Unicode 처리 주의: NFC/NFD 정규화, Hanja→Hangul 변환, 자모 분해/합성

### Training (`training/`)

- `pretrain.py`: 메인 학습 스크립트. `--size`, `--tokenizer`, `--corpus`, `--bf16` (필수 — BitLinear scaler overflow 방지), `--int8 --int8_backend {triton,cuda}`, `--fused_ce` (liger-kernel), `--compile` (non-INT8만), `--grad_ckpt`, DDP 지원
- `dataset.py`: `StreamingPackedDataset` — JSONL 스트리밍, noising 적용, 다중 문장을 `[BOS]...[EOS][BOS]...[EOS]` 형태로 `pack_size` 토큰까지 패킹. PAD 없음. state_dict/load_state_dict로 학습 재개 지원
- `noising.py`: `DenoisingNoiser` + `NoiseConfig`. 2단계 노이즈:
  1. **텍스트 레벨** (토큰화 전): Korean error injection, spacing noise, keyboard typo (유클리드 좌표 기반), n-gram shuffle, word reorder
  2. **토큰 레벨** (토큰화 후): SequenceMatcher diff → token masking (~15%), deletion (~5%), text infilling (Poisson λ=3)
- Metrics: BPC (Bits Per Character, 토크나이저 간 비교용), CER (Character Error Rate, 검증용)

### Error Generation (`error_generation/`)

`KoreanErrorGenerator` — 24종 한국어 오류 유형을 가중치 기반 랜덤 선택으로 주입. 주요: 띄어쓰기(20%), 구두점(10%), 수치, 삭제/첨가, 맞춤법, 모음혼동, 발음, 외래어, 어순, 시제, 의미론적 오류 등.

### CUDA Kernels (`model/`)

JIT 컴파일 (torch.utils.cpp_extension.load). DDP 멀티프로세스 대응 (rank-0 빌드 → barrier → 나머지 로드).

- `cuda_bitlinear_kernel.cu` / `cuda_bitlinear_ext.cpp`: INT8 quantization + matmul + grad weight (dp4a, cublasLt)
- `cuda_doc_linear_attn_kernel.cu` / `cuda_doc_linear_attn_ext.cpp`: Document-isolated linear attention. V3 fused scatter/gather — shared memory에서 context 처리, global memory 트래픽 제거
- `cuda_linear_attention_kernel.cu` / `cuda_linear_attention_ext.cpp`: Non-doc-isolated linear attention fused kernel

소스 해시 기반 JIT 캐시 — 소스 변경 시 자동 재빌드. Stale cache 시 `rm -rf ~/.cache/torch_extensions/` 후 재시도.

### INT8 CUDA 권장 설정

```bash
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
```

- `--int8 --compile` 조합은 설계상 스킵 (custom autograd path)
- `grad_ckpt`: 메모리 절감 크지만 처리량(tok/s) 감소 유의미
- 안정 배치 범위: 1~2; batch 4+는 강한 GPU 필요

### Document Isolation

배치 내 여러 문서를 packed sequence로 처리할 때 문서 간 정보 누출 방지:
- **Mamba**: BOS 위치에서 x_branch/z 제로링 + dt=1e4로 SSM state 완전 리셋 (exp(A*dt)≈0)
- **Mamba-2**: reset_mask → cumsum → `seq_idx`로 네이티브 document isolation
- **Cross-Attention**: src_doc_ids/tgt_doc_ids 기반 per-document context matrix. CUDA scatter(소스→context)/gather(context→타겟) 구조
- `max_docs`는 seq2seq.py에서 1회만 계산하여 `.item()` GPU sync 최소화 (기존 24회→1회)

## Key References

- `AGENTS.md`: AI 어시스턴트용 상세 프로젝트 컨텍스트
- `docs/experiment_handoff_2026-02-23.md`: 최신 실험 결과 및 INT8 가이드
- `training/noise_config.example.json`: 노이즈 설정 템플릿
- Docker: `nvidia/cuda:12.8.0-devel-ubuntu24.04` 기반, Python 3.12, CUDA 12.8
