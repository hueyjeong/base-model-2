# GPU 활용률 & 처리량 최적화 결과

> **작성일**: 2026-03-02  
> **모델**: 128M BitMamba Seq2Seq  
> **환경**: PyTorch 2.10.0+cu130, mamba_ssm 2.3.0, RTX 5060 Ti (로컬)

## 최적화 결과 요약

| Phase | 변경 사항 | tok/s (local) | 향상 | 누적 배율 |
|-------|----------|:------------:|:----:|:--------:|
| baseline | CUDA_LAUNCH_BLOCKING=1 | 5,000 | — | 1.0x |
| ✅ Phase 0 | `CUDA_LAUNCH_BLOCKING=0` | 8,800 | **+76%** | 1.76x |
| ✅ Phase 1 | CUDA Prefetcher + non_blocking | 9,585 | **+9%** | 1.92x |
| ✅ **Phase 2** | **Mamba-2 SSD** | **17,100** | **+78%** | **3.42x** |
| ❌ Phase 3 | Linear Cross-Attention CUDA 커널 | — | -63% (비활성화) | — |

### 총 성능 향상: **3.42배** (5,000 → 17,100 tok/s)

---

## 완료된 최적화

### ✅ Phase 0: `CUDA_LAUNCH_BLOCKING=0`
- 디버깅용 동기 실행 플래그 제거
- 수백 개의 CUDA 커널이 비동기 실행되어 GPU idle time 제거
- **효과: +76%**

### ✅ Phase 1: CUDA Prefetcher + `non_blocking=True`
- `CUDAPrefetcher` 클래스: 별도 CUDA stream에서 다음 배치 미리 GPU 전송
- 데이터 전송과 GPU 연산을 오버랩
- **수정 파일**: `training/pretrain.py`
- **효과: +9%**

### ✅ Phase 2: Mamba-2 SSD 전환
- Mamba-1 (selective scan) → Mamba-2 (SSD chunk-parallel scan)
- `mamba_ssm 2.3.0`의 네이티브 `Mamba2` 모듈 래핑 (`model/mamba2_block.py`)
- Encoder/Decoder에서 `--mamba_version 2` CLI 옵션으로 선택
- **핵심 변경점:**
  - d_state: 16 → 128 (8배 더 큰 모델 표현력)
  - chunk_size=256 내 matmul 기반 병렬 scan
  - fused 커널: conv1d + scan + norm + out_proj 한번에 처리
  - Document isolation: `reset_mask` → `seq_idx` (int32 cumsum) 네이티브 지원
- **파라미터 수**: ~128M → ~156M (d_state 증가 때문)
- **수정 파일**: `model/config.py`, `model/mamba2_block.py` (신규), 
  `model/encoder.py`, `model/decoder.py`, `model/seq2seq.py`, `training/pretrain.py`
- **효과: +78%**

### ❌ Phase 3: Linear Cross-Attention CUDA 커널
- CUDA forward 커널 구현 완료 (`model/cuda_linear_attention_kernel.cu`)
- **결과: PyTorch cuBLAS보다 2.7배 느림** (1.02ms vs 0.38ms)
- **원인:**
  - d_head=64로 context matrix가 매우 작아 cuBLAS가 이미 최적
  - 블록당 64 threads만 사용 → GPU SM underutilize
  - BF16 → FP32 → BF16 변환 오버헤드
- **결론**: 기본 비활성화 (`LINEAR_ATTN_CUDA=0`). PyTorch 구현 유지.
  CUDA 커널은 코드에 남겨두되 사용하지 않음.

---

## 사용법

### Mamba-2 SSD로 학습

```bash
BITLINEAR_CUDA_BACKWARD=fp32_tf32 \
BITLINEAR_CUDA_GRADW_LT=1 \
BITLINEAR_CUDA_FUSED_ACT=1 \
BITLINEAR_CUDA_FUSED_WEIGHT=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python training/pretrain.py \
  --corpus corpus/sample_full.jsonl \
  --text_key text \
  --size 128M \
  --tokenizer keyboard \
  --mamba_version 2 \
  --grad_accum_steps 8 \
  --bf16 --int8 --int8_backend cuda \
  --pack_size 4096 --batch_size 1 \
  --noise_config noise_config.test1.json \
  --num_workers 16 \
  --lr_schedule sgdr --max_steps 619179 \
  --cycle_steps 2164 --lr_decay 0.985059 \
  --warmup_steps 1000 --min_lr 1e-05
```

### 주의사항
- `CUDA_LAUNCH_BLOCKING=1`은 **절대 사용하지 말 것** (디버깅 전용)
- Mamba-2는 Mamba-1과 체크포인트 호환 불가 (scratch 학습 필요)
- `--mamba_version 2` 사용 시 d_state=128, headdim=64 자동 적용

---

## Mamba-2 모델 크기별 파라미터 수 (참고)

| 모델 이름 | Mamba-1 params | Mamba-2 params | 비고 |
|-----------|:-----------:|:-----------:|------|
| 8M | 10.14M | — | — |
| 32M | — | — | 측정 필요 |
| 64M | — | — | 측정 필요 |
| 128M | ~128M | ~156M | d_state 16→128 |
| 256M | — | — | 측정 필요 |

## 추가 가능 최적화 (미시행)

| 항목 | 예상 효과 | 비고 |
|------|----------|------|
| `torch.compile(fullgraph=False)` | +15~30% | INT8 BitLinear와 부분 호환 |
| Flash Linear Attention (`fla`) | 미지수 | pip install 필요, 검증된 Triton 커널 |
| Multi-GPU (DDP) | GPU 수 비례 | 코드 이미 지원됨 |
