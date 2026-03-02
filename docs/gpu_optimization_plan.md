# GPU 활용률 & 처리량 최적화 최종 결과

> **작성일**: 2026-03-02  
> **모델**: 128M BitMamba Seq2Seq (Mamba-2 SSD)  
> **환경**: PyTorch 2.10.0+cu130, mamba_ssm 2.3.0

## 최적화 결과 요약

| Phase | 변경 사항 | tok/s (remote) | tok/s (local) | 비고 |
|-------|----------|:-------------:|:------------:|------|
| baseline | CUDA_LAUNCH_BLOCKING=1 | 28,000 | 5,000 | 초기 |
| ✅ Phase 0 | `CUDA_LAUNCH_BLOCKING=0` | 36,000 | 8,800 | +28% / +76% |
| ✅ Phase 1 | CUDA Prefetcher + non_blocking | ~39,000 | 9,585 | +9% |
| ✅ **Phase 2** | **Mamba-2 SSD** | **66,000** | **17,100** | **+69% / +78%** |

### 총 성능 향상: 리모트 2.36x (28k→66k), 로컬 3.42x (5k→17k)

---

## 시도했으나 효과 없었던 최적화

| 항목 | 결과 | 원인 |
|------|------|------|
| `--fused_ce` | 속도 개선 없음 | 메모리 절약만 됨 |
| CUDA Linear Attention 커널 | **0.37x** (더 느림) | cuBLAS가 d_head=64에서 이미 최적 |
| flash-linear-attention (fla) | **0.51~0.93x** (더 느림) | causal self-attention 전용, non-causal cross-attention에 부적합 |

---

## 적용 내역

### Phase 0: `CUDA_LAUNCH_BLOCKING=0`
- **문제**: 디버깅용 동기 실행 플래그가 모든 CUDA 커널을 순차 실행
- **해결**: 환경변수 제거 또는 0으로 설정
- **효과**: 커널 간 비동기 오버랩 복원, GPU idle time 제거

### Phase 1: CUDA Prefetcher
- **수정 파일**: `training/pretrain.py`
- **구현**: `CUDAPrefetcher` 클래스 — 별도 CUDA stream에서 다음 배치 미리 GPU 전송
- `non_blocking=True`로 CPU→GPU 전송 비동기화
- GPU 연산과 데이터 전송을 오버랩

### Phase 2: Mamba-2 SSD 전환
- **수정 파일**: 
  - `model/config.py` — Mamba-2 필드 추가 (mamba_version, headdim, ngroups, chunk_size)
  - `model/mamba2_block.py` — **신규** Mamba2 래퍼 블록
  - `model/encoder.py` — mamba_version 선택 지원
  - `model/decoder.py` — mamba_version 선택 지원
  - `model/seq2seq.py` — config 전달
  - `training/pretrain.py` — `--mamba_version 2` CLI 옵션 + d_state 자동 업그레이드
- **주요 변경:**
  - d_state: 16 → 128 (8x 모델 표현력 향상)
  - chunk_size=256 내 matmul 기반 병렬 scan (Tensor Core 활용)
  - fused 커널: conv1d + scan + norm + out_proj 한번에 처리
  - Document isolation: `reset_mask` → `seq_idx` (int32 cumsum) 네이티브 지원
  - Mamba-2 블록은 fused 커널 유지 (BitLinear 미적용), FFN만 BitLinear
- **파라미터 수**: 128M → ~156M (d_state 증가, fused 커널 내 파라미터 차이)

---

## 현재 학습 설정

```bash
BITLINEAR_CUDA_BACKWARD=fp32_tf32 \
BITLINEAR_CUDA_GRADW_LT=1 \
BITLINEAR_CUDA_FUSED_ACT=1 \
BITLINEAR_CUDA_FUSED_WEIGHT=1 \
CUDA_LAUNCH_BLOCKING=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python training/pretrain.py \
  --corpus corpus/sample_full.jsonl \
  --text_key text \
  --size 128M \
  --tokenizer keyboard \
  --mamba_version 2 \
  --grad_accum_steps 8 \
  --bf16 --int8 --int8_backend cuda \
  --pack_size 4096 --batch_size 4 \
  --val_corpus corpus/val_50k.jsonl \
  --val_every 538 \
  --save_dir checkpoints/run_v8 \
  --save_every 538 \
  --log_every 10 --num_workers 16 \
  --gdrive_remote "gdrive:base-model-2-ckpts/v8-128M" \
  --log_file training_run_v8-128M.log \
  --lr_schedule sgdr --max_steps 154795 \
  --cycle_steps 538 --lr_decay 0.985133 \
  --warmup_steps 1000 --min_lr 1e-05 \
  --noise_config noise_config.test1.json
```

- **effective batch**: 4 × 8 = 32 packs = 131K tokens/step
- **처리량**: ~66,000 tok/s (리모트)

### 주의사항
- `CUDA_LAUNCH_BLOCKING=1` 절대 사용 금지 (디버깅 전용, 성능 -43%)
- Mamba-2는 Mamba-1과 체크포인트 호환 불가 (scratch 학습 필요)
- `--mamba_version 2` 사용 시 d_state=128, headdim=64 자동 적용

---

## 추가 가능 최적화 (미시행)

| 항목 | 예상 효과 | 비고 |
|------|----------|------|
| `torch.compile(fullgraph=False)` | +15~30% | INT8 BitLinear와 부분 호환 |
| Multi-GPU (DDP) | GPU 수 비례 | 코드 이미 지원됨 |
| batch_size=8 grad_accum=4 | ±0~5% | 같은 effective batch, GPU 병렬성↑ |
