#!/bin/bash
# CompositionCodec fine-tune — warm-start 로 기존 체크포인트 가중치 재사용.
# 기본: 순수 가변 (fixed_slot=F, append_pad_slot=F) 로 5k step sanity run.
#
# INIT_FROM: 가중치만 로드, step 0 재시작, optimizer/scheduler 새로 생성.
#            첫 fine-tune 런에 사용.
# RESUME   : 모델+optimizer+scheduler+step+data_state 전체 복원. 이전 fine-tune
#            런을 이어서 MAX_STEPS 확장 시 사용. cosine lambda 가 새 MAX_STEPS
#            기준으로 재계산돼 SGDR 비슷한 부분 warm-restart 효과.
# 둘 중 하나는 필수 (상호 배타). RESUME 지정 시 INIT_FROM 무시.
#
# 예) 최소 호출 (첫 fine-tune, 4 GPU):
#   INIT_FROM=exp-jamo-codec/checkpoints/composition_5L_step40000.pt \
#   NGPU=4 bash exp-jamo-codec/run_composition_finetune.sh
#
# 예) 이어서 5k 확장 (resume):
#   RESUME=exp-jamo-codec/checkpoints_ft_variable/composition_5L_step25000.pt \
#   MAX_STEPS=30000 \
#   NGPU=4 bash exp-jamo-codec/run_composition_finetune.sh
#
# 예) GDRIVE 백업 포함:
#   INIT_FROM=.../composition_5L_step40000.pt \
#   GDRIVE=gdrive:base-model-2-ckpts/composition-5L-k8-ft-variable \
#   NGPU=4 bash exp-jamo-codec/run_composition_finetune.sh
#
# 아키텍처 파라미터 (D_MODEL/N_LAYERS/KERNEL/MAX_JAMO_PER_TOKEN/SEG_MASKED)는
# warm-start 호환을 위해 체크포인트와 동일해야 함. 바꿀 경우 가중치 strict=False 로
# 일부 드롭돼 warm-start 의미가 약해짐.

set -e
export PYTHONUNBUFFERED=1

# jemalloc preload (run_composition_train.sh 와 동일 목적)
if [ -z "${NO_JEMALLOC}" ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "LD_PRELOAD: jemalloc enabled"
    else
        echo "WARNING: libjemalloc.so.2 없음, glibc malloc으로 진행"
    fi
fi

# ── INIT_FROM / RESUME 상호 배타 체크 ──
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"
if [ -z "${INIT_FROM}" ] && [ -z "${RESUME}" ]; then
    echo "ERROR: INIT_FROM 또는 RESUME 둘 중 하나는 필수"
    echo "  INIT_FROM=.../ckpt.pt  — 가중치만 로드, step 0 재시작 (첫 fine-tune)"
    echo "  RESUME=.../ckpt.pt     — 전체 상태 복원, step 이어받음 (fine-tune 확장)"
    exit 1
fi
if [ -n "${RESUME}" ] && [ -n "${INIT_FROM}" ]; then
    echo "WARNING: RESUME 과 INIT_FROM 모두 지정됨 — RESUME 우선 사용, INIT_FROM 무시"
    INIT_FROM=""
fi
if [ -n "${INIT_FROM}" ] && [ ! -f "${INIT_FROM}" ]; then
    echo "ERROR: INIT_FROM 파일이 존재하지 않음: ${INIT_FROM}"
    exit 1
fi
if [ -n "${RESUME}" ] && [ ! -f "${RESUME}" ]; then
    echo "ERROR: RESUME 파일이 존재하지 않음: ${RESUME}"
    exit 1
fi

# ── 데이터 ──
CORPUS="${CORPUS:-corpus/k-exaone_coverage_100.parquet}"
VAL_CORPUS="${VAL_CORPUS:-corpus/k-exaone_coverage_5_len1000.parquet}"
TEXT_KEY="${TEXT_KEY:-text}"

# ── 학습 예산 (fine-tune 기본값은 sanity 런 기준) ──
MAX_STEPS="${MAX_STEPS:-5000}"
WARMUP="${WARMUP:-500}"
LR="${LR:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"

# ── 아키텍처 (체크포인트 호환 필수) ──
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-5}"
KERNEL="${KERNEL:-8}"
MAX_JAMO_PER_TOKEN="${MAX_JAMO_PER_TOKEN:-32}"
# SEG_MASKED 는 항상 ON (원 체크포인트 아키텍처) — 빈 값 아니면 ON
SEG_MASKED="${SEG_MASKED:-1}"

# ── 슬롯 모드 (기본 순수 가변) ──
# FIXED_SLOT / PAD_SLOT 은 기본 unset → 순수 가변
# 다른 모드를 원하면 명시적으로 FIXED_SLOT=1 또는 PAD_SLOT=1 지정

# ── Decoder (체크포인트는 conv decoder 사용) ──
# PARALLEL_DECODER 는 기본 OFF (원 체크포인트와 동일)

# ── 배치 / 시퀀스 ──
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SEQ_LEN="${SEQ_LEN:-4096}"

# ── 인프라 ──
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"
NGPU="${NGPU:-4}"

# ── 출력 경로 ──
OUT="${OUT:-exp-jamo-codec/checkpoints_ft_variable}"
GDRIVE="${GDRIVE:-}"
LOG_PATH="${LOG_PATH:-exp-jamo-codec/composition_finetune_log.txt}"

# ── 플래그 조립 ──
SEG_MASKED_FLAG=""
[ -n "${SEG_MASKED}" ] && SEG_MASKED_FLAG="--segment_masked"
PAD_SLOT_FLAG=""
[ -n "${PAD_SLOT}" ] && PAD_SLOT_FLAG="--append_pad_slot"
FIXED_SLOT_FLAG=""
[ -n "${FIXED_SLOT}" ] && FIXED_SLOT_FLAG="--fixed_slot"
PARALLEL_DEC_FLAG=""
[ -n "${PARALLEL_DECODER}" ] && PARALLEL_DEC_FLAG="--parallel_decoder"
SLOT_DECODE_FLAG=""
[ -n "${SLOT_DECODE}" ] && SLOT_DECODE_FLAG="--slot_decode"
DEC_LAYERS_FLAG=""
[ -n "${DECODER_LAYERS}" ] && DEC_LAYERS_FLAG="--decoder_layers ${DECODER_LAYERS}"
DEC_HEADS_FLAG=""
[ -n "${DECODER_HEADS}" ] && DEC_HEADS_FLAG="--decoder_heads ${DECODER_HEADS}"
NO_PIN_FLAG=""
[ -n "${NO_PIN_MEMORY}" ] && NO_PIN_FLAG="--no_pin_memory"

# ── VICReg 플래그 (선택) ──
# VICREG_VAR, VICREG_COV 설정 시 활성. VICREG_TARGET 은 z|h_dec|both (기본 z)
VICREG_FLAGS=""
if [ -n "${VICREG_VAR}" ] || [ -n "${VICREG_COV}" ]; then
    VICREG_FLAGS="--vicreg_var ${VICREG_VAR:-0} --vicreg_cov ${VICREG_COV:-0}"
    [ -n "${VICREG_TARGET}" ] && VICREG_FLAGS="${VICREG_FLAGS} --vicreg_target ${VICREG_TARGET}"
    [ -n "${VICREG_WARMUP}" ] && VICREG_FLAGS="${VICREG_FLAGS} --vicreg_warmup ${VICREG_WARMUP}"
fi

echo "=== CompositionCodec fine-tune ==="
if [ -n "${RESUME}" ]; then
    echo "Resume from: ${RESUME} (전체 상태 복원)"
else
    echo "Init from: ${INIT_FROM} (가중치만, step 0 재시작)"
fi
echo "Corpus: ${CORPUS}"
echo "Val corpus: ${VAL_CORPUS}"
echo "Arch: d=${D_MODEL}, L=${N_LAYERS}, k=${KERNEL}, max_jamo_per_token=${MAX_JAMO_PER_TOKEN}"
echo "Mode: segment_masked=${SEG_MASKED:-OFF} fixed_slot=${FIXED_SLOT:-OFF} pad_slot=${PAD_SLOT:-OFF}"
echo "Steps: ${MAX_STEPS}, LR: ${LR}, Warmup: ${WARMUP}"
echo "Batch: ${BATCH_SIZE}×${NGPU}gpu×accum${GRAD_ACCUM}=$((BATCH_SIZE * NGPU * GRAD_ACCUM))"
echo "Save every: ${SAVE_EVERY}, Val every: ${VAL_EVERY}"
echo "Out: ${OUT}"
echo "Log: ${LOG_PATH}"
echo "GDrive: ${GDRIVE:-none}"
echo ""

# 출력 디렉토리 준비
mkdir -p "${OUT}"

torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_composition.py \
    --corpus ${CORPUS} --text_key ${TEXT_KEY} \
    --d_model ${D_MODEL} --n_layers ${N_LAYERS} --kernel_size ${KERNEL} \
    --max_seq_len ${SEQ_LEN} \
    --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} --max_steps ${MAX_STEPS} \
    --lr ${LR} --warmup_steps ${WARMUP} --max_grad_norm ${MAX_GRAD_NORM} \
    --bf16 ${COMPILE_FLAG} --num_workers ${NUM_WORKERS} \
    --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
    --val_corpus ${VAL_CORPUS} --val_every ${VAL_EVERY} --val_samples ${VAL_SAMPLES} \
    --out_dir ${OUT} \
    --max_jamo_per_token ${MAX_JAMO_PER_TOKEN} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    ${SEG_MASKED_FLAG} ${PAD_SLOT_FLAG} ${FIXED_SLOT_FLAG} \
    ${PARALLEL_DEC_FLAG} ${SLOT_DECODE_FLAG} \
    ${DEC_LAYERS_FLAG} ${DEC_HEADS_FLAG} ${NO_PIN_FLAG} \
    ${VICREG_FLAGS} \
    ${RESUME:+--resume ${RESUME}} \
    ${INIT_FROM:+--init_from ${INIT_FROM}} \
    2>&1 | tee "${LOG_PATH}"

# rclone 로그 업로드
if [ -n "${GDRIVE}" ]; then
    echo "=== 최종 로그 업로드 ==="
    rclone copy "${LOG_PATH}" "${GDRIVE}/" 2>/dev/null && \
        echo "로그 업로드 완료" || echo "로그 업로드 실패"
fi
