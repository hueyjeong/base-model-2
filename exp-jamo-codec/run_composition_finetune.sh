#!/bin/bash
# CompositionCodec fine-tune — warm-start 로 기존 체크포인트 가중치 재사용.
# 기본: 순수 가변 (fixed_slot=F, append_pad_slot=F) 로 5k step sanity run.
#
# 예) 최소 호출 (4 GPU, 기본값):
#   INIT_FROM=exp-jamo-codec/checkpoints/composition_5L_step40000.pt \
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

# ── INIT_FROM 필수 체크 ──
if [ -z "${INIT_FROM}" ]; then
    echo "ERROR: INIT_FROM 환경변수 필수 (warm-start 체크포인트 경로)"
    echo "  예: INIT_FROM=exp-jamo-codec/checkpoints/composition_5L_step40000.pt"
    exit 1
fi
if [ ! -f "${INIT_FROM}" ]; then
    echo "ERROR: INIT_FROM 파일이 존재하지 않음: ${INIT_FROM}"
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
DEC_LAYERS_FLAG=""
[ -n "${DECODER_LAYERS}" ] && DEC_LAYERS_FLAG="--decoder_layers ${DECODER_LAYERS}"
DEC_HEADS_FLAG=""
[ -n "${DECODER_HEADS}" ] && DEC_HEADS_FLAG="--decoder_heads ${DECODER_HEADS}"
NO_PIN_FLAG=""
[ -n "${NO_PIN_MEMORY}" ] && NO_PIN_FLAG="--no_pin_memory"

echo "=== CompositionCodec fine-tune ==="
echo "Init from: ${INIT_FROM}"
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
    --lr ${LR} --warmup_steps ${WARMUP} \
    --bf16 ${COMPILE_FLAG} --num_workers ${NUM_WORKERS} \
    --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
    --val_corpus ${VAL_CORPUS} --val_every ${VAL_EVERY} --val_samples ${VAL_SAMPLES} \
    --out_dir ${OUT} \
    --max_jamo_per_token ${MAX_JAMO_PER_TOKEN} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    ${SEG_MASKED_FLAG} ${PAD_SLOT_FLAG} ${FIXED_SLOT_FLAG} \
    ${PARALLEL_DEC_FLAG} ${DEC_LAYERS_FLAG} ${DEC_HEADS_FLAG} ${NO_PIN_FLAG} \
    --init_from ${INIT_FROM} \
    2>&1 | tee "${LOG_PATH}"

# rclone 로그 업로드
if [ -n "${GDRIVE}" ]; then
    echo "=== 최종 로그 업로드 ==="
    rclone copy "${LOG_PATH}" "${GDRIVE}/" 2>/dev/null && \
        echo "로그 업로드 완료" || echo "로그 업로드 실패"
fi
