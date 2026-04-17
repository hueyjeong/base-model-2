#!/bin/bash
# SimpleCodec 학습 — per-token encoder + parallel slot decoder
#
# 예) 로컬 1GPU sanity:
#   NGPU=1 BATCH_SIZE=256 MAX_STEPS=1000 \
#     bash exp-jamo-codec/run_simple.sh
#
# 예) DDP 4GPU 본학습 + GDrive 업로드:
#   NGPU=4 BATCH_SIZE=512 MAX_STEPS=20000 \
#     GDRIVE=gdrive:base-model-2-ckpts/simple-codec \
#     bash exp-jamo-codec/run_simple.sh
#
# 매 체크포인트 저장 시 GDrive 로 ckpt + log 업로드 (백그라운드 스레드).
# 이전 체크포인트는 자동 삭제 (keep_latest_n=1).

set -e
export PYTHONUNBUFFERED=1

if [ -z "${NO_JEMALLOC}" ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "LD_PRELOAD: jemalloc enabled"
    fi
fi

# 데이터
CORPUS="${CORPUS:-corpus/k-exaone_random_coverage_1000_len4096.parquet}"
VAL_CORPUS="${VAL_CORPUS:-corpus/k-exaone_coverage_5_len1000.parquet}"
TEXT_KEY="${TEXT_KEY:-text}"

# 모델
D_MODEL="${D_MODEL:-256}"
N_ENC_LAYERS="${N_ENC_LAYERS:-5}"
N_DEC_LAYERS="${N_DEC_LAYERS:-5}"
KERNEL="${KERNEL:-5}"
MAX_JAMO="${MAX_JAMO:-32}"
DROPOUT="${DROPOUT:-0.1}"

# 학습
BATCH_SIZE="${BATCH_SIZE:-512}"   # 배치당 토큰 수 (문서가 아님)
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-20000}"
WARMUP="${WARMUP:-1000}"
LR="${LR:-3e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 인프라
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"
BF16_FLAG="${BF16_FLAG:---bf16}"
NGPU="${NGPU:-1}"

# 로깅/저장
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-2500}"
VAL_EVERY="${VAL_EVERY:-500}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"
OUT="${OUT:-exp-jamo-codec/checkpoints_simple}"
LOG_PATH="${LOG_PATH:-exp-jamo-codec/simple_train_log.txt}"

# GDrive (옵션 — 지정 시 매 save 마다 + 최종에 업로드)
GDRIVE="${GDRIVE:-}"
export GDRIVE      # train_simple.py 가 os.environ 으로 접근
export LOG_PATH    # 동일

# Resume / init_from
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"

echo "=== SimpleCodec 학습 ==="
echo "Corpus: ${CORPUS}"
echo "Val:    ${VAL_CORPUS}"
echo "Arch: d=${D_MODEL}, enc_L=${N_ENC_LAYERS}, dec_L=${N_DEC_LAYERS}, k=${KERNEL}, max_jamo=${MAX_JAMO}"
echo "Batch: ${BATCH_SIZE} tokens × ${NGPU}gpu × accum${GRAD_ACCUM}"
echo "Steps: ${MAX_STEPS}, warmup=${WARMUP}, LR=${LR}"
echo "Out: ${OUT}"
echo "Log: ${LOG_PATH}"
echo "GDrive: ${GDRIVE:-없음}"
[ -n "${INIT_FROM}" ] && echo "Init from: ${INIT_FROM}"
[ -n "${RESUME}" ] && echo "Resume: ${RESUME}"
echo ""

mkdir -p "${OUT}"

torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_simple.py \
    --corpus ${CORPUS} --text_key ${TEXT_KEY} \
    --val_corpus ${VAL_CORPUS} \
    --d_model ${D_MODEL} --n_enc_layers ${N_ENC_LAYERS} --n_dec_layers ${N_DEC_LAYERS} \
    --kernel_size ${KERNEL} --max_jamo ${MAX_JAMO} --dropout ${DROPOUT} \
    --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} \
    --max_steps ${MAX_STEPS} --warmup_steps ${WARMUP} \
    --lr ${LR} --max_grad_norm ${MAX_GRAD_NORM} \
    --num_workers ${NUM_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} \
    ${BF16_FLAG} ${COMPILE_FLAG} \
    --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
    --val_every ${VAL_EVERY} --val_samples ${VAL_SAMPLES} \
    --out_dir ${OUT} \
    ${RESUME:+--resume ${RESUME}} \
    ${INIT_FROM:+--init_from ${INIT_FROM}} \
    2>&1 | tee "${LOG_PATH}"
