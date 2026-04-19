#!/bin/bash
# Binary ELECTRA 사전학습
#
# 기본값 = 변종 C (BBPE 35K + k=32 hybrid + 16L, jamo-codec-v3 corpus).
# 변종 A (pure binary, K-EXAONE 153K) 재현 시:
#   VOCAB_SIZE=153600 BBPE_BITS=18 EMBED_K=0 GEN_LAYERS=14 DISC_LAYERS=14 \
#   TOKENIZER_PATH=LGAI-EXAONE/K-EXAONE-236B-A23B \
#   OUT=exp-jamo-codec/koelectra/checkpoints_a_small \
#   bash exp-jamo-codec/koelectra/run_binary.sh
#
# 예) 로컬 1GPU sanity (변종 C):
#   NGPU=1 BATCH=16 MAX_STEPS=200 WARMUP=20 VAL_EVERY=100 SAVE_EVERY=100 \
#     bash exp-jamo-codec/koelectra/run_binary.sh
#
# 예) DDP 4GPU 본학습 + GDrive 업로드:
#   NGPU=4 GDRIVE=gdrive:exp-jamo-codec-binary/c_small/ \
#     bash exp-jamo-codec/koelectra/run_binary.sh

set -e
export PYTHONUNBUFFERED=1

if [ -z "${NO_JEMALLOC}" ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "LD_PRELOAD: jemalloc enabled"
    fi
fi

# Model — 기본값은 variant C (BBPE 35K + k=32 + 16L)
# Variant A 재현 시 ENV 로 override:
#   VOCAB_SIZE=153600 BBPE_BITS=18 EMBED_K=0 GEN_LAYERS=14 DISC_LAYERS=14 \
#   TOKENIZER_PATH=LGAI-EXAONE/K-EXAONE-236B-A23B
VOCAB_SIZE="${VOCAB_SIZE:-35000}"
BBPE_BITS="${BBPE_BITS:-16}"
EMBED_K="${EMBED_K:-32}"        # 0 = variant A (pure binary), >0 = variant C (hybrid)
MAX_PATCHES="${MAX_PATCHES:-512}"
EMBED="${EMBED:-128}"
HIDDEN="${HIDDEN:-256}"
NHEADS="${NHEADS:-4}"
DFF="${DFF:-1024}"
GEN_LAYERS="${GEN_LAYERS:-16}"
DISC_LAYERS="${DISC_LAYERS:-16}"
DROPOUT="${DROPOUT:-0.1}"
MASK_RATIO="${MASK_RATIO:-0.20}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-50.0}"

# 데이터
TRAIN_PARQUET="${TRAIN_PARQUET:-corpus/jamo-codec-v3/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-corpus/jamo-codec-v3/val.parquet}"
TEXT_KEY="${TEXT_KEY:-text}"
MIN_LENGTH="${MIN_LENGTH:-10}"
TOKENIZER_PATH="${TOKENIZER_PATH:-checkpoints/bbpe_35k}"

# 학습
BATCH="${BATCH:-128}"
VAL_BATCH="${VAL_BATCH:-64}"
ACCUM="${ACCUM:-1}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-0.0}"
WARMUP="${WARMUP:-10000}"
MAX_STEPS="${MAX_STEPS:-800000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# 인프라
NUM_WORKERS="${NUM_WORKERS:-4}"
NGPU="${NGPU:-4}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"
BF16_FLAG="${BF16_FLAG:---bf16}"

# 로깅/저장
LOG_EVERY="${LOG_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
VAL_EVERY="${VAL_EVERY:-5000}"
VAL_BATCHES="${VAL_BATCHES:-500}"
OUT="${OUT:-exp-jamo-codec/koelectra/checkpoints_c_small}"
LOG_PATH="${LOG_PATH:-exp-jamo-codec/koelectra/c_small_train_log.txt}"

# GDrive
GDRIVE="${GDRIVE:-}"
KEEP_LATEST_N="${KEEP_LATEST_N:-3}"

# Resume
RESUME="${RESUME:-}"

echo "=== Binary ELECTRA 학습 ==="
echo "Variant: k=${EMBED_K} ($([ "${EMBED_K}" = "0" ] && echo 'A pure binary' || echo "C hybrid"))"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Vocab: ${VOCAB_SIZE}, bits=${BBPE_BITS}, P=${MAX_PATCHES}"
echo "Model: embed=${EMBED}, hidden=${HIDDEN}, gen_L=${GEN_LAYERS}, disc_L=${DISC_LAYERS}"
echo "Train: ${TRAIN_PARQUET}"
echo "Val:   ${VAL_PARQUET}"
echo "Batch: ${BATCH} × ${NGPU}gpu × accum${ACCUM}"
echo "Steps: ${MAX_STEPS}, warmup=${WARMUP}, LR=${LR}"
echo "Out:   ${OUT}"
echo "Log:   ${LOG_PATH}"
echo "GDrive: ${GDRIVE:-없음}"
[ -n "${RESUME}" ] && echo "Resume: ${RESUME}"
echo ""

mkdir -p "${OUT}"

GDRIVE_FLAG=""
[ -n "${GDRIVE}" ] && GDRIVE_FLAG="--rclone_remote ${GDRIVE}"

export PYTHONPATH="${PWD}/exp-jamo-codec${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=${NGPU} -m koelectra.train_binary \
    --vocab_size ${VOCAB_SIZE} \
    --bbpe_bits ${BBPE_BITS} \
    --embedding_dim_k ${EMBED_K} \
    --max_patches ${MAX_PATCHES} \
    --embedding_size ${EMBED} --hidden_size ${HIDDEN} \
    --n_heads ${NHEADS} --d_ff ${DFF} \
    --gen_layers ${GEN_LAYERS} --disc_layers ${DISC_LAYERS} \
    --dropout ${DROPOUT} --mask_ratio ${MASK_RATIO} \
    --gen_loss_weight ${GEN_LOSS_WEIGHT} \
    --train_parquet ${TRAIN_PARQUET} --text_key ${TEXT_KEY} \
    --val_parquet ${VAL_PARQUET} \
    --min_length ${MIN_LENGTH} \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --batch_size ${BATCH} --val_batch_size ${VAL_BATCH} \
    --grad_accum_steps ${ACCUM} \
    --lr ${LR} --min_lr ${MIN_LR} \
    --warmup_steps ${WARMUP} --max_steps ${MAX_STEPS} \
    --weight_decay ${WEIGHT_DECAY} --max_grad_norm ${MAX_GRAD_NORM} \
    --num_workers ${NUM_WORKERS} \
    ${BF16_FLAG} ${COMPILE_FLAG} \
    --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
    --val_every ${VAL_EVERY} --val_batches ${VAL_BATCHES} \
    --out_dir ${OUT} --log_file ${LOG_PATH} \
    --keep_latest_n ${KEEP_LATEST_N} \
    ${GDRIVE_FLAG} \
    ${RESUME:+--resume ${RESUME}} \
    2>&1 | tee -a "${LOG_PATH}"
