#!/bin/bash
# KoELECTRA Small v3 + SimpleCodec 사전학습
#
# 예) 로컬 1GPU sanity:
#   NGPU=1 BATCH=16 MAX_STEPS=200 WARMUP=20 VAL_EVERY=100 SAVE_EVERY=100 \
#     TRAIN_PARQUET=corpus/k-exaone_coverage_5_len1000.parquet \
#     VAL_PARQUET=corpus/k-exaone_coverage_5_len1000.parquet \
#     bash exp-jamo-codec/koelectra/run.sh
#
# 예) DDP 4GPU 본학습 + GDrive 업로드:
#   NGPU=4 GDRIVE=gdrive:exp-jamo-codec-koelectra/small/ \
#     bash exp-jamo-codec/koelectra/run.sh
#
# Codec 은 frozen (SimpleCodec checkpoint 고정 로드).
# Transformer + proj + head 만 학습.

set -e
export PYTHONUNBUFFERED=1

if [ -z "${NO_JEMALLOC}" ]; then
    if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
        export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "LD_PRELOAD: jemalloc enabled"
    fi
fi

# Codec
CODEC_CKPT="${CODEC_CKPT:-checkpoints/simple_codec_final.pt}"
CODEC_D_MODEL="${CODEC_D_MODEL:-256}"
CODEC_N_ENC_LAYERS="${CODEC_N_ENC_LAYERS:-5}"
CODEC_N_DEC_LAYERS="${CODEC_N_DEC_LAYERS:-5}"
CODEC_KERNEL="${CODEC_KERNEL:-5}"
MAX_JAMO="${MAX_JAMO:-32}"

# ELECTRA
MAX_PATCHES="${MAX_PATCHES:-512}"
EMBED="${EMBED:-128}"
HIDDEN="${HIDDEN:-256}"
NHEADS="${NHEADS:-4}"
DFF="${DFF:-1024}"
GEN_LAYERS="${GEN_LAYERS:-12}"
DISC_LAYERS="${DISC_LAYERS:-12}"
DROPOUT="${DROPOUT:-0.1}"
MASK_RATIO="${MASK_RATIO:-0.20}"
GEN_LOSS_WEIGHT="${GEN_LOSS_WEIGHT:-50.0}"

# Codec co-training (기본 freeze. co-train 원하면 CODEC_LR_RATIO=0.1 RECON_WEIGHT=0.5 같이 지정)
CODEC_LR_RATIO="${CODEC_LR_RATIO:-0.0}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"

# Generator aux signal (기본 OFF)
FEAT_MATCH_WEIGHT="${FEAT_MATCH_WEIGHT:-0.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"

# 데이터
TRAIN_PARQUET="${TRAIN_PARQUET:-corpus/k-exaone_random_coverage_1000_len4096.parquet}"
VAL_PARQUET="${VAL_PARQUET:-corpus/k-exaone_coverage_5_len1000.parquet}"
TEXT_KEY="${TEXT_KEY:-text}"
MIN_LENGTH="${MIN_LENGTH:-10}"

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
OUT="${OUT:-exp-jamo-codec/koelectra/checkpoints}"
LOG_PATH="${LOG_PATH:-exp-jamo-codec/koelectra/koelectra_train_log.txt}"

# GDrive
GDRIVE="${GDRIVE:-}"
KEEP_LATEST_N="${KEEP_LATEST_N:-3}"

# Resume
RESUME="${RESUME:-}"

echo "=== KoELECTRA Small v3 + SimpleCodec 학습 ==="
echo "Codec ckpt: ${CODEC_CKPT}  (lr_ratio=${CODEC_LR_RATIO}, recon_weight=${RECON_WEIGHT})"
echo "Codec: d=${CODEC_D_MODEL}, enc_L=${CODEC_N_ENC_LAYERS}, dec_L=${CODEC_N_DEC_LAYERS}, k=${CODEC_KERNEL}, max_jamo=${MAX_JAMO}"
echo "ELECTRA: P=${MAX_PATCHES}, embed=${EMBED}, hidden=${HIDDEN}, gen_L=${GEN_LAYERS}, disc_L=${DISC_LAYERS}"
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

# GDRIVE 플래그 조립
GDRIVE_FLAG=""
[ -n "${GDRIVE}" ] && GDRIVE_FLAG="--rclone_remote ${GDRIVE}"

# -m koelectra.train 이 module resolution 할 수 있도록 exp-jamo-codec 을 PYTHONPATH 에
export PYTHONPATH="${PWD}/exp-jamo-codec${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=${NGPU} -m koelectra.train \
    --codec_ckpt ${CODEC_CKPT} \
    --codec_d_model ${CODEC_D_MODEL} \
    --codec_n_enc_layers ${CODEC_N_ENC_LAYERS} \
    --codec_n_dec_layers ${CODEC_N_DEC_LAYERS} \
    --codec_kernel_size ${CODEC_KERNEL} \
    --max_jamo_per_token ${MAX_JAMO} \
    --max_patches ${MAX_PATCHES} \
    --embedding_size ${EMBED} --hidden_size ${HIDDEN} \
    --n_heads ${NHEADS} --d_ff ${DFF} \
    --gen_layers ${GEN_LAYERS} --disc_layers ${DISC_LAYERS} \
    --dropout ${DROPOUT} --mask_ratio ${MASK_RATIO} \
    --gen_loss_weight ${GEN_LOSS_WEIGHT} \
    --codec_lr_ratio ${CODEC_LR_RATIO} --recon_weight ${RECON_WEIGHT} \
    --feat_match_weight ${FEAT_MATCH_WEIGHT} --focal_gamma ${FOCAL_GAMMA} \
    --train_parquet ${TRAIN_PARQUET} --text_key ${TEXT_KEY} \
    --val_parquet ${VAL_PARQUET} \
    --min_length ${MIN_LENGTH} \
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
