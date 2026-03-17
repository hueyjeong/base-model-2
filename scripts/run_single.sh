#!/bin/bash
# DenseEditor 단일 아키텍처 학습 — 5060 Ti 8대 DDP
#
# 사용법:
#   bash scripts/run_single.sh xlstm        # 기본 10000 steps
#   bash scripts/run_single.sh mamba 20000  # 20000 steps
#   NGPU=4 bash scripts/run_single.sh retnet  # 4 GPU

set -e

MIXING=${1:?사용법: bash scripts/run_single.sh <mixing_type> [max_steps]}
MAX_STEPS=${2:-10000}

# 환경변수
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1

NGPU=${NGPU:-8}
CORPUS=${CORPUS:-"corpus/sample_full.jsonl"}
VAL_CORPUS=${VAL_CORPUS:-"corpus/val_50k.jsonl"}
D_MODEL=${D_MODEL:-768}
SEQ_LEN=${SEQ_LEN:-2048}    # 패킹: BOS..EOS BOS..EOS → 2048 토큰
BATCH_SIZE=${BATCH_SIZE:-2}  # 16GB GPU에 seq=2048이면 bs=2
GRAD_ACCUM=${GRAD_ACCUM:-4}  # eff_batch = 2*4*8 = 64
LR=${LR:-1e-3}
WARMUP=${WARMUP:-500}
SAVE_DIR=${SAVE_DIR:-"checkpoints/${MIXING}_d${D_MODEL}"}
RESUME=${RESUME:-}

EFF_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NGPU))

echo "=== DenseEditor: ${MIXING^^} d=${D_MODEL} ==="
echo "GPU=${NGPU}, batch=${BATCH_SIZE}×${GRAD_ACCUM}×${NGPU}=${EFF_BATCH}"
echo "steps=${MAX_STEPS}, lr=${LR}, warmup=${WARMUP}"
echo "save_dir=${SAVE_DIR}"
echo ""

RESUME_FLAG=""
if [ -n "${RESUME}" ]; then
    RESUME_FLAG="--resume ${RESUME}"
    echo "체크포인트 복원: ${RESUME}"
fi

mkdir -p "${SAVE_DIR}"

torchrun --nproc_per_node=${NGPU} \
    -m training.pretrain_dense_editor \
    --mixing_type ${MIXING} \
    --d_model ${D_MODEL} \
    --tokenizer keyboard \
    --corpus ${CORPUS} \
    --text_key text \
    --val_corpus ${VAL_CORPUS} \
    --max_seq_len ${SEQ_LEN} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_steps ${WARMUP} \
    --max_steps ${MAX_STEPS} \
    --bf16 \
    --num_workers 4 \
    --log_interval 50 \
    --save_interval 2500 \
    --save_dir "${SAVE_DIR}" \
    --val_every 500 \
    --val_steps 50 \
    ${RESUME_FLAG} \
    2>&1 | tee "${SAVE_DIR}/train.log"

echo ""
echo "=== ${MIXING^^} 학습 완료 ==="
echo "체크포인트: ${SAVE_DIR}/"
echo "마지막 검증: grep 'val step' ${SAVE_DIR}/train.log | tail -1"
