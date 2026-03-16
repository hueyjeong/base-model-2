#!/bin/bash
# DenseEditor 품질 벤치마크 — 5060 Ti 8대 DDP
#
# 각 아키텍처를 10000 steps 학습하고 validation으로 품질 비교
# d=768, seq=512, 128M params
#
# 사용법:
#   bash scripts/run_quality_bench.sh

set -e

# 환경변수
export BITLINEAR_CUDA_BACKWARD=bf16_tc
export BITLINEAR_CUDA_GRADW_LT=1
export BITLINEAR_CUDA_FUSED_ACT=1
export BITLINEAR_CUDA_FUSED_WEIGHT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export NCCL_P2P_DISABLE=1

CORPUS="corpus/sample_full.jsonl"
VAL_CORPUS="corpus/val_50k.jsonl"
TEXT_KEY="text"
NGPU=8
D_MODEL=640
SEQ_LEN=2048      # 패킹: BOS..EOS BOS..EOS → 2048 토큰 (GPU 활용 극대화)
BATCH_SIZE=2      # per-GPU micro batch (16GB + seq=2048)
GRAD_ACCUM=4      # effective batch = 2 * 4 * 8 = 64
MAX_STEPS=10000
WARMUP=500
LR=1e-3
SAVE_DIR="checkpoints/quality_bench"
LOG_INTERVAL=50
SAVE_INTERVAL=2500
VAL_EVERY=500
VAL_STEPS=50

echo "=== DenseEditor 품질 벤치마크 (DDP ${NGPU}GPU) ==="
echo "d_model=${D_MODEL}, seq_len=${SEQ_LEN}, batch=${BATCH_SIZE}×${GRAD_ACCUM}×${NGPU}=$(( BATCH_SIZE * GRAD_ACCUM * NGPU ))"
echo "max_steps=${MAX_STEPS}, lr=${LR}"
echo ""

for MIXING in xlstm retnet mamba rwkv tcn fnet mlstm; do
    echo "===== ${MIXING^^} ====="
    LOG_FILE="${SAVE_DIR}/${MIXING}_d${D_MODEL}.log"
    mkdir -p "${SAVE_DIR}"

    torchrun --nproc_per_node=${NGPU} \
        -m training.pretrain_dense_editor \
        --mixing_type ${MIXING} \
        --d_model ${D_MODEL} \
        --tokenizer keyboard \
        --corpus ${CORPUS} \
        --text_key ${TEXT_KEY} \
        --val_corpus ${VAL_CORPUS} \
        --max_seq_len ${SEQ_LEN} \
        --batch_size ${BATCH_SIZE} \
        --grad_accum_steps ${GRAD_ACCUM} \
        --lr ${LR} \
        --warmup_steps ${WARMUP} \
        --max_steps ${MAX_STEPS} \
        --bf16 \
        --num_workers 4 \
        --log_interval ${LOG_INTERVAL} \
        --save_interval ${SAVE_INTERVAL} \
        --save_dir "${SAVE_DIR}" \
        --val_every ${VAL_EVERY} \
        --val_steps ${VAL_STEPS} \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "===== ${MIXING^^} 완료 ====="
    echo ""
done

echo "=== 전체 벤치마크 완료 ==="
echo "결과 확인: grep 'val step' ${SAVE_DIR}/*.log"
