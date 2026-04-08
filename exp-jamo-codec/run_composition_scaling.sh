#!/bin/bash
# CompositionCodec 레이어 스케일링 (3L/4L/5L) — concat 방식
set -e
export PYTHONUNBUFFERED=1

CORPUS="${CORPUS:-corpus/val.parquet}"
TEXT_KEY="text"
MAX_STEPS=50000
BATCH_SIZE=64
SEQ_LEN=512
D_MODEL=256
KERNEL=7
LOG_EVERY=1000
OUT="exp-jamo-codec/checkpoints"

echo "=== CompositionCodec (concat) Layer Scaling ==="
echo "Corpus: ${CORPUS}, Steps: ${MAX_STEPS}, Batch: ${BATCH_SIZE}, SeqLen: ${SEQ_LEN}"
echo ""

for NL in 3 4 5; do
    echo "══════════════════════════════════════════"
    echo "[${NL}L k=${KERNEL}] CompositionCodec d=${D_MODEL}"
    echo "══════════════════════════════════════════"

    torchrun --nproc_per_node=${NGPU:-4} exp-jamo-codec/train_composition.py \
      --corpus ${CORPUS} --text_key ${TEXT_KEY} \
      --d_model ${D_MODEL} --n_layers ${NL} --kernel_size ${KERNEL} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
      --lr 3e-4 --warmup_steps 500 \
      --bf16 --compile --num_workers 2 \
      --log_every ${LOG_EVERY} --save_every 0 \
      --out_dir ${OUT}

    echo ""
done

echo "=== 레이어 스케일링 완료 ==="
