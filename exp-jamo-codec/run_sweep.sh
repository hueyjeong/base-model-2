#!/bin/bash
# Phase 1 Conv Codec sweep: 3 tokenizers × 3 strides = 9 조합
# 학습: val.parquet, 평가: test.parquet

set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=10000
BATCH_SIZE=32
SEQ_LEN=512
D_MODEL=256
N_LAYERS=3
LOG_EVERY=500
SAVE_EVERY=0  # 최종만 저장
OUT_BASE="exp-jamo-codec/checkpoints"

echo "=== Phase 1 Conv Codec Sweep ==="
echo "학습: ${CORPUS}, 평가: ${EVAL_CORPUS}"
echo "조합: 3 tokenizers × 3 strides = 9"
echo "스텝: ${MAX_STEPS}, 배치: ${BATCH_SIZE}, seq: ${SEQ_LEN}"
echo ""

for TOK in byte jamo keyboard; do
  for STRIDE in 2 4 8; do
    TAG="${TOK}_s${STRIDE}"
    CKPT="${OUT_BASE}/codec_${TAG}_final.pt"

    echo "──────────────────────────────────────────"
    echo "[${TAG}] 학습 시작 (${TOK}, stride=${STRIDE})"
    echo "──────────────────────────────────────────"

    python exp-jamo-codec/train_codec.py \
      --tokenizer ${TOK} --stride ${STRIDE} \
      --d_model ${D_MODEL} --n_layers ${N_LAYERS} \
      --corpus ${CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
      --lr 3e-4 --warmup_steps 1000 \
      --bf16 --num_workers 2 \
      --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
      --out_dir ${OUT_BASE}

    echo ""
    echo "[${TAG}] 평가"
    python exp-jamo-codec/eval_codec.py \
      --checkpoint ${CKPT} \
      --tokenizer ${TOK} --stride ${STRIDE} \
      --corpus ${EVAL_CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size 64 --max_samples 2000 --n_show 3

    echo ""
    echo ""
  done
done

echo "=== Sweep 완료 ==="
