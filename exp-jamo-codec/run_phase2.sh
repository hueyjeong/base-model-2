#!/bin/bash
# Phase 2: Cross-attention codec sweep + Conv 대비 비교
# jamo 토크나이저, stride 4/8/16

set -e

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=10000
BATCH_SIZE=32
SEQ_LEN=512
D_MODEL=256
N_LAYERS=2
LOG_EVERY=500
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="jamo"

echo "=== Phase 2: Cross-Attention Codec ==="
echo "토크나이저: ${TOK}, stride: 4, 8, 16"
echo ""

for STRIDE in 4 8 16; do
    TAG="xattn_${TOK}_s${STRIDE}"
    CKPT="${OUT_BASE}/xattn_${TOK}_s${STRIDE}_final.pt"

    echo "──────────────────────────────────────────"
    echo "[${TAG}] 학습 (stride=${STRIDE}, 압축 ${SEQ_LEN}→$((SEQ_LEN/STRIDE)))"
    echo "──────────────────────────────────────────"

    python exp-jamo-codec/train_codec.py \
      --codec xattn \
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
      --tokenizer ${TOK} \
      --corpus ${EVAL_CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size 64 --max_samples 2000 --n_show 3

    echo ""
    echo "[${TAG}] z 공간 분석"
    python exp-jamo-codec/analyze_z.py \
      --checkpoint ${CKPT} --tokenizer ${TOK}

    echo ""
    echo ""
done

echo "=== Phase 2 완료 ==="
