#!/bin/bash
# 극한 압축률 테스트: stride 16, 32 (+ 비교용 8)
# jamo 토크나이저 기준 (Phase 1에서 차이 없었으므로 대표 1종)

set -e

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=10000
BATCH_SIZE=32
SEQ_LEN=512
D_MODEL=256
N_LAYERS=3
LOG_EVERY=500
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="jamo"

echo "=== 극한 압축률 테스트 ==="
echo "토크나이저: ${TOK}, stride: 8, 16, 32"
echo ""

for STRIDE in 8 16 32; do
    TAG="${TOK}_s${STRIDE}"
    CKPT="${OUT_BASE}/codec_${TAG}_final.pt"

    echo "──────────────────────────────────────────"
    echo "[${TAG}] 학습 시작 (stride=${STRIDE}, 압축 ${SEQ_LEN}→$((SEQ_LEN/STRIDE)))"
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
      --batch_size 64 --max_samples 2000 --n_show 5

    echo ""
    echo ""
done

echo "=== 극한 압축률 테스트 완료 ==="
