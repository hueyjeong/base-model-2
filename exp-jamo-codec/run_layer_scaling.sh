#!/bin/bash
# 레이어 스케일링 실험: XAttn/Conv × 레이어 수 × stride
# 목표: XAttn이 100% 복원 도달하는 최소 레이어 수 탐색
# + Conv 동일 조건 비교 + BPB 측정

set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=50000
BATCH_SIZE=128
SEQ_LEN=512
D_MODEL=256
LOG_EVERY=2000
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="byte"
STRIDE=16
NGPU=${NGPU:-4}

echo "=== Layer Scaling Experiment ==="
echo "XAttn: 2L, 4L, 6L, 8L + Conv: 3L, 6L, 9L"
echo "Stride: ${STRIDE}, Tokenizer: ${TOK}"
echo "Steps: ${MAX_STEPS}, GPUs: ${NGPU}"
echo ""

# ── XAttn 레이어 스케일링 ──
for NLAYER in 4 6 8; do
    TAG="xattn_${NLAYER}L_s${STRIDE}"
    CKPT="${OUT_BASE}/${TAG}_final.pt"

    echo "══════════════════════════════════════════"
    echo "[${TAG}] XAttn ${NLAYER}L, stride=${STRIDE}"
    echo "══════════════════════════════════════════"

    torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_codec.py \
      --codec xattn \
      --tokenizer ${TOK} --stride ${STRIDE} \
      --d_model ${D_MODEL} --n_layers ${NLAYER} \
      --corpus ${CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
      --lr 1.2e-3 --warmup_steps 500 \
      --bf16 --compile --num_workers 2 \
      --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
      --out_dir ${OUT_BASE}

    CKPT_ACTUAL="${OUT_BASE}/xattn_${TOK}_s${STRIDE}_final.pt"
    if [ -f "${CKPT_ACTUAL}" ] && [ "${CKPT_ACTUAL}" != "${CKPT}" ]; then
        mv "${CKPT_ACTUAL}" "${CKPT}"
    fi

    echo ""
    echo "[${TAG}] 평가"
    python exp-jamo-codec/eval_codec.py \
      --checkpoint ${CKPT} \
      --tokenizer ${TOK} \
      --corpus ${EVAL_CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size 64 --max_samples 2000 --n_show 2

    echo ""
    echo "[${TAG}] z 공간 분석"
    python exp-jamo-codec/analyze_z.py \
      --checkpoint ${CKPT} --tokenizer ${TOK}

    echo ""
    echo ""
done

# ── Conv 레이어 스케일링 (비교용) ──
for NLAYER in 3 6 9; do
    TAG="conv_${NLAYER}L_s${STRIDE}"
    CKPT="${OUT_BASE}/${TAG}_final.pt"

    echo "══════════════════════════════════════════"
    echo "[${TAG}] Conv ${NLAYER}L, stride=${STRIDE}"
    echo "══════════════════════════════════════════"

    torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_codec.py \
      --codec conv \
      --tokenizer ${TOK} --stride ${STRIDE} \
      --d_model ${D_MODEL} --n_layers ${NLAYER} \
      --corpus ${CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
      --lr 1.2e-3 --warmup_steps 500 \
      --bf16 --compile --num_workers 2 \
      --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
      --out_dir ${OUT_BASE}

    CKPT_ACTUAL="${OUT_BASE}/conv_${TOK}_s${STRIDE}_final.pt"
    if [ -f "${CKPT_ACTUAL}" ] && [ "${CKPT_ACTUAL}" != "${CKPT}" ]; then
        mv "${CKPT_ACTUAL}" "${CKPT}"
    fi

    echo ""
    echo "[${TAG}] 평가"
    python exp-jamo-codec/eval_codec.py \
      --checkpoint ${CKPT} \
      --tokenizer ${TOK} \
      --corpus ${EVAL_CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size 64 --max_samples 2000 --n_show 2

    echo ""
    echo "[${TAG}] z 공간 분석"
    python exp-jamo-codec/analyze_z.py \
      --checkpoint ${CKPT} --tokenizer ${TOK}

    echo ""
    echo ""
done

echo "=== Layer Scaling 완료 ==="
