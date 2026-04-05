#!/bin/bash
# Conv 1L/2L/3L × hash_ngram 유무 실험
# 목표: Conv 하한선 탐색 + hash n-gram이 레이어를 대체할 수 있는지 확인

set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=25000
BATCH_SIZE=128
GRAD_ACCUM=2
SEQ_LEN=512
D_MODEL=256
LOG_EVERY=2000
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="byte"
STRIDE=16
NGPU=${NGPU:-4}

echo "=== Conv + Hash N-gram Experiment ==="
echo "Conv 1L/2L/3L × hash_ngram on/off"
echo "Stride: ${STRIDE}, Tokenizer: ${TOK}"
echo "Steps: ${MAX_STEPS}, GPUs: ${NGPU}"
echo ""

for NLAYER in 1 2 3; do
    for HASH_NGRAM in "" "--use_hash_ngram"; do
        if [ -z "${HASH_NGRAM}" ]; then
            HASH_TAG="nohash"
        else
            HASH_TAG="hash"
        fi

        TAG="conv_${NLAYER}L_${HASH_TAG}_s${STRIDE}"
        CKPT="${OUT_BASE}/${TAG}_final.pt"

        echo "══════════════════════════════════════════"
        echo "[${TAG}] Conv ${NLAYER}L, ${HASH_TAG}, stride=${STRIDE}"
        echo "══════════════════════════════════════════"

        torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_codec.py \
          --codec conv \
          --tokenizer ${TOK} --stride ${STRIDE} \
          --d_model ${D_MODEL} --n_layers ${NLAYER} \
          ${HASH_NGRAM} \
          --corpus ${CORPUS} --text_key ${TEXT_KEY} \
          --max_seq_len ${SEQ_LEN} \
          --batch_size ${BATCH_SIZE} --grad_accum_steps ${GRAD_ACCUM} --max_steps ${MAX_STEPS} \
          --lr 2.4e-3 --warmup_steps 250 \
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
done

echo "=== Conv + Hash N-gram 실험 완료 ==="
