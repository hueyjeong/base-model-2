#!/bin/bash
# s16 코덱 체크포인트를 test.parquet으로 평가
set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/test.parquet"
TEXT_KEY="text"
SEQ_LEN=512
BATCH_SIZE=64
MAX_SAMPLES=200000
N_SHOW=0
CKPT_DIR="exp-jamo-codec/checkpoints"
RESULT_FILE="exp-jamo-codec/test_eval_results.txt"

echo "=== Codec Test Evaluation (s16 only) ===" | tee "${RESULT_FILE}"
echo "Corpus: ${CORPUS}" | tee -a "${RESULT_FILE}"
echo "Max samples: ${MAX_SAMPLES}, Seq len: ${SEQ_LEN}" | tee -a "${RESULT_FILE}"
echo "" | tee -a "${RESULT_FILE}"

eval_checkpoint() {
    local CKPT=$1
    local TOK=$2
    local NAME=$(basename "${CKPT}" _final.pt)

    echo "══════════════════════════════════════════" | tee -a "${RESULT_FILE}"
    echo "[${NAME}] tokenizer=${TOK}" | tee -a "${RESULT_FILE}"
    echo "══════════════════════════════════════════" | tee -a "${RESULT_FILE}"

    python exp-jamo-codec/eval_codec.py \
        --checkpoint "${CKPT}" \
        --tokenizer "${TOK}" \
        --corpus "${CORPUS}" --text_key "${TEXT_KEY}" \
        --max_seq_len "${SEQ_LEN}" \
        --batch_size "${BATCH_SIZE}" \
        --max_samples "${MAX_SAMPLES}" \
        --n_show "${N_SHOW}" 2>&1 | tee -a "${RESULT_FILE}"

    echo "" | tee -a "${RESULT_FILE}"
}

# ── 레이어 스케일링 XAttn (byte, s16, 25K) ──
for NL in 2 4 6 8; do
    eval_checkpoint "${CKPT_DIR}/xattn_${NL}L_s16_final.pt" "byte"
done

# ── 레이어 스케일링 Conv (byte, s16, 25K) ──
for NL in 3 6 9; do
    eval_checkpoint "${CKPT_DIR}/conv_${NL}L_s16_final.pt" "byte"
done

# ── Conv + Hash N-gram (byte, s16, 25K) ──
for NL in 1 2 3; do
    for HASH in nohash hash; do
        eval_checkpoint "${CKPT_DIR}/conv_${NL}L_${HASH}_s16_final.pt" "byte"
    done
done

echo "=== 전체 평가 완료 ===" | tee -a "${RESULT_FILE}"
