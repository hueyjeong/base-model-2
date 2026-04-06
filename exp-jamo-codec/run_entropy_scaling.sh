#!/bin/bash
# SmallLM 엔트로피 모델 스케일링 실험 (10M ~ 50M)
# RMSNorm + SwiGLU (d_ff = d×3), byte 토크나이저

set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=25000
BATCH_SIZE=128
GRAD_ACCUM=2
SEQ_LEN=512
LOG_EVERY=2000
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="byte"
NGPU=${NGPU:-4}

echo "=== SmallLM Entropy Model Scaling ==="
echo "RMSNorm + SwiGLU, d_ff = d×3"
echo "Steps: ${MAX_STEPS}, GPUs: ${NGPU}"
echo ""

# (d_model, n_layers, n_heads, label, batch_size)
# 효과 배치 = batch × NGPU(4), grad_accum=1
CONFIGS=(
    "384 5 6 10M 64"
    "512 6 8 20M 64"
    "576 7 9 30M 64"
    "576 9 9 40M 64"
    "640 9 10 50M 64"
)

for CFG in "${CONFIGS[@]}"; do
    read -r D NL NH LABEL BS <<< "${CFG}"

    TAG="entropy_lm_${D}d_${NL}L"

    echo "══════════════════════════════════════════"
    echo "[${LABEL}] SmallLM d=${D}, L=${NL}, H=${NH}, batch=${BS}×${NGPU}gpu"
    echo "══════════════════════════════════════════"

    torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_entropy_lm.py \
      --tokenizer ${TOK} \
      --entropy_d_model ${D} --entropy_n_layers ${NL} --entropy_n_heads ${NH} \
      --corpus ${CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size ${BS} --grad_accum_steps 1 --max_steps ${MAX_STEPS} \
      --lr 2.4e-3 --warmup_steps 1000 \
      --bf16 --compile --num_workers 2 \
      --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
      --out_dir ${OUT_BASE}

    CKPT="${OUT_BASE}/${TAG}_final.pt"

    echo ""
    echo "[${LABEL}] 평가"
    python exp-jamo-codec/eval_entropy_lm.py \
      --checkpoint ${CKPT} \
      --tokenizer ${TOK} \
      --corpus ${EVAL_CORPUS} --text_key ${TEXT_KEY} \
      --max_seq_len ${SEQ_LEN} \
      --batch_size 64 --max_samples 50000

    echo ""
    echo ""
done

echo "=== 엔트로피 모델 스케일링 실험 완료 ==="
