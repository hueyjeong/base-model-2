#!/bin/bash
# Phase 3: 전체 비교 200K steps, DDP
# 4종 codec × 3 stride = 12 조합
# torchrun으로 DDP, 평가는 단일 GPU

set -e
export PYTHONUNBUFFERED=1

CORPUS="corpus/val.parquet"
EVAL_CORPUS="corpus/test.parquet"
TEXT_KEY="text"
MAX_STEPS=200000
BATCH_SIZE=32
SEQ_LEN=512
D_MODEL=256
LOG_EVERY=2000
SAVE_EVERY=0
OUT_BASE="exp-jamo-codec/checkpoints"
TOK="jamo"
NGPU=${NGPU:-4}  # 기본 4 GPU, 환경변수로 변경 가능

echo "=== Phase 3: Full Comparison (200K steps, ${NGPU} GPUs) ==="
echo "Codecs: conv, xattn, entropy_conv, entropy_xattn"
echo "Strides: 4, 8, 16"
echo ""

for CODEC in conv xattn entropy_conv entropy_xattn; do
    # codec별 레이어 수 설정
    if [ "$CODEC" = "conv" ]; then
        N_LAYERS=3
    else
        N_LAYERS=2
    fi

    for STRIDE in 4 8 16; do
        TAG="${CODEC}_${TOK}_s${STRIDE}"
        CKPT="${OUT_BASE}/${TAG}_final.pt"

        echo "══════════════════════════════════════════"
        echo "[${TAG}] 학습 시작 (${CODEC}, stride=${STRIDE}, ${N_LAYERS}L)"
        echo "══════════════════════════════════════════"

        torchrun --nproc_per_node=${NGPU} exp-jamo-codec/train_codec.py \
          --codec ${CODEC} \
          --tokenizer ${TOK} --stride ${STRIDE} \
          --d_model ${D_MODEL} --n_layers ${N_LAYERS} \
          --corpus ${CORPUS} --text_key ${TEXT_KEY} \
          --max_seq_len ${SEQ_LEN} \
          --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
          --lr 3e-4 --warmup_steps 2000 \
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
done

echo "=== Phase 3 완료 ==="
