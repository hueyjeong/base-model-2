#!/bin/bash
# CompositionCodec 본학습 — d=256, 6L, 효과배치 1024, train.parquet 1에포크
set -e
export PYTHONUNBUFFERED=1

CORPUS="${CORPUS:-corpus/train.parquet}"
TEXT_KEY="text"
MAX_STEPS=250000
BATCH_SIZE=256
SEQ_LEN=512
D_MODEL=256
N_LAYERS=6
KERNEL=7
LR=3e-4
WARMUP=2000
LOG_EVERY=5000
SAVE_EVERY=100000
OUT="exp-jamo-codec/checkpoints"
GDRIVE="${GDRIVE:-}"  # 구글 드라이브 마운트 경로 (예: /gdrive/MyDrive/codec)

echo "=== CompositionCodec 본학습 ==="
echo "d=${D_MODEL}, L=${N_LAYERS}, k=${KERNEL}"
echo "Corpus: ${CORPUS}"
echo "Steps: ${MAX_STEPS}, Batch: ${BATCH_SIZE}×${NGPU:-4}gpu=$((BATCH_SIZE * ${NGPU:-4}))"
echo "Save every: ${SAVE_EVERY}, GDrive: ${GDRIVE:-none}"
echo ""

torchrun --nproc_per_node=${NGPU:-4} exp-jamo-codec/train_composition.py \
  --corpus ${CORPUS} --text_key ${TEXT_KEY} \
  --d_model ${D_MODEL} --n_layers ${N_LAYERS} --kernel_size ${KERNEL} \
  --max_seq_len ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
  --lr ${LR} --warmup_steps ${WARMUP} \
  --bf16 --compile --num_workers 2 \
  --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
  --out_dir ${OUT} \
  2>&1 | tee exp-jamo-codec/composition_train_log.txt

# 구글 드라이브 업로드 (경로가 설정된 경우)
if [ -n "${GDRIVE}" ]; then
    echo ""
    echo "=== 구글 드라이브 업로드 ==="
    mkdir -p "${GDRIVE}"
    cp exp-jamo-codec/composition_train_log.txt "${GDRIVE}/"
    for f in ${OUT}/composition_${N_LAYERS}L_*.pt; do
        if [ -f "$f" ]; then
            echo "  업로드: $(basename $f)"
            cp "$f" "${GDRIVE}/"
        fi
    done
    echo "업로드 완료: ${GDRIVE}"
fi
