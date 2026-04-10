#!/bin/bash
# CompositionCodec 본학습 — d=256, 6L, 효과배치 1024, train.parquet 1에포크
set -e
export PYTHONUNBUFFERED=1

CORPUS="${CORPUS:-corpus/train.parquet}"
TEXT_KEY="text"
MAX_STEPS=600000
# 재개/초기화 옵션 (환경변수로 전달)
# RESUME: 전체 상태 복원 (모델+옵티마이저+스케줄러+데이터)
#   예) RESUME=exp-jamo-codec/checkpoints/composition_6L_step110000.pt bash run_composition_train.sh
# INIT_FROM: 가중치만 로드, step 0부터 재학습
#   예) INIT_FROM=exp-jamo-codec/checkpoints/composition_6L_step110000.pt bash run_composition_train.sh
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"
BATCH_SIZE=512
SEQ_LEN=2048
D_MODEL=256
N_LAYERS=6
KERNEL=7
LR=3e-4
WARMUP=2000
LOG_EVERY=1000
SAVE_EVERY=10000
OUT="exp-jamo-codec/checkpoints"
GDRIVE="${GDRIVE:-}"  # rclone 원격지 (예: gdrive:base-model-2-ckpts/composition)

echo "=== CompositionCodec 본학습 ==="
echo "d=${D_MODEL}, L=${N_LAYERS}, k=${KERNEL}"
echo "Corpus: ${CORPUS}"
echo "Steps: ${MAX_STEPS}, Batch: ${BATCH_SIZE}×${NGPU:-4}gpu=$((BATCH_SIZE * ${NGPU:-4}))"
echo "Save every: ${SAVE_EVERY}, GDrive: ${GDRIVE:-none}"
[ -n "${RESUME}" ] && echo "Resume: ${RESUME}"
[ -n "${INIT_FROM}" ] && echo "Init from: ${INIT_FROM}"
echo ""

VAL_CORPUS="${VAL_CORPUS:-corpus/val.parquet}"
VAL_EVERY=5000
VAL_SAMPLES=10000

torchrun --nproc_per_node=${NGPU:-4} exp-jamo-codec/train_composition.py \
  --corpus ${CORPUS} --text_key ${TEXT_KEY} \
  --d_model ${D_MODEL} --n_layers ${N_LAYERS} --kernel_size ${KERNEL} \
  --max_seq_len ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
  --lr ${LR} --warmup_steps ${WARMUP} \
  --bf16 --compile --num_workers 16 \
  --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
  --val_corpus ${VAL_CORPUS} --val_every ${VAL_EVERY} --val_samples ${VAL_SAMPLES} \
  --out_dir ${OUT} \
  ${RESUME:+--resume ${RESUME}} \
  ${INIT_FROM:+--init_from ${INIT_FROM}} \
  2>&1 | tee exp-jamo-codec/composition_train_log.txt

# rclone 업로드는 train_composition.py에서 GDRIVE 환경변수로 자동 처리
# (체크포인트 저장 시마다 백그라운드 스레드로 업로드)
# 최종 로그 업로드
if [ -n "${GDRIVE}" ]; then
    echo "=== 최종 로그 업로드 ==="
    rclone copy exp-jamo-codec/composition_train_log.txt "${GDRIVE}/" 2>/dev/null && \
        echo "로그 업로드 완료" || echo "로그 업로드 실패"
fi
