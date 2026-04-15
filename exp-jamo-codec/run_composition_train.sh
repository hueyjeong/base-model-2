#!/bin/bash
# CompositionCodec 본학습 — d=256, 6L, 효과배치 1024, train.parquet 1에포크
set -e
export PYTHONUNBUFFERED=1

CORPUS="${CORPUS:-corpus/train.parquet}"
TEXT_KEY="text"
MAX_STEPS="${MAX_STEPS:-600000}"
# 재개/초기화 옵션 (환경변수로 전달)
# RESUME: 전체 상태 복원 (모델+옵티마이저+스케줄러+데이터)
#   예) RESUME=exp-jamo-codec/checkpoints/composition_6L_step110000.pt bash run_composition_train.sh
# INIT_FROM: 가중치만 로드, step 0부터 재학습
#   예) INIT_FROM=exp-jamo-codec/checkpoints/composition_6L_step110000.pt bash run_composition_train.sh
RESUME="${RESUME:-}"
INIT_FROM="${INIT_FROM:-}"
# SEQ_LEN을 크게 두면 "문서 경계에서 자연 flush" 동작.
# append_pad_slot으로 자모 수가 늘어나도 억지 flush 발생 안 함.
# 토큰 처리량 유지하려 BATCH_SIZE 절반 (256×4096 = 기존 512×2048과 동일).
BATCH_SIZE="${BATCH_SIZE:-256}"
SEQ_LEN="${SEQ_LEN:-4096}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-6}"
KERNEL="${KERNEL:-7}"
LR="${LR:-3e-4}"
WARMUP="${WARMUP:-2000}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
COMPILE_FLAG="${COMPILE_FLAG:---compile}"
OUT="${OUT:-exp-jamo-codec/checkpoints}"
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
VAL_EVERY="${VAL_EVERY:-5000}"
VAL_SAMPLES="${VAL_SAMPLES:-10000}"

# SEG_MASKED=1 → 토큰 경계 차단 conv (리크 0)
# PAD_SLOT=1 → 각 segment 끝에 PAD 1개 추가 (가변, 일반화 약함)
# FIXED_SLOT=1 → 모든 segment를 MAX_JAMO_PER_TOKEN 슬롯으로 고정 (decode_from_vec 완벽)
# MAX_JAMO_PER_TOKEN: fixed_slot 시 토큰당 슬롯 수 (기본 16)
SEG_MASKED_FLAG=""
[ -n "${SEG_MASKED}" ] && SEG_MASKED_FLAG="--segment_masked"
PAD_SLOT_FLAG=""
[ -n "${PAD_SLOT}" ] && PAD_SLOT_FLAG="--append_pad_slot"
FIXED_SLOT_FLAG=""
[ -n "${FIXED_SLOT}" ] && FIXED_SLOT_FLAG="--fixed_slot"
MAX_JAMO_FLAG=""
[ -n "${MAX_JAMO_PER_TOKEN}" ] && MAX_JAMO_FLAG="--max_jamo_per_token ${MAX_JAMO_PER_TOKEN}"
PARALLEL_DEC_FLAG=""
[ -n "${PARALLEL_DECODER}" ] && PARALLEL_DEC_FLAG="--parallel_decoder"
DEC_LAYERS_FLAG=""
[ -n "${DECODER_LAYERS}" ] && DEC_LAYERS_FLAG="--decoder_layers ${DECODER_LAYERS}"
DEC_HEADS_FLAG=""
[ -n "${DECODER_HEADS}" ] && DEC_HEADS_FLAG="--decoder_heads ${DECODER_HEADS}"
[ -n "${SEG_MASKED}" ] && echo "segment_masked: ON (conv 토큰 경계 차단)"
[ -n "${PAD_SLOT}" ] && echo "append_pad_slot: ON (segment당 PAD 1개 추가)"
[ -n "${FIXED_SLOT}" ] && echo "fixed_slot: ON (모든 토큰 ${MAX_JAMO_PER_TOKEN:-32} 슬롯 고정)"
[ -n "${PARALLEL_DECODER}" ] && echo "parallel_decoder: ON (self-attn ${DECODER_LAYERS:-2}L, encoder 가변 유지)"

NO_PIN_FLAG=""
[ -n "${NO_PIN_MEMORY}" ] && NO_PIN_FLAG="--no_pin_memory"
[ -n "${NO_PIN_MEMORY}" ] && echo "no_pin_memory: ON"

torchrun --nproc_per_node=${NGPU:-4} exp-jamo-codec/train_composition.py \
  --corpus ${CORPUS} --text_key ${TEXT_KEY} \
  --d_model ${D_MODEL} --n_layers ${N_LAYERS} --kernel_size ${KERNEL} \
  --max_seq_len ${SEQ_LEN} \
  --batch_size ${BATCH_SIZE} --max_steps ${MAX_STEPS} \
  --lr ${LR} --warmup_steps ${WARMUP} \
  --bf16 ${COMPILE_FLAG} --num_workers ${NUM_WORKERS} \
  --log_every ${LOG_EVERY} --save_every ${SAVE_EVERY} \
  --val_corpus ${VAL_CORPUS} --val_every ${VAL_EVERY} --val_samples ${VAL_SAMPLES} \
  --out_dir ${OUT} \
  ${SEG_MASKED_FLAG} ${PAD_SLOT_FLAG} ${FIXED_SLOT_FLAG} ${MAX_JAMO_FLAG} \
  ${PARALLEL_DEC_FLAG} ${DEC_LAYERS_FLAG} ${DEC_HEADS_FLAG} ${NO_PIN_FLAG} \
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
