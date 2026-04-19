#!/bin/bash
# Binary ELECTRA 용 8 task KLUE-style downstream benchmark chain.
#
# 사용:
#   bash exp-jamo-codec/koelectra/run_bench_binary.sh checkpoints/b_small/binary_step_20000.pt [tag]
#
#   - $1: 체크포인트 경로 (필수)
#   - $2: 결과 로그 prefix (기본: ckpt 파일명에서 추출)
#
# 결과: /tmp/bench_binary_${TAG}_<task>.log + /tmp/bench_binary_${TAG}_summary.txt

set -e
export PYTHONPATH=exp-jamo-codec
source .venv/bin/activate

CKPT="${1:?usage: bash $0 <ckpt_path> [tag]}"
TAG="${2:-$(basename ${CKPT} .pt)}"
SUMMARY="/tmp/bench_binary_${TAG}_summary.txt"

# --tokenizer_path 는 ckpt args 에서 자동 감지되지만,
# 필요시 TOKENIZER_PATH 환경변수로 override 가능
TOK_FLAG=""
[ -n "${TOKENIZER_PATH}" ] && TOK_FLAG="--tokenizer_path ${TOKENIZER_PATH}"

echo "=== Binary benchmark chain: ${CKPT} (tag=${TAG}) ===" | tee "${SUMMARY}"
echo "Started: $(date)" >> "${SUMMARY}"
echo "" >> "${SUMMARY}"

run_task() {
    local task=$1
    local module=$2
    local extra=$3
    local log="/tmp/bench_binary_${TAG}_${task}.log"
    echo "=== ${task} ===" | tee -a "${SUMMARY}"
    python -u -m koelectra.${module} \
        --ckpt "${CKPT}" \
        --epochs 3 --bf16 --num_workers 2 --compile \
        ${TOK_FLAG} \
        ${extra} 2>&1 | tee "${log}" | grep -E "^\[Eval|^\[Done" | tee -a "${SUMMARY}"
    echo "" >> "${SUMMARY}"
}

# 1. NSMC (sentence-level, 2-class)
run_task nsmc finetune_downstream_binary "--task nsmc --max_patches 64 --batch_size 64 --eval_batch_size 128 --lr 2e-5 --log_every 500"

# 2. KLUE-NLI (sentence-pair, 3-class)
run_task klue_nli finetune_downstream_binary "--task klue_nli --max_patches 128 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 500"

# 3. KLUE-YNAT (single sentence, 7-class)
run_task klue_ynat finetune_downstream_binary "--task klue_ynat --max_patches 64 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 500"

# 4. KLUE-STS (sentence-pair, regression)
run_task klue_sts finetune_downstream_binary "--task klue_sts --max_patches 128 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 200"

# 5. KLUE-RE (sentence + entity-pair, 30-class)
run_task klue_re finetune_downstream_binary "--task klue_re --max_patches 128 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 200"

# 6. PAWS-X-KO (sentence-pair, 2-class)
run_task paws_x_ko finetune_downstream_binary "--task paws_x_ko --max_patches 128 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 500"

# 7. KLUE-NER (token classification, BIO 13-tag, entity F1)
run_task klue_ner finetune_ner_binary "--max_patches 128 --batch_size 32 --eval_batch_size 64 --lr 3e-5 --log_every 200"

# 8. KLUE-MRC (span extraction, EM/F1)
run_task klue_mrc finetune_mrc_binary "--max_patches 512 --batch_size 8 --eval_batch_size 16 --lr 3e-5 --log_every 200"

echo "" >> "${SUMMARY}"
echo "Completed: $(date)" >> "${SUMMARY}"
echo "=== ALL DONE ===" | tee -a "${SUMMARY}"
echo ""
echo "Summary: ${SUMMARY}"
