#!/bin/bash

SIZE="8M"
MAX_STEPS=5000
VAL_EVERY=1000
VAL_STEPS=20
BATCH_SIZE=4
GRAD_ACCUM=8
PACK_SIZE=2048
LR="1e-3"

COMMON_ARGS="--size $SIZE --tokenizer keyboard \
  --corpus corpus/sample_10g.jsonl \
  --text_key text \
  --val_corpus corpus/val_50k.jsonl \
  --max_steps $MAX_STEPS \
  --val_every $VAL_EVERY --val_steps $VAL_STEPS \
  --batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM \
  --pack_size $PACK_SIZE \
  --bf16 \
  --lr $LR --num_workers 0"

echo "=========================================="
echo " Starting Benchmark for Trial A & B"
echo " Model Size: $SIZE, Max Steps: $MAX_STEPS"
echo "=========================================="

echo "[1/4] Running Baseline..."
PYTHONUNBUFFERED=1 .venv/bin/python3 -m training.pretrain $COMMON_ARGS \
  --save_dir checkpoints/bench_base \
  > bench_base.log 2>&1

echo "[2/4] Running Trial A (Bias 0.5)..."
PYTHONUNBUFFERED=1 .venv/bin/python3 -m training.pretrain $COMMON_ARGS \
  --source_bias 0.5 \
  --save_dir checkpoints/bench_trial_a_05 \
  > bench_trial_a_05.log 2>&1

echo "[3/4] Running Trial A+B (Bias 0.5 + Copy Gate)..."
PYTHONUNBUFFERED=1 .venv/bin/python3 -m training.pretrain $COMMON_ARGS \
  --source_bias 0.5 \
  --use_copy_gate \
  --save_dir checkpoints/bench_trial_a_b \
  > bench_trial_a_b.log 2>&1

echo "[4/4] Running Trial B (Copy Gate)..."
PYTHONUNBUFFERED=1 .venv/bin/python3 -m training.pretrain $COMMON_ARGS \
  --use_copy_gate \
  --save_dir checkpoints/bench_trial_b \
  > bench_trial_b.log 2>&1

echo "=========================================="
echo " Benchmark Completed!"
echo " Checking val_bpc results from logs..."
echo ""
echo "--- Baseline ---"
grep "val step    $MAX_STEPS" bench_base.log
echo "--- Trial A (Bias 0.5) ---"
grep "val step    $MAX_STEPS" bench_trial_a_05.log
echo "--- Trial A+B ---"
grep "val step    $MAX_STEPS" bench_trial_a_b.log
echo "--- Trial B (Copy Gate) ---"
grep "val step    $MAX_STEPS" bench_trial_b.log
echo "=========================================="
