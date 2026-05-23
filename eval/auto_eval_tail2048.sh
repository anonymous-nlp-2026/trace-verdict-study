#!/bin/bash
set -e

CKPT_BASE="./checkpoints/codellama13b_full_trace_tail2048_seed42_v2"
LOG="./runs/codellama13b_full_trace_tail2048_seed42_v2.log"
EVAL_SCRIPT="./eval/evaluate.py"
MODEL="./models/CodeLlama-13b-Instruct-hf"
PYTHON="python"

# Wait for training to finish (final directory appears)
echo "Waiting for training to complete..."
while [ ! -d "${CKPT_BASE}/final" ]; do
    sleep 30
done
echo "Training complete. Starting eval..."

# Wait a bit for files to finish writing
sleep 10

export CUDA_VISIBLE_DEVICES=1

# Eval epoch 1 (checkpoint-148)
echo "=== Evaluating checkpoint-148 (epoch 1) ==="
$PYTHON $EVAL_SCRIPT \
    --condition full_trace_tail \
    --checkpoint_dir "${CKPT_BASE}/checkpoint-148" \
    --model_path "$MODEL" \
    --max_length 2048 \
    --truncation_side left \
    --gpu 0 \
    --save_predictions

echo ""
echo "=== Epoch 1 results ==="
cat "${CKPT_BASE}/checkpoint-148/eval_results.json"

# Eval epoch 3 (final)
echo ""
echo "=== Evaluating final (epoch 3) ==="
$PYTHON $EVAL_SCRIPT \
    --condition full_trace_tail \
    --checkpoint_dir "${CKPT_BASE}/final" \
    --model_path "$MODEL" \
    --max_length 2048 \
    --truncation_side left \
    --gpu 0 \
    --save_predictions

echo ""
echo "=== Epoch 3 results ==="
cat "${CKPT_BASE}/final/eval_results.json"

echo ""
echo "=== ALL EVAL DONE ==="
