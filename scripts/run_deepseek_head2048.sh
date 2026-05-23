#!/bin/bash
set -e
# Activate your Python environment
conda activate base
export HF_HOME=~/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd .

echo "=== Starting DeepSeek head-2048 training ==="
echo "Time: $(date)"

python train/train_sft.py \
    --condition full_trace_head \
    --model_path ./models/deepseek-coder-6.7b-instruct \
    --gradient_checkpointing \
    --gpu 0 \
    --seed 42 \
    --output_dir ./checkpoints/deepseek_full_trace_head2048_seed42 \
    --wandb_project trace-verdict-study \
    --wandb_run_name deepseek_full_trace_head2048_seed42

echo "=== Training complete, starting evaluation ==="
echo "Time: $(date)"

python eval/evaluate.py \
    --condition full_trace_head \
    --model_path ./models/deepseek-coder-6.7b-instruct \
    --checkpoint_dir ./checkpoints/deepseek_full_trace_head2048_seed42/final \
    --gpu 0 \
    --output ./checkpoints/deepseek_full_trace_head2048_seed42/final/eval_results.json \
    --save_predictions

echo "=== All done ==="
echo "Time: $(date)"
