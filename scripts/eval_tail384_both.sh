#!/bin/bash
set -e
# Activate your Python environment
export HF_HOME=~/.cache/huggingface
cd .

CKPT_BASE="./checkpoints"
DATA_DIR="./data"
PYTHON="python"
export CUDA_VISIBLE_DEVICES=1

echo "========================================" 
echo "Eval 1: DeepSeek full_trace_tail_384"
echo "========================================" 
$PYTHON eval/evaluate.py \
  --condition full_trace_tail_384 \
  --checkpoint_dir ${CKPT_BASE}/deepseek_full_trace_tail384_seed42/final \
  --model_path "./models/deepseek-coder-6.7b-instruct" \
  --data_dir ${DATA_DIR} \
  --max_length 2048 \
  --truncation_side left \
  --gpu 0 \
  --output ${CKPT_BASE}/deepseek_full_trace_tail384_seed42/final/eval_results.json \
  --save_predictions

echo ""
echo "DeepSeek results:"
cat ${CKPT_BASE}/deepseek_full_trace_tail384_seed42/final/eval_results.json

echo ""
echo "========================================" 
echo "Eval 2: Qwen7B full_trace_tail_384"
echo "========================================" 
$PYTHON eval/evaluate.py \
  --condition full_trace_tail_384 \
  --checkpoint_dir ${CKPT_BASE}/qwen7b_full_trace_tail384_seed42/final \
  --model_path "./models/Qwen2.5-Coder-7B-Instruct" \
  --data_dir ${DATA_DIR} \
  --max_length 2048 \
  --truncation_side left \
  --gpu 0 \
  --output ${CKPT_BASE}/qwen7b_full_trace_tail384_seed42/final/eval_results.json \
  --save_predictions

echo ""
echo "Qwen7B results:"
cat ${CKPT_BASE}/qwen7b_full_trace_tail384_seed42/final/eval_results.json

echo ""
echo "========================================"
echo "ALL EVAL DONE"
echo "========================================"
