#!/usr/bin/env bash
set -euo pipefail

# Activate your Python environment
export HF_HOME=~/.cache/huggingface
cd .

CKPT_BASE="./checkpoints"
DATA_DIR="./data"
DEEPSEEK_BASE="./models/deepseek-coder-6.7b-instruct"
QWEN_BASE="./models/Qwen2.5-Coder-7B-Instruct"
GPU=${1:-0}

echo "========================================"
echo "1/2: deepseek_full_trace_seed42_v3"
echo "  checkpoint: ${CKPT_BASE}/deepseek_full_trace_seed42_v3/final"
echo "  condition:  full_trace"
echo "  max_length: 16384"
echo "  base model: DeepSeek-Coder-6.7B-Instruct"
echo "========================================"
python eval/evaluate.py \
  --condition full_trace \
  --checkpoint_dir "${CKPT_BASE}/deepseek_full_trace_seed42_v3/final" \
  --model_path "${DEEPSEEK_BASE}" \
  --data_dir "${DATA_DIR}" \
  --gpu "${GPU}" \
  --max_length 16384 \
  --output "${CKPT_BASE}/deepseek_full_trace_seed42_v3/final/eval_results.json" \
  --save_predictions
echo "DONE: deepseek_full_trace_seed42_v3 predictions saved"
echo ""

echo "========================================"
echo "2/2: debugbench_vota"
echo "  checkpoint: ${CKPT_BASE}/debugbench_vota/final"
echo "  condition:  debugbench_vota"
echo "  max_length: 2048 (default)"
echo "  base model: Qwen2.5-Coder-7B-Instruct"
echo "========================================"
python eval/evaluate.py \
  --condition debugbench_vota \
  --checkpoint_dir "${CKPT_BASE}/debugbench_vota/final" \
  --model_path "${QWEN_BASE}" \
  --data_dir "${DATA_DIR}" \
  --gpu "${GPU}" \
  --output "${CKPT_BASE}/debugbench_vota/final/eval_results.json" \
  --save_predictions
echo "DONE: debugbench_vota predictions saved"
echo ""

echo "========================================"
echo "ALL PREDICTIONS COMPLETE"
echo "========================================"
echo "Output files:"
echo "  ${CKPT_BASE}/deepseek_full_trace_seed42_v3/final/predictions.jsonl"
echo "  ${CKPT_BASE}/debugbench_vota/final/predictions.jsonl"
