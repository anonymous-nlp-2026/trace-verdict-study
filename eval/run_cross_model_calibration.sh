#!/bin/bash
set -e
cd .
export CUDA_VISIBLE_DEVICES=1
PYTHON=python
OUTDIR=./checkpoints/cross_model_calibration

echo "=== Starting cross-model calibration on GPU 1 ==="
echo "Time: $(date)"

# 1. Qwen-7B (exception_only, original seed42)
echo ""
echo "========== Qwen-7B =========="
$PYTHON eval/calibration.py \
  --condition exception_only \
  --checkpoint_dir ./checkpoints/exception_only/final/ \
  --model_path ./models/Qwen2.5-Coder-7B-Instruct \
  --gpu 0 \
  --output $OUTDIR/qwen7b_exception_only_calibration.json
echo "Qwen-7B done: $(date)"

# 2. Qwen-1.5B
echo ""
echo "========== Qwen-1.5B =========="
$PYTHON eval/calibration.py \
  --condition exception_only \
  --checkpoint_dir ./checkpoints/qwen15b_exception_only_seed42/final/ \
  --model_path ./models/Qwen2.5-Coder-1.5B-Instruct \
  --gpu 0 \
  --output $OUTDIR/qwen15b_exception_only_calibration.json
echo "Qwen-1.5B done: $(date)"

# 3. DeepSeek-6.7B
echo ""
echo "========== DeepSeek-6.7B =========="
$PYTHON eval/calibration.py \
  --condition exception_only \
  --checkpoint_dir ./checkpoints/deepseek_exception_only/final/ \
  --model_path "./models/deepseek-coder-6.7b-instruct" \
  --gpu 0 \
  --output $OUTDIR/deepseek_exception_only_calibration.json
echo "DeepSeek-6.7B done: $(date)"

# 4. CodeLlama-13B
echo ""
echo "========== CodeLlama-13B =========="
$PYTHON eval/calibration.py \
  --condition exception_only \
  --checkpoint_dir ./checkpoints/codellama13b_exception_only_seed42/final/ \
  --model_path ./models/CodeLlama-13b-Instruct-hf \
  --gpu 0 \
  --output $OUTDIR/codellama13b_exception_only_calibration.json
echo "CodeLlama-13B done: $(date)"

echo ""
echo "=== All calibrations complete ==="
echo "Time: $(date)"
echo "DONE" > $OUTDIR/calibration_done.flag
