#!/bin/bash
set -e
# Activate your Python environment
export HF_HOME=~/.cache/huggingface
cd .

CKPT_BASE="./checkpoints"
GPU=1

echo "========================================"
echo "Task 1: VOTA final eval (re-run)"
echo "========================================"
python eval/evaluate.py \
  --condition vota \
  --checkpoint_dir ${CKPT_BASE}/vota/final \
  --gpu ${GPU} \
  --output ${CKPT_BASE}/vota/final/eval_results.json \
  --save_predictions
echo "VOTA done."
cat ${CKPT_BASE}/vota/final/eval_results.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  accuracy: {d[\"metrics\"][\"accuracy\"]:.6f}')"
wc -l ${CKPT_BASE}/vota/final/predictions.jsonl

echo ""
echo "========================================"
echo "Task 2a: no_trace final eval (predictions.jsonl)"
echo "========================================"
python eval/evaluate.py \
  --condition no_trace \
  --checkpoint_dir ${CKPT_BASE}/no_trace/final \
  --gpu ${GPU} \
  --output ${CKPT_BASE}/no_trace/final/eval_results.json \
  --save_predictions
echo "no_trace done."
cat ${CKPT_BASE}/no_trace/final/eval_results.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  accuracy: {d[\"metrics\"][\"accuracy\"]:.6f}')"
wc -l ${CKPT_BASE}/no_trace/final/predictions.jsonl

echo ""
echo "========================================"
echo "Task 2b: full_trace final eval (predictions.jsonl)"
echo "========================================"
python eval/evaluate.py \
  --condition full_trace \
  --checkpoint_dir ${CKPT_BASE}/full_trace/final \
  --gpu ${GPU} \
  --output ${CKPT_BASE}/full_trace/final/eval_results.json \
  --save_predictions
echo "full_trace done."
cat ${CKPT_BASE}/full_trace/final/eval_results.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  accuracy: {d[\"metrics\"][\"accuracy\"]:.6f}')"
wc -l ${CKPT_BASE}/full_trace/final/predictions.jsonl

echo ""
echo "========================================"
echo "Task 2c: exception_only final eval (predictions.jsonl)"
echo "========================================"
python eval/evaluate.py \
  --condition exception_only \
  --checkpoint_dir ${CKPT_BASE}/exception_only/final \
  --gpu ${GPU} \
  --output ${CKPT_BASE}/exception_only/final/eval_results.json \
  --save_predictions
echo "exception_only done."
cat ${CKPT_BASE}/exception_only/final/eval_results.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  accuracy: {d[\"metrics\"][\"accuracy\"]:.6f}')"
wc -l ${CKPT_BASE}/exception_only/final/predictions.jsonl

echo ""
echo "========================================"
echo "Task 3: eval_qwen_epochs.sh"
echo "========================================"
bash scripts/eval_qwen_epochs.sh --gpu ${GPU}

echo ""
echo "========================================"
echo "ALL TASKS COMPLETE"
echo "========================================"
