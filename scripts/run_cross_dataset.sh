#!/bin/bash
set -e

# Activate your Python environment
export HF_HOME=~/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1
cd .

DATA_DIR="./data/cross_dataset/"
MODEL="./models/Qwen2.5-Coder-7B-Instruct"
CKPT_BASE="./checkpoints"
PY="python"

# Wait for GPU 1 to be free
echo "[$(date)] Waiting for GPU 1 to be free..."
while nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    echo "[$(date)] GPU 1 still busy, waiting 60s..."
    sleep 60
done
echo "[$(date)] GPU 1 is free, starting cross-dataset experiments..."

# ============================================================
# Experiment 1: cross_he2mbpp_exception_only
# Train on HumanEval exception_only (1640), Test on MBPP exception_only (4270)
# ============================================================
echo ""
echo "=========================================="
echo "[$(date)] Experiment 1/4: cross_he2mbpp_exception_only"
echo "=========================================="

$PY train/train_sft.py \
    --condition cross_he2mbpp_exception_only \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --output_dir "$CKPT_BASE/cross_he2mbpp_exception_only" \
    --gradient_checkpointing \
    --gpu 0 \
    --seed 42 \
    --wandb_project trace-verdict-study \
    --wandb_run_name cross_he2mbpp_exception_only_seed42

$PY eval/evaluate.py \
    --condition cross_he2mbpp_exception_only \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --checkpoint_dir "$CKPT_BASE/cross_he2mbpp_exception_only/final" \
    --gpu 0 \
    --output "$CKPT_BASE/cross_he2mbpp_exception_only/final/eval_results.json"

echo "[$(date)] Experiment 1 DONE"

# ============================================================
# Experiment 2: cross_he2mbpp_no_trace
# Train on HumanEval no_trace (1640), Test on MBPP no_trace (4270)
# ============================================================
echo ""
echo "=========================================="
echo "[$(date)] Experiment 2/4: cross_he2mbpp_no_trace"
echo "=========================================="

$PY train/train_sft.py \
    --condition cross_he2mbpp_no_trace \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --output_dir "$CKPT_BASE/cross_he2mbpp_no_trace" \
    --gradient_checkpointing \
    --gpu 0 \
    --seed 42 \
    --wandb_project trace-verdict-study \
    --wandb_run_name cross_he2mbpp_no_trace_seed42

$PY eval/evaluate.py \
    --condition cross_he2mbpp_no_trace \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --checkpoint_dir "$CKPT_BASE/cross_he2mbpp_no_trace/final" \
    --gpu 0 \
    --output "$CKPT_BASE/cross_he2mbpp_no_trace/final/eval_results.json"

echo "[$(date)] Experiment 2 DONE"

# ============================================================
# Experiment 3: cross_mbpp2he_exception_only
# Train on MBPP exception_only (4270), Test on HumanEval exception_only (1640)
# ============================================================
echo ""
echo "=========================================="
echo "[$(date)] Experiment 3/4: cross_mbpp2he_exception_only"
echo "=========================================="

$PY train/train_sft.py \
    --condition cross_mbpp2he_exception_only \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --output_dir "$CKPT_BASE/cross_mbpp2he_exception_only" \
    --gradient_checkpointing \
    --gpu 0 \
    --seed 42 \
    --wandb_project trace-verdict-study \
    --wandb_run_name cross_mbpp2he_exception_only_seed42

$PY eval/evaluate.py \
    --condition cross_mbpp2he_exception_only \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --checkpoint_dir "$CKPT_BASE/cross_mbpp2he_exception_only/final" \
    --gpu 0 \
    --output "$CKPT_BASE/cross_mbpp2he_exception_only/final/eval_results.json"

echo "[$(date)] Experiment 3 DONE"

# ============================================================
# Experiment 4: cross_mbpp2he_no_trace
# Train on MBPP no_trace (4270), Test on HumanEval no_trace (1640)
# ============================================================
echo ""
echo "=========================================="
echo "[$(date)] Experiment 4/4: cross_mbpp2he_no_trace"
echo "=========================================="

$PY train/train_sft.py \
    --condition cross_mbpp2he_no_trace \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --output_dir "$CKPT_BASE/cross_mbpp2he_no_trace" \
    --gradient_checkpointing \
    --gpu 0 \
    --seed 42 \
    --wandb_project trace-verdict-study \
    --wandb_run_name cross_mbpp2he_no_trace_seed42

$PY eval/evaluate.py \
    --condition cross_mbpp2he_no_trace \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL" \
    --checkpoint_dir "$CKPT_BASE/cross_mbpp2he_no_trace/final" \
    --gpu 0 \
    --output "$CKPT_BASE/cross_mbpp2he_no_trace/final/eval_results.json"

echo "[$(date)] Experiment 4 DONE"

echo ""
echo "=========================================="
echo "[$(date)] ALL_CROSS_DATASET_DONE"
echo "=========================================="

# Print summary
echo ""
echo "=== Results Summary ==="
for exp in cross_he2mbpp_exception_only cross_he2mbpp_no_trace cross_mbpp2he_exception_only cross_mbpp2he_no_trace; do
    result_file="$CKPT_BASE/$exp/final/eval_results.json"
    if [ -f "$result_file" ]; then
        echo "$exp: $(cat $result_file)"
    else
        echo "$exp: MISSING eval_results.json"
    fi
done
