#!/bin/bash
# Seed robustness experiment: train + eval VOTA condition with seeds 42/123/456.
# Verifies that VOTA 99.2% accuracy is stable across random seeds.
#
# GPU 参数必须与 registry 分配一致，不要假设默认值。
# 使用 CUDA_VISIBLE_DEVICES 隔离物理 GPU，训练/评估内部统一用 --gpu 0。
set -euo pipefail

usage() {
    echo "Usage: $0 [--gpu GPU_ID] [--seeds \"42 123 456\"] [--condition vota]"
    exit 0
}

GPU=1
SEEDS="42 123 456"
CONDITION="vota"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --condition) CONDITION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- GPU 可用性检查 ---
GPU_MEM_USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "ERROR")
if [[ "$GPU_MEM_USED" == "ERROR" ]]; then
    echo "ERROR: Cannot query GPU ${GPU}. Check nvidia-smi." >&2
    exit 1
fi
if [[ "$GPU_MEM_USED" -gt 1000 ]]; then
    echo "ERROR: GPU ${GPU} already has ${GPU_MEM_USED} MiB in use (> 1000 MiB). Aborting to avoid OOM." >&2
    exit 1
fi

# --- CUDA_VISIBLE_DEVICES 隔离 ---
export CUDA_VISIBLE_DEVICES=$GPU
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (physical GPU ${GPU})"

# Activate your Python environment
export HF_HOME=~/.cache/huggingface

CKPT_BASE="./checkpoints"
PROJECT_DIR="."

echo "=== Seed Robustness Experiment ==="
echo "Condition: ${CONDITION}  |  Seeds: ${SEEDS}  |  GPU: ${GPU} (CVD isolated)"
echo ""

EVAL_PATHS=""

for SEED in ${SEEDS}; do
    OUTPUT_DIR="${CKPT_BASE}/${CONDITION}_seed${SEED}"
    CKPT_FINAL="${OUTPUT_DIR}/final"
    EVAL_OUTPUT="${OUTPUT_DIR}/eval_results.json"

    # seed=42 special case: check legacy checkpoint path (vota/final/)
    LEGACY_CKPT="${CKPT_BASE}/${CONDITION}/final"
    if [[ ! -d "${CKPT_FINAL}" ]] && [[ "${SEED}" == "42" ]] && [[ -d "${LEGACY_CKPT}" ]]; then
        CKPT_FINAL="${LEGACY_CKPT}"
        echo "[seed=${SEED}] Using existing checkpoint at ${CKPT_FINAL}"
    fi

    echo "=========================================="
    echo "[seed=${SEED}]"

    # --- Training ---
    if [[ -d "${CKPT_FINAL}" ]]; then
        echo "  Checkpoint exists at ${CKPT_FINAL}, skipping training."
    else
        echo "  Starting training → ${OUTPUT_DIR}"
        python "${PROJECT_DIR}/train/train_sft.py" \
            --condition "${CONDITION}" \
            --output_dir "${OUTPUT_DIR}" \
            --gradient_checkpointing \
            --gpu 0 \
            --seed "${SEED}" \
            --wandb_project trace-verdict-study \
            --wandb_run_name "${CONDITION}_seed${SEED}"
    fi

    # --- Evaluation ---
    echo "  Evaluating → ${EVAL_OUTPUT}"
    mkdir -p "${OUTPUT_DIR}"
    python "${PROJECT_DIR}/eval/evaluate.py" \
        --condition "${CONDITION}" \
        --checkpoint_dir "${CKPT_FINAL}" \
        --gpu 0 \
        --output "${EVAL_OUTPUT}"

    # --- Bootstrap compare (if script exists) ---
    if [[ -f "${PROJECT_DIR}/eval/bootstrap_compare.py" ]]; then
        echo "  (bootstrap_compare.py available — run separately for cross-condition comparison)"
    fi

    # --- Print summary ---
    echo ""
    echo "  ┌─── seed=${SEED} results ───"
    python3 -c "
import json
with open('${EVAL_OUTPUT}') as f:
    r = json.load(f)
m, ci = r['metrics'], r['confidence_intervals']
print(f'  │ Accuracy: {m[\"accuracy\"]*100:.2f}% (95% CI: [{ci[\"accuracy_ci_lower\"]*100:.2f}%, {ci[\"accuracy_ci_upper\"]*100:.2f}%])')
print(f'  │ Pass  P={m[\"pass_precision\"]:.4f}  R={m[\"pass_recall\"]:.4f}  F1={m[\"pass_f1\"]:.4f}')
print(f'  │ Fail  P={m[\"fail_precision\"]:.4f}  R={m[\"fail_recall\"]:.4f}  F1={m[\"fail_f1\"]:.4f}')
print(f'  └────────────────────────')
"
    EVAL_PATHS="${EVAL_PATHS} ${EVAL_OUTPUT}"
    echo ""
done

echo "=========================================="
echo "=== All ${SEEDS} seeds complete ==="
echo ""
echo "--- Aggregating results ---"

python "${PROJECT_DIR}/scripts/aggregate_seeds.py" \
    --paths ${EVAL_PATHS} \
    --output "${CKPT_BASE}/seed_robustness_summary.json"

echo ""
echo "Done. Summary → ${CKPT_BASE}/seed_robustness_summary.json"
