#!/usr/bin/env bash
set -euo pipefail

# DeepSeek calibration analysis
# Usage: bash scripts/calibration_deepseek.sh [--gpu N]

GPU=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Activate your Python environment

DEEPSEEK_BASE="./models/deepseek-coder-6.7b-instruct"
CKPT_ROOT="./checkpoints"
OUTPUT_DIR="${CKPT_ROOT}/calibration_deepseek"
DATA_DIR="./data"
CALIBRATION_PY="./eval/calibration.py"

mkdir -p "${OUTPUT_DIR}"

declare -A CONDITIONS
CONDITIONS=(
    ["deepseek_vota"]="vota"
    ["deepseek_no_trace"]="no_trace"
    ["deepseek_exception_only"]="exception_only"
    ["deepseek_full_trace"]="full_trace"
)

FAILED=0

for name in deepseek_vota deepseek_no_trace deepseek_exception_only deepseek_full_trace; do
    base_cond="${CONDITIONS[$name]}"
    ckpt_dir="${CKPT_ROOT}/${name}/final"

    echo ""
    echo "============================================================"
    echo "Condition: ${name} (data: ${base_cond})"
    echo "Checkpoint: ${ckpt_dir}"
    echo "============================================================"

    if [[ ! -d "${ckpt_dir}" ]]; then
        echo "SKIP: checkpoint not found at ${ckpt_dir}"
        continue
    fi

    python "${CALIBRATION_PY}" \
        --condition "${base_cond}" \
        --checkpoint_dir "${ckpt_dir}" \
        --model_path "${DEEPSEEK_BASE}" \
        --data_dir "${DATA_DIR}" \
        --gpu "${GPU}" \
        --output "${OUTPUT_DIR}/${name}_calibration.json"

    if [[ $? -eq 0 ]]; then
        echo "OK: ${name}"
    else
        echo "FAIL: ${name}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "All done. Results in: ${OUTPUT_DIR}"
if [[ ${FAILED} -gt 0 ]]; then
    echo "WARNING: ${FAILED} condition(s) failed"
    exit 1
fi
