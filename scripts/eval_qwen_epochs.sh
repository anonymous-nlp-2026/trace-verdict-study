#!/bin/bash
set -e

GPU=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Activate your Python environment
export HF_HOME=~/.cache/huggingface
cd .

CKPT_BASE="./checkpoints"
CONDITIONS=(vota no_trace full_trace)
EPOCHS=(checkpoint-148 checkpoint-296)

for cond in "${CONDITIONS[@]}"; do
  for epoch in "${EPOCHS[@]}"; do
    ckpt="${CKPT_BASE}/${cond}/${epoch}"
    output="${ckpt}/eval_results.json"

    echo "========================================"
    echo "Evaluating: ${cond} / ${epoch} (GPU ${GPU})"
    echo "========================================"

    python eval/evaluate.py \
      --condition "${cond}" \
      --checkpoint_dir "${ckpt}" \
      --gpu "${GPU}" \
      --output "${output}" \
      --save_predictions

    echo ""
    echo "--- Results: ${cond} / ${epoch} ---"
    python3 -c "import json; d=json.load(open('${output}')); print(f\"  accuracy: {d['metrics']['accuracy']:.4f}  lift: {d['metrics']['lift']:.4f}  n: {d['metrics']['n']}\")"
    echo ""
  done
done

echo "All 6 evaluations complete."
