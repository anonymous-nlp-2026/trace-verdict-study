# Trace Verdict Study

Code for "Execution Trace Analysis for Verdict Prediction in Code Assessment" — SFT-based models that leverage execution traces to predict pass/fail verdicts.

## Overview

This project investigates how different representations of execution traces affect the accuracy of fine-tuned language models in predicting code correctness. We compare multiple trace conditions (full trace, exception-only, no-trace, label-only, etc.) across different base models (Qwen2.5-Coder, DeepSeek-Coder) and evaluate with bootstrap confidence intervals, calibration analysis, and cross-dataset generalization.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- peft
- datasets
- accelerate
- trl
- scipy, numpy, matplotlib
- wandb (optional, for experiment tracking)

Install dependencies:
```bash
pip install torch transformers peft datasets accelerate trl scipy numpy matplotlib wandb
```

## Project Structure

```
├── train/
│   ├── train_sft.py              # Main SFT training script
│   └── gradient_mask_trainer.py  # Custom trainer with gradient masking
├── eval/
│   ├── evaluate.py               # Main evaluation script
│   ├── bootstrap_compare.py      # Bootstrap CI comparison
│   ├── calibration.py            # Calibration analysis
│   ├── fewshot_eval.py           # Few-shot baseline evaluation
│   ├── fewshot_baseline.py       # Few-shot prompting baseline
│   ├── zeroshot_eval.py          # Zero-shot evaluation
│   └── ...                       # Additional evaluation scripts
├── src/
│   ├── vota.py                   # VoTA (Verdict-of-Trace Analysis) core logic
│   └── assert_decompose.py       # Assert decomposition utilities
├── scripts/
│   ├── run_seed_robustness.sh    # Multi-seed training
│   ├── regex_baseline.py         # Regex heuristic baseline
│   └── ...                       # Additional utility scripts
├── analysis/
│   ├── token_length_analysis.py  # Token length statistics
│   ├── trace_ablation_bootstrap.py # Trace ablation with bootstrap
│   ├── clustered_bootstrap.py    # Cluster-robust bootstrap CI
│   └── ...                       # Additional analysis scripts
└── tests/
    └── test_vota.py              # Unit tests for VoTA
```

## Data Format

Training/evaluation data uses JSONL format with the following fields:
```json
{
  "input": "<execution trace or code context>",
  "output": "pass" or "fail"
}
```

Data preparation scripts are in `data/` (not included in this repository). Place your prepared data files in `./data/`.

## Training

Example training command (Qwen2.5-Coder-7B with exception-only traces):

```bash
python train/train_sft.py \
    --condition exception_only \
    --model_path ./models/Qwen2.5-Coder-7B-Instruct \
    --train_data ./data/exception_only_train.jsonl \
    --eval_data ./data/exception_only_test.jsonl \
    --output_dir ./checkpoints/exception_only_seed42 \
    --max_steps 500 \
    --seed 42 \
    --gradient_checkpointing
```

### Conditions

| Condition | Description |
|-----------|-------------|
| `full_trace` | Complete execution trace |
| `exception_only` | Only exception/error information from trace |
| `no_trace` | Code + test without trace |
| `label_only` | Minimal input (verdict label prediction only) |

## Evaluation

```bash
python eval/evaluate.py \
    --condition exception_only \
    --model_path ./models/Qwen2.5-Coder-7B-Instruct \
    --checkpoint_dir ./checkpoints/exception_only_seed42/final \
    --eval_data ./data/exception_only_test.jsonl \
    --output ./checkpoints/exception_only_seed42/final/eval_results.json
```

## Bootstrap Comparison

```bash
python eval/bootstrap_compare.py \
    --pred_a ./checkpoints/condition_a/final/predictions.jsonl \
    --pred_b ./checkpoints/condition_b/final/predictions.jsonl
```

## License

This code is released for research purposes.
