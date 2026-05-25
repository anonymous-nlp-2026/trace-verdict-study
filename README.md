# Trace Verbosity Degrades SFT Verdict Prediction

Code and data for "Trace Verbosity Degrades SFT Verdict Prediction: A Case Study in Information Density Effects" (Anonymous, EMNLP 2026, under review).

## Overview

This project investigates how different representations of execution traces affect the accuracy of SFT-based language models in predicting code correctness (pass/fail). We systematically ablate trace components across four code-specialized models (1.5B–13B parameters, three architecture families) and find that compact exception events achieve ≥99.33% accuracy while complete traces plateau near the source-code-only baseline (77–82%).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── train/
│   ├── train_sft.py              # Main SFT training script (LoRA)
│   └── gradient_mask_trainer.py  # Custom trainer with gradient masking
├── eval/
│   ├── evaluate.py               # Main evaluation script with bootstrap CI
│   ├── bootstrap_compare.py      # Pairwise bootstrap comparison
│   ├── calibration.py            # Calibration analysis (ECE)
│   ├── fewshot_eval.py           # Few-shot ICL baseline
│   ├── zeroshot_eval.py          # Zero-shot baseline
│   ├── silent_error_analysis.py  # Silent logic error analysis
│   ├── dedup_sensitivity.py      # Deduplication sensitivity analysis
│   └── ...                       # Additional evaluation scripts
├── src/
│   ├── vota.py                   # VOTA (Verdict-Optimized Trace Abstraction) core logic
│   └── assert_decompose.py       # Assertion-point extraction utilities
├── scripts/
│   ├── prepare_ablation_data.py  # Prepare data for all trace conditions
│   ├── run_seed_robustness.sh    # Multi-seed training launcher
│   ├── regex_baseline.py         # Regex heuristic baselines (B1–B5)
│   └── ...                       # Additional utility scripts
├── analysis/
│   ├── clustered_bootstrap.py    # Problem-level clustered bootstrap CI
│   ├── token_length_analysis.py  # Token length statistics
│   ├── diagnostic_token_ratio.py # Diagnostic token density computation
│   ├── tokenizer_stats.py        # Cross-model tokenizer analysis
│   └── ...                       # Additional analysis scripts
└── tests/
    └── test_vota.py              # Unit tests for VOTA
```

## Data Preparation

We use HumanEval (164 problems) and MBPP-sanitized (427 problems) as our benchmark. Data preparation involves:

1. Generate solutions (10 per problem) using a code LLM
2. Execute each solution with `sys.settrace` to collect raw execution traces
3. Extract trace conditions using `scripts/prepare_ablation_data.py`

Place prepared JSONL data files in `./data/`. Each file follows the format:
```json
{"input": "<trace representation + source code>", "output": "pass"}
{"input": "<trace representation + source code>", "output": "fail"}
```

## Trace Conditions

| Condition | Description |
|-----------|-------------|
| `exception_only` | Exception type, stacktrace, and message only |
| `VOTA` | Verdict-Optimized Trace Abstraction: exceptions + assertion comparisons + loop summaries |
| `ae_only` | Assertion-point extraction: exceptions + assertion comparisons |
| `loop_only` | Loop compression only (no exception events) |
| `full_trace` | Complete line-by-line execution trace |
| `no_trace` | Source code + test only (no trace information) |
| `nomarker` | Exception-only with format markers removed |
| `full_trace_label_only` | Full trace input with loss computed on label tokens only |

## Models

We evaluate four code-specialized models:
- Qwen2.5-Coder-7B-Instruct
- Qwen2.5-Coder-1.5B-Instruct
- CodeLlama-13B-Instruct
- DeepSeek-Coder-6.7B-Instruct

All models are fine-tuned with LoRA (rank=16, alpha=32).

## Training

```bash
python train/train_sft.py \
    --condition exception_only \
    --model_path <path_to_model> \
    --train_data ./data/exception_only_train.jsonl \
    --eval_data ./data/exception_only_test.jsonl \
    --output_dir ./checkpoints/exception_only_seed42 \
    --max_steps 500 \
    --seed 42 \
    --gradient_checkpointing
```

For gradient masking experiments:
```bash
python train/train_sft.py \
    --condition exception_only \
    --model_path <path_to_model> \
    --train_data ./data/exception_only_train.jsonl \
    --eval_data ./data/exception_only_test.jsonl \
    --output_dir ./checkpoints/grad_mask_seed42 \
    --gradient_mask_ratio 0.998 \
    --seed 42
```

## Evaluation

```bash
python eval/evaluate.py \
    --condition exception_only \
    --model_path <path_to_model> \
    --checkpoint_dir ./checkpoints/exception_only_seed42/final \
    --eval_data ./data/exception_only_test.jsonl \
    --output ./results/exception_only_seed42.json
```

## Bootstrap Comparison

```bash
python eval/bootstrap_compare.py \
    --pred_a ./results/condition_a_predictions.jsonl \
    --pred_b ./results/condition_b_predictions.jsonl
```

## License

This code is released for research purposes.
