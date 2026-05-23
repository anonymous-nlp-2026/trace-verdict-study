"""Dedup Sensitivity Analysis: compare sample-level vs problem-level accuracy."""

import json
import os
import numpy as np
from collections import defaultdict

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"

# Map: (display_name, checkpoint_subdir, condition_for_test_file)
EXPERIMENTS = [
    # Qwen-7B main conditions
    ("Qwen-7B", "exception_only", "exception_only"),
    ("Qwen-7B", "no_trace", "no_trace"),
    ("Qwen-7B", "full_trace", "full_trace"),
    ("Qwen-7B", "vota", "vota"),
    ("Qwen-7B", "ae_only", "ae_only"),
    ("Qwen-7B", "loop_only", "loop_only"),
    ("Qwen-7B", "vota_no_ae", "vota_no_ae"),
    ("Qwen-7B", "full_trace_label_only", "full_trace"),
    # DeepSeek conditions
    ("DeepSeek", "deepseek_exception_only", "exception_only"),
    ("DeepSeek", "deepseek_no_trace", "no_trace"),
    ("DeepSeek", "deepseek_full_trace", "full_trace"),
    ("DeepSeek", "deepseek_vota", "vota"),
    # CodeLlama-13B conditions
    ("CodeLlama-13B", "codellama13b_exception_only_seed42", "exception_only"),
    ("CodeLlama-13B", "codellama13b_no_trace_seed42", "no_trace"),
    ("CodeLlama-13B", "codellama13b_vota_seed42", "vota"),
    # Qwen-1.5B conditions
    ("Qwen-1.5B", "qwen15b_exception_only_seed42", "exception_only"),
    ("Qwen-1.5B", "qwen15b_no_trace_seed42", "no_trace"),
    ("Qwen-1.5B", "qwen15b_full_trace_seed42", "full_trace"),
    ("Qwen-1.5B", "qwen15b_vota_seed42", "vota"),
]

def load_test_data(condition):
    path = os.path.join(DATA_DIR, f"{condition}_test.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_predictions(ckpt_name):
    path = os.path.join(CKPT_DIR, ckpt_name, "final", "predictions.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def bootstrap_dedup_accuracy(problem_accs, n_bootstrap=10000, seed=42):
    """Bootstrap CI over problem-level accuracies."""
    rng = np.random.RandomState(seed)
    accs = np.array(problem_accs)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(accs, size=len(accs), replace=True)
        means.append(sample.mean())
    means = sorted(means)
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi

def analyze_experiment(model_name, ckpt_name, condition):
    test_data = load_test_data(condition)
    preds = load_predictions(ckpt_name)
    if test_data is None or preds is None:
        return None

    if len(test_data) != len(preds):
        return None

    # Group by problem_id
    problem_groups = defaultdict(list)
    for i, (td, pr) in enumerate(zip(test_data, preds)):
        pid = td.get("problem_id", f"unknown_{i}")
        correct = 1 if td["output"] == pr["pred"] else 0
        problem_groups[pid].append(correct)

    n_total = len(test_data)
    n_problems = len(problem_groups)
    n_correct_total = sum(sum(v) for v in problem_groups.values())

    # Sample-level accuracy
    sample_acc = n_correct_total / n_total

    # Problem-level accuracy (macro average)
    problem_accs = [sum(v) / len(v) for v in problem_groups.values()]
    problem_acc = np.mean(problem_accs)

    # Bootstrap CI for problem-level accuracy
    ci_lo, ci_hi = bootstrap_dedup_accuracy(problem_accs)

    delta = problem_acc - sample_acc

    return {
        "model": model_name,
        "condition": ckpt_name,
        "n_total": n_total,
        "n_problems": n_problems,
        "sample_acc": sample_acc,
        "problem_acc": problem_acc,
        "delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }

def main():
    results = []
    for model_name, ckpt_name, condition in EXPERIMENTS:
        r = analyze_experiment(model_name, ckpt_name, condition)
        if r is not None:
            results.append(r)

    # Print table
    print(f"{'Model':<15} {'Condition':<40} {'n_samp':>6} {'n_prob':>6} "
          f"{'SampleAcc':>10} {'ProblemAcc':>11} {'Delta':>7} {'95% CI':>20}")
    print("-" * 130)

    for r in results:
        ci_str = f"[{r['ci_lo']*100:.1f}%, {r['ci_hi']*100:.1f}%]"
        print(f"{r['model']:<15} {r['condition']:<40} {r['n_total']:>6} {r['n_problems']:>6} "
              f"{r['sample_acc']*100:>9.1f}% {r['problem_acc']*100:>10.1f}% "
              f"{r['delta']*100:>+6.1f}pp {ci_str:>20}")

    # Summary statistics
    deltas = [abs(r['delta']) for r in results]
    print(f"\n--- Summary ---")
    print(f"Number of experiments analyzed: {len(results)}")
    print(f"Absolute delta (sample vs problem-level): "
          f"mean={np.mean(deltas)*100:.2f}pp, max={np.max(deltas)*100:.2f}pp, "
          f"median={np.median(deltas)*100:.2f}pp")
    print(f"All deltas < 1pp: {all(d < 0.01 for d in deltas)}")
    print(f"All deltas < 2pp: {all(d < 0.02 for d in deltas)}")

if __name__ == "__main__":
    main()
