import json
import numpy as np
from pathlib import Path

CHECKPOINT_DIR = Path("./checkpoints")
DATA_DIR = Path("./data")
OUTPUT_DIR = CHECKPOINT_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
B = 10000

# Mapping: condition -> test data file
# DeepSeek conditions use same test data as Qwen counterparts
CONDITION_TO_TEST = {
    "ae_only": "ae_only_test.jsonl",
    "exception_only": "exception_only_test.jsonl",
    "vota": "vota_test.jsonl",
    "loop_only": "loop_only_test.jsonl",
    "full_trace": "full_trace_test.jsonl",
    "no_trace": "no_trace_test.jsonl",
    "full_trace_label_only": "full_trace_test.jsonl",  # same test set, different training
    "vota_no_ae": "vota_no_ae_test.jsonl",
    "deepseek_vota": "vota_test.jsonl",
    "deepseek_no_trace": "no_trace_test.jsonl",
    "deepseek_exception_only": "exception_only_test.jsonl",
    "deepseek_full_trace": "full_trace_test.jsonl",
}

# Override paths for specific conditions
PRED_PATH_OVERRIDE = {
    "deepseek_full_trace": CHECKPOINT_DIR / "deepseek_full_trace_seed42_v3" / "final" / "predictions.jsonl",
}

def load_predictions_with_problem_ids(condition):
    pred_path = PRED_PATH_OVERRIDE.get(condition, CHECKPOINT_DIR / condition / "final" / "predictions.jsonl")
    test_file = CONDITION_TO_TEST.get(condition)
    if test_file is None:
        return None
    test_path = DATA_DIR / test_file
    if not pred_path.exists() or not test_path.exists():
        return None

    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))

    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    if len(preds) != len(test_data):
        print(f"  WARNING: {condition} pred={len(preds)} test={len(test_data)} mismatch!")
        return None

    # Join by line index
    results = []
    for i, (p, t) in enumerate(zip(preds, test_data)):
        correct = (p["pred"] == p["label"])
        results.append({
            "problem_id": t["problem_id"],
            "correct": correct
        })
    return results


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (center - margin, center + margin)


def cluster_robust_bootstrap_ci(results, rng, B=10000):
    # Group by problem_id
    from collections import defaultdict
    problem_data = defaultdict(list)
    for r in results:
        problem_data[r["problem_id"]].append(r["correct"])

    problem_ids = list(problem_data.keys())
    n_problems = len(problem_ids)

    # Pre-compute per-problem arrays
    problem_correct = {pid: np.array(vals) for pid, vals in problem_data.items()}

    # Bootstrap: resample problems, compute accuracy over all samples in resampled problems
    boot_accs = np.empty(B)
    for b in range(B):
        sampled_pids = rng.choice(problem_ids, size=n_problems, replace=True)
        total_correct = 0
        total_samples = 0
        for pid in sampled_pids:
            arr = problem_correct[pid]
            total_correct += arr.sum()
            total_samples += len(arr)
        boot_accs[b] = total_correct / total_samples

    ci_lower = np.percentile(boot_accs, 2.5)
    ci_upper = np.percentile(boot_accs, 97.5)

    # Problem-level accuracy (mean of per-problem accuracies)
    problem_accs = np.array([problem_correct[pid].mean() for pid in problem_ids])
    problem_accuracy = problem_accs.mean()

    return ci_lower, ci_upper, n_problems, problem_accuracy


def main():
    rng = np.random.default_rng(SEED)

    conditions_results = {}
    for condition in CONDITION_TO_TEST:
        print(f"Processing: {condition}")
        results = load_predictions_with_problem_ids(condition)
        if results is None:
            print(f"  SKIPPED (missing files)")
            continue

        n_samples = len(results)
        n_correct = sum(r["correct"] for r in results)
        sample_accuracy = n_correct / n_samples

        ci_lower, ci_upper, n_problems, problem_accuracy = cluster_robust_bootstrap_ci(results, rng, B=B)
        w_lower, w_upper = wilson_ci(n_correct, n_samples)

        cluster_width = ci_upper - ci_lower
        wilson_width = w_upper - w_lower
        ci_width_ratio = cluster_width / wilson_width if wilson_width > 0 else float('inf')

        conditions_results[condition] = {
            "n_samples": n_samples,
            "n_problems": n_problems,
            "sample_accuracy": round(sample_accuracy, 6),
            "problem_accuracy": round(problem_accuracy, 6),
            "cluster_robust_ci": [round(ci_lower, 6), round(ci_upper, 6)],
            "wilson_ci": [round(w_lower, 6), round(w_upper, 6)],
            "ci_width_ratio": round(ci_width_ratio, 4)
        }
        print(f"  n={n_samples}, problems={n_problems}, acc={sample_accuracy:.4f}, "
              f"cluster_ci=[{ci_lower:.4f}, {ci_upper:.4f}], "
              f"wilson_ci=[{w_lower:.4f}, {w_upper:.4f}], ratio={ci_width_ratio:.2f}")

    # Save JSON
    output = {
        "method": "cluster-robust bootstrap",
        "B": B,
        "seed": SEED,
        "cluster_unit": "problem_id",
        "conditions": conditions_results
    }
    json_path = OUTPUT_DIR / "cluster_robust_ci.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save markdown summary
    md_lines = [
        "# Cluster-Robust Bootstrap 95% CI Summary",
        "",
        f"Method: Cluster-robust bootstrap (B={B}, seed={SEED}, cluster_unit=problem_id)",
        "",
        "| Condition | N | Problems | Sample Acc | Problem Acc | Cluster CI | Wilson CI | Width Ratio |",
        "|-----------|---|----------|-----------|-------------|------------|-----------|-------------|",
    ]
    for cond, r in sorted(conditions_results.items()):
        md_lines.append(
            f"| {cond} | {r['n_samples']} | {r['n_problems']} | "
            f"{r['sample_accuracy']:.4f} | {r['problem_accuracy']:.4f} | "
            f"[{r['cluster_robust_ci'][0]:.4f}, {r['cluster_robust_ci'][1]:.4f}] | "
            f"[{r['wilson_ci'][0]:.4f}, {r['wilson_ci'][1]:.4f}] | "
            f"{r['ci_width_ratio']:.2f} |"
        )
    md_lines.append("")

    md_path = OUTPUT_DIR / "cluster_robust_ci_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
