"""Seed robustness analysis for ae_only condition (3 seeds: 42, 123, 456)."""

import json
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

CHECKPOINT_BASE = Path("./checkpoints")
SEEDS = {
    42: CHECKPOINT_BASE / "ae_only" / "final",
    123: CHECKPOINT_BASE / "ae_only_seed123" / "final",
    456: CHECKPOINT_BASE / "ae_only_seed456" / "final",
}


def load_eval_results(seed_dirs: dict) -> dict:
    results = {}
    for seed, path in seed_dirs.items():
        eval_file = path / "eval_results.json"
        if not eval_file.exists():
            print(f"[WARN] seed={seed}: {eval_file} not found, skipping")
            continue
        with open(eval_file) as f:
            results[seed] = json.load(f)
    return results


def load_predictions(seed_dirs: dict) -> dict:
    preds = {}
    for seed, path in seed_dirs.items():
        pred_file = path / "predictions.jsonl"
        if not pred_file.exists():
            continue
        items = []
        with open(pred_file) as f:
            for line in f:
                items.append(json.loads(line))
        preds[seed] = items
    return preds


def mcnemar_test(preds_a: list, preds_b: list) -> dict:
    """McNemar's test for paired nominal data."""
    assert len(preds_a) == len(preds_b)
    b = 0  # a correct, b wrong
    c = 0  # a wrong, b correct
    for pa, pb in zip(preds_a, preds_b):
        a_correct = pa["prediction"] == pa["label"]
        b_correct = pb["prediction"] == pb["label"]
        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1
    n_discordant = b + c
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"statistic": round(statistic, 4), "p_value": round(p_value, 4), "b": b, "c": c}


def paired_bootstrap_ci(preds_a: list, preds_b: list, n_bootstrap: int = 10000, alpha: float = 0.05) -> dict:
    """Paired bootstrap confidence interval for accuracy difference."""
    np.random.seed(42)
    n = len(preds_a)
    a_correct = np.array([p["prediction"] == p["label"] for p in preds_a])
    b_correct = np.array([p["prediction"] == p["label"] for p in preds_b])
    observed_diff = a_correct.mean() - b_correct.mean()

    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diffs.append(a_correct[idx].mean() - b_correct[idx].mean())
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 100 * alpha / 2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return {
        "observed_diff": round(observed_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "significant": not (ci_lower <= 0 <= ci_upper),
    }


def main():
    results = load_eval_results(SEEDS)
    if len(results) < 2:
        print(f"Only {len(results)} seed results available. Need at least 2.")
        sys.exit(1)

    # Mean ± std accuracy
    accuracies = {s: r["metrics"]["accuracy"] for s, r in results.items()}
    acc_values = list(accuracies.values())
    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values, ddof=1) if len(acc_values) > 1 else 0.0

    print("=" * 60)
    print("SEED ROBUSTNESS ANALYSIS: ae_only condition")
    print("=" * 60)
    print(f"\nSeeds available: {sorted(results.keys())}")
    print(f"\nPer-seed accuracy:")
    for s in sorted(accuracies):
        print(f"  seed {s:>3d}: {accuracies[s]*100:.2f}%")
    print(f"\n  Mean ± Std: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Range: [{min(acc_values)*100:.2f}%, {max(acc_values)*100:.2f}%]")

    # Per-class F1
    print(f"\nPer-seed F1 (pass / fail):")
    for s in sorted(results):
        m = results[s]["metrics"]
        print(f"  seed {s:>3d}: pass_f1={m['pass_f1']:.4f}, fail_f1={m['fail_f1']:.4f}")

    # McNemar + paired bootstrap (if predictions available)
    preds = load_predictions(SEEDS)
    if len(preds) >= 2:
        print(f"\n{'='*60}")
        print("PAIRWISE COMPARISONS (McNemar + Paired Bootstrap)")
        print("=" * 60)
        for (s1, s2) in combinations(sorted(preds.keys()), 2):
            print(f"\n  seed {s1} vs seed {s2}:")
            try:
                mc = mcnemar_test(preds[s1], preds[s2])
                print(f"    McNemar: χ²={mc['statistic']}, p={mc['p_value']}, discordant=({mc['b']},{mc['c']})")
            except Exception as e:
                print(f"    McNemar: failed ({e})")
            pb = paired_bootstrap_ci(preds[s1], preds[s2])
            print(f"    Bootstrap Δacc: {pb['observed_diff']*100:+.3f}% CI=[{pb['ci_lower']*100:.3f}%, {pb['ci_upper']*100:.3f}%] sig={pb['significant']}")

    # LaTeX table output
    print(f"\n{'='*60}")
    print("LATEX TABLE OUTPUT")
    print("=" * 60)
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Seed robustness for \texttt{ae\_only} condition (3 seeds).}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Seed & Accuracy (\%) & Pass $F_1$ & Fail $F_1$ \\")
    print(r"\midrule")
    for s in sorted(results):
        m = results[s]["metrics"]
        print(f"{s} & {m['accuracy']*100:.2f} & {m['pass_f1']:.4f} & {m['fail_f1']:.4f} \\\\")
    print(r"\midrule")
    print(f"Mean $\\pm$ Std & {mean_acc*100:.2f} $\\pm$ {std_acc*100:.2f} & --- & --- \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save summary JSON
    summary = {
        "condition": "ae_only",
        "seeds": sorted(results.keys()),
        "per_seed_accuracy": accuracies,
        "mean_accuracy": round(mean_acc, 6),
        "std_accuracy": round(std_acc, 6),
        "n_samples": results[list(results.keys())[0]]["metrics"]["n"],
    }
    out_path = Path(__file__).parent / "seed_robustness_ae_only.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()
