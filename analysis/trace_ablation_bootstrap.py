"""Paired bootstrap significance test for trace length ablation."""
import json
import numpy as np
from pathlib import Path
from itertools import combinations

CKPT_ROOT = Path("./checkpoints")

CONDITIONS = {
    "256tok": CKPT_ROOT / "full_trace_256tok" / "final" / "predictions.jsonl",
    "512tok": CKPT_ROOT / "full_trace_512tok" / "final" / "predictions.jsonl",
    "1024tok": CKPT_ROOT / "full_trace_1024tok" / "final" / "predictions.jsonl",
    "no_limit": CKPT_ROOT / "full_trace" / "final" / "predictions.jsonl",
    "no_trace": CKPT_ROOT / "no_trace" / "final" / "predictions.jsonl",
    "exception_only": CKPT_ROOT / "exception_only" / "final" / "predictions.jsonl",
}

PAIRS = [
    ("256tok", "512tok"),
    ("512tok", "1024tok"),
    ("1024tok", "no_limit"),
    ("256tok", "no_limit"),
    ("256tok", "no_trace"),
    ("512tok", "no_trace"),
    ("1024tok", "no_trace"),
    ("no_limit", "no_trace"),
    ("256tok", "exception_only"),
    ("512tok", "exception_only"),
    ("1024tok", "exception_only"),
    ("no_limit", "exception_only"),
]


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            preds[d["id"]] = (d["label"], d["pred"])
    return preds


def paired_bootstrap(preds_a, preds_b, n_boot=10000, seed=42):
    ids = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    n = len(ids)
    correct_a = np.array([1 if preds_a[i][0] == preds_a[i][1] else 0 for i in ids])
    correct_b = np.array([1 if preds_b[i][0] == preds_b[i][1] else 0 for i in ids])
    
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()
    obs_diff = acc_a - acc_b
    
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[b] = correct_a[idx].mean() - correct_b[idx].mean()
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    # two-sided p-value: proportion of bootstrap diffs on opposite side of 0
    if obs_diff >= 0:
        p_value = (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    else:
        p_value = (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    p_value = min(p_value * 2, 1.0)  # two-sided
    
    return {
        "n_samples": n,
        "acc_a": float(acc_a),
        "acc_b": float(acc_b),
        "diff": float(obs_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
    }


def main():
    # Load all available predictions
    loaded = {}
    for name, path in CONDITIONS.items():
        if path.exists():
            loaded[name] = load_predictions(path)
            print(f"Loaded {name}: {len(loaded[name])} samples, "
                  f"acc={sum(1 for v in loaded[name].values() if v[0]==v[1])/len(loaded[name])*100:.2f}%")
        else:
            print(f"MISSING: {name} ({path})")
    
    print(f"\n{'='*80}")
    print(f"Paired Bootstrap Significance Tests (N=10000, seed=42)")
    print(f"{'='*80}")
    print(f"{'Pair':<35} {'Diff':>8} {'95% CI':>20} {'p-value':>10} {'Sig':>5}")
    print(f"{'-'*80}")
    
    results = []
    for a, b in PAIRS:
        if a not in loaded or b not in loaded:
            print(f"{a} vs {b:<24} {'SKIPPED (missing predictions)':>53}")
            continue
        r = paired_bootstrap(loaded[a], loaded[b])
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        print(f"{a} vs {b:<24} {r['diff']*100:>+7.2f}pp "
              f"[{r['ci_lower']*100:>+6.2f}, {r['ci_upper']*100:>+6.2f}] "
              f"{r['p_value']:>9.4f} {sig:>5}")
        results.append({"pair": f"{a} vs {b}", **r, "sig": sig})
    
    # Also compute per-condition bootstrap CIs
    print(f"\n{'='*80}")
    print(f"Per-condition Bootstrap 95% CIs")
    print(f"{'='*80}")
    
    rng = np.random.RandomState(42)
    ci_results = {}
    for name, preds in loaded.items():
        ids = sorted(preds.keys())
        correct = np.array([1 if preds[i][0] == preds[i][1] else 0 for i in ids])
        acc = correct.mean()
        boot_accs = np.empty(10000)
        for b in range(10000):
            idx = rng.randint(0, len(correct), size=len(correct))
            boot_accs[b] = correct[idx].mean()
        ci_lo = np.percentile(boot_accs, 2.5)
        ci_hi = np.percentile(boot_accs, 97.5)
        print(f"  {name:<20} {acc*100:>6.2f}% [{ci_lo*100:.2f}, {ci_hi*100:.2f}]")
        ci_results[name] = {"acc": float(acc), "ci_lower": float(ci_lo), "ci_upper": float(ci_hi)}
    
    # Save JSON
    out = {
        "paired_tests": results,
        "per_condition_ci": ci_results,
        "config": {"n_bootstrap": 10000, "seed": 42},
    }
    out_path = Path("./artifacts/trace_ablation_bootstrap.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
