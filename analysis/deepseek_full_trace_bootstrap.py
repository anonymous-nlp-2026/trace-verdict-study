"""DeepSeek full_trace paired bootstrap + McNemar for 3 missing pairs."""
import json, numpy as np
from scipy import stats

def load_predictions(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    return np.array([1 if l['pred'] == l['label'] else 0 for l in lines])

def paired_bootstrap(correct_a, correct_b, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(correct_a)
    observed_diff = correct_a.mean() - correct_b.mean()
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        diffs.append(correct_a[idx].mean() - correct_b[idx].mean())
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    if observed_diff > 0:
        p_value = (np.sum(diffs <= 0) + 1) / (n_bootstrap + 1)
    else:
        p_value = (np.sum(diffs >= 0) + 1) / (n_bootstrap + 1)
    return observed_diff, ci_lower, ci_upper, p_value

def mcnemar_test(correct_a, correct_b):
    b = np.sum((correct_a == 1) & (correct_b == 0))
    c = np.sum((correct_a == 0) & (correct_b == 1))
    if b + c == 0:
        return 1.0
    result = stats.binomtest(b, b + c, 0.5)
    return result.pvalue

pairs = [
    ("full_trace", "VOTA",
     "./checkpoints/deepseek_full_trace_seed42_v3/final/predictions.jsonl",
     "./checkpoints/deepseek_vota/final/predictions.jsonl"),
    ("full_trace", "exception_only",
     "./checkpoints/deepseek_full_trace_seed42_v3/final/predictions.jsonl",
     "./checkpoints/deepseek_exception_only/final/predictions.jsonl"),
    ("full_trace", "no_trace",
     "./checkpoints/deepseek_full_trace_seed42_v3/final/predictions.jsonl",
     "./checkpoints/deepseek_no_trace/final/predictions.jsonl"),
]

results = []
for name_a, name_b, path_a, path_b in pairs:
    ca = load_predictions(path_a)
    cb = load_predictions(path_b)
    assert len(ca) == 1190, f"{name_a} n={len(ca)} != 1190"
    assert len(cb) == 1190, f"{name_b} n={len(cb)} != 1190"
    diff, ci_lo, ci_hi, p_boot = paired_bootstrap(ca, cb)
    p_mcnemar = mcnemar_test(ca, cb)
    r = {
        "pair": f"{name_a} vs {name_b}",
        "acc_a": float(ca.mean()),
        "acc_b": float(cb.mean()),
        "diff": float(diff),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "p_bootstrap": float(p_boot),
        "p_mcnemar": float(p_mcnemar),
        "n": len(ca),
        "significant": bool(p_mcnemar < 0.05)
    }
    results.append(r)
    sig = '***' if p_mcnemar<0.001 else '**' if p_mcnemar<0.01 else '*' if p_mcnemar<0.05 else 'ns'
    print(f"{name_a}({ca.mean():.4f}) vs {name_b}({cb.mean():.4f}): D={diff:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] p_boot={p_boot:.2e} p_mcnemar={p_mcnemar:.2e} {sig}")

outpath = "./analysis/deepseek_full_trace_paired_bootstrap.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")

existing_path = "./analysis/deepseek_paired_comparisons.json"
try:
    with open(existing_path) as f:
        existing = json.load(f)
    if isinstance(existing, list):
        existing.extend(results)
    elif isinstance(existing, dict):
        existing["full_trace_pairs"] = results
    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Merged into {existing_path}")
except Exception as e:
    print(f"Could not merge into existing file: {e}")
