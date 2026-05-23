import json
import os
import numpy as np
from itertools import combinations
from scipy.stats import binomtest

BASE = "./checkpoints"

DEEPSEEK_EXPS = {
    "VOTA":           f"{BASE}/deepseek_vota/final/predictions.jsonl",
    "exception_only": f"{BASE}/deepseek_exception_only/final/predictions.jsonl",
    "full_trace":     f"{BASE}/deepseek_full_trace_seed42_v3/final/predictions.jsonl",
    "no_trace":       f"{BASE}/deepseek_no_trace/final/predictions.jsonl",
}


def load_predictions(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    records.sort(key=lambda x: x["id"])
    return records


def accuracy(preds):
    correct = sum(1 for p in preds if p["pred"] == p["label"])
    return correct / len(preds)


def mcnemar_test(preds_a, preds_b):
    assert len(preds_a) == len(preds_b)
    b = 0  # A wrong, B right
    c = 0  # A right, B wrong
    for a, bb in zip(preds_a, preds_b):
        assert a["id"] == bb["id"]
        a_correct = a["pred"] == a["label"]
        b_correct = bb["pred"] == bb["label"]
        if not a_correct and b_correct:
            b += 1
        elif a_correct and not b_correct:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    p_value = binomtest(min(b, c), n, 0.5).pvalue
    return b, c, p_value


def paired_bootstrap_ci(preds_a, preds_b, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(preds_a)
    correct_a = np.array([1 if p["pred"] == p["label"] else 0 for p in preds_a])
    correct_b = np.array([1 if p["pred"] == p["label"] else 0 for p in preds_b])
    diffs = correct_a - correct_b

    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_deltas.append(diffs[idx].mean())
    boot_deltas = np.array(boot_deltas)
    lo = np.percentile(boot_deltas, 100 * alpha / 2)
    hi = np.percentile(boot_deltas, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def main():
    loaded = {}
    for name, path in DEEPSEEK_EXPS.items():
        preds = load_predictions(path)
        loaded[name] = preds
        print(f"  Loaded {name}: n={len(preds)}, acc={accuracy(preds)*100:.2f}%")

    group_results = []
    txt_lines = []

    txt_lines.append(f"\n{'='*110}")
    txt_lines.append(f" DeepSeek Paired Comparisons (n=1190, all seed42)")
    txt_lines.append(f"{'='*110}")
    txt_lines.append(f"  VOTA = deepseek_vota (max_length=16384)")
    txt_lines.append(f"  exception_only = deepseek_exception_only (seed42)")
    txt_lines.append(f"  full_trace = deepseek_full_trace_seed42_v3")
    txt_lines.append(f"  no_trace = deepseek_no_trace (seed42)")
    txt_lines.append(f"{'='*110}")
    header = f"{'Pair (A vs B)':<40} {'Acc_A':>7} {'Acc_B':>7} {'D(A-B)':>8} {'95% CI':>20} {'McNemar p':>12} {'sig':>5}"
    txt_lines.append(header)
    txt_lines.append("-" * 110)

    pairs = list(combinations(sorted(loaded.keys()), 2))

    for name_a, name_b in pairs:
        preds_a = loaded[name_a]
        preds_b = loaded[name_b]

        acc_a = accuracy(preds_a)
        acc_b = accuracy(preds_b)
        delta = acc_a - acc_b

        b, c, p_val = mcnemar_test(preds_a, preds_b)
        ci_lo, ci_hi = paired_bootstrap_ci(preds_a, preds_b)

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        result = {
            "pair": f"{name_a} vs {name_b}",
            "name_a": name_a,
            "name_b": name_b,
            "acc_a": round(acc_a * 100, 2),
            "acc_b": round(acc_b * 100, 2),
            "delta": round(delta * 100, 2),
            "ci_lo": round(ci_lo * 100, 2),
            "ci_hi": round(ci_hi * 100, 2),
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_p": p_val,
            "sig": sig,
        }
        group_results.append(result)

        pair_str = f"{name_a} vs {name_b}"
        ci_str = f"[{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]"
        line = f"{pair_str:<40} {acc_a*100:>6.2f}% {acc_b*100:>6.2f}% {delta*100:>+7.2f}% {ci_str:>20} {p_val:>12.2e} {sig:>5}"
        txt_lines.append(line)

    all_results = {"DeepSeek (n=1190, seed42)": group_results}

    out_dir = "./analysis"
    with open(f"{out_dir}/deepseek_paired_comparisons.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    txt_content = "\n".join(txt_lines) + "\n"
    with open(f"{out_dir}/deepseek_paired_comparisons.txt", "w") as f:
        f.write(txt_content)

    print(f"\nSaved: {out_dir}/deepseek_paired_comparisons.json")
    print(f"Saved: {out_dir}/deepseek_paired_comparisons.txt")
    print(txt_content)


if __name__ == "__main__":
    main()
