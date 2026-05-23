# Aggregate eval results across seeds: mean +/- std, pairwise z-tests, optional McNemar.

import argparse
import json
import glob
import os
import sys
from itertools import combinations
from math import sqrt, erfc

import numpy as np


def normal_cdf(x):
    """Standard normal CDF via erfc."""
    return 0.5 * erfc(-x / sqrt(2))


def two_proportion_ztest(p1, p2, n):
    """Two-proportion z-test (pooled) for equal sample sizes. Returns (z, p_value)."""
    p_hat = (p1 + p2) / 2
    if p_hat == 0 or p_hat == 1:
        return 0.0, 1.0
    se = sqrt(2 * p_hat * (1 - p_hat) / n)
    z = (p1 - p2) / se
    p_value = 2 * (1 - normal_cdf(abs(z)))
    return float(z), float(p_value)


def mcnemar_test(preds_a, preds_b):
    """McNemar test from two parallel per-sample prediction lists.
    Each element must have 'label' and 'pred' keys."""
    assert len(preds_a) == len(preds_b)
    b = 0  # A wrong, B correct
    c = 0  # A correct, B wrong
    for a_item, b_item in zip(preds_a, preds_b):
        a_ok = a_item["label"] == a_item["pred"]
        b_ok = b_item["label"] == b_item["pred"]
        if a_ok and not b_ok:
            c += 1
        elif not a_ok and b_ok:
            b += 1
    n_disc = b + c
    if n_disc == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0, "b": b, "c": c}
    chi2 = (abs(b - c) - 1) ** 2 / n_disc  # continuity-corrected
    p_value = erfc(sqrt(chi2 / 2))  # chi2 CDF with df=1
    return {"chi2": float(chi2), "p_value": float(p_value), "n_discordant": n_disc, "b": b, "c": c}


def load_results(paths):
    results = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        data["_path"] = p
        parent = os.path.basename(os.path.dirname(p))
        data["_seed_label"] = parent
        results.append(data)
    return results


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate seed robustness results: mean±std + pairwise statistical tests."
    )
    parser.add_argument("--pattern", default=None, help="Glob pattern for eval_results.json files")
    parser.add_argument("--paths", nargs="+", default=None, help="Explicit paths to eval_results.json")
    parser.add_argument(
        "--prediction_paths", nargs="+", default=None,
        help="Per-sample prediction JSONs (for McNemar test). "
             "Each JSON: {\"predictions\": [{\"label\": ..., \"pred\": ...}, ...]}"
    )
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    if args.paths:
        results = load_results(args.paths)
    elif args.pattern:
        results = load_results(sorted(glob.glob(args.pattern)))
    else:
        results = load_results(sorted(glob.glob(
            "./checkpoints/vota_seed*/eval_results.json"
        )))

    if not results:
        print("ERROR: No result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} seed results:")
    for r in results:
        print(f"  {r['_path']}")
    print()

    # --- Per-seed metrics ---
    metric_keys = ["accuracy", "pass_f1", "fail_f1", "pass_precision", "pass_recall",
                   "fail_precision", "fail_recall"]
    seed_data = {}
    for r in results:
        label = r["_seed_label"]
        m = r["metrics"]
        seed_data[label] = {k: m.get(k, 0.0) for k in metric_keys}
        print(f"  {label}: acc={m['accuracy']:.4f}  pass_f1={m['pass_f1']:.4f}  fail_f1={m['fail_f1']:.4f}")

    n_samples = results[0]["metrics"]["n"]
    labels = list(seed_data.keys())
    print()

    # --- Summary statistics ---
    print("=== Summary (mean ± std) ===")
    summary = {}
    for key in ["accuracy", "pass_f1", "fail_f1"]:
        vals = [seed_data[s][key] for s in seed_data]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "range_pp": float((max(vals) - min(vals)) * 100),
            "n_seeds": len(vals),
        }
        s = summary[key]
        print(f"  {key}: {s['mean']*100:.2f}% ± {s['std']*100:.2f}pp "
              f"(range: [{s['min']*100:.2f}%, {s['max']*100:.2f}%], Δ={s['range_pp']:.2f}pp)")

    # --- Pairwise two-proportion z-tests ---
    print(f"\n=== Pairwise z-tests (n={n_samples} per seed) ===")
    pairwise_z = {}
    for a, b in combinations(labels, 2):
        z, p = two_proportion_ztest(seed_data[a]["accuracy"], seed_data[b]["accuracy"], n_samples)
        key = f"{a}_vs_{b}"
        pairwise_z[key] = {
            "acc_a": seed_data[a]["accuracy"],
            "acc_b": seed_data[b]["accuracy"],
            "diff_pp": float((seed_data[a]["accuracy"] - seed_data[b]["accuracy"]) * 100),
            "z": z,
            "p_value": p,
            "significant_005": p < 0.05,
        }
        print(f"  {a} vs {b}: Δ={pairwise_z[key]['diff_pp']:+.2f}pp  "
              f"z={z:.3f}  p={p:.4f}  {sig_label(p)}")

    # --- McNemar tests (if per-sample predictions provided) ---
    mcnemar_results = {}
    if args.prediction_paths:
        print("\n=== McNemar tests (per-sample) ===")
        all_preds = {}
        for p in args.prediction_paths:
            with open(p) as f:
                data = json.load(f)
            dirname = os.path.basename(os.path.dirname(p))
            all_preds[dirname] = data["predictions"]

        pred_labels = sorted(all_preds.keys())
        for a, b in combinations(pred_labels, 2):
            result = mcnemar_test(all_preds[a], all_preds[b])
            key = f"{a}_vs_{b}"
            mcnemar_results[key] = result
            print(f"  {a} vs {b}: χ²={result['chi2']:.3f}  p={result['p_value']:.4f}  "
                  f"{sig_label(result['p_value'])}  "
                  f"(discordant: {result['c']}/{result['b']}, total={result['n_discordant']})")
    else:
        print("\nNote: pass --prediction_paths for McNemar test (requires per-sample predictions).")

    # --- Verdict ---
    acc_std = summary["accuracy"]["std"]
    acc_range = summary["accuracy"]["range_pp"]
    any_sig = any(v["significant_005"] for v in pairwise_z.values())

    print("\n=== Verdict ===")
    if acc_range < 1.0 and not any_sig:
        verdict = "STABLE"
        print(f"  {verdict}: accuracy range {acc_range:.2f}pp < 1pp, no significant pairwise differences.")
    elif acc_range < 2.0:
        verdict = "ACCEPTABLE"
        print(f"  {verdict}: accuracy range {acc_range:.2f}pp < 2pp.")
    else:
        verdict = "UNSTABLE"
        print(f"  {verdict}: accuracy range {acc_range:.2f}pp >= 2pp — investigate seed sensitivity.")

    # --- Output ---
    output_data = {
        "seeds": seed_data,
        "summary": summary,
        "pairwise_ztests": pairwise_z,
        "mcnemar": mcnemar_results or None,
        "verdict": verdict,
        "n_seeds": len(results),
        "n_samples": n_samples,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved → {args.output}")
    else:
        print("\n" + json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
