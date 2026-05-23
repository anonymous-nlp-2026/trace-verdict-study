#!/usr/bin/env python3
"""Token-length ablation analysis for full_trace verdict prediction."""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT_BASE = Path("./checkpoints")
FIG_DIR = Path("./analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    256:  CKPT_BASE / "full_trace_256tok"  / "final" / "eval_results.json",
    512:  CKPT_BASE / "full_trace_512tok"  / "final" / "eval_results.json",
    1024: CKPT_BASE / "full_trace_1024tok" / "final" / "eval_results.json",
    2048: CKPT_BASE / "full_trace_2048tok" / "final" / "eval_results.json",
}
UNTRUNCATED_PATH = CKPT_BASE / "full_trace" / "final" / "eval_results.json"
NO_TRACE_PATH    = CKPT_BASE / "no_trace"   / "final" / "eval_results.json"

MAJORITY_BASELINE = 0.7151


def load_result(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_data():
    rows = []
    for tok, path in sorted(EXPERIMENTS.items()):
        r = load_result(path)
        if r is None:
            print(f"  [SKIP] {tok}tok — not found: {path}")
            continue
        m = r["metrics"]
        ci = r["confidence_intervals"]
        rows.append({
            "label": f"{tok}",
            "tok": tok,
            "acc": m["accuracy"] * 100,
            "ci_lo": ci["accuracy_ci_lower"] * 100,
            "ci_hi": ci["accuracy_ci_upper"] * 100,
            "n": m["n"],
            "pass_f1": m["pass_f1"] * 100,
            "fail_f1": m["fail_f1"] * 100,
        })

    r = load_result(UNTRUNCATED_PATH)
    if r:
        m = r["metrics"]
        ci = r["confidence_intervals"]
        rows.append({
            "label": "full",
            "tok": 9999,
            "acc": m["accuracy"] * 100,
            "ci_lo": ci["accuracy_ci_lower"] * 100,
            "ci_hi": ci["accuracy_ci_upper"] * 100,
            "n": m["n"],
            "pass_f1": m["pass_f1"] * 100,
            "fail_f1": m["fail_f1"] * 100,
        })
    else:
        print("  [SKIP] untruncated — not found")

    no_trace = load_result(NO_TRACE_PATH)
    no_trace_acc = no_trace["metrics"]["accuracy"] * 100 if no_trace else 75.13

    return rows, no_trace_acc


def plot_curve(rows, no_trace_acc):
    labels = [r["label"] for r in rows]
    accs   = [r["acc"]    for r in rows]
    ci_lo  = [r["ci_lo"]  for r in rows]
    ci_hi  = [r["ci_hi"]  for r in rows]
    yerr_lo = [a - lo for a, lo in zip(accs, ci_lo)]
    yerr_hi = [hi - a for a, hi in zip(accs, ci_hi)]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.errorbar(x, accs, yerr=[yerr_lo, yerr_hi],
                fmt="o-", color="#2563EB", capsize=3, capthick=1.2,
                linewidth=1.5, markersize=5, zorder=3)

    ax.axhline(MAJORITY_BASELINE * 100, color="#9CA3AF", ls="--", lw=1,
               label=f"Majority baseline ({MAJORITY_BASELINE*100:.1f}%)")
    ax.axhline(no_trace_acc, color="#F59E0B", ls="--", lw=1,
               label=f"No-trace baseline ({no_trace_acc:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Trace token limit", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.tick_params(labelsize=10)

    ymin = min(MAJORITY_BASELINE * 100, min(ci_lo)) - 2
    ymax = max(max(ci_hi), no_trace_acc) + 2
    ax.set_ylim(ymin, ymax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout(pad=0.4)

    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"token_length_ablation.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def latex_table(rows, no_trace_acc):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Token-length ablation for full-trace verdict prediction.}")
    lines.append(r"\label{tab:token-length-ablation}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Token Limit & Accuracy (\%) & 95\% CI & $\Delta$ vs.\ No-trace \\")
    lines.append(r"\midrule")
    lines.append(f"Majority baseline & {MAJORITY_BASELINE*100:.2f} & --- & --- \\\\")
    lines.append(f"No-trace baseline & {no_trace_acc:.2f} & --- & --- \\\\")
    lines.append(r"\midrule")
    for r in rows:
        delta = r["acc"] - no_trace_acc
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{r['label']} & {r['acc']:.2f} & "
            f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] & "
            f"{sign}{delta:.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    out = FIG_DIR / "token_length_ablation_table.tex"
    with open(out, "w") as f:
        f.write(tex)
    print(f"  Saved: {out}")
    print()
    print(tex)


def print_summary(rows, no_trace_acc):
    print(f"\n{'Label':<8} {'Acc%':>7} {'CI':>18} {'ΔNoTrace':>10}")
    print("-" * 48)
    for r in rows:
        delta = r["acc"] - no_trace_acc
        print(f"{r['label']:<8} {r['acc']:>7.2f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]{delta:>+10.2f}")


if __name__ == "__main__":
    print("Collecting results...")
    rows, no_trace_acc = collect_data()
    if not rows:
        print("No results found. Exiting.")
        sys.exit(1)

    print(f"\nFound {len(rows)} data points.")
    print_summary(rows, no_trace_acc)
    print("\nGenerating figure...")
    plot_curve(rows, no_trace_acc)
    print("\nGenerating LaTeX table...")
    latex_table(rows, no_trace_acc)
    print("\nDone.")
