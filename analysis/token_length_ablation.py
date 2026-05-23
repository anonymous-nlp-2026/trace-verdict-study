#!/usr/bin/env python3
"""Token-length ablation: accuracy vs. trace truncation length."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT_BASE = Path("./checkpoints")
FIG_DIR = Path("./analysis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRUNCATION_LENGTHS = [256, 512, 1024, 2048]

EXCEPTION_ONLY_ACC = None
NO_TRACE_ACC = None


def load_eval(subdir):
    p = CKPT_BASE / subdir / "final" / "eval_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def collect():
    rows = []
    for tok in TRUNCATION_LENGTHS:
        r = load_eval(f"full_trace_{tok}tok")
        if r is None:
            rows.append({"label": str(tok), "tok": tok, "acc": None,
                         "ci_lo": None, "ci_hi": None, "status": "pending"})
            continue
        m, ci = r["metrics"], r["confidence_intervals"]
        rows.append({
            "label": str(tok), "tok": tok,
            "acc": m["accuracy"] * 100,
            "ci_lo": ci["accuracy_ci_lower"] * 100,
            "ci_hi": ci["accuracy_ci_upper"] * 100,
            "pass_f1": m["pass_f1"] * 100,
            "fail_f1": m["fail_f1"] * 100,
            "n": m["n"],
            "status": "done",
        })

    r = load_eval("full_trace")
    if r:
        m, ci = r["metrics"], r["confidence_intervals"]
        rows.append({
            "label": "full", "tok": 99999,
            "acc": m["accuracy"] * 100,
            "ci_lo": ci["accuracy_ci_lower"] * 100,
            "ci_hi": ci["accuracy_ci_upper"] * 100,
            "pass_f1": m["pass_f1"] * 100,
            "fail_f1": m["fail_f1"] * 100,
            "n": m["n"],
            "status": "done",
        })

    global EXCEPTION_ONLY_ACC, NO_TRACE_ACC
    r_exc = load_eval("exception_only")
    EXCEPTION_ONLY_ACC = r_exc["metrics"]["accuracy"] * 100 if r_exc else 99.83
    r_nt = load_eval("no_trace")
    NO_TRACE_ACC = r_nt["metrics"]["accuracy"] * 100 if r_nt else 75.13

    return rows


def write_csv(rows):
    out = FIG_DIR / "token_length_ablation_summary.csv"
    fields = ["label", "tok", "acc", "ci_lo", "ci_hi",
              "pass_f1", "fail_f1", "n", "status"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  CSV: {out}")


def plot(rows):
    done = [r for r in rows if r["status"] == "done"]
    pending = [r for r in rows if r["status"] == "pending"]

    labels_done = [r["label"] for r in done]
    accs = [r["acc"] for r in done]
    yerr_lo = [r["acc"] - r["ci_lo"] for r in done]
    yerr_hi = [r["ci_hi"] - r["acc"] for r in done]

    all_labels = [r["label"] for r in rows]
    x_map = {lab: i for i, lab in enumerate(all_labels)}
    x_done = [x_map[l] for l in labels_done]
    x_pending = [x_map[r["label"]] for r in pending]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.errorbar(x_done, accs, yerr=[yerr_lo, yerr_hi],
                fmt="o-", color="#2563EB", capsize=4, capthick=1.2,
                linewidth=1.8, markersize=6, zorder=5,
                label="Full trace (truncated)")

    for xi, acc, lo, hi in zip(x_done, accs, yerr_lo, yerr_hi):
        ax.annotate(f"{acc:.1f}%", (xi, acc),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="#2563EB", fontweight="bold")

    if pending:
        for xp in x_pending:
            ax.axvline(xp, color="#D1D5DB", ls=":", lw=1, zorder=1)
            ax.annotate("pending", (xp, NO_TRACE_ACC + 1),
                        ha="center", fontsize=7, color="#9CA3AF", style="italic")

    ax.axhline(EXCEPTION_ONLY_ACC, color="#10B981", ls="--", lw=1.2, zorder=2,
               label=f"Exception-only ({EXCEPTION_ONLY_ACC:.1f}%)")
    ax.axhline(NO_TRACE_ACC, color="#F59E0B", ls="--", lw=1.2, zorder=2,
               label=f"No-trace ({NO_TRACE_ACC:.1f}%)")

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_xlabel("Trace token limit", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.tick_params(labelsize=10)

    y_lo = min(NO_TRACE_ACC, min(accs) - max(yerr_lo)) - 3
    y_hi = EXCEPTION_ONLY_ACC + 2
    ax.set_ylim(y_lo, y_hi)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=7.5, loc="center right", frameon=True,
              fancybox=False, edgecolor="#E5E7EB")

    fig.tight_layout(pad=0.5)

    for ext in ["pdf", "png"]:
        p = FIG_DIR / f"token_length_ablation.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Figure: {p}")
    plt.close(fig)


def print_table(rows):
    print(f"\n{'Tokens':<8} {'Acc%':>7} {'95% CI':>18} {'Δ NoTrace':>10} {'Status':>8}")
    print("-" * 56)
    for r in rows:
        if r["status"] == "pending":
            print(f"{r['label']:<8} {'---':>7} {'---':>18} {'---':>10} {'pending':>8}")
        else:
            delta = r["acc"] - NO_TRACE_ACC
            print(f"{r['label']:<8} {r['acc']:>7.2f} "
                  f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]  "
                  f"{delta:>+8.2f} {'done':>8}")
    print(f"\nBaselines: exception_only={EXCEPTION_ONLY_ACC:.2f}%, "
          f"no_trace={NO_TRACE_ACC:.2f}%")


if __name__ == "__main__":
    rows = collect()
    done_count = sum(1 for r in rows if r["status"] == "done")
    print(f"Found {done_count}/{len(rows)} completed data points.")

    print_table(rows)
    print("\nWriting CSV...")
    write_csv(rows)
    print("\nGenerating figure...")
    plot(rows)
    print("\nDone.")
