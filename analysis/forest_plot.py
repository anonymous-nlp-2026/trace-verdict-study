"""Forest plot: sample-level vs clustered bootstrap CIs for all conditions."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("./artifacts/clustered_bootstrap")

with open(OUT_DIR / "bootstrap_results.json") as f:
    data = json.load(f)

conditions = ["no_trace", "full_trace", "loop_only", "vota_no_ae", "vota", "ae_only", "exception_only"]
display = {
    "no_trace": "No Trace",
    "full_trace": "Full Trace",
    "loop_only": "Loop Only",
    "vota_no_ae": "VoTA (no AE)",
    "vota": "VoTA",
    "ae_only": "AE Only",
    "exception_only": "Exception Only",
}

fig, ax = plt.subplots(figsize=(8, 4.5))

y_positions = np.arange(len(conditions))
bar_height = 0.18

for i, cond in enumerate(conditions):
    r = data["per_condition"][cond]
    pt = r["point_estimate"] * 100

    s_lo = r["sample_ci_lower"] * 100
    s_hi = r["sample_ci_upper"] * 100
    c_lo = r["clustered_ci_lower"] * 100
    c_hi = r["clustered_ci_upper"] * 100

    # Clustered CI (wider, behind)
    ax.barh(i + bar_height/2, c_hi - c_lo, left=c_lo, height=bar_height,
            color="#e74c3c", alpha=0.35, edgecolor="#e74c3c", linewidth=0.8,
            label="Clustered CI" if i == 0 else None)
    # Sample CI (narrower, in front)
    ax.barh(i - bar_height/2, s_hi - s_lo, left=s_lo, height=bar_height,
            color="#3498db", alpha=0.5, edgecolor="#3498db", linewidth=0.8,
            label="Sample CI" if i == 0 else None)
    # Point estimate
    ax.plot(pt, i, "ko", markersize=5, zorder=5)
    # Accuracy label
    ax.text(pt + 0.3, i, f"{pt:.2f}%", va="center", fontsize=8)

ax.set_yticks(y_positions)
ax.set_yticklabels([display[c] for c in conditions], fontsize=9)
ax.set_xlabel("Accuracy (%)", fontsize=10)
ax.set_title("Sample-level vs Problem-level Clustered Bootstrap 95% CI", fontsize=11)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_xlim(62, 102)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
fig.savefig(OUT_DIR / "forest_plot_ci.pdf", bbox_inches="tight", dpi=300)
fig.savefig(OUT_DIR / "forest_plot_ci.png", bbox_inches="tight", dpi=200)
print(f"Saved to {OUT_DIR}/forest_plot_ci.{{pdf,png}}")

# Second plot: pairwise differences
pairs = [
    ("exception_only_vs_ae_only", "Exc. Only vs AE Only"),
    ("loop_only_vs_full_trace", "Loop Only vs Full Trace"),
    ("exception_only_vs_loop_only", "Exc. Only vs Loop Only"),
    ("vota_vs_no_trace", "VoTA vs No Trace"),
    ("vota_vs_full_trace", "VoTA vs Full Trace"),
    ("exception_only_vs_no_trace", "Exc. Only vs No Trace"),
    ("exception_only_vs_full_trace", "Exc. Only vs Full Trace"),
    ("ae_only_vs_loop_only", "AE Only vs Loop Only"),
    ("vota_no_ae_vs_vota", "VoTA (no AE) vs VoTA"),
]

fig2, ax2 = plt.subplots(figsize=(8, 5))
for i, (key, label) in enumerate(pairs):
    pr = data["pairwise_tests"][key]
    diff = pr["observed_diff"] * 100
    lo = pr["ci_lower"] * 100
    hi = pr["ci_upper"] * 100
    sig = pr["significant"]
    color = "#2ecc71" if sig else "#e74c3c"

    ax2.plot(diff, i, "o", color=color, markersize=6, zorder=5)
    ax2.hlines(i, lo, hi, color=color, linewidth=2, alpha=0.7)
    sig_label = f"Δ={diff:+.2f}pp {'*' if sig else '(ns)'}"
    ax2.text(max(hi + 0.5, diff + 0.5), i, sig_label, va="center", fontsize=7.5)

ax2.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_yticks(range(len(pairs)))
ax2.set_yticklabels([p[1] for p in pairs], fontsize=8.5)
ax2.set_xlabel("Accuracy Difference (pp)", fontsize=10)
ax2.set_title("Pairwise Differences with Clustered Bootstrap 95% CI", fontsize=11)
ax2.grid(axis="x", alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
fig2.savefig(OUT_DIR / "pairwise_diff_ci.pdf", bbox_inches="tight", dpi=300)
fig2.savefig(OUT_DIR / "pairwise_diff_ci.png", bbox_inches="tight", dpi=200)
print(f"Saved to {OUT_DIR}/pairwise_diff_ci.{{pdf,png}}")
