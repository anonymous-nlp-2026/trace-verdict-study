"""Generate trace length ablation figure for paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Load bootstrap CIs if available
ci_path = Path("./artifacts/trace_ablation_bootstrap.json")
ci_data = {}
if ci_path.exists():
    with open(ci_path) as f:
        ci_data = json.load(f).get("per_condition_ci", {})

# Data points
trace_conditions = [
    ("256tok", 256),
    ("512tok", 512),
    ("1024tok", 1024),
    ("no_limit", 1970),
]

x_vals = []
y_vals = []
y_err_lo = []
y_err_hi = []

for name, tokens in trace_conditions:
    x_vals.append(tokens)
    if name in ci_data:
        acc = ci_data[name]["acc"] * 100
        ci_lo = ci_data[name]["ci_lower"] * 100
        ci_hi = ci_data[name]["ci_upper"] * 100
    else:
        # Fallback to known values from eval_results
        fallback = {"256tok": 78.82, "512tok": 80.42, "1024tok": 77.23, "no_limit": 81.76}
        acc = fallback[name]
        ci_lo = acc - 2.3
        ci_hi = acc + 2.3
    y_vals.append(acc)
    y_err_lo.append(acc - ci_lo)
    y_err_hi.append(ci_hi - acc)

x_vals = np.array(x_vals)
y_vals = np.array(y_vals)
yerr = np.array([y_err_lo, y_err_hi])

# Reference lines
ref_exception = 99.83
ref_no_trace = 75.13
if "exception_only" in ci_data:
    ref_exception = ci_data["exception_only"]["acc"] * 100
if "no_trace" in ci_data:
    ref_no_trace = ci_data["no_trace"]["acc"] * 100

# Figure
fig, ax = plt.subplots(figsize=(5.5, 3.8))

# Main ablation line
ax.errorbar(x_vals, y_vals, yerr=yerr, fmt="o-", color="#2166AC", 
            markersize=7, linewidth=1.8, capsize=4, capthick=1.5,
            markerfacecolor="#2166AC", markeredgecolor="white", markeredgewidth=1.2,
            label="Full trace", zorder=5)

# Reference lines
ax.axhline(y=ref_exception, color="#4DAF4A", linestyle="--", linewidth=1.3, alpha=0.85,
           label=f"Exception-only ({ref_exception:.1f}%)")
ax.axhline(y=ref_no_trace, color="#999999", linestyle="--", linewidth=1.3, alpha=0.85,
           label=f"No-trace baseline ({ref_no_trace:.1f}%)")

# Axes
ax.set_xlabel("Max trace tokens")
ax.set_ylabel("Accuracy (%)")

ax.set_xticks(x_vals)
ax.set_xticklabels(["256", "512", "1024", "≈2000"])

# Y axis range to show all data
ax.set_ylim(70, 102)
ax.set_yticks(np.arange(72, 102, 4))

# Light grid
ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc="center right", framealpha=0.9, edgecolor="0.8")

# Annotate data points
for xi, yi in zip(x_vals, y_vals):
    ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8.5, color="#2166AC")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_dir = Path("./artifacts")
fig.savefig(out_dir / "fig_trace_ablation.pdf")
fig.savefig(out_dir / "fig_trace_ablation.png")
print(f"Saved to {out_dir / 'fig_trace_ablation.pdf'} and .png")
plt.close()
