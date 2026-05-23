import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6

base = "./checkpoints/calibration"
with open(f"{base}/exception_only_calibration.json") as f:
    exc = json.load(f)
with open(f"{base}/no_trace_calibration.json") as f:
    nt = json.load(f)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))

# Diagonal (perfect calibration)
ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=0.8, alpha=0.5, label='Perfect calibration')

# no_trace
nt_bins = nt['reliability_diagram']
nt_conf = [b['mean_confidence'] for b in nt_bins if b['count'] > 0]
nt_acc = [b['accuracy'] for b in nt_bins if b['count'] > 0]
ax.plot(nt_conf, nt_acc, 's-', color='#d62728', markersize=5, linewidth=1.2,
        label=f'No Trace (ECE={nt["ece"]:.4f})')

# exception_only
exc_bins = exc['reliability_diagram']
exc_conf = [b['mean_confidence'] for b in exc_bins if b['count'] > 0]
exc_acc = [b['accuracy'] for b in exc_bins if b['count'] > 0]
ax.plot(exc_conf, exc_acc, 'o-', color='#1f77b4', markersize=5, linewidth=1.2,
        label=f'Exception Only (ECE={exc["ece"]:.4f})')

ax.set_xlabel('Mean Predicted Confidence')
ax.set_ylabel('Observed Accuracy')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.0, 1.05)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.2, linewidth=0.5)

plt.tight_layout(pad=0.5)
out = "./docs/paper/figures/calibration_reliability.pdf"
plt.savefig(out, bbox_inches='tight', dpi=300)
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f"Saved: {out}")
