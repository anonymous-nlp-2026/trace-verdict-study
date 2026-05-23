"""Plot training dynamics: accuracy and F1 over training steps."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_dynamics(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda x: x["step"])
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exception_only", required=True,
                        help="dynamics_log.jsonl for exception_only")
    parser.add_argument("--full_trace", required=True,
                        help="dynamics_log.jsonl for full_trace")
    parser.add_argument("--output", default="./artifacts/training_dynamics.pdf")
    args = parser.parse_args()

    eo = load_dynamics(args.exception_only)
    ft = load_dynamics(args.full_trace)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Accuracy
    ax = axes[0]
    ax.plot([e["step"] for e in eo], [e["accuracy"] * 100 for e in eo],
            "o-", color="#2196F3", label="Exception-only", markersize=4)
    ax.plot([e["step"] for e in ft], [e["accuracy"] * 100 for e in ft],
            "s-", color="#FF5722", label="Full-trace (LO)", markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Pass F1 & Fail F1
    ax = axes[1]
    ax.plot([e["step"] for e in eo], [e["pass_f1"] * 100 for e in eo],
            "o-", color="#2196F3", label="Exception-only pass", markersize=4)
    ax.plot([e["step"] for e in eo], [e["fail_f1"] * 100 for e in eo],
            "o--", color="#2196F3", label="Exception-only fail", markersize=4, alpha=0.6)
    ax.plot([e["step"] for e in ft], [e["pass_f1"] * 100 for e in ft],
            "s-", color="#FF5722", label="Full-trace (LO) pass", markersize=4)
    ax.plot([e["step"] for e in ft], [e["fail_f1"] * 100 for e in ft],
            "s--", color="#FF5722", label="Full-trace (LO) fail", markersize=4, alpha=0.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Per-class F1")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Train loss
    ax = axes[2]
    eo_loss = [(e["step"], e["train_loss"]) for e in eo if e.get("train_loss") is not None]
    ft_loss = [(e["step"], e["train_loss"]) for e in ft if e.get("train_loss") is not None]
    if eo_loss:
        ax.plot([s for s, _ in eo_loss], [l for _, l in eo_loss],
                "o-", color="#2196F3", label="Exception-only", markersize=4)
    if ft_loss:
        ax.plot([s for s, _ in ft_loss], [l for _, l in ft_loss],
                "s-", color="#FF5722", label="Full-trace (LO)", markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved to {args.output}")

    png_path = args.output.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
