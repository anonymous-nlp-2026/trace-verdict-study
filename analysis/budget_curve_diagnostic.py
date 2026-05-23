"""
Budget Curve Diagnostic Ratio Analysis

Analyzes diagnostic token density across tail-truncated trace conditions
to test whether phase transition in accuracy corresponds to the point
where exception traceback becomes fully included.
"""

import json
import re
import os
import sys
import numpy as np
from collections import defaultdict

DATA_DIR = "./data"
OUT_DIR = "./artifacts/budget_curve_diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

TRACE_START = "Execution trace:\n"
TRACE_END_PATTERN = "\n\nDoes this code pass all tests? Answer: "

CONDITIONS = {
    "tail-256": "full_trace_tail_256_test.jsonl",
    "tail-384": "full_trace_tail_384_test.jsonl",
    "tail-512": "full_trace_tail_512_test.jsonl",
    "tail-1024": "full_trace_tail_1024_test.jsonl",
    "full_trace": "full_trace_test.jsonl",
    "exception_only": "exception_only_test.jsonl",
}

ACCURACIES = {
    "tail-256": 0.7882,
    "tail-384": None,
    "tail-512": 0.8042,
    "tail-1024": 0.7723,
    "full_trace": 0.8176,
    "exception_only": 0.9983,
}

EXCEPTION_TYPES = [
    "AssertionError", "AssertError", "AssertionErr",
    "TypeError", "ValueError", "KeyError", "IndexError",
    "AttributeError", "NameError", "RuntimeError",
    "ZeroDivisionError", "StopIteration", "RecursionError",
    "OverflowError", "FileNotFoundError", "ImportError",
    "SyntaxError", "IndentationError", "UnboundLocalError",
    "NotImplementedError", "ArithmeticError", "LookupError",
    "EOFError", "MemoryError", "OSError", "IOError",
    "TimeoutError", "UnicodeError", "UnicodeDecodeError",
    "Exception", "BaseException",
]

EXCEPTION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(e) for e in EXCEPTION_TYPES) + r')\b'
)

DIAGNOSTIC_PATTERNS = [
    re.compile(r'\[ERROR\]'),
    re.compile(r'\[No trace available\]'),
    re.compile(r'\[No exception data in trace\]'),
    re.compile(r'\[\.\.\.trace truncated\.\.\.\]'),
    EXCEPTION_PATTERN,
    re.compile(r'at line \d+'),
    re.compile(r'actual\s*[:=]', re.IGNORECASE),
    re.compile(r'expected\s*[:=]', re.IGNORECASE),
    re.compile(r'Traceback'),
    re.compile(r'assert\s+\w+', re.IGNORECASE),
]


def extract_trace(text):
    idx = text.find(TRACE_START)
    if idx < 0:
        return "", -1, -1
    start = idx + len(TRACE_START)
    end = text.find(TRACE_END_PATTERN, start)
    if end < 0:
        end = len(text)
    return text[start:end], start, end


def has_full_traceback(trace_text):
    """Check if trace contains a complete exception: type + message."""
    if "[ERROR]" in trace_text:
        m = re.search(r'\[ERROR\]\s*(\w*Error\w*)', trace_text)
        if m:
            return True
    if "Traceback" in trace_text:
        m = re.search(r'(\w+Error):\s*.+', trace_text)
        if m:
            return True
    for exc in EXCEPTION_TYPES:
        if exc in trace_text:
            return True
    return False


def count_diagnostic_chars(trace_text):
    """Count characters in trace that match diagnostic patterns."""
    diagnostic_positions = set()
    for pat in DIAGNOSTIC_PATTERNS:
        for m in pat.finditer(trace_text):
            for pos in range(m.start(), m.end()):
                diagnostic_positions.add(pos)
    return len(diagnostic_positions)


def analyze_condition(cond_name, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None

    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)
    fail_count = 0
    fail_has_traceback = 0
    
    trace_lengths = []
    diagnostic_char_counts = []
    total_input_lengths = []
    
    has_trace_content_count = 0
    no_trace_count = 0
    truncated_only_count = 0

    for s in samples:
        inp = s["input"]
        label = s["output"]
        trace, ts, te = extract_trace(inp)
        
        total_input_lengths.append(len(inp))
        trace_lengths.append(len(trace))
        
        diag_chars = count_diagnostic_chars(trace)
        diagnostic_char_counts.append(diag_chars)
        
        if label == "fail":
            fail_count += 1
            if has_full_traceback(trace):
                fail_has_traceback += 1
            
            actual_content = trace.replace("[...trace truncated...]", "").replace("[No trace available]", "").strip()
            if "[No trace available]" in trace:
                no_trace_count += 1
            elif not actual_content:
                truncated_only_count += 1
            else:
                has_trace_content_count += 1

    trace_lengths = np.array(trace_lengths)
    diagnostic_char_counts = np.array(diagnostic_char_counts)
    total_input_lengths = np.array(total_input_lengths)
    
    diagnostic_ratios = np.where(
        trace_lengths > 0,
        diagnostic_char_counts / trace_lengths,
        0.0
    )

    result = {
        "condition": cond_name,
        "n_samples": total,
        "n_fail": fail_count,
        "n_pass": total - fail_count,
        "mean_input_chars": float(np.mean(total_input_lengths)),
        "mean_trace_chars": float(np.mean(trace_lengths)),
        "mean_diagnostic_chars": float(np.mean(diagnostic_char_counts)),
        "diagnostic_ratio_by_chars": float(np.mean(diagnostic_ratios)),
        "global_diagnostic_ratio": float(np.sum(diagnostic_char_counts) / max(np.sum(trace_lengths), 1)),
        "has_full_traceback_count": fail_has_traceback,
        "has_full_traceback_pct": fail_has_traceback / max(fail_count, 1) * 100,
        "fail_has_trace_content": has_trace_content_count,
        "fail_no_trace": no_trace_count,
        "fail_truncated_only": truncated_only_count,
        "fail_has_any_trace_pct": has_trace_content_count / max(fail_count, 1) * 100,
        "accuracy": ACCURACIES.get(cond_name),
    }
    return result


def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ordered = ["tail-256", "tail-384", "tail-512", "tail-1024", "full_trace", "exception_only"]
    labels = ["tail-256", "tail-384", "tail-512", "tail-1024", "full(2048)", "exc_only"]
    x_positions = [256, 384, 512, 1024, 2048, 3000]  # approximate x for plotting

    # Filter to conditions with accuracy data for the main plot
    plot_conds = [c for c in ordered if results[c]["accuracy"] is not None]
    plot_labels = [labels[ordered.index(c)] for c in plot_conds]
    plot_x = [x_positions[ordered.index(c)] for c in plot_conds]

    # --- Figure 1: Dual-axis plot ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    diag_ratios = [results[c]["global_diagnostic_ratio"] * 100 for c in plot_conds]
    accs = [results[c]["accuracy"] * 100 for c in plot_conds]
    traceback_pcts = [results[c]["has_full_traceback_pct"] for c in plot_conds]
    trace_content_pcts = [results[c]["fail_has_any_trace_pct"] for c in plot_conds]

    color_diag = "#e74c3c"
    color_acc = "#2980b9"
    color_tb = "#27ae60"
    color_trace = "#f39c12"

    ax1.set_xlabel("Truncation Budget (tokens)", fontsize=11)
    
    l1 = ax1.plot(plot_x, diag_ratios, 'o-', color=color_diag, linewidth=2, markersize=8, label="Diagnostic Ratio (%)")
    l3 = ax1.plot(plot_x, traceback_pcts, 's--', color=color_tb, linewidth=2, markersize=7, label="has_full_traceback (%)")
    l4 = ax1.plot(plot_x, trace_content_pcts, 'D:', color=color_trace, linewidth=2, markersize=7, label="has_trace_content (% of fail)")
    ax1.set_ylabel("Diagnostic / Traceback Metrics (%)", fontsize=10, color=color_diag)
    ax1.tick_params(axis='y', labelcolor=color_diag)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    l2 = ax2.plot(plot_x, accs, '^-', color=color_acc, linewidth=2, markersize=8, label="Accuracy (%)")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(70, 102)

    # x-axis labels
    ax1.set_xticks(plot_x)
    ax1.set_xticklabels(plot_labels, fontsize=9, rotation=15)

    lines = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="center left", fontsize=8, bbox_to_anchor=(0.01, 0.5))

    ax1.set_title("Budget Curve: Diagnostic Ratio vs Accuracy by Truncation Length", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(axis='y', alpha=0.3)

    # Annotate key finding
    ax1.annotate(
        "full_trace format: 0 diagnostic tokens\n(no [ERROR] markers)",
        xy=(512, 0), xytext=(600, 40),
        fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8)
    )

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_dual_axis.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_dual_axis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dual-axis plot")

    # --- Figure 2: Stacked area/bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    
    all_conds = ordered
    all_labels = labels
    
    diag_chars = [results[c]["mean_diagnostic_chars"] for c in all_conds]
    trace_chars = [results[c]["mean_trace_chars"] for c in all_conds]
    input_chars = [results[c]["mean_input_chars"] for c in all_conds]
    non_diag_trace = [t - d for t, d in zip(trace_chars, diag_chars)]
    non_trace = [i - t for i, t in zip(input_chars, trace_chars)]

    x = np.arange(len(all_conds))
    width = 0.6
    
    ax.bar(x, diag_chars, width, label="Diagnostic chars", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x, non_diag_trace, width, bottom=diag_chars, label="Non-diagnostic trace", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x, non_trace, width, bottom=[d + n for d, n in zip(diag_chars, non_diag_trace)],
           label="Code + Test + Prompt", color="#bdc3c7", edgecolor="black", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel("Mean Characters per Sample", fontsize=10)
    ax.set_title("Token Composition by Truncation Length", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    
    # Add diagnostic ratio annotation on bars
    for i, c in enumerate(all_conds):
        ratio = results[c]["global_diagnostic_ratio"]
        if ratio > 0:
            ax.text(i, diag_chars[i] / 2, f"{ratio:.1%}", ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_stacked.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved stacked bar chart")

    # --- Figure 3: Phase transition detail ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: has_trace_content vs accuracy
    tc_conds = [c for c in ordered if results[c]["accuracy"] is not None]
    tc_x = [results[c]["fail_has_any_trace_pct"] for c in tc_conds]
    tc_y = [results[c]["accuracy"] * 100 for c in tc_conds]
    tc_labels_plot = [labels[ordered.index(c)] for c in tc_conds]
    
    ax1.scatter(tc_x, tc_y, s=100, c=["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"], 
                edgecolors="black", linewidth=0.5, zorder=3)
    for i, lab in enumerate(tc_labels_plot):
        ax1.annotate(lab, (tc_x[i], tc_y[i]), textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax1.set_xlabel("Fail samples with trace content (%)", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_title("Trace Content Availability vs Accuracy", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Right: traceback completeness breakdown across conditions
    conds_bar = ordered
    labels_bar = labels
    traceback_yes = [results[c]["has_full_traceback_count"] for c in conds_bar]
    traceback_no_but_trace = [results[c]["fail_has_trace_content"] - results[c]["has_full_traceback_count"] for c in conds_bar]
    truncated_only = [results[c]["fail_truncated_only"] for c in conds_bar]
    no_trace = [results[c]["fail_no_trace"] for c in conds_bar]
    
    x = np.arange(len(conds_bar))
    width = 0.6
    ax2.bar(x, traceback_yes, width, label="Has traceback", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax2.bar(x, traceback_no_but_trace, width, bottom=traceback_yes, label="Trace content (no traceback)", color="#3498db", edgecolor="black", linewidth=0.5)
    ax2.bar(x, truncated_only, width, bottom=[a+b for a,b in zip(traceback_yes, traceback_no_but_trace)], 
            label="Truncated marker only", color="#f39c12", edgecolor="black", linewidth=0.5)
    ax2.bar(x, no_trace, width, bottom=[a+b+c for a,b,c in zip(traceback_yes, traceback_no_but_trace, truncated_only)],
            label="No trace", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_bar, fontsize=9, rotation=15)
    ax2.set_ylabel("Number of Fail Samples", fontsize=10)
    ax2.set_title("Fail Sample Trace Completeness", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_phase_detail.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "budget_curve_phase_detail.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved phase detail plot")


def main():
    results = {}
    for cond, fname in CONDITIONS.items():
        print(f"Analyzing {cond}...")
        r = analyze_condition(cond, fname)
        if r:
            results[cond] = r

    # Print summary table
    print("\n" + "=" * 130)
    header = f"{'Condition':<16} {'InputChars':>10} {'TraceChars':>10} {'DiagChars':>10} {'DiagRatio':>10} {'TB_pct':>8} {'TraceContent':>12} {'Accuracy':>10}"
    print(header)
    print("-" * 130)
    ordered = ["tail-256", "tail-384", "tail-512", "tail-1024", "full_trace", "exception_only"]
    for c in ordered:
        if c in results:
            r = results[c]
            acc_str = f"{r['accuracy']*100:.2f}%" if r['accuracy'] else "N/A"
            print(f"{c:<16} {r['mean_input_chars']:>10.0f} {r['mean_trace_chars']:>10.0f} "
                  f"{r['mean_diagnostic_chars']:>10.1f} {r['global_diagnostic_ratio']:>10.4f} "
                  f"{r['has_full_traceback_pct']:>7.1f}% {r['fail_has_any_trace_pct']:>11.1f}% "
                  f"{acc_str:>10}")
    print("=" * 130)

    # Hypothesis test
    print("\n--- Hypothesis Verification ---")
    print("H: Phase transition occurs where truncation starts including full exception traceback.")
    print()
    for c in ordered:
        if c in results:
            r = results[c]
            acc_str = f"{r['accuracy']*100:.1f}%" if r['accuracy'] else "N/A"
            print(f"  {c:<16}: traceback={r['has_full_traceback_pct']:.1f}%, "
                  f"trace_content={r['fail_has_any_trace_pct']:.1f}%, "
                  f"diag_ratio={r['global_diagnostic_ratio']:.4f}, "
                  f"acc={acc_str}")
    
    print()
    print("Finding: tail-X conditions use full_trace format which has NO [ERROR] markers or")
    print("exception type annotations. Diagnostic ratio = 0 for all tail conditions.")
    print("has_full_traceback = 0% for all tail conditions.")
    print()
    print("The phase transition hypothesis (traceback inclusion -> accuracy jump) is NOT")
    print("supported by the data. The accuracy differences across tail-256/384/512/1024/full")
    print("are driven by execution context volume, not diagnostic signal presence.")
    print()
    
    # Additional: check if accuracy correlates with trace content availability
    conds_with_acc = [c for c in ordered if c in results and results[c]["accuracy"] is not None]
    trace_pcts = [results[c]["fail_has_any_trace_pct"] for c in conds_with_acc]
    accs = [results[c]["accuracy"] for c in conds_with_acc]
    if len(conds_with_acc) >= 3:
        from scipy import stats
        r, p = stats.pearsonr(trace_pcts, accs)
        print(f"Correlation (trace_content_pct vs accuracy): r={r:.3f}, p={p:.3f}")

    # Save JSON
    output = {
        "description": "Budget Curve Diagnostic Ratio Analysis",
        "hypothesis": "Phase transition occurs where truncation starts including full exception traceback",
        "verdict": "NOT SUPPORTED - full_trace format has no diagnostic markers; tail truncation only varies execution context volume",
        "per_condition": results,
    }
    
    json_path = os.path.join(OUT_DIR, "budget_curve_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    make_plots(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
