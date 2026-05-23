"""
Diagnostic Token Ratio Analysis (Model-Free)

Computes the fraction of input tokens that are "diagnostic" — i.e., 
rule-identified tokens that explicitly encode verdict-relevant signals
(error types, assertion outcomes, absence markers), as opposed to 
generic context (variable dumps, line-by-line states, code, tests).

This metric is fully model-free: it uses only a tokenizer (no inference)
and rule-based pattern matching (no learned classifier).
"""

import json
import re
import os
import sys
import numpy as np
from collections import defaultdict

TRACE_START = "Execution trace:\n"
TRACE_END_PATTERN = "\n\nDoes this code pass all tests? Answer: "

DATA_DIR = "./data"
OUT_DIR = "./artifacts/diagnostic_token_ratio"

CONDITIONS = ["exception_only", "full_trace", "ae_only", "loop_only", "no_trace"]

ACCURACIES = {
    "exception_only": 0.9983,
    "full_trace": 0.8144,
    "ae_only": 0.9992,
    "loop_only": 0.8513,
    "no_trace": 0.7513,
}


def extract_trace_section(text):
    """Return (trace_text, trace_start_char, trace_end_char) or (None, -1, -1)."""
    idx = text.find(TRACE_START)
    if idx < 0:
        return None, -1, -1
    start = idx + len(TRACE_START)
    end_idx = text.find(TRACE_END_PATTERN, start)
    if end_idx < 0:
        end_idx = len(text)
    return text[start:end_idx], start, end_idx


def find_diagnostic_char_spans(text, condition):
    """
    Return list of (start, end) char spans within `text` that are diagnostic.
    These spans are absolute positions in the full input text.
    """
    trace, trace_start, trace_end = extract_trace_section(text)
    
    if trace is None:
        return []
    
    if condition == "no_trace":
        return []
    
    if condition == "exception_only":
        # Entire trace section is diagnostic:
        # - [ERROR] lines: exception type + message + context (all captured at exception point)
        # - [No exception data in trace]: verdict marker (pass signal)
        # - [No trace available]: absence marker
        return [(trace_start, trace_end)]
    
    if condition == "full_trace":
        # No explicit verdict markers — all [Line N] variable dumps
        # Diagnostic = 0
        return []
    
    if condition == "ae_only":
        # Each [ASSERT] line is diagnostic (shows actual vs expected)
        # [No assertion data in trace] is also diagnostic
        # Exclude [fallback: full locals] portions
        spans = []
        lines = trace.split("\n")
        pos = trace_start
        for line in lines:
            line_len = len(line)
            if line.startswith("[ASSERT]") or line.startswith("[No assertion"):
                # Check for fallback locals — exclude that part
                fallback_idx = line.find("| [fallback: full locals]")
                if fallback_idx >= 0:
                    spans.append((pos, pos + fallback_idx))
                else:
                    spans.append((pos, pos + line_len))
            pos += line_len + 1  # +1 for \n
        return spans
    
    if condition == "loop_only":
        # [LOOP] header metadata is diagnostic (iteration count)
        # Variable dumps (first={...}, last={...}) are NOT diagnostic
        # [No trace available] is absence marker
        spans = []
        lines = trace.split("\n")
        pos = trace_start
        for line in lines:
            line_len = len(line)
            if line.startswith("[LOOP]"):
                # Extract just the header: "[LOOP] lines X, Y iterations:"
                header_match = re.match(r'(\[LOOP\] lines .+?, \d+ iterations:)', line)
                if header_match:
                    spans.append((pos, pos + header_match.end()))
                else:
                    # Fallback: just the [LOOP] tag
                    spans.append((pos, pos + min(line_len, 6)))
            elif line.startswith("[No trace"):
                spans.append((pos, pos + line_len))
            pos += line_len + 1
        return spans
    
    return []


def char_spans_to_token_count(offsets, spans):
    """
    Given token offset_mapping [(start, end), ...] and diagnostic char spans,
    count how many tokens overlap with any diagnostic span.
    """
    if not spans:
        return 0
    
    count = 0
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            continue  # special token
        for span_start, span_end in spans:
            if tok_start < span_end and tok_end > span_start:
                count += 1
                break
    return count


def main():
    # Load tokenizer (CPU only — no model weights)
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./models/Qwen2.5-Coder-7B-Instruct",
    )
    
    results = {}
    
    for cond in CONDITIONS:
        path = os.path.join(DATA_DIR, f"{cond}_test.jsonl")
        if not os.path.exists(path):
            print(f"  {cond}: FILE NOT FOUND, skipping")
            continue
        
        print(f"Processing {cond}...")
        
        with open(path) as f:
            samples = [json.loads(line) for line in f]
        
        total_tokens_list = []
        diag_tokens_list = []
        trace_tokens_list = []
        
        for sample in samples:
            text = sample["input"]
            
            # Tokenize with offset mapping
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            
            n_total = len(input_ids)
            
            # Count trace section tokens
            trace_text, trace_start, trace_end = extract_trace_section(text)
            if trace_text is not None:
                n_trace = char_spans_to_token_count(offsets, [(trace_start, trace_end)])
            else:
                n_trace = 0
            
            # Count diagnostic tokens
            diag_spans = find_diagnostic_char_spans(text, cond)
            n_diag = char_spans_to_token_count(offsets, diag_spans)
            
            total_tokens_list.append(n_total)
            diag_tokens_list.append(n_diag)
            trace_tokens_list.append(n_trace)
        
        total_arr = np.array(total_tokens_list)
        diag_arr = np.array(diag_tokens_list)
        trace_arr = np.array(trace_tokens_list)
        ratio_arr = diag_arr / np.maximum(total_arr, 1)
        trace_ratio_arr = trace_arr / np.maximum(total_arr, 1)
        
        # Diagnostic density within trace section
        diag_density_arr = np.where(trace_arr > 0, diag_arr / trace_arr, 0.0)
        
        results[cond] = {
            "n_samples": len(samples),
            "mean_total_tokens": float(np.mean(total_arr)),
            "std_total_tokens": float(np.std(total_arr)),
            "mean_trace_tokens": float(np.mean(trace_arr)),
            "std_trace_tokens": float(np.std(trace_arr)),
            "mean_diagnostic_tokens": float(np.mean(diag_arr)),
            "std_diagnostic_tokens": float(np.std(diag_arr)),
            "diagnostic_ratio": float(np.mean(ratio_arr)),
            "std_diagnostic_ratio": float(np.std(ratio_arr)),
            "trace_fraction": float(np.mean(trace_ratio_arr)),
            "diagnostic_density": float(np.mean(diag_density_arr)),
            "accuracy": ACCURACIES.get(cond, None),
        }
        
        print(f"  {cond}: total={np.mean(total_arr):.0f}±{np.std(total_arr):.0f}, "
              f"trace={np.mean(trace_arr):.0f}±{np.std(trace_arr):.0f}, "
              f"diag={np.mean(diag_arr):.0f}±{np.std(diag_arr):.0f}, "
              f"ratio={np.mean(ratio_arr):.4f}, "
              f"density={np.mean(diag_density_arr):.4f}")
    
    # Compute correlations
    conds_with_acc = [c for c in CONDITIONS if c in results and results[c]["accuracy"] is not None]
    ratios = [results[c]["diagnostic_ratio"] for c in conds_with_acc]
    accs = [results[c]["accuracy"] for c in conds_with_acc]
    
    from scipy import stats
    
    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(ratios, accs)
    # Spearman correlation  
    r_spearman, p_spearman = stats.spearmanr(ratios, accs)
    
    print(f"\nCorrelation (diagnostic_ratio vs accuracy):")
    print(f"  Pearson:  r={r_pearson:.4f}, p={p_pearson:.4f}")
    print(f"  Spearman: r={r_spearman:.4f}, p={p_spearman:.4f}")
    
    # Also compute with diagnostic_density
    densities = [results[c]["diagnostic_density"] for c in conds_with_acc]
    r_dens, p_dens = stats.pearsonr(densities, accs)
    print(f"  Pearson (density vs acc): r={r_dens:.4f}, p={p_dens:.4f}")
    
    # Save JSON results
    output = {
        "description": "Diagnostic Token Ratio: model-free information density metric",
        "method": "Rule-based diagnostic token identification + Qwen2.5-Coder-7B tokenizer",
        "per_condition": results,
        "correlation": {
            "diagnostic_ratio_vs_accuracy": {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            },
            "diagnostic_density_vs_accuracy": {
                "pearson_r": float(r_dens),
                "pearson_p": float(p_dens),
            }
        }
    }
    
    json_path = os.path.join(OUT_DIR, "diagnostic_token_ratio.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")
    
    # === Visualization ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    cond_labels = {
        "exception_only": "ExcOnly",
        "full_trace": "FullTrace",
        "ae_only": "AEOnly",
        "loop_only": "LoopOnly",
        "no_trace": "NoTrace",
    }
    
    ordered = ["no_trace", "full_trace", "loop_only", "exception_only", "ae_only"]
    labels = [cond_labels[c] for c in ordered]
    
    # (a) Bar chart: diagnostic token ratio per condition
    ax = axes[0]
    ratio_vals = [results[c]["diagnostic_ratio"] for c in ordered]
    colors = ["#bdc3c7", "#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax.bar(range(len(ordered)), ratio_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Diagnostic Token Ratio", fontsize=10)
    ax.set_title("(a) Diagnostic Token Ratio", fontsize=11, fontweight="bold")
    for i, v in enumerate(ratio_vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(ratio_vals) * 1.2 + 0.01)
    
    # (b) Scatter: diagnostic_ratio vs accuracy
    ax = axes[1]
    ratio_plot = [results[c]["diagnostic_ratio"] for c in ordered]
    acc_plot = [results[c]["accuracy"] for c in ordered]
    ax.scatter(ratio_plot, acc_plot, c=colors, s=80, edgecolors="black", linewidth=0.5, zorder=3)
    for i, c in enumerate(ordered):
        ax.annotate(labels[i], (ratio_plot[i], acc_plot[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Diagnostic Token Ratio", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"(b) Ratio vs Accuracy (r={r_pearson:.3f})", fontsize=11, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    
    # Fit line
    z = np.polyfit(ratio_plot, acc_plot, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ratio_plot) - 0.01, max(ratio_plot) + 0.01, 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.6, zorder=1)
    
    # (c) Stacked bar: diagnostic vs non-diagnostic vs non-trace tokens
    ax = axes[2]
    diag_vals = [results[c]["mean_diagnostic_tokens"] for c in ordered]
    trace_vals = [results[c]["mean_trace_tokens"] for c in ordered]
    total_vals = [results[c]["mean_total_tokens"] for c in ordered]
    non_diag_trace = [t - d for t, d in zip(trace_vals, diag_vals)]
    non_trace = [tot - tr for tot, tr in zip(total_vals, trace_vals)]
    
    x = range(len(ordered))
    ax.bar(x, diag_vals, label="Diagnostic", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.bar(x, non_diag_trace, bottom=diag_vals, label="Non-diag trace", color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x, non_trace, bottom=[d + n for d, n in zip(diag_vals, non_diag_trace)],
           label="Code + Test + Prompt", color="#bdc3c7", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Token Count", fontsize=10)
    ax.set_title("(c) Token Composition", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "diagnostic_token_ratio.pdf")
    fig_png = os.path.join(OUT_DIR, "diagnostic_token_ratio.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(fig_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_png}")
    
    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Condition':<16} {'Total Tok':>10} {'Trace Tok':>10} {'Diag Tok':>10} {'DiagRatio':>10} {'Density':>10} {'Accuracy':>10}")
    print("-" * 90)
    for c in ordered:
        r = results[c]
        print(f"{c:<16} {r['mean_total_tokens']:>10.0f} {r['mean_trace_tokens']:>10.0f} "
              f"{r['mean_diagnostic_tokens']:>10.0f} {r['diagnostic_ratio']:>10.4f} "
              f"{r['diagnostic_density']:>10.4f} {r['accuracy']:>10.4f}")
    print("=" * 90)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
