"""
Extended Diagnostic Token Ratio Analysis (Model-Free)

Covers all experimental conditions including vota variants, tail/head
truncations, nomarker, and padded2048. Fixes full_trace diagnostic=0
bug (was missing [No trace available] absence markers).
"""

import json
import re
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import stats

TRACE_START = "Execution trace:\n"
TRACE_END_PATTERN = "\n\nDoes this code pass all tests? Answer: "

DATA_DIR = "./data"
OUT_DIR = "./artifacts/diagnostic_token_ratio"

CONDITIONS = {
    # Original 5 (keep original accuracy values)
    "no_trace":        {"file": "no_trace_test.jsonl",               "accuracy": 0.7513},
    "full_trace":      {"file": "full_trace_test.jsonl",             "accuracy": 0.8144},
    "loop_only":       {"file": "loop_only_test.jsonl",              "accuracy": 0.8513},
    "ae_only":         {"file": "ae_only_test.jsonl",                "accuracy": 0.9992},
    "exception_only":  {"file": "exception_only_test.jsonl",         "accuracy": 0.9983},
    # New conditions
    "vota":            {"file": "vota_test.jsonl",                   "accuracy": 0.9941},
    "vota_no_ae":      {"file": "vota_no_ae_test.jsonl",             "accuracy": 0.9941},
    "tail_256":        {"file": "full_trace_tail_256_test.jsonl",    "accuracy": 0.7748},
    "tail_384":        {"file": "full_trace_tail_384_test.jsonl",    "accuracy": 0.7983},
    "tail_512":        {"file": "full_trace_tail_512_test.jsonl",    "accuracy": 0.9361},
    "tail_1024":       {"file": "full_trace_tail_1024_test.jsonl",   "accuracy": 0.9647},
    "tail_2048":       {"file": "full_trace_tail_test.jsonl",        "accuracy": 0.9731},
    "head_2048":       {"file": "full_trace_head_test.jsonl",        "accuracy": 0.8101},
    "nomarker":        {"file": "exception_only_nomarker_test.jsonl","accuracy": 0.9782},
    "padded2048":      {"file": "exception_only_padded2048_test.jsonl","accuracy": 0.9983},
}

# Conditions where entire trace section is diagnostic
ENTIRE_TRACE_DIAGNOSTIC = {"exception_only", "padded2048"}
# Conditions with no diagnostic content by design
NO_DIAGNOSTIC = {"no_trace", "nomarker"}


def extract_trace_section(text):
    idx = text.find(TRACE_START)
    if idx < 0:
        return None, -1, -1
    start = idx + len(TRACE_START)
    end_idx = text.find(TRACE_END_PATTERN, start)
    if end_idx < 0:
        end_idx = len(text)
    return text[start:end_idx], start, end_idx


def find_diagnostic_char_spans(text, condition):
    trace, trace_start, trace_end = extract_trace_section(text)
    if trace is None:
        return []

    if condition in NO_DIAGNOSTIC:
        return []

    if condition in ENTIRE_TRACE_DIAGNOSTIC:
        return [(trace_start, trace_end)]

    # Universal pattern-based matching for all other conditions
    spans = []
    lines = trace.split("\n")
    pos = trace_start
    for line in lines:
        line_len = len(line)

        if line in ("[No trace available]",
                     "[No exception data in trace]",
                     "[No assertion data in trace]"):
            spans.append((pos, pos + line_len))
        elif line.startswith("[ERROR]"):
            spans.append((pos, pos + line_len))
        elif line.startswith("[ASSERT]"):
            fallback_idx = line.find("| [fallback: full locals]")
            if fallback_idx >= 0:
                spans.append((pos, pos + fallback_idx))
            else:
                spans.append((pos, pos + line_len))
        elif line.startswith("[LOOP]"):
            header_match = re.match(r'(\[LOOP\] lines .+?, \d+ iterations:)', line)
            if header_match:
                spans.append((pos, pos + header_match.end()))
            else:
                spans.append((pos, pos + min(line_len, 6)))

        pos += line_len + 1  # +1 for \n

    return spans


def char_spans_to_token_mask(text, spans, tokenizer):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    token_ids = encoding["input_ids"]
    n_tokens = len(token_ids)

    mask = [False] * n_tokens
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for span_start, span_end in spans:
            if tok_start < span_end and tok_end > span_start:
                mask[tok_idx] = True
                break

    return token_ids, mask, offsets


def analyze_condition(condition, config, tokenizer):
    filepath = os.path.join(DATA_DIR, config["file"])
    if not os.path.exists(filepath):
        print(f"  SKIP: {filepath} not found")
        return None

    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    total_tokens_list = []
    trace_tokens_list = []
    diag_tokens_list = []
    ratio_list = []

    for sample in samples:
        text = sample.get("input", sample.get("text", ""))

        encoding = tokenizer(text, add_special_tokens=False)
        total_tok = len(encoding["input_ids"])

        trace, trace_start, trace_end = extract_trace_section(text)
        if trace is not None:
            trace_full = text[trace_start:trace_end]
            trace_enc = tokenizer(trace_full, add_special_tokens=False)
            trace_tok = len(trace_enc["input_ids"])
        else:
            trace_tok = 0

        spans = find_diagnostic_char_spans(text, condition)

        if spans:
            _, mask, _ = char_spans_to_token_mask(text, spans, tokenizer)
            diag_tok = sum(mask)
        else:
            diag_tok = 0

        total_tokens_list.append(total_tok)
        trace_tokens_list.append(trace_tok)
        diag_tokens_list.append(diag_tok)
        ratio_list.append(diag_tok / total_tok if total_tok > 0 else 0.0)

    total_arr = np.array(total_tokens_list)
    trace_arr = np.array(trace_tokens_list)
    diag_arr = np.array(diag_tokens_list)
    ratio_arr = np.array(ratio_list)

    mean_trace = float(np.mean(trace_arr))
    mean_diag = float(np.mean(diag_arr))

    result = {
        "n_samples": len(samples),
        "mean_total_tokens": float(np.mean(total_arr)),
        "std_total_tokens": float(np.std(total_arr)),
        "mean_trace_tokens": mean_trace,
        "std_trace_tokens": float(np.std(trace_arr)),
        "mean_diagnostic_tokens": mean_diag,
        "std_diagnostic_tokens": float(np.std(diag_arr)),
        "diagnostic_ratio": float(np.mean(ratio_arr)),
        "std_diagnostic_ratio": float(np.std(ratio_arr)),
        "trace_fraction": mean_trace / float(np.mean(total_arr)) if np.mean(total_arr) > 0 else 0.0,
        "diagnostic_density": mean_diag / mean_trace if mean_trace > 0 else 0.0,
        "accuracy": config["accuracy"],
    }
    return result


def main():
    from transformers import AutoTokenizer

    tokenizer_name = "./models/Qwen2.5-Coder-7B-Instruct"
    cache_dir = "~/.cache/huggingface"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for cond, config in CONDITIONS.items():
        print(f"Processing: {cond}")
        r = analyze_condition(cond, config, tokenizer)
        if r is not None:
            results[cond] = r
            print(f"  n={r['n_samples']}, diag_ratio={r['diagnostic_ratio']:.4f}, "
                  f"diag_tok={r['mean_diagnostic_tokens']:.1f}, acc={r['accuracy']:.4f}")

    # Correlations
    conds_with_data = [c for c in results if results[c]["n_samples"] > 0]
    ratios = [results[c]["diagnostic_ratio"] for c in conds_with_data]
    accs = [results[c]["accuracy"] for c in conds_with_data]

    pearson_r, pearson_p = stats.pearsonr(ratios, accs)
    spearman_r, spearman_p = stats.spearmanr(ratios, accs)

    # Also compute excluding no_trace (which has both ratio=0 and lowest acc)
    conds_with_trace = [c for c in conds_with_data if c != "no_trace"]
    if len(conds_with_trace) >= 3:
        ratios_t = [results[c]["diagnostic_ratio"] for c in conds_with_trace]
        accs_t = [results[c]["accuracy"] for c in conds_with_trace]
        pearson_r_t, pearson_p_t = stats.pearsonr(ratios_t, accs_t)
    else:
        pearson_r_t, pearson_p_t = 0, 1

    output = {
        "description": "Extended Diagnostic Token Ratio: model-free information density metric",
        "method": "Rule-based diagnostic token identification + Qwen2.5-Coder-7B tokenizer",
        "fix_notes": "full_trace diagnostic ratio corrected: [No trace available] markers now counted as diagnostic",
        "per_condition": results,
        "correlation": {
            "all_conditions": {
                "n": len(conds_with_data),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            },
            "excluding_no_trace": {
                "n": len(conds_with_trace),
                "pearson_r": pearson_r_t,
                "pearson_p": pearson_p_t,
            },
        },
    }

    json_path = os.path.join(OUT_DIR, "diagnostic_token_ratio_extended.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # --- Plotting ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    # Group conditions for visualization
    original_5 = ["no_trace", "full_trace", "loop_only", "ae_only", "exception_only"]
    vota_group = ["vota", "vota_no_ae"]
    tail_group = ["tail_256", "tail_384", "tail_512", "tail_1024", "tail_2048"]
    other_group = ["head_2048", "nomarker", "padded2048"]

    all_conds = [c for c in original_5 + vota_group + tail_group + other_group if c in results]

    short_labels = {
        "no_trace": "NoTrace", "full_trace": "FullTrace", "loop_only": "LoopOnly",
        "ae_only": "AEOnly", "exception_only": "ExcOnly",
        "vota": "VoTA", "vota_no_ae": "VoTA-noAE",
        "tail_256": "Tail256", "tail_384": "Tail384", "tail_512": "Tail512",
        "tail_1024": "Tail1024", "tail_2048": "Tail2048",
        "head_2048": "Head2048", "nomarker": "NoMarker", "padded2048": "Padded",
    }

    group_colors = {}
    for c in original_5: group_colors[c] = "#3498db"
    for c in vota_group: group_colors[c] = "#2ecc71"
    for c in tail_group: group_colors[c] = "#e74c3c"
    for c in other_group: group_colors[c] = "#9b59b6"

    # Figure: scatter plot (ratio vs accuracy)
    fig, ax = plt.subplots(figsize=(8, 6))

    for c in all_conds:
        r = results[c]
        color = group_colors.get(c, "#95a5a6")
        ax.scatter(r["diagnostic_ratio"], r["accuracy"],
                   c=color, s=80, edgecolors="black", linewidth=0.5, zorder=3)
        # Smart label placement
        offset_x, offset_y = 5, 5
        if c in ("ae_only", "exception_only", "padded2048"):
            offset_y = -12
        if c == "full_trace":
            offset_x = -50
        if c == "nomarker":
            offset_x = -55
        ax.annotate(short_labels.get(c, c),
                     (r["diagnostic_ratio"], r["accuracy"]),
                     textcoords="offset points", xytext=(offset_x, offset_y),
                     fontsize=7, color=color, fontweight="bold")

    # Fit line (all points)
    r_all = [results[c]["diagnostic_ratio"] for c in all_conds]
    a_all = [results[c]["accuracy"] for c in all_conds]
    z = np.polyfit(r_all, a_all, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(r_all) - 0.01, max(r_all) + 0.05, 100)
    ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, zorder=1)

    ax.set_xlabel("Diagnostic Token Ratio", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Diagnostic Token Ratio vs Accuracy (r={pearson_r:.3f}, p={pearson_p:.4f})",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.set_xlim(-0.01, max(r_all) * 1.15 + 0.01)

    # Legend for groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Original 5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='VoTA variants'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Tail truncations'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9b59b6', markersize=8, label='Other'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"diagnostic_token_ratio_extended.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # --- Summary table ---
    print("\n" + "=" * 110)
    print(f"{'Condition':<16} {'Total Tok':>10} {'Trace Tok':>10} {'Diag Tok':>10} "
          f"{'DiagRatio':>10} {'Density':>10} {'Accuracy':>10}")
    print("-" * 110)
    for c in all_conds:
        r = results[c]
        print(f"{c:<16} {r['mean_total_tokens']:>10.0f} {r['mean_trace_tokens']:>10.0f} "
              f"{r['mean_diagnostic_tokens']:>10.1f} {r['diagnostic_ratio']:>10.4f} "
              f"{r['diagnostic_density']:>10.4f} {r['accuracy']:>10.4f}")
    print("=" * 110)
    print(f"\nPearson r = {pearson_r:.4f} (p = {pearson_p:.4f}), "
          f"Spearman r = {spearman_r:.4f} (p = {spearman_p:.4f})")
    print(f"Excluding no_trace: Pearson r = {pearson_r_t:.4f} (p = {pearson_p_t:.4f})")


if __name__ == "__main__":
    main()
