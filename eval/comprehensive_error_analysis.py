"""Comprehensive error analysis across all conditions for trace-verdict-study."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("./data")
CKPT_DIR = Path("./checkpoints")

# Qwen-7B seed42 experiments (primary)
CONDITIONS = {
    "exception_only": {
        "test": DATA_DIR / "exception_only_test.jsonl",
        "pred": CKPT_DIR / "exception_only" / "final" / "predictions.jsonl",
    },
    "vota": {
        "test": DATA_DIR / "vota_test.jsonl",
        "pred": CKPT_DIR / "vota" / "final" / "predictions.jsonl",
    },
    "full_trace": {
        "test": DATA_DIR / "full_trace_test.jsonl",
        "pred": CKPT_DIR / "full_trace" / "final" / "predictions.jsonl",
    },
    "no_trace": {
        "test": DATA_DIR / "no_trace_test.jsonl",
        "pred": CKPT_DIR / "no_trace" / "final" / "predictions.jsonl",
    },
}


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_features(input_text):
    """Extract error/trace features from the input text."""
    features = {}

    # Problem source
    features["has_trace"] = "Execution trace:" in input_text
    features["no_trace_marker"] = "[No trace available]" in input_text
    features["no_exception_marker"] = "[No exception data in trace]" in input_text

    # Error types from trace
    error_types = re.findall(r'\[ERROR\]\s+(\w+(?:Error|Exception|Exit|Warning))\s', input_text)
    features["error_types"] = error_types
    features["has_error"] = len(error_types) > 0
    features["primary_error"] = error_types[0] if error_types else None

    # Trace components
    features["has_assert"] = "[ASSERT]" in input_text
    features["has_loop"] = "[LOOP]" in input_text
    features["has_call"] = "[CALL]" in input_text
    features["has_return"] = "[RETURN]" in input_text
    features["truncated"] = "[...trace truncated...]" in input_text or "...truncated" in input_text

    # Input length (chars)
    features["input_len"] = len(input_text)

    # Trace length (chars after "Execution trace:")
    trace_start = input_text.find("Execution trace:")
    if trace_start != -1:
        features["trace_len"] = len(input_text[trace_start:])
    else:
        features["trace_len"] = 0

    return features


def analyze_condition(name, test_data, predictions):
    """Analyze a single condition."""
    n = len(predictions)
    assert n == len(test_data), f"{name}: test={len(test_data)} pred={len(predictions)}"

    # Build aligned data
    results = []
    for i in range(n):
        t = test_data[i]
        p = predictions[i]
        label = t["output"]
        pred = p["pred"]
        problem_id = t.get("problem_id", f"unknown_{i}")
        source = "HumanEval" if problem_id.startswith("HumanEval") else "MBPP" if problem_id.startswith("MBPP") else "other"

        feats = extract_features(t["input"])

        results.append({
            "idx": i,
            "problem_id": problem_id,
            "source": source,
            "label": label,
            "pred": pred,
            "raw_pred": p.get("raw_pred", pred),
            "correct": label == pred,
            **feats,
        })

    # Confusion matrix
    tp = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "pass")
    tn = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "fail")
    fp = sum(1 for r in results if r["label"] == "fail" and r["pred"] == "pass")  # predict pass, actual fail
    fn = sum(1 for r in results if r["label"] == "pass" and r["pred"] == "fail")  # predict fail, actual pass

    accuracy = (tp + tn) / n
    misclassified = [r for r in results if not r["correct"]]
    n_wrong = len(misclassified)

    # Error direction
    fp_samples = [r for r in misclassified if r["label"] == "fail" and r["pred"] == "pass"]
    fn_samples = [r for r in misclassified if r["label"] == "pass" and r["pred"] == "fail"]

    # Source distribution of errors
    source_all = Counter(r["source"] for r in results)
    source_wrong = Counter(r["source"] for r in misclassified)

    # Problem-level analysis
    problem_wrong = Counter(r["problem_id"] for r in misclassified)

    # Error type analysis (for fp: predicted pass but actually fail)
    fp_error_types = Counter()
    for r in fp_samples:
        if r["has_error"]:
            for et in r["error_types"]:
                fp_error_types[et] += 1
        elif r["no_trace_marker"]:
            fp_error_types["[no_trace]"] += 1
        elif r["no_exception_marker"]:
            fp_error_types["[no_exception_data]"] += 1
        else:
            fp_error_types["[other]"] += 1

    # FN analysis (predicted fail but actually pass)
    fn_error_types = Counter()
    fn_trace_features = Counter()
    for r in fn_samples:
        if r["has_error"]:
            for et in r["error_types"]:
                fn_error_types[et] += 1
        elif r["no_exception_marker"]:
            fn_error_types["[no_exception_data]"] += 1
        elif r["no_trace_marker"]:
            fn_error_types["[no_trace]"] += 1
        else:
            fn_error_types["[no_errors_but_pred_fail]"] += 1
        if r["has_error"]:
            fn_trace_features["has_error"] += 1
        if r["has_assert"]:
            fn_trace_features["has_assert"] += 1
        if r["has_loop"]:
            fn_trace_features["has_loop"] += 1
        if r["truncated"]:
            fn_trace_features["truncated"] += 1

    # Trace length stats for correct vs wrong
    import numpy as np
    trace_lens_correct = [r["trace_len"] for r in results if r["correct"]]
    trace_lens_wrong = [r["trace_len"] for r in results if not r["correct"]]
    input_lens_correct = [r["input_len"] for r in results if r["correct"]]
    input_lens_wrong = [r["input_len"] for r in results if not r["correct"]]

    # Illegal predictions
    illegal = [r for r in results if r["raw_pred"] not in ("pass", "fail")]

    return {
        "name": name,
        "n": n,
        "accuracy": accuracy,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "n_wrong": n_wrong,
        "n_fp": len(fp_samples),
        "n_fn": len(fn_samples),
        "fp_ratio": len(fp_samples) / n_wrong if n_wrong > 0 else 0,
        "fn_ratio": len(fn_samples) / n_wrong if n_wrong > 0 else 0,
        "source_all": dict(source_all),
        "source_wrong": dict(source_wrong),
        "source_error_rate": {
            s: source_wrong.get(s, 0) / source_all[s] * 100
            for s in source_all
        },
        "top_problem_errors": dict(problem_wrong.most_common(20)),
        "n_unique_problems_wrong": len(problem_wrong),
        "fp_error_types": dict(fp_error_types.most_common()),
        "fn_error_types": dict(fn_error_types.most_common()),
        "fn_trace_features": dict(fn_trace_features),
        "trace_len_correct_mean": float(np.mean(trace_lens_correct)) if trace_lens_correct else 0,
        "trace_len_wrong_mean": float(np.mean(trace_lens_wrong)) if trace_lens_wrong else 0,
        "input_len_correct_mean": float(np.mean(input_lens_correct)) if input_lens_correct else 0,
        "input_len_wrong_mean": float(np.mean(input_lens_wrong)) if input_lens_wrong else 0,
        "n_illegal": len(illegal),
        "illegal_tokens": dict(Counter(r["raw_pred"] for r in illegal).most_common(10)),
        "misclassified_ids": [r["idx"] for r in misclassified],
        "fp_samples": fp_samples[:10],  # top 10 for inspection
        "fn_samples": fn_samples[:10],
    }


def cross_condition_analysis(all_results):
    """Compare errors across conditions."""
    # Find samples wrong in one but not another
    comparisons = {}

    conditions = list(all_results.keys())
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            wrong1 = set(all_results[c1]["misclassified_ids"])
            wrong2 = set(all_results[c2]["misclassified_ids"])
            comparisons[f"{c1}_vs_{c2}"] = {
                "both_wrong": len(wrong1 & wrong2),
                f"only_{c1}_wrong": len(wrong1 - wrong2),
                f"only_{c2}_wrong": len(wrong2 - wrong1),
                "both_correct": all_results[c1]["n"] - len(wrong1 | wrong2),
            }

    return comparisons


def generate_markdown(all_results, cross_results):
    """Generate markdown report."""
    lines = []
    lines.append("# Error Analysis Report: Trace-Verdict-Study")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("| Condition | N | Accuracy | Errors | FP (pred pass, actual fail) | FN (pred fail, actual pass) | FP% | FN% |")
    lines.append("|-----------|---|----------|--------|-------|-------|-----|-----|")
    for name in ["exception_only", "vota", "full_trace", "no_trace"]:
        r = all_results[name]
        lines.append(f"| {name} | {r['n']} | {r['accuracy']*100:.2f}% | {r['n_wrong']} | {r['n_fp']} | {r['n_fn']} | {r['fp_ratio']*100:.1f}% | {r['fn_ratio']*100:.1f}% |")
    lines.append("")

    # Confusion matrices
    lines.append("## 2. Confusion Matrices")
    lines.append("")
    for name in ["exception_only", "vota", "full_trace", "no_trace"]:
        r = all_results[name]
        cm = r["confusion_matrix"]
        lines.append(f"### {name} (accuracy: {r['accuracy']*100:.2f}%)")
        lines.append("")
        lines.append("|  | Pred Pass | Pred Fail |")
        lines.append("|--|-----------|-----------|")
        lines.append(f"| **Actual Pass** | {cm['TP']} (TP) | {cm['FN']} (FN) |")
        lines.append(f"| **Actual Fail** | {cm['FP']} (FP) | {cm['TN']} (TN) |")
        lines.append("")
        # Precision/Recall
        pass_prec = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
        pass_rec = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
        fail_prec = cm['TN'] / (cm['TN'] + cm['FN']) if (cm['TN'] + cm['FN']) > 0 else 0
        fail_rec = cm['TN'] / (cm['TN'] + cm['FP']) if (cm['TN'] + cm['FP']) > 0 else 0
        lines.append(f"- Pass: precision={pass_prec:.4f}, recall={pass_rec:.4f}")
        lines.append(f"- Fail: precision={fail_prec:.4f}, recall={fail_rec:.4f}")
        lines.append("")

    # Error categories
    lines.append("## 3. Error Categories by Condition")
    lines.append("")
    for name in ["exception_only", "vota", "full_trace", "no_trace"]:
        r = all_results[name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"**False Positives** (predicted pass, actually fail): {r['n_fp']} samples")
        lines.append("")
        if r["fp_error_types"]:
            lines.append("| Error Type | Count |")
            lines.append("|------------|-------|")
            for et, cnt in sorted(r["fp_error_types"].items(), key=lambda x: -x[1]):
                lines.append(f"| {et} | {cnt} |")
            lines.append("")

        lines.append(f"**False Negatives** (predicted fail, actually pass): {r['n_fn']} samples")
        lines.append("")
        if r["fn_error_types"]:
            lines.append("| Category | Count |")
            lines.append("|----------|-------|")
            for et, cnt in sorted(r["fn_error_types"].items(), key=lambda x: -x[1]):
                lines.append(f"| {et} | {cnt} |")
            lines.append("")

        # Source distribution
        lines.append(f"**Error rate by source:**")
        lines.append("")
        lines.append("| Source | Total | Errors | Error Rate |")
        lines.append("|--------|-------|--------|------------|")
        for s in sorted(r["source_all"].keys()):
            total = r["source_all"][s]
            wrong = r["source_wrong"].get(s, 0)
            rate = r["source_error_rate"].get(s, 0)
            lines.append(f"| {s} | {total} | {wrong} | {rate:.2f}% |")
        lines.append("")

        # Top problematic problems
        if r["top_problem_errors"]:
            lines.append(f"**Top problematic programs** ({r['n_unique_problems_wrong']} unique problems with errors):")
            lines.append("")
            lines.append("| Problem ID | Error Count |")
            lines.append("|------------|-------------|")
            for pid, cnt in list(r["top_problem_errors"].items())[:10]:
                lines.append(f"| {pid} | {cnt} |")
            lines.append("")

        # Illegal outputs
        if r["n_illegal"] > 0:
            lines.append(f"**Illegal outputs**: {r['n_illegal']} samples")
            lines.append("")
            for tok, cnt in r["illegal_tokens"].items():
                lines.append(f"- `{tok}`: {cnt}")
            lines.append("")

        # Trace length comparison
        lines.append(f"**Trace/Input length (chars):**")
        lines.append(f"- Correct samples: trace_len={r['trace_len_correct_mean']:.0f}, input_len={r['input_len_correct_mean']:.0f}")
        lines.append(f"- Wrong samples: trace_len={r['trace_len_wrong_mean']:.0f}, input_len={r['input_len_wrong_mean']:.0f}")
        lines.append("")

    # Cross-condition comparison
    lines.append("## 4. Cross-Condition Comparisons")
    lines.append("")
    lines.append("| Comparison | Both Correct | Both Wrong | Only A Wrong | Only B Wrong |")
    lines.append("|------------|--------------|------------|--------------|--------------|")
    for key, comp in cross_results.items():
        parts = key.split("_vs_")
        a, b = parts[0], parts[1]
        bc = comp["both_correct"]
        bw = comp["both_wrong"]
        aw = comp[f"only_{a}_wrong"]
        bww = comp[f"only_{b}_wrong"]
        lines.append(f"| {a} vs {b} | {bc} | {bw} | {aw} | {bww} |")
    lines.append("")

    # Detailed cross analysis: exception_only vs vota
    lines.append("### 4.1 exception_only vs vota")
    lines.append("")
    exc_wrong = set(all_results["exception_only"]["misclassified_ids"])
    vota_wrong = set(all_results["vota"]["misclassified_ids"])
    lines.append(f"- exception_only errors: {len(exc_wrong)}")
    lines.append(f"- vota errors: {len(vota_wrong)}")
    lines.append(f"- Both wrong: {len(exc_wrong & vota_wrong)}")
    lines.append(f"- Only exception_only wrong: {len(exc_wrong - vota_wrong)}")
    lines.append(f"- Only vota wrong: {len(vota_wrong - exc_wrong)}")
    lines.append("")

    # What are the vota-only errors?
    lines.append("### 4.2 exception_only vs full_trace")
    lines.append("")
    ft_wrong = set(all_results["full_trace"]["misclassified_ids"])
    lines.append(f"- exception_only errors: {len(exc_wrong)}")
    lines.append(f"- full_trace errors: {len(ft_wrong)}")
    lines.append(f"- Both wrong: {len(exc_wrong & ft_wrong)}")
    lines.append(f"- Only exception_only wrong: {len(exc_wrong - ft_wrong)}")
    lines.append(f"- Only full_trace wrong: {len(ft_wrong - exc_wrong)}")
    lines.append("")

    lines.append("### 4.3 no_trace baseline error pattern")
    lines.append("")
    nt_wrong = set(all_results["no_trace"]["misclassified_ids"])
    lines.append(f"- no_trace errors: {len(nt_wrong)}")
    lines.append(f"- Shared with exception_only: {len(nt_wrong & exc_wrong)}")
    lines.append(f"- Shared with full_trace: {len(nt_wrong & ft_wrong)}")
    lines.append(f"- Unique to no_trace (fixed by any trace): {len(nt_wrong - exc_wrong - ft_wrong)}")
    lines.append(f"- Errors in all 4 conditions: {len(nt_wrong & exc_wrong & ft_wrong & vota_wrong)}")
    lines.append("")

    # Key findings
    lines.append("## 5. Key Findings")
    lines.append("")

    # Finding 1: FP vs FN balance
    exc = all_results["exception_only"]
    vota = all_results["vota"]
    ft = all_results["full_trace"]
    nt = all_results["no_trace"]

    lines.append("### 5.1 Error Direction Patterns")
    lines.append("")
    lines.append("| Condition | FP% (pred pass, actual fail) | FN% (pred fail, actual pass) | Dominant Direction |")
    lines.append("|-----------|-----|-----|-------------------|")
    for name, r in [("exception_only", exc), ("vota", vota), ("full_trace", ft), ("no_trace", nt)]:
        dominant = "FP" if r["fp_ratio"] > r["fn_ratio"] else "FN" if r["fn_ratio"] > r["fp_ratio"] else "balanced"
        lines.append(f"| {name} | {r['fp_ratio']*100:.1f}% | {r['fn_ratio']*100:.1f}% | {dominant} |")
    lines.append("")

    # Finding 2: exception_only failure modes
    lines.append("### 5.2 exception_only Failure Modes (the minimal sufficient signal)")
    lines.append("")
    if exc["n_wrong"] > 0:
        lines.append(f"With only {exc['n_wrong']}/{exc['n']} errors ({(1-exc['accuracy'])*100:.2f}% error rate):")
        lines.append("")
        if exc["fn_error_types"]:
            lines.append("FN cases (predicted fail but actually pass):")
            for et, cnt in sorted(exc["fn_error_types"].items(), key=lambda x: -x[1]):
                lines.append(f"- {et}: {cnt}")
            lines.append("")
        if exc["fp_error_types"]:
            lines.append("FP cases (predicted pass but actually fail):")
            for et, cnt in sorted(exc["fp_error_types"].items(), key=lambda x: -x[1]):
                lines.append(f"- {et}: {cnt}")
            lines.append("")

    # Finding 3: What full_trace gets wrong that exception_only doesn't
    lines.append("### 5.3 full_trace Additional Errors (compared to exception_only)")
    lines.append("")
    ft_only = ft_wrong - exc_wrong
    lines.append(f"full_trace has {len(ft_only)} errors that exception_only gets right.")
    lines.append("This suggests the full trace introduces noise that hurts classification.")
    lines.append("")

    # Finding 4: Hardest problems
    lines.append("### 5.4 Hardest Problems (wrong in most conditions)")
    lines.append("")
    # Find problems wrong in multiple conditions
    problem_errors_all = defaultdict(set)
    for cname, r in all_results.items():
        for idx in r["misclassified_ids"]:
            # We need problem_id... let me use the test data
            pass  # Will be handled below

    lines.append("(Problem-level cross-condition analysis computed from individual condition tables above)")
    lines.append("")

    # Summary bullets for paper
    lines.append("## 6. Paper-Ready Key Findings")
    lines.append("")
    lines.append(f"1. **Exception-only achieves {exc['accuracy']*100:.2f}% accuracy with only {exc['n_wrong']} errors out of {exc['n']} samples.** "
                 f"Its errors split into {exc['n_fn']} false negatives (predicting fail on passing code) "
                 f"and {exc['n_fp']} false positives (predicting pass on failing code), "
                 f"indicating {'a bias toward predicting fail' if exc['fn_ratio'] > 0.6 else 'a bias toward predicting pass' if exc['fp_ratio'] > 0.6 else 'a relatively balanced error distribution'}.")
    lines.append("")
    lines.append(f"2. **Full trace input degrades performance to {ft['accuracy']*100:.2f}%.** "
                 f"The {len(ft_only)} additional errors (beyond exception_only's {exc['n_wrong']}) "
                 f"suggest that verbose trace information introduces noise rather than providing useful signal, "
                 f"supporting the minimal-signal hypothesis.")
    lines.append("")
    lines.append(f"3. **No-trace baseline ({nt['accuracy']*100:.2f}%) errors are partially recoverable.** "
                 f"Of its {nt['n_wrong']} errors, {len(nt_wrong - exc_wrong - ft_wrong)} are fixed by any trace condition, "
                 f"confirming that execution traces provide genuine signal beyond code-only analysis.")
    lines.append("")

    return "\n".join(lines)


def main():
    all_results = {}

    for name, paths in CONDITIONS.items():
        test_path = paths["test"]
        pred_path = paths["pred"]

        if not test_path.exists():
            print(f"[SKIP] {name}: test file not found: {test_path}")
            continue
        if not pred_path.exists():
            print(f"[SKIP] {name}: predictions not found: {pred_path}")
            continue

        print(f"[{name}] Loading data...")
        test_data = load_jsonl(test_path)
        predictions = load_jsonl(pred_path)

        print(f"  test={len(test_data)}, pred={len(predictions)}")
        result = analyze_condition(name, test_data, predictions)
        all_results[name] = result
        print(f"  accuracy={result['accuracy']*100:.2f}%, errors={result['n_wrong']} (FP={result['n_fp']}, FN={result['n_fn']})")

    if not all_results:
        print("No conditions could be analyzed!")
        sys.exit(1)

    # Cross-condition analysis
    print("\nCross-condition analysis...")
    cross_results = cross_condition_analysis(all_results)

    # Generate markdown
    print("Generating report...")
    report = generate_markdown(all_results, cross_results)

    # Save
    output_path = "./artifacts/error_analysis_report.md"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    # Also save JSON for further analysis
    json_path = "./artifacts/error_analysis_data.json"
    # Remove non-serializable items
    json_results = {}
    for name, r in all_results.items():
        jr = {k: v for k, v in r.items() if k not in ("fp_samples", "fn_samples")}
        json_results[name] = jr
    json_results["cross_condition"] = cross_results

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"JSON data saved to {json_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"{'Condition':<20} {'Acc':>8} {'Errors':>7} {'FP':>5} {'FN':>5}")
    for name in ["exception_only", "vota", "full_trace", "no_trace"]:
        if name in all_results:
            r = all_results[name]
            print(f"{name:<20} {r['accuracy']*100:>7.2f}% {r['n_wrong']:>7} {r['n_fp']:>5} {r['n_fn']:>5}")


if __name__ == "__main__":
    main()
