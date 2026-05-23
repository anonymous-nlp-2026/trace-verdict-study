import json
import re
import numpy as np
from collections import Counter

# ── Load data ──
DATA_PATH = "./data/exception_only_test.jsonl"
SFT_PRED_PATH = "./checkpoints/exception_only/final/predictions.jsonl"
OUTPUT_PATH = "./analysis/regex_baseline_results.json"

samples = []
with open(DATA_PATH) as f:
    for line in f:
        d = json.loads(line)
        inp = d["input"]
        label = d["output"]
        ts = inp.find("Execution trace:\n")
        te = inp.find("\n\nDoes this code pass all tests?")
        trace = inp[ts + len("Execution trace:\n"):te] if ts >= 0 and te >= 0 else ""
        samples.append({"trace": trace, "label": label, "problem_id": d.get("problem_id", "")})

sft_preds = []
with open(SFT_PRED_PATH) as f:
    for line in f:
        sft_preds.append(json.loads(line))

labels = [s["label"] for s in samples]
sft_pred_labels = [p["pred"] for p in sft_preds]

N = len(samples)
print(f"N={N}, pass={labels.count('pass')}, fail={labels.count('fail')}")

# ── Metrics helper ──
def compute_metrics(y_true, y_pred, name):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == "fail" and p == "fail": tp += 1
        elif t == "pass" and p == "fail": fp += 1
        elif t == "fail" and p == "pass": fn += 1
        else: tn += 1
    acc = (tp + tn) / len(y_true)
    prec_fail = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_fail = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_fail = 2 * prec_fail * rec_fail / (prec_fail + rec_fail) if (prec_fail + rec_fail) > 0 else 0
    prec_pass = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_pass = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_pass = 2 * prec_pass * rec_pass / (prec_pass + rec_pass) if (prec_pass + rec_pass) > 0 else 0
    macro_f1 = (f1_fail + f1_pass) / 2
    return {
        "name": name,
        "accuracy": round(acc, 6),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "fail": {"precision": round(prec_fail, 4), "recall": round(rec_fail, 4), "f1": round(f1_fail, 4)},
        "pass": {"precision": round(prec_pass, 4), "recall": round(rec_pass, 4), "f1": round(f1_pass, 4)},
        "macro_f1": round(macro_f1, 4),
    }

# ── McNemar test ──
def mcnemar_test(y_true, pred_a, pred_b):
    b = c = 0
    for t, a, bb in zip(y_true, pred_a, pred_b):
        a_correct = (a == t)
        b_correct = (bb == t)
        if a_correct and not b_correct: b += 1
        elif not a_correct and b_correct: c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0, "p_value": 1.0, "note": "no discordant pairs"}
    chi2 = (abs(b - c) - 1)**2 / n  # with continuity correction
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {"b_sft_right_regex_wrong": b, "c_regex_right_sft_wrong": c, "chi2": round(chi2, 4), "p_value": round(p, 6)}

# ── Define baselines ──
def baseline_error_tag(trace):
    return "fail" if "[ERROR]" in trace else "pass"

def baseline_traceback(trace):
    return "fail" if "Traceback" in trace else "pass"

def baseline_any_keyword(trace):
    return "fail" if any(kw in trace for kw in ["[ERROR]", "Traceback", "Exception"]) else "pass"

def baseline_exception_regex(trace):
    pat = r'(AssertionError|AssertionError|ValueError|TypeError|IndexError|KeyError|AttributeError|RuntimeError|ZeroDivisionError|Exception|Error)'
    return "fail" if re.search(pat, trace) else "pass"

def baseline_no_trace_available(trace):
    """Exploit the dataset artifact: [No trace available] always means fail in this dataset"""
    if trace == "[No exception data in trace]":
        return "pass"
    else:
        return "fail"

def baseline_error_count(trace, threshold=1):
    return "fail" if trace.count("[ERROR]") >= threshold else "pass"

# ── Run all baselines ──
baselines = {
    "B1_ERROR_tag": lambda t: baseline_error_tag(t),
    "B2_Traceback": lambda t: baseline_traceback(t),
    "B3_any_keyword": lambda t: baseline_any_keyword(t),
    "B4_exception_regex": lambda t: baseline_exception_regex(t),
    "B5_no_exception_data_heuristic": lambda t: baseline_no_trace_available(t),
    "B6_error_count_t1": lambda t: baseline_error_count(t, 1),
    "B6_error_count_t2": lambda t: baseline_error_count(t, 2),
    "B6_error_count_t3": lambda t: baseline_error_count(t, 3),
    "B6_error_count_t5": lambda t: baseline_error_count(t, 5),
}

results = {}
all_preds = {}

for bname, bfn in baselines.items():
    preds = [bfn(s["trace"]) for s in samples]
    all_preds[bname] = preds
    m = compute_metrics(labels, preds, bname)
    mc = mcnemar_test(labels, sft_pred_labels, preds)
    m["mcnemar_vs_sft"] = mc
    results[bname] = m
    print(f"\n{bname}: acc={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
    print(f"  fail: P={m['fail']['precision']:.4f} R={m['fail']['recall']:.4f} F1={m['fail']['f1']:.4f}")
    print(f"  pass: P={m['pass']['precision']:.4f} R={m['pass']['recall']:.4f} F1={m['pass']['f1']:.4f}")
    print(f"  CM: TP={m['confusion_matrix']['TP']} FP={m['confusion_matrix']['FP']} FN={m['confusion_matrix']['FN']} TN={m['confusion_matrix']['TN']}")
    print(f"  McNemar: b={mc.get('b_sft_right_regex_wrong', mc.get('b',0))}, c={mc.get('c_regex_right_sft_wrong', mc.get('c',0))}, p={mc.get('p_value', 'N/A')}")

# ── SFT metrics ──
sft_metrics = compute_metrics(labels, sft_pred_labels, "SFT_exception_only")
print(f"\nSFT: acc={sft_metrics['accuracy']:.4f}, macro_f1={sft_metrics['macro_f1']:.4f}")
print(f"  fail: P={sft_metrics['fail']['precision']:.4f} R={sft_metrics['fail']['recall']:.4f} F1={sft_metrics['fail']['f1']:.4f}")
print(f"  pass: P={sft_metrics['pass']['precision']:.4f} R={sft_metrics['pass']['recall']:.4f} F1={sft_metrics['pass']['f1']:.4f}")
print(f"  CM: TP={sft_metrics['confusion_matrix']['TP']} FP={sft_metrics['confusion_matrix']['FP']} FN={sft_metrics['confusion_matrix']['FN']} TN={sft_metrics['confusion_matrix']['TN']}")
results["SFT_exception_only"] = sft_metrics

# ── Detailed analysis for best regex baseline ──
# Find best regex baseline
best_name = max((k for k in results if k != "SFT_exception_only"), key=lambda k: results[k]["accuracy"])
best_preds = all_preds[best_name]
print(f"\n{'='*60}")
print(f"Best regex baseline: {best_name} (acc={results[best_name]['accuracy']:.4f})")

# Sample-level disagreement analysis
print(f"\n=== Disagreement analysis: {best_name} vs SFT ===")
disagree_regex_wrong = []
disagree_sft_wrong = []
both_wrong = []
for i in range(N):
    regex_correct = (best_preds[i] == labels[i])
    sft_correct = (sft_pred_labels[i] == labels[i])
    if regex_correct and not sft_correct:
        disagree_sft_wrong.append(i)
    elif not regex_correct and sft_correct:
        disagree_regex_wrong.append(i)
    elif not regex_correct and not sft_correct:
        both_wrong.append(i)

print(f"Regex wrong, SFT right: {len(disagree_regex_wrong)}")
print(f"SFT wrong, Regex right: {len(disagree_sft_wrong)}")
print(f"Both wrong: {len(both_wrong)}")

# Analyze characteristics of disagreement samples
print(f"\n--- Regex wrong, SFT right (n={len(disagree_regex_wrong)}) ---")
for i in disagree_regex_wrong[:10]:
    s = samples[i]
    trace_preview = s['trace'][:120].replace('\n', ' ')
    print(f"  [{i}] label={s['label']} regex={best_preds[i]} sft={sft_pred_labels[i]} trace={trace_preview}")

print(f"\n--- SFT wrong, Regex right (n={len(disagree_sft_wrong)}) ---")
for i in disagree_sft_wrong[:10]:
    s = samples[i]
    trace_preview = s['trace'][:120].replace('\n', ' ')
    print(f"  [{i}] label={s['label']} regex={best_preds[i]} sft={sft_pred_labels[i]} trace={trace_preview}")

print(f"\n--- Both wrong (n={len(both_wrong)}) ---")
for i in both_wrong[:10]:
    s = samples[i]
    trace_preview = s['trace'][:120].replace('\n', ' ')
    print(f"  [{i}] label={s['label']} regex={best_preds[i]} sft={sft_pred_labels[i]} trace={trace_preview}")

# ── Analyze the 8 pass samples with trace content ──
print(f"\n=== 8 pass samples with actual trace content ===")
for i, s in enumerate(samples):
    if s['label'] == 'pass' and s['trace'] not in ('[No exception data in trace]', '[No trace available]', ''):
        trace_preview = s['trace'][:200].replace('\n', ' ')
        print(f"  [{i}] trace={trace_preview}")

# ── Calibration analysis (ECE + Brier) ──
# Regex: confidence = 1.0 always (binary output)
# SFT: need to check if logprobs available; if not, treat as confidence = 1.0 for correct pred
print(f"\n=== Calibration analysis ===")
# Check if SFT predictions have probabilities
has_probs = "prob" in sft_preds[0] or "confidence" in sft_preds[0] or "logprob" in sft_preds[0]
print(f"SFT predictions have probability info: {has_probs}")
print(f"SFT prediction fields: {list(sft_preds[0].keys())}")

# For regex: always confidence=1.0
regex_correct = sum(1 for i in range(N) if best_preds[i] == labels[i])
regex_wrong = N - regex_correct
regex_ece = regex_wrong / N  # since conf=1.0, ECE = error rate
regex_brier = regex_wrong / N  # Brier = mean((1.0 - correct)^2) = error rate when conf=1

print(f"\nRegex ({best_name}):")
print(f"  ECE (conf=1.0): {regex_ece:.6f}")
print(f"  Brier score: {regex_brier:.6f}")

# SFT also appears to have binary predictions (no probs)
sft_correct = sum(1 for i in range(N) if sft_pred_labels[i] == labels[i])
sft_wrong = N - sft_correct
sft_ece = sft_wrong / N
sft_brier = sft_wrong / N
print(f"\nSFT (no probs, conf=1.0 assumed):")
print(f"  ECE: {sft_ece:.6f}")
print(f"  Brier score: {sft_brier:.6f}")

# ── Save results ──
output = {
    "dataset": "exception_only_test",
    "n_samples": N,
    "label_distribution": {"pass": labels.count("pass"), "fail": labels.count("fail")},
    "trace_pattern_distribution": {
        "no_exception_data_pass": 843,
        "no_trace_available_fail": 70,
        "has_content_fail": 269,
        "has_content_pass": 8,
    },
    "baselines": results,
    "best_regex_baseline": best_name,
    "disagreement_analysis": {
        "regex_wrong_sft_right": len(disagree_regex_wrong),
        "sft_wrong_regex_right": len(disagree_sft_wrong),
        "both_wrong": len(both_wrong),
        "regex_wrong_indices": disagree_regex_wrong,
        "sft_wrong_indices": disagree_sft_wrong,
        "both_wrong_indices": both_wrong,
    },
    "calibration": {
        "regex": {"ece": regex_ece, "brier": regex_brier, "note": "confidence always 1.0"},
        "sft": {"ece": sft_ece, "brier": sft_brier, "note": "no probability output, assumed confidence 1.0"},
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}")
