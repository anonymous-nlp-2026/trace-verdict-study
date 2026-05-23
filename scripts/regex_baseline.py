import json
import re
from collections import Counter, defaultdict

# Load test data
test_data = []
with open("./data/exception_only_test.jsonl") as f:
    for i, line in enumerate(f):
        item = json.loads(line.strip())
        item["idx"] = i
        test_data.append(item)

# Load SFT predictions (exception_only seed42)
sft_preds = []
with open("./checkpoints/exception_only/final/predictions.jsonl") as f:
    for line in f:
        sft_preds.append(json.loads(line.strip()))

# Also load multi-seed predictions
seed_files = {
    "seed42": "./checkpoints/exception_only/final/predictions.jsonl",
    "seed123": "./checkpoints/exception_only_seed123/final/predictions.jsonl",
    "seed456": "./checkpoints/exception_only_seed456/final/predictions.jsonl",
}
all_seed_preds = {}
for seed_name, path in seed_files.items():
    try:
        preds = []
        with open(path) as f:
            for line in f:
                preds.append(json.loads(line.strip()))
        all_seed_preds[seed_name] = preds
    except:
        pass

print(f"=== DATA OVERVIEW ===")
print(f"Total test samples: {len(test_data)}")
labels = [d["output"] for d in test_data]
print(f"Pass: {labels.count('pass')}, Fail: {labels.count('fail')}")

# === Categorize each sample by trace content ===
print(f"\n=== TRACE CONTENT ANALYSIS ===")
categories = []
for d in test_data:
    inp = d["input"]
    trace_start = inp.find("Execution trace:\n")
    if trace_start == -1:
        trace_text = ""
    else:
        trace_text = inp[trace_start + len("Execution trace:\n"):].replace("\n\nDoes this code pass all tests? Answer: ", "")
    
    has_no_exception = "[No exception data in trace]" in trace_text
    has_no_trace = "[No trace available]" in trace_text
    has_error_tag = "[ERROR]" in trace_text
    has_error_word = bool(re.search(r'(Error|Exception|Traceback)', trace_text))
    
    cat = {
        "idx": d["idx"],
        "label": d["output"],
        "trace_text": trace_text.strip(),
        "has_no_exception": has_no_exception,
        "has_no_trace": has_no_trace,
        "has_error_tag": has_error_tag,
        "has_error_word": has_error_word,
        "sft_pred": sft_preds[d["idx"]]["pred"] if d["idx"] < len(sft_preds) else "?",
    }
    categories.append(cat)

# Cross-tab: trace pattern vs label
print("\nTrace pattern distribution:")
pattern_label = defaultdict(lambda: Counter())
for c in categories:
    if c["has_no_exception"]:
        pat = "NO_EXCEPTION_MARKER"
    elif c["has_no_trace"]:
        pat = "NO_TRACE_AVAILABLE"
    elif c["has_error_tag"]:
        pat = "HAS_ERROR_TAG"
    else:
        pat = "OTHER"
    pattern_label[pat][c["label"]] += 1
    c["pattern"] = pat

for pat in ["NO_EXCEPTION_MARKER", "NO_TRACE_AVAILABLE", "HAS_ERROR_TAG", "OTHER"]:
    counts = pattern_label[pat]
    total = sum(counts.values())
    print(f"  {pat}: total={total}, pass={counts['pass']}, fail={counts['fail']}")

# === REGEX BASELINES ===
print(f"\n=== REGEX BASELINES ===")

def b_marker(trace_text):
    """[No exception data in trace] -> pass, else -> fail"""
    if "[No exception data in trace]" in trace_text:
        return "pass"
    return "fail"

def b_error_tag(trace_text):
    """[ERROR] -> fail, [No exception data in trace] -> pass, else -> pass"""
    if "[ERROR]" in trace_text:
        return "fail"
    return "pass"

def b_no_trace_fail(trace_text):
    """[No exception data in trace] -> pass, [No trace available] -> fail, [ERROR] -> fail, else -> fail"""
    if "[No exception data in trace]" in trace_text:
        return "pass"
    return "fail"

def b_optimal(trace_text):
    """Best possible regex: marker->pass, error/no_trace->fail"""
    if "[No exception data in trace]" in trace_text:
        return "pass"
    if "[ERROR]" in trace_text or "[No trace available]" in trace_text:
        return "fail"
    # For OTHER category, choose whichever gives better results
    return "fail"

def b_reviewer(trace_text):
    """Reviewer's proposed rule: has error text -> fail, else -> pass"""
    if re.search(r'(Error|Exception|Traceback|\[ERROR\])', trace_text):
        return "fail"
    if "[No trace available]" in trace_text:
        return "fail"
    return "pass"

def evaluate(name, pred_fn, categories, show_errors=True):
    preds = [pred_fn(c["trace_text"]) for c in categories]
    labels = [c["label"] for c in categories]
    
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(labels)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {correct}/{len(labels)} = {acc:.4f} ({acc*100:.2f}%)")
    
    for cls in ["pass", "fail"]:
        tp = sum(p == cls and l == cls for p, l in zip(preds, labels))
        fp = sum(p == cls and l != cls for p, l in zip(preds, labels))
        fn = sum(p != cls and l == cls for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {cls}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})")
    
    # Errors
    errors = [(c, p) for c, p in zip(categories, preds) if p != c["label"]]
    if show_errors and errors:
        print(f"  Errors ({len(errors)}):")
        for c, p in errors[:20]:
            sft_correct = "SFT✓" if c["sft_pred"] == c["label"] else "SFT✗"
            trace_preview = c["trace_text"][:80].replace("\n", " ")
            print(f"    idx={c['idx']} label={c['label']} pred={p} pattern={c['pattern']} {sft_correct} | {trace_preview}")
        if len(errors) > 20:
            print(f"    ... and {len(errors)-20} more")
    
    return preds

# Run all baselines
regex_results = {}
for name, fn in [("B_marker", b_marker), ("B_error_tag", b_error_tag), 
                  ("B_no_trace_fail", b_no_trace_fail), ("B_optimal", b_optimal),
                  ("B_reviewer", b_reviewer)]:
    preds = evaluate(name, fn, categories)
    regex_results[name] = preds

# === SFT EVALUATION ===
print(f"\n--- SFT (exception_only seed42) ---")
sft_labels = [c["label"] for c in categories]
sft_predictions = [c["sft_pred"] for c in categories]
sft_correct = sum(p == l for p, l in zip(sft_predictions, sft_labels))
print(f"Accuracy: {sft_correct}/{len(sft_labels)} = {sft_correct/len(sft_labels):.4f} ({sft_correct/len(sft_labels)*100:.2f}%)")
for cls in ["pass", "fail"]:
    tp = sum(p == cls and l == cls for p, l in zip(sft_predictions, sft_labels))
    fp = sum(p == cls and l != cls for p, l in zip(sft_predictions, sft_labels))
    fn = sum(p != cls and l == cls for p, l in zip(sft_predictions, sft_labels))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  {cls}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})")

# SFT errors
sft_errors = [(c, p) for c, p in zip(categories, sft_predictions) if p != c["label"]]
print(f"  Errors ({len(sft_errors)}):")
for c, p in sft_errors[:20]:
    trace_preview = c["trace_text"][:80].replace("\n", " ")
    print(f"    idx={c['idx']} label={c['label']} pred={p} pattern={c['pattern']} | {trace_preview}")

# === McNEMAR ANALYSIS: SFT vs each regex ===
print(f"\n=== McNEMAR DISCORDANT PAIRS (SFT vs Regex) ===")
for rname, rpreds in regex_results.items():
    sft_right_regex_wrong = []
    sft_wrong_regex_right = []
    for i, (s, r, l) in enumerate(zip(sft_predictions, rpreds, sft_labels)):
        sc = (s == l)
        rc = (r == l)
        if sc and not rc:
            sft_right_regex_wrong.append(i)
        if rc and not sc:
            sft_wrong_regex_right.append(i)
    
    n_sr = len(sft_right_regex_wrong)
    n_rs = len(sft_wrong_regex_right)
    print(f"\n{rname}:")
    print(f"  SFT✓ & Regex✗: {n_sr}")
    print(f"  SFT✗ & Regex✓: {n_rs}")
    print(f"  Net advantage SFT: {n_sr - n_rs}")
    
    if sft_right_regex_wrong:
        print(f"  Samples where SFT correct but {rname} wrong:")
        for idx in sft_right_regex_wrong[:15]:
            c = categories[idx]
            trace_preview = c["trace_text"][:100].replace("\n", " ")
            print(f"    idx={idx} label={c['label']} sft={c['sft_pred']} regex={rpreds[idx]} pattern={c['pattern']} | {trace_preview}")
        if len(sft_right_regex_wrong) > 15:
            print(f"    ... and {len(sft_right_regex_wrong)-15} more")

# === Q2: EDGE CASE ANALYSIS ===
print(f"\n=== Q2: EDGE CASE ANALYSIS ===")

# Case 1: fail samples with no exception/error text
print("\n--- Case 1: fail samples with NO exception/error text ---")
fail_no_error = [c for c in categories if c["label"] == "fail" and c["pattern"] not in ["HAS_ERROR_TAG"]]
print(f"Count: {len(fail_no_error)}")
for c in fail_no_error:
    sft_correct = "SFT✓" if c["sft_pred"] == c["label"] else "SFT✗"
    trace_preview = c["trace_text"][:120].replace("\n", " ")
    print(f"  idx={c['idx']} pattern={c['pattern']} {sft_correct} | {trace_preview}")

# Case 2: pass samples with exception text
print("\n--- Case 2: pass samples WITH exception/error-like text ---")
pass_with_error = [c for c in categories if c["label"] == "pass" and (c["has_error_tag"] or c["has_error_word"])]
print(f"Count: {len(pass_with_error)}")
for c in pass_with_error[:20]:
    sft_correct = "SFT✓" if c["sft_pred"] == c["label"] else "SFT✗"
    trace_preview = c["trace_text"][:120].replace("\n", " ")
    print(f"  idx={c['idx']} pattern={c['pattern']} has_error_tag={c['has_error_tag']} has_error_word={c['has_error_word']} {sft_correct} | {trace_preview}")

# Case 3: "No trace available" samples - label distribution
print("\n--- Case 3: [No trace available] samples ---")
no_trace = [c for c in categories if c["pattern"] == "NO_TRACE_AVAILABLE"]
print(f"Count: {len(no_trace)}")
no_trace_labels = Counter(c["label"] for c in no_trace)
print(f"  Labels: {dict(no_trace_labels)}")
no_trace_sft = Counter(c["sft_pred"] for c in no_trace)
print(f"  SFT predictions: {dict(no_trace_sft)}")
no_trace_sft_correct = sum(1 for c in no_trace if c["sft_pred"] == c["label"])
print(f"  SFT accuracy on these: {no_trace_sft_correct}/{len(no_trace)}")

# Case 4: OTHER category (no marker, no trace available, no error tag)
print("\n--- Case 4: OTHER category samples (no standard markers) ---")
other = [c for c in categories if c["pattern"] == "OTHER"]
print(f"Count: {len(other)}")
if other:
    for c in other[:10]:
        sft_correct = "SFT✓" if c["sft_pred"] == c["label"] else "SFT✗"
        print(f"  idx={c['idx']} label={c['label']} {sft_correct}")
        print(f"    trace: {c['trace_text'][:200]}")

# === MULTI-SEED SFT RESULTS ===
print(f"\n=== MULTI-SEED SFT COMPARISON ===")
for seed_name, preds_list in all_seed_preds.items():
    if len(preds_list) != len(test_data):
        print(f"{seed_name}: {len(preds_list)} predictions (MISMATCH)")
        continue
    correct = sum(preds_list[i]["pred"] == test_data[i]["output"] for i in range(len(test_data)))
    print(f"{seed_name}: {correct}/{len(test_data)} = {correct/len(test_data)*100:.2f}%")

# Best regex accuracy
print(f"\n=== SUMMARY ===")
for rname, rpreds in regex_results.items():
    acc = sum(p == l for p, l in zip(rpreds, sft_labels)) / len(sft_labels)
    print(f"{rname}: {acc*100:.2f}%")
sft_acc = sft_correct / len(sft_labels)
print(f"SFT (seed42): {sft_acc*100:.2f}%")
