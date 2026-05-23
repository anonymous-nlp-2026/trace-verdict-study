"""Error analysis across trace conditions for plan_018."""
import json
import os
import re
from collections import Counter, defaultdict

CHECKPOINT_BASE = "./checkpoints"
DATA_BASE = "./data"
ARTIFACT_DIR = "./artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Conditions to analyze ---
# Qwen-7B main conditions (seed42 defaults)
CONDITIONS = {
    "exception_only": f"{CHECKPOINT_BASE}/exception_only/final/predictions.jsonl",
    "vota": f"{CHECKPOINT_BASE}/vota/final/predictions.jsonl",
    "ae_only": f"{CHECKPOINT_BASE}/ae_only/final/predictions.jsonl",
    "full_trace": f"{CHECKPOINT_BASE}/full_trace/final/predictions.jsonl",
    "full_trace_label_only": f"{CHECKPOINT_BASE}/full_trace_label_only/final/predictions.jsonl",
    "loop_only": f"{CHECKPOINT_BASE}/loop_only/final/predictions.jsonl",
    "no_trace": f"{CHECKPOINT_BASE}/no_trace/final/predictions.jsonl",
    "vota_no_ae": f"{CHECKPOINT_BASE}/vota_no_ae/final/predictions.jsonl",
}

TEST_DATA = {
    "exception_only": f"{DATA_BASE}/exception_only_test.jsonl",
    "vota": f"{DATA_BASE}/vota_test.jsonl",
    "ae_only": f"{DATA_BASE}/ae_only_test.jsonl",
    "full_trace": f"{DATA_BASE}/full_trace_test.jsonl",
    "full_trace_label_only": f"{DATA_BASE}/full_trace_test.jsonl",  # same underlying data
    "loop_only": f"{DATA_BASE}/loop_only_test.jsonl",
    "no_trace": f"{DATA_BASE}/no_trace_test.jsonl",
    "vota_no_ae": f"{DATA_BASE}/vota_no_ae_test.jsonl",
}

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def extract_error_type(error_str):
    if not error_str:
        return "no_error"
    match = re.match(r"(\w+Error|\w+Exception|Timeout)", error_str)
    if match:
        return match.group(1)
    if "Timeout" in error_str or "timeout" in error_str:
        return "Timeout"
    return "OtherError"

def extract_source(problem_id):
    if problem_id.startswith("HumanEval"):
        return "humaneval"
    elif problem_id.startswith("MBPP"):
        return "mbpp"
    return "unknown"

# --- Load raw traces for metadata ---
print("Loading raw traces...")
raw_traces = load_jsonl(f"{DATA_BASE}/raw_traces.jsonl")
print(f"  Total raw traces: {len(raw_traces)}")

# Build lookup: (problem_id, solution_hash) -> metadata
# Since multiple solutions per problem, use index in raw_traces
raw_by_pid = defaultdict(list)
for i, item in enumerate(raw_traces):
    raw_by_pid[item["problem_id"]].append(item)

# --- Reproduce train/test split to identify test problem_ids ---
import random
rng = random.Random(42)
by_problem = defaultdict(list)
for item in raw_traces:
    by_problem[item["problem_id"]].append(item)
problem_ids = sorted(by_problem.keys())
rng.shuffle(problem_ids)
n_train = int(len(problem_ids) * 0.8)
test_pids = set(problem_ids[n_train:])
print(f"  Test problem IDs: {len(test_pids)}")

# --- Load test data (use exception_only to get problem_id ordering) ---
print("Loading test data for problem_id mapping...")
ref_test = load_jsonl(TEST_DATA["exception_only"])
print(f"  Test samples: {len(ref_test)}")

# Build per-sample metadata from raw_traces
# Match test samples to raw_traces by problem_id + verdict
test_metadata = []
for i, sample in enumerate(ref_test):
    pid = sample["problem_id"]
    label = sample["output"]  # "pass" or "fail"
    verdict_bool = (label == "pass")
    
    # Find matching raw trace entry
    candidates = [r for r in raw_by_pid[pid] if r["verdict"] == verdict_bool]
    
    meta = {
        "idx": i,
        "problem_id": pid,
        "source": extract_source(pid),
        "label": label,
    }
    
    if candidates:
        raw = candidates[0]
        tr = raw.get("test_results", {})
        meta["error_type"] = extract_error_type(tr.get("error", ""))
        meta["n_passed"] = tr.get("passed", 0)
        meta["n_failed"] = tr.get("failed", 0)
        meta["has_trace"] = bool(raw.get("trace_raw"))
        meta["solution_len"] = len(raw.get("solution", "").split("\n"))
    else:
        meta["error_type"] = "unknown"
        meta["n_passed"] = -1
        meta["n_failed"] = -1
        meta["has_trace"] = False
        meta["solution_len"] = -1
    
    test_metadata.append(meta)

# Verify: check problem_id ordering consistency across conditions
for cond_name, test_path in TEST_DATA.items():
    if not os.path.exists(test_path):
        continue
    cond_test = load_jsonl(test_path)
    if len(cond_test) != len(ref_test):
        print(f"  WARNING: {cond_name} test size mismatch: {len(cond_test)} vs {len(ref_test)}")
        continue
    mismatches = sum(1 for a, b in zip(ref_test, cond_test) if a["problem_id"] != b["problem_id"])
    if mismatches > 0:
        print(f"  WARNING: {cond_name} has {mismatches} problem_id mismatches!")

# --- Load predictions ---
print("\nLoading predictions...")
predictions = {}
for cond_name, pred_path in CONDITIONS.items():
    if not os.path.exists(pred_path):
        print(f"  SKIP {cond_name}: file not found")
        continue
    preds = load_jsonl(pred_path)
    if len(preds) != len(ref_test):
        print(f"  WARNING: {cond_name} has {len(preds)} predictions vs {len(ref_test)} test samples")
    predictions[cond_name] = preds
    acc = sum(1 for p in preds if p["pred"] == p["label"]) / len(preds) * 100
    print(f"  {cond_name}: {len(preds)} samples, accuracy={acc:.2f}%")

# ================================================================
# ANALYSIS
# ================================================================
results = {}

# --- 1. Basic TP/FP/TN/FN per condition ---
print("\n" + "=" * 60)
print("1. Confusion matrix per condition")
print("=" * 60)
confusion = {}
for cond_name, preds in predictions.items():
    tp = fn = fp = tn = 0
    for p in preds:
        if p["label"] == "pass" and p["pred"] == "pass":
            tp += 1
        elif p["label"] == "pass" and p["pred"] == "fail":
            fn += 1
        elif p["label"] == "fail" and p["pred"] == "pass":
            fp += 1
        else:
            tn += 1
    total = len(preds)
    acc = (tp + tn) / total * 100
    prec_pass = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec_pass = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    prec_fail = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
    rec_fail = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    confusion[cond_name] = {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(acc, 2),
        "precision_pass": round(prec_pass, 2),
        "recall_pass": round(rec_pass, 2),
        "precision_fail": round(prec_fail, 2),
        "recall_fail": round(rec_fail, 2),
        "total": total,
        "n_errors": fp + fn,
    }
    print(f"  {cond_name}: TP={tp} FP={fp} TN={tn} FN={fn} | acc={acc:.2f}% | errors={fp+fn}")
results["confusion"] = confusion

# --- 2. Error samples by error type and source ---
print("\n" + "=" * 60)
print("2. Error distribution by error_type and source")
print("=" * 60)
error_by_type = {}
error_by_source = {}
for cond_name, preds in predictions.items():
    type_counter = Counter()
    source_counter = Counter()
    fp_type = Counter()
    fn_type = Counter()
    fp_source = Counter()
    fn_source = Counter()
    
    for i, p in enumerate(preds):
        if p["pred"] != p["label"]:
            meta = test_metadata[i]
            et = meta["error_type"]
            src = meta["source"]
            type_counter[et] += 1
            source_counter[src] += 1
            
            if p["label"] == "fail" and p["pred"] == "pass":  # FP (predicted pass, actual fail)
                fp_type[et] += 1
                fp_source[src] += 1
            else:  # FN (predicted fail, actual pass)
                fn_type[et] += 1
                fn_source[src] += 1
    
    error_by_type[cond_name] = {
        "all": dict(type_counter.most_common()),
        "FP": dict(fp_type.most_common()),
        "FN": dict(fn_type.most_common()),
    }
    error_by_source[cond_name] = {
        "all": dict(source_counter.most_common()),
        "FP": dict(fp_source.most_common()),
        "FN": dict(fn_source.most_common()),
    }
    print(f"\n  {cond_name}:")
    print(f"    By error_type: {dict(type_counter.most_common(5))}")
    print(f"    By source: {dict(source_counter)}")
    print(f"    FP by type: {dict(fp_type.most_common(5))}")
    print(f"    FN by type: {dict(fn_type.most_common(5))}")

results["error_by_type"] = error_by_type
results["error_by_source"] = error_by_source

# --- 3. Tier analysis ---
print("\n" + "=" * 60)
print("3. Tier 1 vs Tier 2 failure modes")
print("=" * 60)
tier1_conds = [c for c in ["exception_only", "vota", "ae_only"] if c in predictions]
tier2_conds = [c for c in ["full_trace", "loop_only", "no_trace", "full_trace_label_only"] if c in predictions]

tier1_errors = defaultdict(set)  # idx -> set of conditions that got it wrong
tier2_errors = defaultdict(set)
for cond in tier1_conds:
    for i, p in enumerate(predictions[cond]):
        if p["pred"] != p["label"]:
            tier1_errors[i].add(cond)
for cond in tier2_conds:
    for i, p in enumerate(predictions[cond]):
        if p["pred"] != p["label"]:
            tier2_errors[i].add(cond)

tier1_error_ids = set(tier1_errors.keys())
tier2_error_ids = set(tier2_errors.keys())
all_error_ids = tier1_error_ids | tier2_error_ids

overlap = tier1_error_ids & tier2_error_ids
tier1_unique = tier1_error_ids - tier2_error_ids
tier2_unique = tier2_error_ids - tier1_error_ids

tier_analysis = {
    "tier1_conditions": tier1_conds,
    "tier2_conditions": tier2_conds,
    "tier1_total_errors": len(tier1_error_ids),
    "tier2_total_errors": len(tier2_error_ids),
    "overlap": len(overlap),
    "tier1_unique": len(tier1_unique),
    "tier2_unique": len(tier2_unique),
    "tier1_subset_of_tier2": tier1_error_ids.issubset(tier2_error_ids),
}

print(f"  Tier 1 ({tier1_conds}): {len(tier1_error_ids)} unique error samples")
print(f"  Tier 2 ({tier2_conds}): {len(tier2_error_ids)} unique error samples")
print(f"  Overlap: {len(overlap)}")
print(f"  Tier 1 unique (not in Tier 2 errors): {len(tier1_unique)}")
print(f"  Tier 2 unique (not in Tier 1 errors): {len(tier2_unique)}")
print(f"  Tier 1 errors ⊂ Tier 2 errors: {tier1_error_ids.issubset(tier2_error_ids)}")

# Characterize overlap samples
if overlap:
    overlap_types = Counter()
    overlap_sources = Counter()
    for idx in overlap:
        meta = test_metadata[idx]
        overlap_types[meta["error_type"]] += 1
        overlap_sources[meta["source"]] += 1
    tier_analysis["overlap_error_types"] = dict(overlap_types.most_common())
    tier_analysis["overlap_sources"] = dict(overlap_sources)
    print(f"\n  Overlap error types: {dict(overlap_types.most_common())}")
    print(f"  Overlap sources: {dict(overlap_sources)}")

# Tier1 unique errors (Tier 1 got wrong but Tier 2 conditions all got right)
if tier1_unique:
    t1u_types = Counter()
    for idx in tier1_unique:
        t1u_types[test_metadata[idx]["error_type"]] += 1
    tier_analysis["tier1_unique_error_types"] = dict(t1u_types.most_common())
    print(f"\n  Tier 1 unique error types: {dict(t1u_types.most_common())}")

# Tier2 unique: what Tier2 gets wrong but Tier1 all get right
if tier2_unique:
    t2u_types = Counter()
    t2u_sources = Counter()
    for idx in tier2_unique:
        meta = test_metadata[idx]
        t2u_types[meta["error_type"]] += 1
        t2u_sources[meta["source"]] += 1
    tier_analysis["tier2_unique_error_types"] = dict(t2u_types.most_common())
    tier_analysis["tier2_unique_sources"] = dict(t2u_sources)
    print(f"\n  Tier 2 unique error types: {dict(t2u_types.most_common())}")
    print(f"  Tier 2 unique sources: {dict(t2u_sources)}")

results["tier_analysis"] = tier_analysis

# --- 4. Cross-condition error consistency ---
print("\n" + "=" * 60)
print("4. Cross-condition error consistency")
print("=" * 60)
all_conds = list(predictions.keys())
per_sample_errors = defaultdict(set)  # idx -> set of conditions that got it wrong
for cond in all_conds:
    for i, p in enumerate(predictions[cond]):
        if p["pred"] != p["label"]:
            per_sample_errors[i].add(cond)

# Samples wrong in ALL conditions
all_wrong = {idx for idx, conds in per_sample_errors.items() if len(conds) == len(all_conds)}
# Samples wrong in no condition (all correct)
n_never_wrong = len(ref_test) - len(per_sample_errors)

cross_consistency = {
    "total_samples": len(ref_test),
    "never_wrong": n_never_wrong,
    "wrong_in_at_least_one": len(per_sample_errors),
    "wrong_in_all": len(all_wrong),
}

# Distribution of "how many conditions got it wrong"
wrong_count_dist = Counter()
for idx, conds in per_sample_errors.items():
    wrong_count_dist[len(conds)] += 1
cross_consistency["wrong_count_distribution"] = dict(sorted(wrong_count_dist.items()))

print(f"  Never wrong (all {len(all_conds)} conditions correct): {n_never_wrong}")
print(f"  Wrong in at least 1 condition: {len(per_sample_errors)}")
print(f"  Wrong in ALL {len(all_conds)} conditions: {len(all_wrong)}")
print(f"  Distribution (n_conditions_wrong -> n_samples): {dict(sorted(wrong_count_dist.items()))}")

# Characterize "all wrong" samples
if all_wrong:
    aw_types = Counter()
    aw_sources = Counter()
    aw_labels = Counter()
    for idx in all_wrong:
        meta = test_metadata[idx]
        aw_types[meta["error_type"]] += 1
        aw_sources[meta["source"]] += 1
        aw_labels[meta["label"]] += 1
    cross_consistency["all_wrong_types"] = dict(aw_types.most_common())
    cross_consistency["all_wrong_sources"] = dict(aw_sources)
    cross_consistency["all_wrong_labels"] = dict(aw_labels)
    print(f"\n  All-wrong samples:")
    print(f"    Error types: {dict(aw_types.most_common())}")
    print(f"    Sources: {dict(aw_sources)}")
    print(f"    Labels: {dict(aw_labels)}")

    # Show specific examples
    examples = []
    for idx in sorted(all_wrong)[:10]:
        meta = test_metadata[idx]
        ex = {"idx": idx, "problem_id": meta["problem_id"], "label": meta["label"],
               "error_type": meta["error_type"], "source": meta["source"]}
        for cond in all_conds:
            ex[f"pred_{cond}"] = predictions[cond][idx]["pred"]
        examples.append(ex)
    cross_consistency["all_wrong_examples"] = examples

# Samples where ONLY Tier1 got right (Tier2 all wrong, Tier1 all correct)
tier1_only_correct = set()
for idx in range(len(ref_test)):
    t1_all_correct = all(predictions[c][idx]["pred"] == predictions[c][idx]["label"] for c in tier1_conds)
    t2_any_wrong = any(predictions[c][idx]["pred"] != predictions[c][idx]["label"] for c in tier2_conds)
    if t1_all_correct and t2_any_wrong:
        tier1_only_correct.add(idx)

cross_consistency["tier1_only_correct"] = len(tier1_only_correct)
if tier1_only_correct:
    t1oc_types = Counter()
    t1oc_sources = Counter()
    t1oc_labels = Counter()
    for idx in tier1_only_correct:
        meta = test_metadata[idx]
        t1oc_types[meta["error_type"]] += 1
        t1oc_sources[meta["source"]] += 1
        t1oc_labels[meta["label"]] += 1
    cross_consistency["tier1_only_correct_types"] = dict(t1oc_types.most_common())
    cross_consistency["tier1_only_correct_sources"] = dict(t1oc_sources)
    cross_consistency["tier1_only_correct_labels"] = dict(t1oc_labels)
    print(f"\n  Tier 1 all correct but Tier 2 has errors: {len(tier1_only_correct)} samples")
    print(f"    Error types: {dict(t1oc_types.most_common(5))}")
    print(f"    Sources: {dict(t1oc_sources)}")
    print(f"    Labels: {dict(t1oc_labels)}")

results["cross_consistency"] = cross_consistency

# --- 5. FP vs FN bias ---
print("\n" + "=" * 60)
print("5. FP vs FN bias per condition")
print("=" * 60)
fp_fn_analysis = {}
for cond_name in all_conds:
    cm = confusion[cond_name]
    total_pos = cm["TP"] + cm["FN"]  # actual pass
    total_neg = cm["TN"] + cm["FP"]  # actual fail
    fp_rate = cm["FP"] / total_neg * 100 if total_neg > 0 else 0
    fn_rate = cm["FN"] / total_pos * 100 if total_pos > 0 else 0
    bias = "FP-biased" if cm["FP"] > cm["FN"] else ("FN-biased" if cm["FN"] > cm["FP"] else "balanced")
    fp_fn_analysis[cond_name] = {
        "FP": cm["FP"], "FN": cm["FN"],
        "FP_rate": round(fp_rate, 2), "FN_rate": round(fn_rate, 2),
        "bias": bias,
        "actual_pass": total_pos, "actual_fail": total_neg,
    }
    print(f"  {cond_name}: FP={cm['FP']}({fp_rate:.1f}%) FN={cm['FN']}({fn_rate:.1f}%) -> {bias}")
results["fp_fn_analysis"] = fp_fn_analysis

# --- 6. Error by solution length (code complexity proxy) ---
print("\n" + "=" * 60)
print("6. Error rate by solution length")
print("=" * 60)
length_bins = [(0, 5, "1-5 lines"), (5, 10, "6-10"), (10, 20, "11-20"), (20, 50, "21-50"), (50, 999, "50+")]
error_by_length = {}
for cond_name, preds in predictions.items():
    bin_stats = {}
    for lo, hi, label in length_bins:
        indices = [i for i, m in enumerate(test_metadata) if lo < m["solution_len"] <= hi]
        if not indices:
            continue
        n_err = sum(1 for i in indices if preds[i]["pred"] != preds[i]["label"])
        bin_stats[label] = {"total": len(indices), "errors": n_err, 
                            "error_rate": round(n_err / len(indices) * 100, 2)}
    error_by_length[cond_name] = bin_stats
    print(f"  {cond_name}:")
    for label, stats in bin_stats.items():
        print(f"    {label}: {stats['errors']}/{stats['total']} ({stats['error_rate']}%)")
results["error_by_length"] = error_by_length

# --- 7. Detailed error examples for paper ---
print("\n" + "=" * 60)
print("7. Interesting error examples")
print("=" * 60)

# Find samples where exception_only is correct but no_trace is wrong
if "exception_only" in predictions and "no_trace" in predictions:
    eo_correct_nt_wrong = []
    for i in range(len(ref_test)):
        eo = predictions["exception_only"][i]
        nt = predictions["no_trace"][i]
        if eo["pred"] == eo["label"] and nt["pred"] != nt["label"]:
            meta = test_metadata[i]
            eo_correct_nt_wrong.append({
                "idx": i, "problem_id": meta["problem_id"],
                "label": meta["label"], "error_type": meta["error_type"],
                "source": meta["source"], "solution_len": meta["solution_len"],
            })
    results["exception_only_correct_no_trace_wrong"] = {
        "count": len(eo_correct_nt_wrong),
        "by_error_type": dict(Counter(x["error_type"] for x in eo_correct_nt_wrong).most_common()),
        "by_source": dict(Counter(x["source"] for x in eo_correct_nt_wrong)),
        "by_label": dict(Counter(x["label"] for x in eo_correct_nt_wrong)),
        "examples": eo_correct_nt_wrong[:20],
    }
    print(f"  exception_only correct & no_trace wrong: {len(eo_correct_nt_wrong)}")
    print(f"    By error_type: {dict(Counter(x['error_type'] for x in eo_correct_nt_wrong).most_common(5))}")

# Find samples where full_trace is wrong but exception_only is correct
if "exception_only" in predictions and "full_trace" in predictions:
    eo_correct_ft_wrong = []
    for i in range(len(ref_test)):
        eo = predictions["exception_only"][i]
        ft = predictions["full_trace"][i]
        if eo["pred"] == eo["label"] and ft["pred"] != ft["label"]:
            meta = test_metadata[i]
            eo_correct_ft_wrong.append({
                "idx": i, "problem_id": meta["problem_id"],
                "label": meta["label"], "error_type": meta["error_type"],
                "source": meta["source"],
            })
    results["exception_only_correct_full_trace_wrong"] = {
        "count": len(eo_correct_ft_wrong),
        "by_error_type": dict(Counter(x["error_type"] for x in eo_correct_ft_wrong).most_common()),
        "by_source": dict(Counter(x["source"] for x in eo_correct_ft_wrong)),
        "by_label": dict(Counter(x["label"] for x in eo_correct_ft_wrong)),
    }
    print(f"  exception_only correct & full_trace wrong: {len(eo_correct_ft_wrong)}")

# --- 8. Label distribution in test set ---
label_dist = Counter(m["label"] for m in test_metadata)
source_dist = Counter(m["source"] for m in test_metadata)
error_type_dist = Counter(m["error_type"] for m in test_metadata)
results["test_set_stats"] = {
    "total": len(test_metadata),
    "label_distribution": dict(label_dist),
    "source_distribution": dict(source_dist),
    "error_type_distribution": dict(error_type_dist.most_common()),
}
print(f"\n  Test set: {dict(label_dist)}, sources: {dict(source_dist)}")
print(f"  Error types: {dict(error_type_dist.most_common(10))}")

# ================================================================
# SAVE RESULTS
# ================================================================

# Save JSON
json_path = f"{ARTIFACT_DIR}/error_analysis_plan018.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved JSON: {json_path}")

# Generate markdown report
md_lines = []
md_lines.append("# Error Analysis: Trace Condition Failure Modes (Plan 018)")
md_lines.append("")
md_lines.append(f"**Date**: 2026-05-21  ")
md_lines.append(f"**Test set**: {len(ref_test)} samples  ")
md_lines.append(f"**Conditions analyzed**: {', '.join(sorted(predictions.keys()))}  ")
md_lines.append("")

# Section 1
md_lines.append("## 1. Confusion Matrix Summary")
md_lines.append("")
md_lines.append("| Condition | Accuracy | TP | FP | TN | FN | Total Errors |")
md_lines.append("|-----------|----------|----|----|----|----|-------------|")
for cond in sorted(confusion.keys(), key=lambda c: -confusion[c]["accuracy"]):
    cm = confusion[cond]
    md_lines.append(f"| {cond} | {cm['accuracy']}% | {cm['TP']} | {cm['FP']} | {cm['TN']} | {cm['FN']} | {cm['n_errors']} |")
md_lines.append("")

# Section 2
md_lines.append("## 2. Error Distribution by Error Type")
md_lines.append("")
for cond in sorted(error_by_type.keys(), key=lambda c: -confusion[c]["accuracy"]):
    ebt = error_by_type[cond]
    md_lines.append(f"### {cond} (acc={confusion[cond]['accuracy']}%)")
    md_lines.append(f"- **All errors**: {ebt['all']}")
    md_lines.append(f"- **FP** (pred=pass, actual=fail): {ebt['FP']}")
    md_lines.append(f"- **FN** (pred=fail, actual=pass): {ebt['FN']}")
    md_lines.append("")

# Section 3
md_lines.append("## 3. Tier Analysis")
md_lines.append("")
ta = tier_analysis
md_lines.append(f"- **Tier 1** ({', '.join(ta['tier1_conditions'])}): {ta['tier1_total_errors']} unique error samples")
md_lines.append(f"- **Tier 2** ({', '.join(ta['tier2_conditions'])}): {ta['tier2_total_errors']} unique error samples")
md_lines.append(f"- **Overlap**: {ta['overlap']} samples wrong in both tiers")
md_lines.append(f"- **Tier 1 unique** (wrong in Tier 1 only): {ta['tier1_unique']}")
md_lines.append(f"- **Tier 2 unique** (wrong in Tier 2 only): {ta['tier2_unique']}")
md_lines.append(f"- **Tier 1 errors ⊂ Tier 2 errors**: {ta['tier1_subset_of_tier2']}")
md_lines.append("")
if "tier2_unique_error_types" in ta:
    md_lines.append(f"Tier 2 unique error types: {ta['tier2_unique_error_types']}")
    md_lines.append(f"Tier 2 unique sources: {ta.get('tier2_unique_sources', {})}")
    md_lines.append("")

# Section 4
md_lines.append("## 4. Cross-Condition Error Consistency")
md_lines.append("")
cc = cross_consistency
md_lines.append(f"- Never wrong (all conditions correct): **{cc['never_wrong']}** / {cc['total_samples']}")
md_lines.append(f"- Wrong in at least 1 condition: **{cc['wrong_in_at_least_one']}**")
md_lines.append(f"- Wrong in ALL conditions: **{cc['wrong_in_all']}** (inherently hard)")
md_lines.append(f"- Tier 1 all correct but Tier 2 has errors: **{cc['tier1_only_correct']}**")
md_lines.append("")
md_lines.append("**Distribution of errors across conditions:**")
md_lines.append("")
md_lines.append("| # Conditions Wrong | # Samples |")
md_lines.append("|-------------------|-----------|")
for k, v in sorted(cc["wrong_count_distribution"].items()):
    md_lines.append(f"| {k} | {v} |")
md_lines.append("")

if "all_wrong_types" in cc:
    md_lines.append(f"**All-wrong samples**: error types = {cc['all_wrong_types']}, sources = {cc['all_wrong_sources']}, labels = {cc['all_wrong_labels']}")
    md_lines.append("")

if "tier1_only_correct_types" in cc:
    md_lines.append(f"**Tier 1 correct / Tier 2 wrong**: error types = {cc['tier1_only_correct_types']}")
    md_lines.append(f"- Sources: {cc['tier1_only_correct_sources']}")
    md_lines.append(f"- Labels: {cc['tier1_only_correct_labels']}")
    md_lines.append("")

# Section 5
md_lines.append("## 5. FP vs FN Bias")
md_lines.append("")
md_lines.append("| Condition | FP | FP Rate | FN | FN Rate | Bias |")
md_lines.append("|-----------|----|---------|----|---------|------|")
for cond in sorted(fp_fn_analysis.keys(), key=lambda c: -confusion[c]["accuracy"]):
    fa = fp_fn_analysis[cond]
    md_lines.append(f"| {cond} | {fa['FP']} | {fa['FP_rate']}% | {fa['FN']} | {fa['FN_rate']}% | {fa['bias']} |")
md_lines.append("")

# Section 6
md_lines.append("## 6. Error Rate by Solution Length")
md_lines.append("")
for cond in sorted(error_by_length.keys(), key=lambda c: -confusion[c]["accuracy"]):
    md_lines.append(f"### {cond}")
    md_lines.append("| Length Bin | Errors / Total | Error Rate |")
    md_lines.append("|-----------|----------------|------------|")
    for label, stats in error_by_length[cond].items():
        md_lines.append(f"| {label} | {stats['errors']}/{stats['total']} | {stats['error_rate']}% |")
    md_lines.append("")

# Section 7
md_lines.append("## 7. Key Pairwise Comparisons")
md_lines.append("")
if "exception_only_correct_no_trace_wrong" in results:
    r = results["exception_only_correct_no_trace_wrong"]
    md_lines.append(f"### exception_only correct & no_trace wrong: {r['count']} samples")
    md_lines.append(f"- Error types: {r['by_error_type']}")
    md_lines.append(f"- Sources: {r['by_source']}")
    md_lines.append(f"- Labels: {r['by_label']}")
    md_lines.append("")

if "exception_only_correct_full_trace_wrong" in results:
    r = results["exception_only_correct_full_trace_wrong"]
    md_lines.append(f"### exception_only correct & full_trace wrong: {r['count']} samples")
    md_lines.append(f"- Error types: {r['by_error_type']}")
    md_lines.append(f"- Sources: {r['by_source']}")
    md_lines.append(f"- Labels: {r['by_label']}")
    md_lines.append("")

# Section 8
md_lines.append("## 8. Test Set Statistics")
md_lines.append("")
ts = results["test_set_stats"]
md_lines.append(f"- Total: {ts['total']}")
md_lines.append(f"- Labels: {ts['label_distribution']}")
md_lines.append(f"- Sources: {ts['source_distribution']}")
md_lines.append(f"- Error types: {ts['error_type_distribution']}")

md_text = "\n".join(md_lines)
md_path = f"{ARTIFACT_DIR}/error_analysis_plan018.md"
with open(md_path, "w") as f:
    f.write(md_text)
print(f"Saved MD: {md_path}")

print("\nDone.")
