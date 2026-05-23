import json
import sys
from collections import Counter, defaultdict

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"

CONDITIONS = {
    "exception_only": f"{CKPT_DIR}/exception_only/final/predictions.jsonl",
    "vota": f"{CKPT_DIR}/vota/final/predictions.jsonl",
    "full_trace": f"{CKPT_DIR}/full_trace/final/predictions.jsonl",
    "no_trace": f"{CKPT_DIR}/no_trace/final/predictions.jsonl",
    "loop_only": f"{CKPT_DIR}/loop_only/final/predictions.jsonl",
}

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

# 1. Load raw_traces for metadata
print("=" * 70)
print("PHASE 1: Loading raw data and metadata")
print("=" * 70)
raw_traces = load_jsonl(f"{DATA_DIR}/raw_traces.jsonl")
print(f"Total raw traces: {len(raw_traces)}")

# Build metadata index: the test set has 1190 items
# Load exception_only_test to get problem_ids for the test set
test_data = load_jsonl(f"{DATA_DIR}/exception_only_test.jsonl")
print(f"Test set size: {len(test_data)}")

# Extract problem_ids from test data
test_pids = [d["problem_id"] for d in test_data]

# Build raw_traces lookup by (problem_id, index_within_problem)
# First understand the structure: each raw trace = one solution variant
pid_to_traces = defaultdict(list)
for t in raw_traces:
    pid_to_traces[t["problem_id"]].append(t)

# The test data maps 1:1 by line index to predictions
# We need to figure out which raw_trace corresponds to each test item
# Use the output/label to match
test_labels = [d["output"] for d in test_data]
print(f"Label distribution in test set: {Counter(test_labels)}")

# Build per-sample metadata
# From raw_traces, build index for matching
# The test items are ordered by problem_id, with multiple variants per problem
# Let's verify this by checking the raw_trace ordering vs test ordering

# Build metadata for each test sample
sample_meta = []
pid_cursor = defaultdict(int)
for i, td in enumerate(test_data):
    pid = td["problem_id"]
    label = td["output"]  # "pass" or "fail"
    
    # Find matching raw trace
    traces_for_pid = pid_to_traces.get(pid, [])
    cursor = pid_cursor[pid]
    
    meta = {
        "idx": i,
        "problem_id": pid,
        "label": label,
        "source": None,
        "error_type": None,
        "error_msg": None,
        "trace_len": None,
        "n_tests_passed": None,
        "n_tests_failed": None,
        "has_trace": None,
    }
    
    if cursor < len(traces_for_pid):
        rt = traces_for_pid[cursor]
        meta["source"] = rt.get("source", "unknown")
        tr = rt.get("test_results", {})
        err = tr.get("error", "")
        meta["error_msg"] = err if err else None
        if err:
            meta["error_type"] = err.split(":")[0].split("(")[0].strip()
        elif not rt["verdict"]:
            meta["error_type"] = "assertion_fail"
        else:
            meta["error_type"] = "no_error"
        meta["n_tests_passed"] = tr.get("passed", 0)
        meta["n_tests_failed"] = tr.get("failed", 0)
        trace_raw = rt.get("trace_raw", [])
        meta["trace_len"] = len(trace_raw) if isinstance(trace_raw, list) else 0
        meta["has_trace"] = bool(trace_raw)
    
    pid_cursor[pid] += 1
    sample_meta.append(meta)

# Check source distribution
src_counts = Counter(m["source"] for m in sample_meta)
print(f"Source distribution: {dict(src_counts)}")

# Check error type distribution
err_counts = Counter(m["error_type"] for m in sample_meta)
print(f"Error type distribution: {dict(err_counts)}")

# 2. Load predictions for all conditions
print("\n" + "=" * 70)
print("PHASE 2: Loading predictions and computing per-condition accuracy")
print("=" * 70)

preds_by_cond = {}
for cond, path in CONDITIONS.items():
    preds = load_jsonl(path)
    preds_by_cond[cond] = preds
    correct = sum(1 for p in preds if p["label"] == p["pred"])
    acc = correct / len(preds) * 100
    fp = sum(1 for p in preds if p["label"] == "fail" and p["pred"] == "pass")
    fn = sum(1 for p in preds if p["label"] == "pass" and p["pred"] == "fail")
    tp = sum(1 for p in preds if p["label"] == "pass" and p["pred"] == "pass")
    tn = sum(1 for p in preds if p["label"] == "fail" and p["pred"] == "fail")
    print(f"{cond:20s}: acc={acc:.2f}% | TP={tp} TN={tn} FP={fp} FN={fn}")

# 3. Per error type accuracy
print("\n" + "=" * 70)
print("PHASE 3: Accuracy by error type per condition")
print("=" * 70)

error_types_list = sorted(set(m["error_type"] for m in sample_meta if m["error_type"]))
for etype in error_types_list:
    indices = [m["idx"] for m in sample_meta if m["error_type"] == etype]
    n = len(indices)
    print(f"\n--- {etype} (n={n}) ---")
    for cond in CONDITIONS:
        preds = preds_by_cond[cond]
        correct = sum(1 for i in indices if preds[i]["label"] == preds[i]["pred"])
        acc = correct / n * 100 if n > 0 else 0
        fp = sum(1 for i in indices if preds[i]["label"] == "fail" and preds[i]["pred"] == "pass")
        fn = sum(1 for i in indices if preds[i]["label"] == "pass" and preds[i]["pred"] == "fail")
        print(f"  {cond:20s}: acc={acc:.2f}% (correct={correct}/{n}, FP={fp}, FN={fn})")

# 4. By source dataset
print("\n" + "=" * 70)
print("PHASE 4: Accuracy by source dataset per condition")
print("=" * 70)

sources_list = sorted(set(m["source"] for m in sample_meta if m["source"]))
for src in sources_list:
    indices = [m["idx"] for m in sample_meta if m["source"] == src]
    n = len(indices)
    print(f"\n--- {src} (n={n}) ---")
    for cond in CONDITIONS:
        preds = preds_by_cond[cond]
        correct = sum(1 for i in indices if preds[i]["label"] == preds[i]["pred"])
        acc = correct / n * 100 if n > 0 else 0
        print(f"  {cond:20s}: acc={acc:.2f}% ({correct}/{n})")

# 5. By trace length buckets (for samples with traces)
print("\n" + "=" * 70)
print("PHASE 5: Accuracy by trace length bucket")
print("=" * 70)

trace_lens = [m["trace_len"] for m in sample_meta if m["trace_len"] is not None]
if trace_lens:
    import statistics
    print(f"Trace length stats: min={min(trace_lens)}, max={max(trace_lens)}, "
          f"mean={statistics.mean(trace_lens):.1f}, median={statistics.median(trace_lens)}")
    
    # Define buckets
    buckets = [(0, 0, "no_trace(0)"), (1, 5, "short(1-5)"), (6, 20, "medium(6-20)"), 
               (21, 50, "long(21-50)"), (51, float('inf'), "very_long(51+)")]
    
    for lo, hi, name in buckets:
        indices = [m["idx"] for m in sample_meta if m["trace_len"] is not None and lo <= m["trace_len"] <= hi]
        n = len(indices)
        if n == 0:
            continue
        print(f"\n--- {name} (n={n}) ---")
        for cond in CONDITIONS:
            preds = preds_by_cond[cond]
            correct = sum(1 for i in indices if preds[i]["label"] == preds[i]["pred"])
            acc = correct / n * 100
            print(f"  {cond:20s}: acc={acc:.2f}% ({correct}/{n})")

# 6. Cross-condition overlap analysis
print("\n" + "=" * 70)
print("PHASE 6: Cross-condition overlap analysis")
print("=" * 70)

n_samples = len(test_data)
# For each sample, which conditions got it right
correct_by_sample = defaultdict(set)
wrong_by_sample = defaultdict(set)
for cond in CONDITIONS:
    preds = preds_by_cond[cond]
    for i in range(n_samples):
        if preds[i]["label"] == preds[i]["pred"]:
            correct_by_sample[i].add(cond)
        else:
            wrong_by_sample[i].add(cond)

# All correct
all_correct = sum(1 for i in range(n_samples) if len(correct_by_sample[i]) == len(CONDITIONS))
all_wrong = sum(1 for i in range(n_samples) if len(wrong_by_sample[i]) == len(CONDITIONS))
print(f"Correct by ALL conditions: {all_correct}/{n_samples} ({all_correct/n_samples*100:.1f}%)")
print(f"Wrong by ALL conditions: {all_wrong}/{n_samples} ({all_wrong/n_samples*100:.1f}%)")

# Only wrong by specific conditions
for cond in CONDITIONS:
    only_this_wrong = sum(1 for i in range(n_samples) if wrong_by_sample[i] == {cond})
    print(f"Wrong ONLY by {cond:20s}: {only_this_wrong}")

# Pairwise disagreement
print("\nPairwise disagreement counts:")
cond_list = list(CONDITIONS.keys())
for i in range(len(cond_list)):
    for j in range(i+1, len(cond_list)):
        c1, c2 = cond_list[i], cond_list[j]
        disagree = sum(1 for k in range(n_samples) 
                       if preds_by_cond[c1][k]["pred"] != preds_by_cond[c2][k]["pred"])
        print(f"  {c1:20s} vs {c2:20s}: {disagree} disagreements")

# 7. FP/FN breakdown by condition
print("\n" + "=" * 70)
print("PHASE 7: FP vs FN analysis")
print("=" * 70)

for cond in CONDITIONS:
    preds = preds_by_cond[cond]
    fp_indices = [i for i in range(n_samples) if preds[i]["label"] == "fail" and preds[i]["pred"] == "pass"]
    fn_indices = [i for i in range(n_samples) if preds[i]["label"] == "pass" and preds[i]["pred"] == "fail"]
    
    print(f"\n--- {cond} ---")
    print(f"  FP={len(fp_indices)}, FN={len(fn_indices)}")
    
    if fp_indices:
        fp_err_types = Counter(sample_meta[i]["error_type"] for i in fp_indices)
        print(f"  FP error types: {dict(fp_err_types)}")
        fp_sources = Counter(sample_meta[i]["source"] for i in fp_indices)
        print(f"  FP sources: {dict(fp_sources)}")
    
    if fn_indices:
        fn_sources = Counter(sample_meta[i]["source"] for i in fn_indices)
        print(f"  FN sources: {dict(fn_sources)}")

# 8. Tier analysis: exception_only vs loop_only failure mode comparison
print("\n" + "=" * 70)
print("PHASE 8: Tier 1 (exception_only) vs Tier 2 (loop_only) failure comparison")
print("=" * 70)

exc_preds = preds_by_cond["exception_only"]
loop_preds = preds_by_cond["loop_only"]

exc_wrong = set(i for i in range(n_samples) if exc_preds[i]["label"] != exc_preds[i]["pred"])
loop_wrong = set(i for i in range(n_samples) if loop_preds[i]["label"] != loop_preds[i]["pred"])

both_wrong = exc_wrong & loop_wrong
exc_only_wrong = exc_wrong - loop_wrong
loop_only_wrong = loop_wrong - exc_wrong

print(f"exception_only errors: {len(exc_wrong)}")
print(f"loop_only errors: {len(loop_wrong)}")
print(f"Both wrong: {len(both_wrong)}")
print(f"Only exception_only wrong: {len(exc_only_wrong)}")
print(f"Only loop_only wrong: {len(loop_only_wrong)}")

if exc_only_wrong:
    print(f"\nSamples only exception_only gets wrong:")
    for i in sorted(exc_only_wrong):
        m = sample_meta[i]
        print(f"  idx={i}, pid={m['problem_id']}, label={m['label']}, err={m['error_type']}, "
              f"pred_exc={exc_preds[i]['pred']}, pred_loop={loop_preds[i]['pred']}")

if loop_only_wrong:
    print(f"\nSamples only loop_only gets wrong (first 20):")
    for i in sorted(loop_only_wrong)[:20]:
        m = sample_meta[i]
        print(f"  idx={i}, pid={m['problem_id']}, label={m['label']}, err={m['error_type']}, "
              f"pred_exc={exc_preds[i]['pred']}, pred_loop={loop_preds[i]['pred']}")

# 9. Universally hard samples (wrong by 3+ conditions)
print("\n" + "=" * 70)
print("PHASE 9: Universally hard samples (wrong by 3+ conditions)")
print("=" * 70)

hard_samples = [(i, wrong_by_sample[i]) for i in range(n_samples) if len(wrong_by_sample[i]) >= 3]
hard_samples.sort(key=lambda x: -len(x[1]))
print(f"Samples wrong by 3+ conditions: {len(hard_samples)}")
for i, wrong_conds in hard_samples:
    m = sample_meta[i]
    print(f"  idx={i}, pid={m['problem_id']}, label={m['label']}, err={m['error_type']}, "
          f"src={m['source']}, trace_len={m['trace_len']}, "
          f"wrong_by={sorted(wrong_conds)}")

# 10. Label balance check
print("\n" + "=" * 70)
print("PHASE 10: Label balance and per-label accuracy")
print("=" * 70)

pass_indices = [i for i in range(n_samples) if test_labels[i] == "pass"]
fail_indices = [i for i in range(n_samples) if test_labels[i] == "fail"]
print(f"Pass samples: {len(pass_indices)}, Fail samples: {len(fail_indices)}")

for cond in CONDITIONS:
    preds = preds_by_cond[cond]
    pass_acc = sum(1 for i in pass_indices if preds[i]["pred"] == "pass") / len(pass_indices) * 100
    fail_acc = sum(1 for i in fail_indices if preds[i]["pred"] == "fail") / len(fail_indices) * 100
    print(f"  {cond:20s}: pass_acc={pass_acc:.2f}%, fail_acc={fail_acc:.2f}%")

print("\nDone.")
