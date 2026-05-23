import json
import os
from collections import Counter, defaultdict

CKPT_BASE = "./checkpoints"
DATA_BASE = "./data"

# Qwen-7B seed42 conditions (primary analysis)
QWEN7B_CONDITIONS = {
    "exception_only":  f"{CKPT_BASE}/exception_only/final/predictions.jsonl",
    "ae_only":         f"{CKPT_BASE}/ae_only/final/predictions.jsonl",
    "vota":            f"{CKPT_BASE}/vota/final/predictions.jsonl",
    "loop_only":       f"{CKPT_BASE}/loop_only/final/predictions.jsonl",
    "full_trace_tail": f"{CKPT_BASE}/full_trace_tail_seed42/final/predictions.jsonl",
    "label_only":      f"{CKPT_BASE}/full_trace_label_only/final/predictions.jsonl",
    "no_trace":        f"{CKPT_BASE}/no_trace/final/predictions.jsonl",
    "nomarker":        f"{CKPT_BASE}/exception_only_nomarker/final/predictions.jsonl",
}

# Cross-model exception_only
CROSS_MODEL = {
    "Qwen-7B":       f"{CKPT_BASE}/exception_only/final/predictions.jsonl",
    "Qwen-1.5B":     f"{CKPT_BASE}/qwen15b_exception_only_seed42/final/predictions.jsonl",
    "CodeLlama-13B": f"{CKPT_BASE}/codellama13b_exception_only_seed42/final/predictions.jsonl",
    "DeepSeek-7B":   f"{CKPT_BASE}/deepseek_exception_only/final/predictions.jsonl",
}

TIERS = {
    "Tier1_high":  ["exception_only", "ae_only", "vota"],
    "Tier2_mid":   ["loop_only", "full_trace_tail", "label_only"],
    "Tier3_low":   ["no_trace", "nomarker"],
}

def load_predictions(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def load_test_data(condition="exception_only"):
    path = f"{DATA_BASE}/{condition}_test.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f]

def get_error_indices(preds):
    errors = []
    fp, fn = [], []
    for i, p in enumerate(preds):
        if p["pred"] != p["label"]:
            errors.append(i)
            if p["label"] == "fail" and p["pred"] == "pass":
                fp.append(i)
            elif p["label"] == "pass" and p["pred"] == "fail":
                fn.append(i)
    return errors, fp, fn

# ===== Load test data for feature extraction =====
test_data = load_test_data("exception_only")
N = len(test_data)

# ===== Section 1: Error Distribution =====
print("=" * 80)
print("1. ERROR DISTRIBUTION BY CONDITION (Qwen-7B seed42)")
print("=" * 80)
print(f"{'Condition':<18} {'Total':>6} {'Errors':>7} {'FP':>5} {'FN':>5} {'FP%':>7} {'FN%':>7} {'Err%':>7} {'Acc%':>7}")
print("-" * 80)

all_errors = {}
all_fp = {}
all_fn = {}

n_pass = sum(1 for d in test_data if d["output"] == "pass")
n_fail = N - n_pass

for cond in ["exception_only", "ae_only", "vota", "loop_only", "full_trace_tail", "label_only", "no_trace", "nomarker"]:
    path = QWEN7B_CONDITIONS[cond]
    if not os.path.exists(path):
        print(f"{cond:<18} MISSING")
        continue
    preds = load_predictions(path)
    errors, fp, fn = get_error_indices(preds)
    all_errors[cond] = set(errors)
    all_fp[cond] = set(fp)
    all_fn[cond] = set(fn)
    fp_rate = len(fp) / n_fail * 100 if n_fail > 0 else 0
    fn_rate = len(fn) / n_pass * 100 if n_pass > 0 else 0
    err_rate = len(errors) / N * 100
    acc = (N - len(errors)) / N * 100
    print(f"{cond:<18} {N:>6} {len(errors):>7} {len(fp):>5} {len(fn):>5} {fp_rate:>6.2f}% {fn_rate:>6.2f}% {err_rate:>6.2f}% {acc:>6.2f}%")

# ===== Section 2: Error Sample Characteristics =====
print()
print("=" * 80)
print("2. ERROR SAMPLE CHARACTERISTICS")
print("=" * 80)

# Use exception_only test data for input length analysis
input_lens = [len(d["input"]) for d in test_data]
problem_ids = [d["problem_id"] for d in test_data]

# For conditions with many errors, compare error vs correct sample characteristics
for cond in ["no_trace", "nomarker", "loop_only", "full_trace_tail"]:
    if cond not in all_errors or len(all_errors[cond]) < 5:
        continue
    err_lens = [input_lens[i] for i in all_errors[cond]]
    cor_lens = [input_lens[i] for i in range(N) if i not in all_errors[cond]]
    err_labels = Counter(test_data[i]["output"] for i in all_errors[cond])
    print(f"\n--- {cond} (n_errors={len(all_errors[cond])}) ---")
    print(f"  Error samples - input len: mean={sum(err_lens)/len(err_lens):.0f}, "
          f"min={min(err_lens)}, max={max(err_lens)}, median={sorted(err_lens)[len(err_lens)//2]}")
    print(f"  Correct samples - input len: mean={sum(cor_lens)/len(cor_lens):.0f}, "
          f"min={min(cor_lens)}, max={max(cor_lens)}, median={sorted(cor_lens)[len(cor_lens)//2]}")
    print(f"  Error label dist: {dict(err_labels)}")
    err_pids = Counter(problem_ids[i] for i in all_errors[cond])
    top_pids = err_pids.most_common(5)
    print(f"  Top error problem_ids: {top_pids}")

# ===== Section 3: Cross-condition Error Overlap =====
print()
print("=" * 80)
print("3. CROSS-CONDITION ERROR OVERLAP")
print("=" * 80)

# Hard examples: errors in ALL conditions
conds_with_errors = [c for c in all_errors if len(all_errors[c]) > 0]
if conds_with_errors:
    universal_errors = set.intersection(*[all_errors[c] for c in conds_with_errors])
    print(f"\nUniversal errors (wrong in ALL {len(conds_with_errors)} conditions): {len(universal_errors)}")
    if universal_errors:
        univ_labels = Counter(test_data[i]["output"] for i in universal_errors)
        univ_pids = Counter(problem_ids[i] for i in universal_errors)
        print(f"  Labels: {dict(univ_labels)}")
        print(f"  Problem IDs: {dict(univ_pids)}")

    # Errors in all Tier3 but not Tier1
    tier1_conds = [c for c in TIERS["Tier1_high"] if c in all_errors]
    tier3_conds = [c for c in TIERS["Tier3_low"] if c in all_errors]

    if tier1_conds and tier3_conds:
        tier1_errors = set.union(*[all_errors[c] for c in tier1_conds])
        tier3_errors = set.union(*[all_errors[c] for c in tier3_conds])
        tier3_only = tier3_errors - tier1_errors
        print(f"\nTier3-only errors (wrong in Tier3 but correct in ALL Tier1): {len(tier3_only)}")
        if tier3_only:
            t3_labels = Counter(test_data[i]["output"] for i in tier3_only)
            print(f"  Labels: {dict(t3_labels)}")

    # Pairwise overlap matrix
    cond_list = ["exception_only", "ae_only", "vota", "loop_only", "full_trace_tail", "label_only", "no_trace", "nomarker"]
    cond_list = [c for c in cond_list if c in all_errors]

    print(f"\nPairwise error overlap (Jaccard similarity):")
    print(f"{'':>18}", end="")
    for c in cond_list:
        print(f" {c[:8]:>8}", end="")
    print()

    for c1 in cond_list:
        print(f"{c1:<18}", end="")
        for c2 in cond_list:
            if not all_errors[c1] and not all_errors[c2]:
                j = 1.0
            elif not all_errors[c1] or not all_errors[c2]:
                j = 0.0
            else:
                inter = len(all_errors[c1] & all_errors[c2])
                union = len(all_errors[c1] | all_errors[c2])
                j = inter / union if union > 0 else 0
            print(f" {j:>8.3f}", end="")
        print()

    # Raw overlap counts
    print(f"\nPairwise error overlap (raw counts |A∩B|):")
    print(f"{'':>18}", end="")
    for c in cond_list:
        print(f" {c[:8]:>8}", end="")
    print()
    for c1 in cond_list:
        print(f"{c1:<18}", end="")
        for c2 in cond_list:
            inter = len(all_errors[c1] & all_errors[c2])
            print(f" {inter:>8}", end="")
        print()

# ===== Section 4: Tier Analysis =====
print()
print("=" * 80)
print("4. TIER ANALYSIS")
print("=" * 80)

for tier_name, tier_conds in TIERS.items():
    available = [c for c in tier_conds if c in all_errors]
    if not available:
        continue
    tier_err_union = set.union(*[all_errors[c] for c in available])
    tier_err_inter = set.intersection(*[all_errors[c] for c in available]) if len(available) > 1 else all_errors[available[0]]

    tier_fp = set.union(*[all_fp.get(c, set()) for c in available])
    tier_fn = set.union(*[all_fn.get(c, set()) for c in available])

    print(f"\n{tier_name} ({', '.join(available)}):")
    print(f"  Union of errors: {len(tier_err_union)}")
    print(f"  Intersection of errors: {len(tier_err_inter)}")
    print(f"  Total FP (union): {len(tier_fp)}, Total FN (union): {len(tier_fn)}")
    if tier_err_union:
        fp_frac = len(tier_fp) / len(tier_err_union) * 100
        fn_frac = len(tier_fn) / len(tier_err_union) * 100
        print(f"  FP fraction: {fp_frac:.1f}%, FN fraction: {fn_frac:.1f}%")

# FP/FN pattern by tier
print(f"\nFP vs FN breakdown by tier:")
print(f"{'Tier':<20} {'Avg FP':>8} {'Avg FN':>8} {'FP/(FP+FN)':>12}")
print("-" * 50)
for tier_name, tier_conds in TIERS.items():
    available = [c for c in tier_conds if c in all_fp]
    if not available:
        continue
    avg_fp = sum(len(all_fp[c]) for c in available) / len(available)
    avg_fn = sum(len(all_fn[c]) for c in available) / len(available)
    ratio = avg_fp / (avg_fp + avg_fn) * 100 if (avg_fp + avg_fn) > 0 else 0
    print(f"{tier_name:<20} {avg_fp:>8.1f} {avg_fn:>8.1f} {ratio:>11.1f}%")

# ===== Section 5: Cross-model comparison =====
print()
print("=" * 80)
print("5. CROSS-MODEL COMPARISON (exception_only)")
print("=" * 80)

model_errors = {}
print(f"{'Model':<18} {'Errors':>7} {'FP':>5} {'FN':>5} {'Acc%':>7}")
print("-" * 50)
for model, path in CROSS_MODEL.items():
    if not os.path.exists(path):
        print(f"{model:<18} MISSING")
        continue
    preds = load_predictions(path)
    errors, fp, fn = get_error_indices(preds)
    model_errors[model] = set(errors)
    acc = (N - len(errors)) / N * 100
    print(f"{model:<18} {len(errors):>7} {len(fp):>5} {len(fn):>5} {acc:>6.2f}%")

# Cross-model error overlap
if len(model_errors) > 1:
    models_list = list(model_errors.keys())
    print(f"\nCross-model error overlap (raw counts):")
    print(f"{'':>18}", end="")
    for m in models_list:
        print(f" {m[:8]:>8}", end="")
    print()
    for m1 in models_list:
        print(f"{m1:<18}", end="")
        for m2 in models_list:
            inter = len(model_errors[m1] & model_errors[m2])
            print(f" {inter:>8}", end="")
        print()

    all_model_errors = set.intersection(*model_errors.values())
    print(f"\nUniversal cross-model errors: {len(all_model_errors)}")
    if all_model_errors:
        labels = Counter(test_data[i]["output"] for i in all_model_errors)
        pids = Counter(problem_ids[i] for i in all_model_errors)
        print(f"  Labels: {dict(labels)}")
        print(f"  Problem IDs: {dict(pids)}")

# ===== Section 6: Per-problem error rate =====
print()
print("=" * 80)
print("6. PER-PROBLEM ERROR ANALYSIS")
print("=" * 80)

# For each problem, how many conditions get it wrong?
problem_error_counts = defaultdict(lambda: defaultdict(int))
for cond in cond_list:
    for i in all_errors[cond]:
        pid = problem_ids[i]
        problem_error_counts[pid][cond] += 1

# Which problems are hardest (errors across most conditions)?
pid_difficulty = {}
for pid in set(problem_ids):
    indices = [i for i, p in enumerate(problem_ids) if p == pid]
    n_samples = len(indices)
    total_errors = sum(1 for cond in cond_list for i in indices if i in all_errors[cond])
    pid_difficulty[pid] = total_errors / (n_samples * len(cond_list)) * 100

sorted_pids = sorted(pid_difficulty.items(), key=lambda x: -x[1])
print(f"\nTop 15 hardest problems (avg error rate across conditions):")
print(f"{'Problem ID':<25} {'Err%':>7} {'#Samples':>9}")
print("-" * 45)
for pid, err_pct in sorted_pids[:15]:
    n_samp = sum(1 for p in problem_ids if p == pid)
    print(f"{pid:<25} {err_pct:>6.1f}% {n_samp:>9}")

print(f"\nTop 15 easiest problems (lowest error rate):")
easiest = [x for x in sorted_pids if x[1] == 0]
print(f"  {len(easiest)} problems with 0% error rate across all conditions")

print("\n\nDone.")
