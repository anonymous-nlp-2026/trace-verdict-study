import json
import os
import re
import sys
from collections import defaultdict, Counter
from itertools import combinations

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"
OUT_CKPT_DIR = "./outputs"

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def split_by_problem_id(data, train_ratio=0.8, seed=42):
    import random
    rng = random.Random(seed)
    by_problem = defaultdict(list)
    for item in data:
        by_problem[item["problem_id"]].append(item)
    problem_ids = sorted(by_problem.keys())
    rng.shuffle(problem_ids)
    n_train = int(len(problem_ids) * train_ratio)
    train_ids = set(problem_ids[:n_train])
    train_data, test_data = [], []
    for pid in problem_ids:
        target = train_data if pid in train_ids else test_data
        target.extend(by_problem[pid])
    return train_data, test_data

def extract_error_type(error_str):
    if not error_str:
        return "no_error"
    m = re.match(r"(\w+Error|Exception)\b", error_str)
    if m:
        return m.group(1)
    return "other_error"

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)

# ── Step 1: Load raw traces and build test metadata ──
print("Loading raw traces...")
raw_data = load_jsonl(os.path.join(DATA_DIR, "raw_traces.jsonl"))
print(f"  Total raw samples: {len(raw_data)}")
_, test_raw = split_by_problem_id(raw_data)
print(f"  Test split: {len(test_raw)} samples")

test_meta = []
for i, item in enumerate(test_raw):
    error_str = ""
    tr = item.get("test_results", {})
    if isinstance(tr, dict):
        error_str = tr.get("error", "") or ""
    error_type = extract_error_type(error_str)
    has_trace = bool(item.get("trace_raw"))
    test_meta.append({
        "id": i,
        "problem_id": item["problem_id"],
        "verdict": item["verdict"],
        "error_type": error_type if not item["verdict"] else "none(pass)",
        "error_str": error_str,
        "has_trace": has_trace,
        "source": item.get("source", "unknown"),
        "input_len": len(item.get("solution", "")),
    })

# ── Step 2: Discover all predictions files ──
print("\nDiscovering predictions files...")
experiments = {}

for base_dir in [CKPT_DIR, OUT_CKPT_DIR]:
    if not os.path.exists(base_dir):
        continue
    for exp_name in os.listdir(base_dir):
        pred_path = os.path.join(base_dir, exp_name, "final", "predictions.jsonl")
        if os.path.isfile(pred_path):
            preds = load_jsonl(pred_path)
            if len(preds) == len(test_meta):
                experiments[exp_name] = preds

print(f"  Found {len(experiments)} experiments with {len(test_meta)}-sample predictions")
for name in sorted(experiments.keys()):
    preds = experiments[name]
    correct = sum(1 for p in preds if p["pred"] == p["label"])
    print(f"    {name}: acc={correct/len(preds):.4f}")

# Filter out broken runs (accuracy < 50% = worse than random for binary classification)
to_remove = []
for name, preds in experiments.items():
    acc = sum(1 for p in preds if p["pred"] == p["label"]) / len(preds)
    if acc < 0.5:
        to_remove.append(name)
for name in to_remove:
    del experiments[name]
    print(f"  EXCLUDED (broken run, acc<50%): {name}")
print(f"  After filtering: {len(experiments)} valid experiments")

# ── Step 3: Classify experiments by condition and model ──
def parse_experiment_name(name):
    name_lower = name.lower()
    model = "qwen7b"
    if "qwen15b" in name_lower or "qwen_15b" in name_lower:
        model = "qwen15b"
    elif "deepseek" in name_lower:
        model = "deepseek"
    elif "codellama" in name_lower or "codellama13b" in name_lower:
        model = "codellama13b"
    
    condition = "unknown"
    for cond in ["exception_only_nomarker", "exception_only", "ae_only", "loop_only",
                 "vota_no_actexp", "vota_no_ae", "vota", "no_trace",
                 "full_trace_label_only", "full_trace_tail512", "full_trace_tail384",
                 "full_trace_tail", "full_trace_head", "full_trace_1024tok",
                 "full_trace_512tok", "full_trace_256tok", "full_trace",
                 "label_only", "nomarker"]:
        cond_pattern = cond.replace("_", "[_-]?")
        name_clean = name_lower.replace(model, "").strip("_")
        if re.search(cond_pattern, name_clean):
            condition = cond
            break
    
    seed_m = re.search(r"seed(\d+)", name_lower)
    seed = seed_m.group(1) if seed_m else "42"
    
    return model, condition, seed

exp_info = {}
for name in experiments:
    model, condition, seed = parse_experiment_name(name)
    exp_info[name] = {"model": model, "condition": condition, "seed": seed}

# ── Step 4: Analysis ──
report = []
def R(s=""):
    report.append(s)
    print(s)

R("# Cross-Condition Error Analysis Report")
R(f"Generated: {len(experiments)} experiments, {len(test_meta)} test samples each")
R()

# ─── 4.0 Data coverage ───
R("## 1. Data Coverage")
R()
R("### Experiments by model × condition")
R()
model_cond = defaultdict(lambda: defaultdict(list))
for name, info in exp_info.items():
    model_cond[info["model"]][info["condition"]].append(name)

all_conditions = sorted(set(info["condition"] for info in exp_info.values()))
all_models = sorted(set(info["model"] for info in exp_info.values()))
header = "| Condition | " + " | ".join(all_models) + " |"
sep = "|---|" + "|".join(["---"]*len(all_models)) + "|"
R(header)
R(sep)
for cond in all_conditions:
    row = f"| {cond} |"
    for model in all_models:
        exps = model_cond[model].get(cond, [])
        row += f" {len(exps)} |"
    R(row)
R()

# ─── 4.1 Error rate by exception type ───
R("## 2. Error Rate by Exception Type")
R()

error_type_counts = Counter(m["error_type"] for m in test_meta)
R("### Test set exception type distribution")
R("| Exception Type | Count | % |")
R("|---|---|---|")
for et, cnt in error_type_counts.most_common():
    R(f"| {et} | {cnt} | {100*cnt/len(test_meta):.1f}% |")
R()

# For each key condition, compute error rate by exception type
key_conditions = ["exception_only", "ae_only", "loop_only", "vota", "no_trace", "full_trace"]
for cond in key_conditions:
    cond_exps = [(name, preds) for name, preds in experiments.items() 
                 if exp_info[name]["condition"] == cond]
    if not cond_exps:
        continue
    
    R(f"### Condition: {cond} ({len(cond_exps)} runs)")
    R("| Exception Type | N | Avg Error Rate | Per-Run Error Rates |")
    R("|---|---|---|---|")
    
    for et, _ in error_type_counts.most_common():
        indices = [m["id"] for m in test_meta if m["error_type"] == et]
        if not indices:
            continue
        run_error_rates = []
        for name, preds in cond_exps:
            errors = sum(1 for idx in indices if preds[idx]["pred"] != preds[idx]["label"])
            run_error_rates.append(errors / len(indices))
        avg_er = sum(run_error_rates) / len(run_error_rates)
        rates_str = ", ".join(f"{r:.3f}" for r in run_error_rates)
        R(f"| {et} | {len(indices)} | {avg_er:.4f} | {rates_str} |")
    R()

# ─── 4.2 Tier 1 vs Tier 2 failure mode comparison ───
R("## 3. Tier 1 vs Tier 2 Failure Mode Comparison")
R()
R("Tier 1 (high accuracy): exception_only, ae_only")
R("Tier 2 (lower accuracy): loop_only, no_trace")
R()

def get_error_ids(exp_name):
    preds = experiments[exp_name]
    return set(i for i, p in enumerate(preds) if p["pred"] != p["label"])

tier1_conds = ["exception_only", "ae_only"]
tier2_conds = ["loop_only", "no_trace"]

for model in all_models:
    tier1_errors = set()
    tier1_count = 0
    tier2_errors = set()
    tier2_count = 0
    
    for cond in tier1_conds:
        for name in model_cond[model].get(cond, []):
            tier1_errors |= get_error_ids(name)
            tier1_count += 1
    for cond in tier2_conds:
        for name in model_cond[model].get(cond, []):
            tier2_errors |= get_error_ids(name)
            tier2_count += 1
    
    if tier1_count == 0 or tier2_count == 0:
        continue
    
    shared = tier1_errors & tier2_errors
    tier1_only = tier1_errors - tier2_errors
    tier2_only = tier2_errors - tier1_errors
    
    R(f"### Model: {model}")
    R(f"- Tier 1 union errors: {len(tier1_errors)}")
    R(f"- Tier 2 union errors: {len(tier2_errors)}")
    R(f"- Shared errors: {len(shared)}")
    R(f"- Tier 1 only: {len(tier1_only)}")
    R(f"- Tier 2 only: {len(tier2_only)}")
    
    if tier2_only:
        tier2_only_types = Counter(test_meta[i]["error_type"] for i in tier2_only)
        R(f"- Tier 2 unique error types: {dict(tier2_only_types.most_common())}")
    
    if shared:
        shared_types = Counter(test_meta[i]["error_type"] for i in shared)
        R(f"- Shared error types: {dict(shared_types.most_common())}")
    R()

# ─── 4.3 Irreducible errors in exception_only ───
R("## 4. Irreducible Errors in exception_only")
R()

eo_exps = [(name, preds) for name, preds in experiments.items()
           if exp_info[name]["condition"] == "exception_only"]

if eo_exps:
    # Find samples that are errors in ANY exception_only run
    all_eo_errors = defaultdict(list)
    for name, preds in eo_exps:
        for i, p in enumerate(preds):
            if p["pred"] != p["label"]:
                all_eo_errors[i].append(name)
    
    R(f"Total unique error samples across {len(eo_exps)} exception_only runs: {len(all_eo_errors)}")
    R()
    
    # Samples that error in ALL runs (truly irreducible)
    universal_errors = [i for i, runs in all_eo_errors.items() if len(runs) == len(eo_exps)]
    R(f"Errors in ALL {len(eo_exps)} runs (universally hard): {len(universal_errors)}")
    
    R()
    R("### Per-sample error details")
    R("| Sample ID | problem_id | label | error_type | has_trace | #runs_wrong | wrong_in |")
    R("|---|---|---|---|---|---|---|")
    
    for idx in sorted(all_eo_errors.keys()):
        m = test_meta[idx]
        runs = all_eo_errors[idx]
        run_names = ", ".join(sorted(runs))
        R(f"| {idx} | {m['problem_id']} | {'pass' if m['verdict'] else 'fail'} | {m['error_type']} | {m['has_trace']} | {len(runs)}/{len(eo_exps)} | {run_names} |")
    
    R()
    
    # Cross-model: which models share errors?
    eo_by_model = defaultdict(list)
    for name, preds in eo_exps:
        model = exp_info[name]["model"]
        eo_by_model[model].append((name, preds))
    
    if len(eo_by_model) > 1:
        R("### Cross-model error overlap in exception_only")
        model_error_sets = {}
        for model, runs in eo_by_model.items():
            err_set = set()
            for name, preds in runs:
                for i, p in enumerate(preds):
                    if p["pred"] != p["label"]:
                        err_set.add(i)
            model_error_sets[model] = err_set
        
        models_list = sorted(model_error_sets.keys())
        R("| Model A | Model B | Errors A | Errors B | Intersection | Jaccard |")
        R("|---|---|---|---|---|---|")
        for ma, mb in combinations(models_list, 2):
            sa, sb = model_error_sets[ma], model_error_sets[mb]
            inter = sa & sb
            j = jaccard(sa, sb)
            R(f"| {ma} | {mb} | {len(sa)} | {len(sb)} | {len(inter)} | {j:.3f} |")
        R()

# ─── 4.4 Cross-model Jaccard overlap ───
R("## 5. Cross-Model Error Overlap (Jaccard Similarity)")
R()

for cond in key_conditions:
    cond_exps_by_model = defaultdict(list)
    for name, preds in experiments.items():
        if exp_info[name]["condition"] == cond:
            model = exp_info[name]["model"]
            cond_exps_by_model[model].append((name, preds))
    
    if len(cond_exps_by_model) < 2:
        continue
    
    R(f"### Condition: {cond}")
    
    # Use seed42 run per model if available, else first run
    model_errors = {}
    for model, runs in cond_exps_by_model.items():
        seed42_runs = [(n, p) for n, p in runs if exp_info[n]["seed"] == "42"]
        chosen = seed42_runs[0] if seed42_runs else runs[0]
        name, preds = chosen
        err_ids = set(i for i, p in enumerate(preds) if p["pred"] != p["label"])
        model_errors[model] = (name, err_ids)
    
    models_list = sorted(model_errors.keys())
    R(f"Models: {', '.join(f'{m} ({model_errors[m][0]}, {len(model_errors[m][1])} errors)' for m in models_list)}")
    R()
    
    # Jaccard matrix
    R("| | " + " | ".join(models_list) + " |")
    R("|---|" + "|".join(["---"]*len(models_list)) + "|")
    for ma in models_list:
        row = f"| {ma} |"
        for mb in models_list:
            if ma == mb:
                row += " 1.000 |"
            else:
                j = jaccard(model_errors[ma][1], model_errors[mb][1])
                row += f" {j:.3f} |"
        R(row)
    R()
    
    if len(models_list) >= 2:
        all_err = set()
        for m in models_list:
            all_err |= model_errors[m][1]
        universal = set.intersection(*(model_errors[m][1] for m in models_list))
        any_model = all_err
        R(f"- Total unique errors (any model): {len(any_model)}")
        R(f"- Universal errors (all models): {len(universal)}")
        if universal:
            uni_types = Counter(test_meta[i]["error_type"] for i in universal)
            R(f"- Universal error types: {dict(uni_types.most_common())}")
        R()

# ─── 4.5 Per-condition confusion matrix summary ───
R("## 6. Per-Condition Confusion Summary (seed42, Qwen-7B)")
R()
R("| Condition | TP | TN | FP | FN | Acc | FPR | FNR |")
R("|---|---|---|---|---|---|---|---|")

for cond in key_conditions:
    qwen7b_seed42 = [name for name in experiments 
                     if exp_info[name]["model"] == "qwen7b" 
                     and exp_info[name]["condition"] == cond
                     and exp_info[name]["seed"] == "42"]
    if not qwen7b_seed42:
        continue
    name = qwen7b_seed42[0]
    preds = experiments[name]
    tp = sum(1 for p in preds if p["label"] == "pass" and p["pred"] == "pass")
    tn = sum(1 for p in preds if p["label"] == "fail" and p["pred"] == "fail")
    fp = sum(1 for p in preds if p["label"] == "fail" and p["pred"] == "pass")
    fn = sum(1 for p in preds if p["label"] == "pass" and p["pred"] == "fail")
    acc = (tp + tn) / len(preds)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    R(f"| {cond} | {tp} | {tn} | {fp} | {fn} | {acc:.4f} | {fpr:.4f} | {fnr:.4f} |")
R()

# ─── 4.6 Rebuttal summary ───
R("## 7. Rebuttal Key Points")
R()

# Compute some key stats for rebuttal
eo_seed42_qwen = [n for n in experiments if exp_info[n]["condition"] == "exception_only" 
                  and exp_info[n]["model"] == "qwen7b" and exp_info[n]["seed"] == "42"]
lo_seed42_qwen = [n for n in experiments if exp_info[n]["condition"] == "loop_only" 
                  and exp_info[n]["model"] == "qwen7b" and exp_info[n]["seed"] == "42"]

if eo_seed42_qwen:
    eo_preds = experiments[eo_seed42_qwen[0]]
    eo_errors = sum(1 for p in eo_preds if p["pred"] != p["label"])
    R(f"1. **exception_only near-ceiling**: Qwen-7B seed42 has only {eo_errors}/{len(eo_preds)} errors ({100*eo_errors/len(eo_preds):.2f}%)")

if lo_seed42_qwen:
    lo_preds = experiments[lo_seed42_qwen[0]]
    lo_errors = sum(1 for p in lo_preds if p["pred"] != p["label"])
    R(f"2. **loop_only gap**: Qwen-7B seed42 has {lo_errors}/{len(lo_preds)} errors ({100*lo_errors/len(lo_preds):.1f}%)")

# Cross-model consistency
eo_all_models = {}
for name in experiments:
    if exp_info[name]["condition"] == "exception_only" and exp_info[name]["seed"] == "42":
        eo_all_models[exp_info[name]["model"]] = set(
            i for i, p in enumerate(experiments[name]) if p["pred"] != p["label"]
        )
if len(eo_all_models) > 1:
    union_eo = set.union(*eo_all_models.values())
    inter_eo = set.intersection(*eo_all_models.values())
    R(f"3. **Cross-model exception_only**: {len(eo_all_models)} models, union errors={len(union_eo)}, "
      f"intersection={len(inter_eo)} → {'high' if len(inter_eo)/max(len(union_eo),1) > 0.5 else 'low'} overlap "
      f"(Jaccard={jaccard(list(eo_all_models.values())[0], list(eo_all_models.values())[1]) if len(eo_all_models)==2 else 'N/A'})")

lo_all_models = {}
for name in experiments:
    if exp_info[name]["condition"] == "loop_only" and exp_info[name]["seed"] == "42":
        lo_all_models[exp_info[name]["model"]] = set(
            i for i, p in enumerate(experiments[name]) if p["pred"] != p["label"]
        )
if len(lo_all_models) > 1:
    union_lo = set.union(*lo_all_models.values())
    inter_lo = set.intersection(*lo_all_models.values())
    R(f"4. **Cross-model loop_only**: {len(lo_all_models)} models, union errors={len(union_lo)}, "
      f"intersection={len(inter_lo)} → hard-case overlap ratio={len(inter_lo)/max(len(union_lo),1):.2f}")

# Error type characterization
if lo_seed42_qwen:
    lo_err_ids = set(i for i, p in enumerate(experiments[lo_seed42_qwen[0]]) if p["pred"] != p["label"])
    lo_err_types = Counter(test_meta[i]["error_type"] for i in lo_err_ids)
    R(f"5. **loop_only error composition** (Qwen-7B seed42): {dict(lo_err_types.most_common(5))}")

R()
R("---")
R("*Report generated by error_analysis.py*")

# Save report
output_path = "./artifacts/error_analysis_plan018.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write("\n".join(report))
print(f"\nReport saved to {output_path}")
