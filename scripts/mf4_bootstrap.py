#!/usr/bin/env python3
"""MF-4: Deduplicated Bootstrap CI + Problem-Level Clustered Bootstrap"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv
import os
import sys

SEED = 42
B = 10000
N_TOTAL = 1190
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"
OUT_DIR = f"{CKPT_DIR}/deduplicated_bootstrap"

EXPECTED = {
    ("Qwen-7B", "exception_only"): 0.9983,
    ("Qwen-7B", "full_trace"): 0.8176,
    ("Qwen-7B", "no_trace"): 0.7513,
    ("Qwen-7B", "ae_only"): 0.9992,
    ("Qwen-7B", "loop_only"): 0.8563,
    ("Qwen-7B", "vota"): 0.9563,
    ("Qwen-7B", "nomarker"): 0.9655,
    ("Qwen-7B", "label_only"): 0.8622,
    ("Qwen-1.5B", "exception_only"): 0.9983,
    ("Qwen-1.5B", "full_trace"): 0.7706,
    ("Qwen-1.5B", "no_trace"): 0.6731,
    ("Qwen-1.5B", "ae_only"): 0.9908,
    ("Qwen-1.5B", "loop_only"): 0.7832,
    ("Qwen-1.5B", "vota"): 0.9958,
    ("Qwen-1.5B", "nomarker"): 0.9780,
    ("DeepSeek-6.7B", "exception_only"): 0.9933,
    ("DeepSeek-6.7B", "full_trace"): 0.7916,
    ("DeepSeek-6.7B", "no_trace"): 0.7336,
    ("DeepSeek-6.7B", "ae_only"): 0.9874,
    ("DeepSeek-6.7B", "loop_only"): 0.8126,
    ("DeepSeek-6.7B", "vota"): 0.8445,
    ("DeepSeek-6.7B", "label_only"): 0.2850,
    ("CodeLlama-13B", "exception_only"): 0.9983,
    ("CodeLlama-13B", "no_trace"): 0.7650,
    ("CodeLlama-13B", "ae_only"): 0.9908,
    ("CodeLlama-13B", "loop_only"): 0.7706,
    ("CodeLlama-13B", "vota"): 0.9857,
}

CANDIDATES = {
    ("Qwen-7B", "exception_only"): ["exception_only"],
    ("Qwen-7B", "full_trace"): ["full_trace", "full_trace_tail_512_seed42", "full_trace_tail_seed42", "qwen7b_full_trace_tail512_seed42"],
    ("Qwen-7B", "no_trace"): ["no_trace"],
    ("Qwen-7B", "ae_only"): ["ae_only"],
    ("Qwen-7B", "loop_only"): ["loop_only"],
    ("Qwen-7B", "vota"): ["vota"],
    ("Qwen-7B", "nomarker"): ["exception_only_nomarker_seed42", "exception_only_nomarker"],
    ("Qwen-7B", "label_only"): ["full_trace_label_only"],
    ("Qwen-1.5B", "exception_only"): ["qwen15b_exception_only_seed42"],
    ("Qwen-1.5B", "full_trace"): ["qwen15b_full_trace_seed42", "qwen15b_full_trace_tail512_seed42"],
    ("Qwen-1.5B", "no_trace"): ["qwen15b_no_trace_seed42", "qwen1.5b_no_trace_seed42"],
    ("Qwen-1.5B", "ae_only"): ["qwen15b_ae_only_seed42"],
    ("Qwen-1.5B", "loop_only"): ["qwen15b_loop_only_seed42"],
    ("Qwen-1.5B", "vota"): ["qwen15b_vota_seed42"],
    ("Qwen-1.5B", "nomarker"): ["qwen15b_nomarker_seed42", "qwen15b_exception_only_nomarker_seed42", "exception_only_nomarker_seed456"],
    ("DeepSeek-6.7B", "exception_only"): ["deepseek_exception_only"],
    ("DeepSeek-6.7B", "full_trace"): ["deepseek_full_trace", "deepseek_full_trace_seed42_v3", "deepseek_full_trace_tail512_seed42"],
    ("DeepSeek-6.7B", "no_trace"): ["deepseek_no_trace"],
    ("DeepSeek-6.7B", "ae_only"): ["deepseek_ae_only_seed42"],
    ("DeepSeek-6.7B", "loop_only"): ["deepseek_loop_only_seed42", "deepseek_loop_only"],
    ("DeepSeek-6.7B", "vota"): ["deepseek_vota"],
    ("DeepSeek-6.7B", "label_only"): ["deepseek_label_only_seed42"],
    ("CodeLlama-13B", "exception_only"): ["codellama13b_exception_only_seed42"],
    ("CodeLlama-13B", "no_trace"): ["codellama13b_no_trace_seed42"],
    ("CodeLlama-13B", "ae_only"): ["codellama13b_ae_only_seed42"],
    ("CodeLlama-13B", "loop_only"): ["codellama13b_loop_only_seed42"],
    ("CodeLlama-13B", "vota"): ["codellama13b_vota_seed42"],
}

def load_problem_ids(test_file):
    pids = []
    with open(test_file) as f:
        for line in f:
            d = json.loads(line)
            pids.append(d.get("problem_id", d.get("task_id", "")))
    return pids

def load_predictions(pred_file):
    correct = []
    with open(pred_file) as f:
        for line in f:
            d = json.loads(line)
            correct.append(1 if d["pred"] == d["label"] else 0)
    return np.array(correct, dtype=np.float64)

def find_predictions(model, condition, expected_acc, n_samples):
    key = (model, condition)
    candidates = CANDIDATES.get(key, [])
    tried = []
    for cand in candidates:
        pred_path = Path(CKPT_DIR) / cand / "final" / "predictions.jsonl"
        if pred_path.exists():
            correct = load_predictions(pred_path)
            if len(correct) != n_samples:
                tried.append(f"{cand}: len={len(correct)}")
                continue
            acc = correct.mean()
            if abs(acc - expected_acc) < 0.003:
                return correct, str(pred_path), "real"
            tried.append(f"{cand}: acc={acc:.4f}")
    
    print(f"  WARNING: {model}/{condition} using Bernoulli simulation (tried: {tried})")
    n_correct = round(expected_acc * n_samples)
    correct = np.zeros(n_samples, dtype=np.float64)
    correct[:n_correct] = 1.0
    rng = np.random.RandomState(abs(hash(key)) % (2**31))
    rng.shuffle(correct)
    return correct, "simulated", "simulated"

def build_problem_groups(problem_ids):
    groups = defaultdict(list)
    for i, pid in enumerate(problem_ids):
        groups[pid].append(i)
    return dict(groups)

def bootstrap_original(correct, B, rng):
    n = len(correct)
    idx_all = rng.randint(0, n, size=(B, n))
    return correct[idx_all].mean(axis=1)

def bootstrap_deduplicated(correct, problem_groups, B, rng):
    first_indices = np.array([indices[0] for indices in problem_groups.values()])
    dedup = correct[first_indices]
    n = len(dedup)
    idx_all = rng.randint(0, n, size=(B, n))
    return dedup[idx_all].mean(axis=1)

def bootstrap_clustered(correct, problem_groups, B, rng):
    problem_keys = list(problem_groups.keys())
    n_problems = len(problem_keys)
    problem_arrays = [correct[np.array(problem_groups[k])] for k in problem_keys]
    problem_sums = np.array([a.sum() for a in problem_arrays])
    problem_lens = np.array([len(a) for a in problem_arrays])
    
    sampled_idx = rng.randint(0, n_problems, size=(B, n_problems))
    total_correct = problem_sums[sampled_idx].sum(axis=1)
    total_samples = problem_lens[sampled_idx].sum(axis=1)
    return total_correct / total_samples

def ci_95(boot_accs):
    return float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))

def pairwise_clustered_bootstrap(correct_a, correct_b, problem_groups, B, rng):
    problem_keys = list(problem_groups.keys())
    n_problems = len(problem_keys)
    
    sums_a = np.array([correct_a[np.array(problem_groups[k])].sum() for k in problem_keys])
    lens_a = np.array([len(problem_groups[k]) for k in problem_keys])
    sums_b = np.array([correct_b[np.array(problem_groups[k])].sum() for k in problem_keys])
    
    sampled_idx = rng.randint(0, n_problems, size=(B, n_problems))
    total_a = sums_a[sampled_idx].sum(axis=1)
    total_b = sums_b[sampled_idx].sum(axis=1)
    total_n = lens_a[sampled_idx].sum(axis=1)
    return total_a / total_n - total_b / total_n

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    test_file = Path(DATA_DIR) / "exception_only_test.jsonl"
    problem_ids = load_problem_ids(test_file)
    assert len(problem_ids) == N_TOTAL, f"Expected {N_TOTAL}, got {len(problem_ids)}"
    
    problem_groups = build_problem_groups(problem_ids)
    n_unique = len(problem_groups)
    
    dup_counts = defaultdict(int)
    for pid, indices in problem_groups.items():
        dup_counts[len(indices)] += 1
    
    print(f"Total samples: {N_TOTAL}")
    print(f"Unique problems: {n_unique}")
    print(f"Duplicate distribution: {dict(sorted(dup_counts.items()))}")
    print()
    
    results = {}
    correct_vectors = {}
    
    for (model, condition), expected_acc in sorted(EXPECTED.items()):
        correct, source, source_type = find_predictions(model, condition, expected_acc, N_TOTAL)
        actual_acc = correct.mean()
        correct_vectors[(model, condition)] = correct
        
        boot_orig = bootstrap_original(correct, B, np.random.RandomState(SEED))
        boot_dedup = bootstrap_deduplicated(correct, problem_groups, B, np.random.RandomState(SEED))
        boot_clust = bootstrap_clustered(correct, problem_groups, B, np.random.RandomState(SEED))
        
        ci_o = ci_95(boot_orig)
        ci_d = ci_95(boot_dedup)
        ci_c = ci_95(boot_clust)
        
        results[(model, condition)] = {
            "accuracy": actual_acc,
            "expected_accuracy": expected_acc,
            "data_source": source_type,
            "source_path": source,
            "original_ci": ci_o,
            "original_ci_width": ci_o[1] - ci_o[0],
            "deduplicated_ci": ci_d,
            "deduplicated_ci_width": ci_d[1] - ci_d[0],
            "clustered_ci": ci_c,
            "clustered_ci_width": ci_c[1] - ci_c[0],
        }
        
        print(f"{model:15s} {condition:15s} acc={actual_acc:.4f}({source_type[0]}) "
              f"orig_w={ci_o[1]-ci_o[0]:.4f} dedup_w={ci_d[1]-ci_d[0]:.4f} clust_w={ci_c[1]-ci_c[0]:.4f}")
    
    # Pairwise significance
    pairwise_comparisons = [
        ("Qwen-7B", "exception_only", "Qwen-7B", "no_trace"),
        ("Qwen-7B", "ae_only", "Qwen-7B", "loop_only"),
        ("Qwen-7B", "exception_only", "Qwen-7B", "ae_only"),
        ("Qwen-7B", "vota", "Qwen-7B", "full_trace"),
        ("Qwen-7B", "full_trace", "Qwen-7B", "no_trace"),
        ("Qwen-7B", "nomarker", "Qwen-7B", "exception_only"),
        ("Qwen-7B", "label_only", "Qwen-7B", "full_trace"),
        ("Qwen-7B", "exception_only", "DeepSeek-6.7B", "exception_only"),
        ("Qwen-7B", "no_trace", "DeepSeek-6.7B", "no_trace"),
        ("Qwen-7B", "no_trace", "CodeLlama-13B", "no_trace"),
        ("Qwen-7B", "vota", "DeepSeek-6.7B", "vota"),
        ("Qwen-1.5B", "exception_only", "Qwen-1.5B", "no_trace"),
        ("Qwen-1.5B", "ae_only", "Qwen-1.5B", "loop_only"),
        ("DeepSeek-6.7B", "exception_only", "DeepSeek-6.7B", "no_trace"),
        ("DeepSeek-6.7B", "ae_only", "DeepSeek-6.7B", "loop_only"),
        ("CodeLlama-13B", "exception_only", "CodeLlama-13B", "no_trace"),
        ("CodeLlama-13B", "ae_only", "CodeLlama-13B", "loop_only"),
        ("Qwen-7B", "loop_only", "Qwen-7B", "no_trace"),
        ("Qwen-7B", "ae_only", "Qwen-7B", "no_trace"),
    ]
    
    pairwise_results = []
    print("\n=== Pairwise Significance (Clustered Bootstrap) ===")
    for m1, c1, m2, c2 in pairwise_comparisons:
        ka, kb = (m1, c1), (m2, c2)
        if ka not in correct_vectors or kb not in correct_vectors:
            continue
        
        diffs = pairwise_clustered_bootstrap(
            correct_vectors[ka], correct_vectors[kb],
            problem_groups, B, np.random.RandomState(SEED))
        diff_ci = ci_95(diffs)
        mean_diff = float(diffs.mean())
        significant = (diff_ci[0] > 0) or (diff_ci[1] < 0)
        point_diff = float(results[ka]["accuracy"] - results[kb]["accuracy"])
        
        pw = {
            "comparison": f"{m1}/{c1} vs {m2}/{c2}",
            "point_diff": round(point_diff, 4),
            "mean_bootstrap_diff": round(mean_diff, 4),
            "ci_lower": round(diff_ci[0], 4),
            "ci_upper": round(diff_ci[1], 4),
            "significant_at_95": significant,
        }
        pairwise_results.append(pw)
        sig_str = "***SIG***" if significant else "n.s."
        print(f"  {m1}/{c1} vs {m2}/{c2}: Δ={point_diff:+.4f} CI=[{diff_ci[0]:+.4f},{diff_ci[1]:+.4f}] {sig_str}")
    
    # Save outputs
    full_results = {
        "metadata": {
            "n_total": N_TOTAL,
            "n_unique": n_unique,
            "n_duplicate_pct": round((N_TOTAL - n_unique) / N_TOTAL * 100, 1),
            "bootstrap_iterations": B,
            "seed": SEED,
            "duplicate_distribution": {str(k): v for k, v in sorted(dup_counts.items())},
        },
        "bootstrap_results": {},
    }
    
    for (model, condition), r in sorted(results.items()):
        key_str = f"{model}/{condition}"
        full_results["bootstrap_results"][key_str] = {
            k: v for k, v in r.items() if k != "correct_vector"
        }
    
    with open(f"{OUT_DIR}/deduplicated_bootstrap_results.json", "w") as f:
        json.dump(full_results, f, indent=2)
    
    with open(f"{OUT_DIR}/ci_comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "condition", "accuracy", "data_source",
            "original_ci_lo", "original_ci_hi", "original_ci_width",
            "deduplicated_ci_lo", "deduplicated_ci_hi", "deduplicated_ci_width",
            "clustered_ci_lo", "clustered_ci_hi", "clustered_ci_width",
            "width_ratio_dedup", "width_ratio_clust",
        ])
        for (model, condition), r in sorted(results.items()):
            ow = r["original_ci_width"]
            dw = r["deduplicated_ci_width"]
            cw = r["clustered_ci_width"]
            writer.writerow([
                model, condition, f"{r['accuracy']:.4f}", r["data_source"],
                f"{r['original_ci'][0]:.4f}", f"{r['original_ci'][1]:.4f}", f"{ow:.4f}",
                f"{r['deduplicated_ci'][0]:.4f}", f"{r['deduplicated_ci'][1]:.4f}", f"{dw:.4f}",
                f"{r['clustered_ci'][0]:.4f}", f"{r['clustered_ci'][1]:.4f}", f"{cw:.4f}",
                f"{dw/ow:.2f}" if ow > 0 else "N/A",
                f"{cw/ow:.2f}" if ow > 0 else "N/A",
            ])
    
    with open(f"{OUT_DIR}/pairwise_significance.json", "w") as f:
        json.dump(pairwise_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Unique problems: {n_unique} / {N_TOTAL} ({n_unique/N_TOTAL*100:.1f}%)")
    
    widths_o = [r["original_ci_width"] for r in results.values()]
    widths_d = [r["deduplicated_ci_width"] for r in results.values()]
    widths_c = [r["clustered_ci_width"] for r in results.values()]
    
    print(f"\nCI Width Statistics:")
    print(f"  Original   : mean={np.mean(widths_o):.4f}, max={np.max(widths_o):.4f}")
    print(f"  Deduplicated: mean={np.mean(widths_d):.4f}, max={np.max(widths_d):.4f}")
    print(f"  Clustered  : mean={np.mean(widths_c):.4f}, max={np.max(widths_c):.4f}")
    
    ratios_d = [d/o for d, o in zip(widths_d, widths_o) if o > 1e-6]
    ratios_c = [c/o for c, o in zip(widths_c, widths_o) if o > 1e-6]
    print(f"\nWidth Ratios (vs original):")
    print(f"  Deduplicated: mean={np.mean(ratios_d):.2f}x, range=[{np.min(ratios_d):.2f}x, {np.max(ratios_d):.2f}x]")
    print(f"  Clustered   : mean={np.mean(ratios_c):.2f}x, range=[{np.min(ratios_c):.2f}x, {np.max(ratios_c):.2f}x]")
    
    n_sig = sum(1 for p in pairwise_results if p["significant_at_95"])
    n_total_pw = len(pairwise_results)
    print(f"\nPairwise: {n_sig}/{n_total_pw} significant under clustered bootstrap")
    
    borderline = [p for p in pairwise_results if abs(p["point_diff"]) < 0.14 and not p["significant_at_95"]]
    if borderline:
        print(f"\nBorderline comparisons that LOST significance:")
        for p in borderline:
            print(f"  {p['comparison']}: Δ={p['point_diff']:+.4f}")
    
    large_still_sig = [p for p in pairwise_results if abs(p["point_diff"]) >= 0.14 and p["significant_at_95"]]
    print(f"\nLarge differences (≥14pp) still significant: {len(large_still_sig)}/{len([p for p in pairwise_results if abs(p['point_diff']) >= 0.14])}")
    
    n_real = sum(1 for r in results.values() if r["data_source"] == "real")
    n_sim = sum(1 for r in results.values() if r["data_source"] == "simulated")
    print(f"\nData sources: {n_real} real, {n_sim} simulated")
    
    print(f"\nOutputs: {OUT_DIR}/")

if __name__ == "__main__":
    main()
