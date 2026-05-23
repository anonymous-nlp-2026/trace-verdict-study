"""Task 1-3: eval label_only + paired bootstrap vs standard full_trace and no_trace."""
import json, os, sys
import numpy as np

CKPT_PATH = "./checkpoints/full_trace_label_only/final"
MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
DATA_DIR = "./data"
PRED_DIR = "./checkpoints/predictions_v5"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def compute_metrics(labels, preds):
    n = len(labels)
    correct = sum(l == p for l, p in zip(labels, preds))
    accuracy = correct / n if n else 0.0
    count_pass = sum(l == "pass" for l in labels)
    count_fail = n - count_pass
    majority_class = "pass" if count_pass >= count_fail else "fail"
    majority_baseline = max(count_pass, count_fail) / n if n else 0.0
    lift = accuracy - majority_baseline
    metrics = {"accuracy": accuracy, "majority_baseline": majority_baseline,
               "lift": lift, "majority_class": majority_class, "n": n}
    for cls in ["pass", "fail"]:
        tp = sum(l == cls and p == cls for l, p in zip(labels, preds))
        fp = sum(l != cls and p == cls for l, p in zip(labels, preds))
        fn = sum(l == cls and p != cls for l, p in zip(labels, preds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[f"{cls}_precision"] = prec
        metrics[f"{cls}_recall"] = rec
        metrics[f"{cls}_f1"] = f1
    return metrics

def bootstrap_ci(labels, preds, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels_arr, preds_arr = np.array(labels), np.array(preds)
    all_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        all_metrics.append(compute_metrics(labels_arr[idx].tolist(), preds_arr[idx].tolist()))
    ci = {}
    for key in all_metrics[0]:
        if key in ("n", "majority_class"): continue
        values = [m[key] for m in all_metrics]
        ci[f"{key}_ci_lower"] = float(np.percentile(values, 2.5))
        ci[f"{key}_ci_upper"] = float(np.percentile(values, 97.5))
    return ci

def paired_bootstrap(correct_a, correct_b, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(correct_a)
    obs_diff = float(correct_a.mean() - correct_b.mean())
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    return {
        "observed_diff": obs_diff,
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0)),
        "n_bootstrap": n_bootstrap, "n_samples": n,
    }

# ---- Task 1: Eval label_only ----
print("=== Task 1: Evaluating full_trace_label_only ===", flush=True)
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

test_data = []
with open(os.path.join(DATA_DIR, "full_trace_test.jsonl")) as f:
    for line in f:
        line = line.strip()
        if line: test_data.append(json.loads(line))
print(f"Test samples: {len(test_data)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading base model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0}
)
print(f"Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, CKPT_PATH)
model.eval()

predictions = []
for i, item in enumerate(test_data):
    messages = [{"role": "user", "content": item["input"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, temperature=None, top_p=None)
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
    pred = "pass" if "pass" in raw else ("fail" if "fail" in raw else raw)
    label = item["output"]
    predictions.append({
        "idx": i, "problem_id": item.get("problem_id", f"sample_{i}"),
        "label": label, "pred": pred, "correct": label == pred,
    })
    if (i + 1) % 100 == 0:
        acc = sum(p["correct"] for p in predictions) / len(predictions)
        print(f"  [{i+1}/{len(test_data)}] acc={acc:.4f}", flush=True)

labels = [p["label"] for p in predictions]
preds_list = [p["pred"] for p in predictions]
metrics = compute_metrics(labels, preds_list)
ci = bootstrap_ci(labels, preds_list)

eval_results = {"condition": "full_trace_label_only", "checkpoint_dir": CKPT_PATH,
                "metrics": metrics, "confidence_intervals": ci}
eval_out = os.path.join(CKPT_PATH, "eval_results.json")
with open(eval_out, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"\neval_results.json SAVED: {eval_out}")
print(f"  accuracy: {metrics['accuracy']*100:.1f}% [{ci['accuracy_ci_lower']*100:.1f}, {ci['accuracy_ci_upper']*100:.1f}]")

pred_out = os.path.join(PRED_DIR, "full_trace_label_only_predictions.json")
with open(pred_out, "w") as f:
    json.dump({"condition": "full_trace_label_only", "predictions": predictions}, f, indent=2)
print(f"  predictions saved: {pred_out}")

del model
torch.cuda.empty_cache()

# ---- Task 2 & 3: Paired bootstrap ----
print("\n=== Task 2 & 3: Paired Bootstrap ===", flush=True)
with open(os.path.join(PRED_DIR, "full_trace_predictions.json")) as f:
    ft_preds = json.load(f)["predictions"]
with open(os.path.join(PRED_DIR, "no_trace_predictions.json")) as f:
    nt_preds = json.load(f)["predictions"]

lo_correct = np.array([p["correct"] for p in predictions], dtype=np.float64)
ft_correct = np.array([p["correct"] for p in ft_preds], dtype=np.float64)
nt_correct = np.array([p["correct"] for p in nt_preds], dtype=np.float64)
print(f"Samples: label_only={len(lo_correct)}, full_trace={len(ft_correct)}, no_trace={len(nt_correct)}")
assert len(lo_correct) == len(ft_correct) == len(nt_correct)

# Task 2: label-only vs standard full_trace
r_vs_ft = paired_bootstrap(lo_correct, ft_correct, n_bootstrap=10000)
print(f"\n--- label-only vs standard full_trace ---")
print(f"  paired diff: {r_vs_ft['observed_diff']*100:+.1f}pp")
print(f"  95% CI: [{r_vs_ft['ci_lower']*100:.1f}, {r_vs_ft['ci_upper']*100:.1f}]")
print(f"  p-value: {r_vs_ft['p_value']:.4f}")
print(f"  significant: {'YES' if r_vs_ft['p_value'] < 0.05 else 'NO'}")
with open(os.path.join(CKPT_PATH, "bootstrap_vs_standard.json"), "w") as f:
    json.dump(r_vs_ft, f, indent=2)

# Task 3: label-only vs no_trace
r_vs_nt = paired_bootstrap(lo_correct, nt_correct, n_bootstrap=10000)
print(f"\n--- label-only vs no_trace ---")
print(f"  lift: {r_vs_nt['observed_diff']*100:+.1f}pp")
print(f"  95% CI: [{r_vs_nt['ci_lower']*100:.1f}, {r_vs_nt['ci_upper']*100:.1f}]")
print(f"  p-value: {r_vs_nt['p_value']:.4f}")
print(f"  significant: {'YES' if r_vs_nt['p_value'] < 0.05 else 'NO'}")
with open(os.path.join(CKPT_PATH, "bootstrap_vs_notrace.json"), "w") as f:
    json.dump(r_vs_nt, f, indent=2)

print("\n=== ALL TASKS COMPLETE ===")
