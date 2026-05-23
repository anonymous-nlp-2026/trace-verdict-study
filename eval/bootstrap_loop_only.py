"""Generate loop_only predictions and run pairwise bootstrap comparison."""
import json
import os
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_DIR = "./checkpoints/loop_only/final"
MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
DATA_DIR = "./data"
PRED_V5_DIR = "./checkpoints/predictions_v5"
OUTPUT_DIR = "./checkpoints/loop_only/final"
BASELINES = ["no_trace", "vota", "full_trace"]
N_BOOTSTRAP = 10000
SEED = 42

def load_test_data(condition):
    path = os.path.join(DATA_DIR, f"{condition}_test.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_predictions(condition):
    path = os.path.join(PRED_V5_DIR, f"{condition}_predictions.json")
    with open(path) as f:
        return json.load(f)["predictions"]

def compute_metrics(labels, preds):
    n = len(labels)
    correct = sum(1 for l, p in zip(labels, preds) if l == p)
    accuracy = correct / n
    count_pass = sum(1 for l in labels if l == "pass")
    count_fail = n - count_pass
    majority_class = "pass" if count_pass >= count_fail else "fail"
    majority_baseline = max(count_pass, count_fail) / n
    return {"accuracy": accuracy, "majority_baseline": majority_baseline,
            "majority_class": majority_class, "n": n}

def bootstrap_pairwise(correct_a, correct_b, n_bootstrap):
    rng = np.random.RandomState(SEED)
    n = len(correct_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = correct_a[idx].mean() - correct_b[idx].mean()
        diffs.append(diff)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(correct_a.mean() - correct_b.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "ci_contains_zero": bool(np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)),
    }

def main():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    # Step 1: Generate loop_only predictions
    print("Loading test data for loop_only...")
    test_data = load_test_data("loop_only")
    print(f"Test samples: {len(test_data)}")

    print(f"Loading base model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": 0},
    )
    print(f"Loading LoRA adapter from {CHECKPOINT_DIR}")
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.eval()

    print("Running loop_only inference...")
    predictions = []
    for i, item in enumerate(test_data):
        messages = [{"role": "user", "content": item["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            print(f"  [{i+1}/{len(test_data)}] acc={acc:.4f}")

    # Save predictions
    pred_path = os.path.join(OUTPUT_DIR, "loop_only_predictions.json")
    with open(pred_path, "w") as f:
        json.dump({"condition": "loop_only", "predictions": predictions}, f, indent=2)
    print(f"Saved predictions: {pred_path}")

    acc = sum(p["correct"] for p in predictions) / len(predictions)
    print(f"\nloop_only accuracy: {acc:.4f}")

    # Step 2: Load baseline predictions
    del model
    torch.cuda.empty_cache()

    loop_correct = np.array([p["correct"] for p in predictions], dtype=np.float64)

    results = {"loop_only_accuracy": acc, "comparisons": {}, "config": {"n_bootstrap": N_BOOTSTRAP, "seed": SEED}}

    for baseline in BASELINES:
        print(f"\nLoading {baseline} predictions...")
        baseline_preds = load_predictions(baseline)
        baseline_correct = np.array([p["correct"] for p in baseline_preds], dtype=np.float64)
        baseline_acc = baseline_correct.mean()

        if len(baseline_correct) != len(loop_correct):
            print(f"  WARNING: size mismatch loop_only={len(loop_correct)} vs {baseline}={len(baseline_correct)}")
            min_n = min(len(loop_correct), len(baseline_correct))
            lc = loop_correct[:min_n]
            bc = baseline_correct[:min_n]
        else:
            lc, bc = loop_correct, baseline_correct

        comp = bootstrap_pairwise(lc, bc, N_BOOTSTRAP)
        comp["loop_only_acc"] = float(acc)
        comp["baseline_acc"] = float(baseline_acc)
        results["comparisons"][f"loop_only_vs_{baseline}"] = comp

        sig = "significant" if not comp["ci_contains_zero"] else "NOT significant"
        print(f"  loop_only ({acc:.4f}) vs {baseline} ({baseline_acc:.4f}): "
              f"diff={comp['mean_diff']:+.4f} 95%CI=[{comp['ci_lower']:+.4f}, {comp['ci_upper']:+.4f}] ({sig})")

    out_path = os.path.join(OUTPUT_DIR, "bootstrap_compare.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
