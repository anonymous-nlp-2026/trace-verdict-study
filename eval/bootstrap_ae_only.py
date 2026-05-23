"""Generate ae_only and exception_only predictions, then run pairwise bootstrap."""
import json
import os
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_DIR = "./checkpoints"
MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
DATA_DIR = "./data"
PRED_V5_DIR = "./checkpoints/predictions_v5"
OUTPUT_PATH = "./checkpoints/ae_only/final/bootstrap_pairwise.json"
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

def load_predictions_v5(condition):
    path = os.path.join(PRED_V5_DIR, f"{condition}_predictions.json")
    with open(path) as f:
        return json.load(f)["predictions"]

def run_inference(checkpoint_path, test_data, tokenizer, model, device):
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
        if (i + 1) % 200 == 0:
            acc = sum(p["correct"] for p in predictions) / len(predictions)
            print(f"  [{i+1}/{len(test_data)}] acc={acc:.4f}")
    return predictions

def bootstrap_pairwise(correct_a, correct_b, n_bootstrap):
    rng = np.random.RandomState(SEED)
    n = len(correct_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = correct_a[idx].mean() - correct_b[idx].mean()
        diffs.append(diff)
    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    observed_diff = float(correct_a.mean() - correct_b.mean())
    # Two-sided p-value: proportion of bootstrap diffs on wrong side of zero
    if observed_diff >= 0:
        p_value = float(2 * np.mean(diffs <= 0))
    else:
        p_value = float(2 * np.mean(diffs >= 0))
    p_value = min(p_value, 1.0)
    return {
        "diff": observed_diff,
        "ci": [ci_lower, ci_upper],
        "p_value": p_value,
        "significant": not (ci_lower <= 0 <= ci_upper),
    }

def main():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0")

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": device},
    )

    # --- ae_only inference ---
    ae_ckpt = os.path.join(CHECKPOINT_DIR, "ae_only/final")
    print(f"Loading ae_only adapter from {ae_ckpt}")
    model = PeftModel.from_pretrained(base_model, ae_ckpt)
    model.eval()
    test_data = load_test_data("ae_only")
    print(f"Running ae_only inference ({len(test_data)} samples)...")
    ae_preds = run_inference(ae_ckpt, test_data, tokenizer, model, device)
    ae_acc = sum(p["correct"] for p in ae_preds) / len(ae_preds)
    print(f"ae_only accuracy: {ae_acc:.6f}")

    # Save ae_only predictions
    ae_pred_path = os.path.join(CHECKPOINT_DIR, "ae_only/final/ae_only_predictions.json")
    with open(ae_pred_path, "w") as f:
        json.dump({"condition": "ae_only", "predictions": ae_preds}, f, indent=2)
    print(f"Saved: {ae_pred_path}")

    # Unload ae_only adapter, load exception_only
    del model
    torch.cuda.empty_cache()

    # --- exception_only inference ---
    exc_ckpt = os.path.join(CHECKPOINT_DIR, "exception_only/final")
    print(f"\nLoading exception_only adapter from {exc_ckpt}")
    model = PeftModel.from_pretrained(base_model, exc_ckpt)
    model.eval()
    test_data_exc = load_test_data("exception_only")
    print(f"Running exception_only inference ({len(test_data_exc)} samples)...")
    exc_preds = run_inference(exc_ckpt, test_data_exc, tokenizer, model, device)
    exc_acc = sum(p["correct"] for p in exc_preds) / len(exc_preds)
    print(f"exception_only accuracy: {exc_acc:.6f}")

    # Save exception_only predictions
    exc_pred_path = os.path.join(CHECKPOINT_DIR, "exception_only/final/exception_only_predictions.json")
    with open(exc_pred_path, "w") as f:
        json.dump({"condition": "exception_only", "predictions": exc_preds}, f, indent=2)
    print(f"Saved: {exc_pred_path}")

    del model
    del base_model
    torch.cuda.empty_cache()

    # --- Load baseline predictions ---
    print("\nLoading baseline predictions...")
    no_trace_preds = load_predictions_v5("no_trace")
    vota_preds = load_predictions_v5("vota")

    # Build correct arrays
    ae_correct = np.array([p["correct"] for p in ae_preds], dtype=np.float64)
    exc_correct = np.array([p["correct"] for p in exc_preds], dtype=np.float64)
    no_trace_correct = np.array([p["correct"] for p in no_trace_preds], dtype=np.float64)
    vota_correct = np.array([p["correct"] for p in vota_preds], dtype=np.float64)

    print(f"\nSample counts: ae_only={len(ae_correct)}, exception_only={len(exc_correct)}, "
          f"no_trace={len(no_trace_correct)}, vota={len(vota_correct)}")

    # --- Pairwise bootstrap ---
    print(f"\nBootstrap (N={N_BOOTSTRAP}, seed={SEED})...")

    comparisons = {}
    pairs = [
        ("ae_only", ae_correct, "no_trace", no_trace_correct),
        ("ae_only", ae_correct, "exception_only", exc_correct),
        ("ae_only", ae_correct, "vota", vota_correct),
    ]
    for name_a, arr_a, name_b, arr_b in pairs:
        key = f"{name_a}_vs_{name_b}"
        comp = bootstrap_pairwise(arr_a, arr_b, N_BOOTSTRAP)
        comparisons[key] = comp
        sig = "significant" if comp["significant"] else "NOT significant"
        print(f"  {key}: diff={comp['diff']:+.6f} 95%CI=[{comp['ci'][0]:+.6f}, {comp['ci'][1]:+.6f}] "
              f"p={comp['p_value']:.4f} ({sig})")

    result = {
        "ae_only_vs_no_trace": comparisons["ae_only_vs_no_trace"],
        "ae_only_vs_exception_only": comparisons["ae_only_vs_exception_only"],
        "ae_only_vs_vota": comparisons["ae_only_vs_vota"],
        "config": {"n_bootstrap": N_BOOTSTRAP, "seed": SEED, "method": "paired_bootstrap"},
        "accuracies": {
            "ae_only": float(ae_acc),
            "exception_only": float(exc_acc),
            "no_trace": float(no_trace_correct.mean()),
            "vota": float(vota_correct.mean()),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
