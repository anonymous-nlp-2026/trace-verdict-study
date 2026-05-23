"""Evaluation script for trace-verdict classification models with bootstrap CI."""

import argparse
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(data_dir: str, condition: str) -> list[dict]:
    """Load test JSONL file for a given condition."""
    path = os.path.join(data_dir, f"{condition}_test.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def predict(model, tokenizer, input_text: str, device: torch.device, max_length: int = 2048, truncation_side: str = "right") -> str:
    """Generate a single-token prediction via greedy decoding."""
    messages = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if inputs["input_ids"].shape[1] > max_length:
        overhead_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
        )
        overhead_len = len(tokenizer(overhead_prompt).input_ids)
        content_budget = max_length - overhead_len
        content_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        if truncation_side == "left":
            content_ids = content_ids[-content_budget:]
        else:
            content_ids = content_ids[:content_budget]
        truncated_text = tokenizer.decode(content_ids, skip_special_tokens=False)
        messages = [{"role": "user", "content": truncated_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()


def classify_prediction(pred: str) -> str:
    """Map raw model output to pass/fail label."""
    if "pass" in pred:
        return "pass"
    if "fail" in pred:
        return "fail"
    return pred


def compute_metrics(labels: list[str], preds: list[str]) -> dict:
    """Compute accuracy, majority baseline, lift, and per-class precision/recall/F1."""
    n = len(labels)
    correct = sum(1 for l, p in zip(labels, preds) if l == p)
    accuracy = correct / n if n > 0 else 0.0

    count_pass = sum(1 for l in labels if l == "pass")
    count_fail = n - count_pass
    majority_class = "pass" if count_pass >= count_fail else "fail"
    majority_baseline = max(count_pass, count_fail) / n if n > 0 else 0.0
    lift = accuracy - majority_baseline

    metrics = {"accuracy": accuracy, "majority_baseline": majority_baseline,
               "lift": lift, "majority_class": majority_class, "n": n}

    for cls in ["pass", "fail"]:
        tp = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, preds) if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, preds) if l == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"{cls}_precision"] = precision
        metrics[f"{cls}_recall"] = recall
        metrics[f"{cls}_f1"] = f1

    return metrics


def bootstrap_ci(labels: list[str], preds: list[str], n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Compute 95% bootstrap confidence intervals for all metrics."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    all_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(labels_arr[idx].tolist(), preds_arr[idx].tolist())
        all_metrics.append(m)

    ci = {}
    for key in all_metrics[0]:
        if key in ("n", "majority_class"):
            continue
        values = [m[key] for m in all_metrics]
        ci[f"{key}_ci_lower"] = float(np.percentile(values, 2.5))
        ci[f"{key}_ci_upper"] = float(np.percentile(values, 97.5))

    return ci


def main():
    parser = argparse.ArgumentParser(description="Evaluate trace-verdict classification model.")
    parser.add_argument("--condition", required=True, choices=["no_trace", "vota", "full_trace", "vota_no_ae", "ae_only", "exception_only", "exception_only_nomarker", "loop_only", "debugbench_no_trace", "debugbench_vota", "debugbench_full_trace", "debugbench_ae_only", "debugbench_exception_only", "full_trace_tail", "full_trace_tail_512", "full_trace_tail_1024", "full_trace_tail_256", "full_trace_head", "exception_only_padded2048", "full_trace_tail_384", "cross_he2mbpp_exception_only", "cross_he2mbpp_no_trace", "cross_mbpp2he_exception_only", "cross_mbpp2he_no_trace", "exc_type_only", "exc_stacktrace_only", "exc_message_only", "cross_he2mbpp_exception_only_nomarker"],
                        help="Evaluation condition: no_trace, vota, or full_trace")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to checkpoint directory with LoRA adapter")
    parser.add_argument("--model_path", default="./models/Qwen2.5-Coder-7B-Instruct",
                        help="Path to base model")
    parser.add_argument("--data_dir", default="./data/",
                        help="Directory containing test data")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: {checkpoint_dir}/eval_results.json)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-sample predictions to predictions.jsonl")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max input token length, must match training max_length (default: 2048)")
    parser.add_argument("--truncation_side", default="right", choices=["left", "right"],
                        help="Truncation side: right=keep head (default), left=keep tail")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.checkpoint_dir, "eval_results.json")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    print(f"Condition: {args.condition}")
    print(f"Checkpoint: {args.checkpoint_dir}")

    test_data = load_test_data(args.data_dir, args.condition)
    print(f"Test samples: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )

    print(f"Loading LoRA adapter from {args.checkpoint_dir}")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()

    print("Running inference...")
    labels, preds, raw_preds = [], [], []

    for i, item in enumerate(test_data):
        raw_pred = predict(model, tokenizer, item["input"], device, max_length=args.max_length, truncation_side=args.truncation_side)
        pred = classify_prediction(raw_pred)

        labels.append(item["output"])
        preds.append(pred)
        raw_preds.append(raw_pred)

        if (i + 1) % 50 == 0:
            acc = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels)
            print(f"  [{i+1}/{len(test_data)}] running accuracy: {acc:.4f}")

    print("\nComputing metrics...")
    metrics = compute_metrics(labels, preds)
    ci = bootstrap_ci(labels, preds)

    results = {
        "condition": args.condition,
        "checkpoint_dir": args.checkpoint_dir,
        "metrics": metrics,
        "confidence_intervals": ci,
    }

    mb = metrics["majority_baseline"]
    lift_val = metrics["lift"]
    maj_cls = metrics["majority_class"]

    print(f"\n=== Evaluation Results (condition: {args.condition}) ===")
    print(f"Accuracy:           {metrics['accuracy']*100:.1f}% "
          f"(95% CI: [{ci['accuracy_ci_lower']*100:.1f}%, {ci['accuracy_ci_upper']*100:.1f}%])")
    print(f"Majority baseline:  {mb*100:.1f}% (always predict '{maj_cls}')")
    print(f"Lift over majority: {lift_val*100:+.1f}pp "
          f"(95% CI: [{ci['lift_ci_lower']*100:+.1f}pp, {ci['lift_ci_upper']*100:+.1f}pp])")
    print()
    print("Per-class metrics:")
    for cls in ["pass", "fail"]:
        print(f"  {cls}:  P={metrics[f'{cls}_precision']:.2f}  "
              f"R={metrics[f'{cls}_recall']:.2f}  F1={metrics[f'{cls}_f1']:.2f}")
    print("\nBootstrap: 1000 resamples")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    if args.save_predictions:
        pred_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "predictions.jsonl")
        with open(pred_path, "w") as f:
            for i in range(len(labels)):
                json.dump({"id": i, "label": labels[i], "pred": preds[i], "raw_pred": raw_preds[i]}, f)
                f.write("\n")
        print(f"Predictions saved to {pred_path}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
