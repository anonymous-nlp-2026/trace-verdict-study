import json
import argparse
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict(model, tokenizer, input_text, device, max_length=2048):
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


def classify(pred):
    if "pass" in pred:
        return "pass"
    if "fail" in pred:
        return "fail"
    return pred


def compute_metrics(labels, preds):
    n = len(labels)
    correct = sum(1 for l, p in zip(labels, preds) if l == p)
    accuracy = correct / n

    count_pass = sum(1 for l in labels if l == "pass")
    count_fail = n - count_pass
    majority_baseline = max(count_pass, count_fail) / n

    metrics = {"accuracy": accuracy, "majority_baseline": majority_baseline, "lift": accuracy - majority_baseline}

    for cls in ["pass", "fail"]:
        tp = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, preds) if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, preds) if l == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics[f"{cls}_precision"] = prec
        metrics[f"{cls}_recall"] = rec
        metrics[f"{cls}_f1"] = f1

    return metrics


def bootstrap_ci(labels, preds, n_boot=2000, ci=0.95):
    n = len(labels)
    accs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bl = [labels[i] for i in idx]
        bp = [preds[i] for i in idx]
        accs.append(sum(1 for l, p in zip(bl, bp) if l == p) / n)
    lo = np.percentile(accs, (1 - ci) / 2 * 100)
    hi = np.percentile(accs, (1 + ci) / 2 * 100)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    print("Loading data...")
    samples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")

    labels = []
    preds = []
    raw_outputs = []
    errors = []

    for i, s in enumerate(samples):
        raw = predict(model, tokenizer, s["input"], device)
        pred = classify(raw)
        label = s["output"].strip().lower()

        labels.append(label)
        preds.append(pred)
        raw_outputs.append(raw)

        if pred != label:
            errors.append({"idx": i, "label": label, "pred": pred, "raw": raw, "input_preview": s["input"][:200]})

        if (i + 1) % 50 == 0:
            acc_so_far = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels)
            print(f"  [{i+1}/{len(samples)}] running acc={acc_so_far:.4f}")

    metrics = compute_metrics(labels, preds)
    ci_lo, ci_hi = bootstrap_ci(labels, preds)
    metrics["accuracy_95ci_lo"] = ci_lo
    metrics["accuracy_95ci_hi"] = ci_hi

    n_unknown = sum(1 for p in preds if p not in ("pass", "fail"))
    metrics["n_unknown"] = n_unknown
    metrics["n_test"] = len(samples)
    metrics["model"] = "Qwen2.5-Coder-7B-Instruct"
    metrics["method"] = "zero-shot"
    metrics["condition"] = "exception_only"

    for k, v in metrics.items():
        if isinstance(v, float):
            metrics[k] = round(v, 4)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        pred_records = [{"idx": i, "label": labels[i], "prediction": preds[i], "raw_output": raw_outputs[i]}
                        for i in range(len(labels))]
        json.dump(pred_records, f, indent=2)

    with open(os.path.join(args.output_dir, "error_examples.json"), "w") as f:
        json.dump(errors[:20], f, indent=2)

    print(f"\n{'='*50}")
    print(f"ZERO-SHOT RESULTS (exception_only, n={len(samples)})")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"Majority BL: {metrics['majority_baseline']:.4f}")
    print(f"Lift:        {metrics['lift']:.4f}")
    print(f"Unknown:     {n_unknown}")
    print(f"Pass  P={metrics['pass_precision']:.4f} R={metrics['pass_recall']:.4f} F1={metrics['pass_f1']:.4f}")
    print(f"Fail  P={metrics['fail_precision']:.4f} R={metrics['fail_recall']:.4f} F1={metrics['fail_f1']:.4f}")
    print(f"\nFirst 5 errors:")
    for e in errors[:5]:
        print(f"  [{e['idx']}] label={e['label']} pred={e['pred']} raw={repr(e['raw'])}")
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
