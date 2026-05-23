import json
import argparse
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

EXAMPLE_INDICES = [1851, 875, 30]

SYSTEM_PROMPT = """You are a software testing expert. Given execution traces from test cases, determine whether each test passed or failed.

Here are some examples:"""

def load_examples(train_path, indices):
    with open(train_path) as f:
        all_data = [json.loads(l) for l in f]
    examples = []
    for idx in indices:
        d = all_data[idx]
        examples.append({"input": d["input"], "output": d["output"], "index": idx})
    return examples

def build_fewshot_prompt(examples, test_input):
    parts = [SYSTEM_PROMPT]
    for i, ex in enumerate(examples, 1):
        ex_input = ex["input"].rstrip()
        if ex_input.endswith("Answer: "):
            ex_input = ex_input[:-len("Answer: ")].rstrip()
        elif ex_input.endswith("Answer:"):
            ex_input = ex_input[:-len("Answer:")].rstrip()
        parts.append(f"\nExample {i}:\n{ex_input}\n\nAnswer: {ex['output']}")

    test_text = test_input.rstrip()
    if test_text.endswith("Answer: "):
        test_text = test_text[:-len("Answer: ")].rstrip()
    elif test_text.endswith("Answer:"):
        test_text = test_text[:-len("Answer:")].rstrip()

    parts.append(f"\nNow determine:\n{test_text}\n\nBased on this trace, did the test pass or fail? Answer with exactly one word: pass or fail.\n\nAnswer:")
    return "\n".join(parts)


def predict(model, tokenizer, fewshot_prompt, device, max_length=4096):
    messages = [{"role": "user", "content": fewshot_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if inputs["input_ids"].shape[1] > max_length:
        overhead_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
        )
        overhead_len = len(tokenizer(overhead_prompt).input_ids)
        content_budget = max_length - overhead_len
        content_ids = tokenizer(fewshot_prompt, add_special_tokens=False).input_ids
        content_ids = content_ids[:content_budget]
        truncated_text = tokenizer.decode(content_ids, skip_special_tokens=False)
        messages = [{"role": "user", "content": truncated_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()


def classify(pred):
    pred = pred.strip().lower()
    if "pass" in pred:
        return "pass"
    if "fail" in pred:
        return "fail"
    if pred in ("yes", "true"):
        return "pass"
    if pred in ("no", "false"):
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
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    print("Loading examples from train set...")
    examples = load_examples(args.train_path, EXAMPLE_INDICES[:args.n_examples])
    for i, ex in enumerate(examples):
        print(f"  Example {i+1}: train_idx={ex['index']} label={ex['output']} input_preview={ex['input'][:80]}...")

    print("Loading test data...")
    samples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples")

    first_prompt = build_fewshot_prompt(examples, samples[0]["input"])
    first_tokens = len(tokenizer.encode(first_prompt))
    print(f"First prompt token count: {first_tokens}")

    labels = []
    preds = []
    raw_outputs = []
    errors = []
    truncated_count = 0

    for i, s in enumerate(samples):
        fewshot_prompt = build_fewshot_prompt(examples, s["input"])
        prompt_tokens = len(tokenizer.encode(fewshot_prompt))
        if prompt_tokens > args.max_length:
            truncated_count += 1

        raw = predict(model, tokenizer, fewshot_prompt, device, max_length=args.max_length)
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
    metrics["n_truncated"] = truncated_count
    metrics["model"] = "Qwen2.5-Coder-7B-Instruct"
    metrics["method"] = "3-shot"
    metrics["condition"] = "exception_only"
    metrics["example_indices"] = EXAMPLE_INDICES[:args.n_examples]
    metrics["example_labels"] = [ex["output"] for ex in examples]

    for k, v in metrics.items():
        if isinstance(v, float):
            metrics[k] = round(v, 4)

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for i in range(len(labels)):
            record = {"idx": i, "label": labels[i], "prediction": preds[i], "raw_output": raw_outputs[i]}
            f.write(json.dumps(record) + "\n")

    error_path = os.path.join(output_dir, "error_examples.json")
    with open(error_path, "w") as f:
        json.dump(errors[:30], f, indent=2)

    print(f"\n{'='*50}")
    print(f"3-SHOT RESULTS (exception_only, n={len(samples)})")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"Majority BL: {metrics['majority_baseline']:.4f}")
    print(f"Lift:        {metrics['lift']:.4f}")
    print(f"Unknown:     {n_unknown}")
    print(f"Truncated:   {truncated_count}")
    print(f"Pass  P={metrics['pass_precision']:.4f} R={metrics['pass_recall']:.4f} F1={metrics['pass_f1']:.4f}")
    print(f"Fail  P={metrics['fail_precision']:.4f} R={metrics['fail_recall']:.4f} F1={metrics['fail_f1']:.4f}")
    print(f"\nExamples used (train indices): {EXAMPLE_INDICES[:args.n_examples]}")
    print(f"Example labels: {[ex['output'] for ex in examples]}")
    print(f"\nFirst 5 errors:")
    for e in errors[:5]:
        print(f"  [{e['idx']}] label={e['label']} pred={e['pred']} raw={repr(e['raw'])}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
