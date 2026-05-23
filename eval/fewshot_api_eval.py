import json
import argparse
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

EXAMPLE_INDICES = [1851, 875, 30]

SYSTEM_PROMPT = """You are a software testing expert. Given code and its test cases along with execution trace information, determine whether the code passes all tests.

Rules:
- If the execution trace shows "[No exception data in trace]" or "[No trace available]", it means the trace does not contain exception information. You must still judge based on the code and tests.
- If the execution trace shows error/exception information, the code likely fails.
- Answer with exactly one word: pass or fail."""


def load_examples(train_path, indices):
    with open(train_path) as f:
        all_data = [json.loads(l) for l in f]
    examples = []
    for idx in indices:
        d = all_data[idx]
        examples.append({"input": d["input"], "output": d["output"], "index": idx})
    return examples


def build_messages(examples, test_input, n_shot):
    test_text = test_input.rstrip()
    if test_text.endswith("Answer: "):
        test_text = test_text[:-len("Answer: ")].rstrip()
    elif test_text.endswith("Answer:"):
        test_text = test_text[:-len("Answer:")].rstrip()

    if n_shot == 0:
        user_content = f"{test_text}\n\nBased on this information, did the test pass or fail? Answer with exactly one word: pass or fail."
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    parts = ["Here are some examples:\n"]
    for i, ex in enumerate(examples[:n_shot], 1):
        ex_input = ex["input"].rstrip()
        if ex_input.endswith("Answer: "):
            ex_input = ex_input[:-len("Answer: ")].rstrip()
        elif ex_input.endswith("Answer:"):
            ex_input = ex_input[:-len("Answer:")].rstrip()
        parts.append(f"Example {i}:\n{ex_input}\n\nAnswer: {ex['output']}\n")

    parts.append(f"Now determine:\n{test_text}\n\nBased on this information, did the test pass or fail? Answer with exactly one word: pass or fail.")
    user_content = "\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_openai(messages, model="gpt-4o", max_retries=3):
    import openai
    client = openai.OpenAI()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=5,
                temperature=0,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def call_anthropic(messages, model="claude-sonnet-4-20250514", max_retries=3):
    import anthropic
    client = anthropic.Anthropic()
    system = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            chat_messages.append(m)
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                system=system,
                messages=chat_messages,
                max_tokens=5,
                temperature=0,
            )
            return resp.content[0].text.strip().lower()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


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


def run_eval(samples, examples, n_shot, provider, model, max_workers=10):
    call_fn = call_openai if provider == "openai" else call_anthropic

    def process_one(i, s):
        msgs = build_messages(examples, s["input"], n_shot)
        raw = call_fn(msgs, model=model)
        pred = classify(raw)
        label = s["output"].strip().lower()
        return i, label, pred, raw

    labels = [None] * len(samples)
    preds = [None] * len(samples)
    raw_outputs = [None] * len(samples)
    errors = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, i, s): i for i, s in enumerate(samples)}
        for future in as_completed(futures):
            i, label, pred, raw = future.result()
            labels[i] = label
            preds[i] = pred
            raw_outputs[i] = raw
            if pred != label:
                errors.append({"idx": i, "label": label, "pred": pred, "raw": raw})
            done += 1
            if done % 50 == 0:
                acc_so_far = sum(1 for l, p in zip(labels[:done], preds[:done]) if l is not None and p is not None and l == p) / done
                print(f"  [{done}/{len(samples)}] running acc ~{acc_so_far:.4f}")

    return labels, preds, raw_outputs, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=0)
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()

    if args.model is None:
        args.model = "gpt-4o" if args.provider == "openai" else "claude-sonnet-4-20250514"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Provider: {args.provider}, Model: {args.model}, N-shot: {args.n_shot}")

    examples = []
    if args.n_shot > 0:
        print("Loading few-shot examples from training set...")
        examples = load_examples(args.train_path, EXAMPLE_INDICES[:args.n_examples])
        for i, ex in enumerate(examples):
            print(f"  Example {i+1}: train_idx={ex['index']} label={ex['output']}")

    print("Loading test data...")
    samples = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples")

    labels, preds, raw_outputs, errors = run_eval(
        samples, examples, args.n_shot, args.provider, args.model, args.max_workers
    )

    metrics = compute_metrics(labels, preds)
    ci_lo, ci_hi = bootstrap_ci(labels, preds)
    metrics["accuracy_95ci_lo"] = ci_lo
    metrics["accuracy_95ci_hi"] = ci_hi
    metrics["n_unknown"] = sum(1 for p in preds if p not in ("pass", "fail"))
    metrics["n_test"] = len(samples)
    metrics["model"] = args.model
    metrics["provider"] = args.provider
    metrics["method"] = f"{args.n_shot}-shot"
    metrics["condition"] = "exception_only"
    if args.n_shot > 0:
        metrics["example_indices"] = EXAMPLE_INDICES[:args.n_examples]
        metrics["example_labels"] = [ex["output"] for ex in examples]

    for k, v in metrics.items():
        if isinstance(v, float):
            metrics[k] = round(v, 4)

    result_path = os.path.join(args.output_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for i in range(len(labels)):
            record = {"idx": i, "label": labels[i], "prediction": preds[i], "raw_output": raw_outputs[i]}
            f.write(json.dumps(record) + "\n")

    error_path = os.path.join(args.output_dir, "error_examples.json")
    with open(error_path, "w") as f:
        json.dump(errors[:30], f, indent=2)

    print(f"\n{'='*50}")
    print(f"{args.n_shot}-SHOT RESULTS ({args.provider}/{args.model}, n={len(samples)})")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}  (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    print(f"Majority BL: {metrics['majority_baseline']:.4f}")
    print(f"Lift:        {metrics['lift']:.4f}")
    print(f"Unknown:     {metrics['n_unknown']}")
    print(f"Pass  P={metrics['pass_precision']:.4f} R={metrics['pass_recall']:.4f} F1={metrics['pass_f1']:.4f}")
    print(f"Fail  P={metrics['fail_precision']:.4f} R={metrics['fail_recall']:.4f} F1={metrics['fail_f1']:.4f}")
    if errors:
        print(f"\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  [{e['idx']}] label={e['label']} pred={e['pred']} raw={repr(e['raw'])}")
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
