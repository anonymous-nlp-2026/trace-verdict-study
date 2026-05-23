import json
import time
import argparse
import random
from pathlib import Path
from openai import OpenAI

API_BASE = "http://47.94.22.126/v1"
API_KEY = "tum_XXB3sZILRCg8llb3NjxVbbRNzFpsYVFdiogFkavpGn8"

DATA_DIR = "./data"
ARTIFACT_DIR = "./artifacts"

SYSTEM_PROMPT = (
    "You are a test verdict predictor. Given a code snippet and its execution trace, "
    "predict whether the test passes or fails. Respond with exactly one word: \"pass\" or \"fail\"."
)

MODEL_MAP = {
    "gpt4o": "gpt-4o",
    "gpt41": "gpt-4.1",
}


def load_data(condition, split="test"):
    path = Path(DATA_DIR) / f"{condition}_{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def parse_verdict(text):
    t = text.strip().lower()
    if t in ("pass", "fail"):
        return t
    if "fail" in t:
        return "fail"
    if "pass" in t:
        return "pass"
    return "unparseable"


def select_fewshot_examples(train_data, n=3, seed=42):
    rng = random.Random(seed)
    passes = [d for d in train_data if d["output"] == "pass"]
    fails = [d for d in train_data if d["output"] == "fail"]
    rng.shuffle(passes)
    rng.shuffle(fails)
    # 2 pass + 1 fail
    examples = passes[:2] + fails[:1]
    rng.shuffle(examples)
    return examples


def build_messages(item, shots, fewshot_examples=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if shots > 0 and fewshot_examples:
        for ex in fewshot_examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": item["input"]})
    return messages


def run_experiment(model_key, condition, shots, limit=None, delay=0.5):
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    model_id = MODEL_MAP[model_key]

    test_data = load_data(condition, "test")
    if limit:
        test_data = test_data[:limit]

    fewshot_examples = None
    if shots > 0:
        train_data = load_data(condition, "train")
        fewshot_examples = select_fewshot_examples(train_data, n=shots)
        print(f"Few-shot examples selected (labels): {[e['output'] for e in fewshot_examples]}")

    predictions = []
    correct = 0
    tp = fp = fn = tn = 0
    errors = 0
    unparseables = 0

    total = len(test_data)
    for i, item in enumerate(test_data):
        messages = build_messages(item, shots, fewshot_examples)
        pred = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=10,
                    temperature=0,
                )
                raw = resp.choices[0].message.content
                pred = parse_verdict(raw)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    pred = "error"
                    print(f"  [{i+1}/{total}] API error after 3 retries: {e}")

        gold = item["output"]
        is_correct = pred == gold

        if pred == "unparseable":
            unparseables += 1
        elif pred == "error":
            errors += 1

        if is_correct:
            correct += 1
        if pred == "fail" and gold == "fail":
            tp += 1
        elif pred == "fail" and gold == "pass":
            fp += 1
        elif pred == "pass" and gold == "fail":
            fn += 1
        elif pred == "pass" and gold == "pass":
            tn += 1

        predictions.append({
            "problem_id": item["problem_id"],
            "gold": gold,
            "pred": pred,
            "raw": raw if pred != "error" else None,
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            acc_so_far = correct / (i + 1) * 100
            print(f"  [{i+1}/{total}] acc={acc_so_far:.2f}% errors={errors} unparseable={unparseables}")

        if delay > 0:
            time.sleep(delay)

    accuracy = correct / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": round(accuracy, 2),
        "precision_fail": round(precision, 2),
        "recall_fail": round(recall, 2),
        "f1_fail": round(f1, 2),
        "total": total,
        "correct": correct,
        "errors": errors,
        "unparseables": unparseables,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    exp_name = f"fewshot_{model_key}_{condition}_{shots}shot"
    result = {
        "experiment": exp_name,
        "model": model_id,
        "condition": condition,
        "shots": shots,
        "metrics": metrics,
        "predictions": predictions,
    }

    out_path = Path(ARTIFACT_DIR) / f"{exp_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {exp_name} ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision(fail): {precision:.2f}%  Recall(fail): {recall:.2f}%  F1(fail): {f1:.2f}%")
    print(f"Errors: {errors}  Unparseable: {unparseables}")
    print(f"Saved to {out_path}")

    return exp_name, metrics


def generate_summary(all_results):
    lines = ["# Few-shot Baseline Summary\n"]
    lines.append("## Comparison with SFT Baselines\n")
    lines.append("| Experiment | Model | Condition | Shots | Accuracy | P(fail) | R(fail) | F1(fail) | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for exp_name, metrics in all_results:
        parts = exp_name.split("_")
        # fewshot_{model}_{condition}_{shots}shot
        model_key = parts[1]
        condition = "_".join(parts[2:-1])
        shots = parts[-1].replace("shot", "")
        model_id = MODEL_MAP.get(model_key, model_key)
        lines.append(
            f"| {exp_name} | {model_id} | {condition} | {shots} | "
            f"{metrics['accuracy']:.2f}% | {metrics['precision_fail']:.2f}% | "
            f"{metrics['recall_fail']:.2f}% | {metrics['f1_fail']:.2f}% | {metrics['errors']} |"
        )

    lines.append("\n## SFT Baselines (Qwen-7B)\n")
    lines.append("| Condition | Accuracy |")
    lines.append("|---|---|")
    lines.append("| exception_only | 99.83% |")
    lines.append("| no_trace | 75.13% |")
    lines.append("")

    summary_path = Path(ARTIFACT_DIR) / "fewshot_baseline_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--condition", required=True, choices=["exception_only", "no_trace"])
    parser.add_argument("--shots", type=int, required=True, choices=[0, 3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        all_results = []
        for model_key in ["gpt4o", "gpt41"]:
            for condition in ["exception_only", "no_trace"]:
                for shots in [0, 3]:
                    if condition == "no_trace" and shots == 3:
                        continue
                    exp_name = f"fewshot_{model_key}_{condition}_{shots}shot"
                    path = Path(ARTIFACT_DIR) / f"{exp_name}.json"
                    if path.exists():
                        with open(path) as f:
                            data = json.load(f)
                        all_results.append((exp_name, data["metrics"]))
        generate_summary(all_results)
    elif args.run_all:
        conditions = [
            ("gpt4o", "exception_only", 0),
            ("gpt4o", "exception_only", 3),
            ("gpt4o", "no_trace", 0),
            ("gpt41", "exception_only", 0),
            ("gpt41", "exception_only", 3),
            ("gpt41", "no_trace", 0),
        ]
        all_results = []
        for model_key, condition, shots in conditions:
            exp_name, metrics = run_experiment(model_key, condition, shots, limit=args.limit, delay=args.delay)
            all_results.append((exp_name, metrics))
        generate_summary(all_results)
    else:
        run_experiment(args.model, args.condition, args.shots, limit=args.limit, delay=args.delay)
