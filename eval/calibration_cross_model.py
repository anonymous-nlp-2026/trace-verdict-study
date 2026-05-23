"""Cross-model calibration analysis for exception_only condition.
Computes ECE, MCE, Brier score for 4 models x exception_only.
"""
import json
import os
import gc
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "Qwen-7B": {
        "model_path": "./models/Qwen2.5-Coder-7B-Instruct",
        "checkpoint": "./checkpoints/exception_only/final",
    },
    "DeepSeek-6.7B": {
        "model_path": "./models/deepseek-coder-6.7b-instruct",
        "checkpoint": "./checkpoints/deepseek_exception_only/final",
    },
    "Qwen-1.5B": {
        "model_path": "./models/Qwen2.5-Coder-1.5B-Instruct",
        "checkpoint": "./checkpoints/qwen15b_exception_only_seed42/final",
    },
    "CodeLlama-13B": {
        "model_path": "./models/CodeLlama-13b-Instruct-hf",
        "checkpoint": "./checkpoints/codellama13b_exception_only_seed42/final",
    },
}

TEST_DATA = "./data/exception_only_test.jsonl"
JSON_OUT = "./artifacts/cross_model_calibration_plan019.json"
MD_OUT = "./artifacts/cross_model_calibration_plan019.md"
MAX_LENGTH = 2048


def load_test_data(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_label_token_ids(tokenizer):
    pass_ids = tokenizer.encode("pass", add_special_tokens=False)
    fail_ids = tokenizer.encode("fail", add_special_tokens=False)
    return pass_ids[0], fail_ids[0]


def compute_calibration(confidences, correctness, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(confidences)
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            gap = abs(avg_conf - avg_acc)
            ece += (n_bin / total) * gap
            mce = max(mce, gap)
            bin_details.append({
                "bin": f"({lo:.1f}, {hi:.1f}]",
                "n": int(n_bin),
                "avg_conf": round(float(avg_conf), 4),
                "avg_acc": round(float(avg_acc), 4),
                "gap": round(float(gap), 4),
            })
        else:
            bin_details.append({
                "bin": f"({lo:.1f}, {hi:.1f}]",
                "n": 0,
            })
    return float(ece), float(mce), bin_details


def compute_brier(probs_positive, labels_binary):
    return float(np.mean((probs_positive - labels_binary) ** 2))


def evaluate_model(name, config, test_data, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pass_id, fail_id = get_label_token_ids(tokenizer)
    print(f"Token IDs - pass: {pass_id}, fail: {fail_id}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(model, config["checkpoint"])
    model.eval()

    all_probs_pass = []
    all_labels_binary = []
    all_preds = []
    all_confidences = []

    for i, item in enumerate(test_data):
        input_text = item["input"]
        label = item["output"]

        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if inputs["input_ids"].shape[1] > MAX_LENGTH:
            overhead_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
            )
            overhead_len = len(tokenizer(overhead_prompt).input_ids)
            content_budget = MAX_LENGTH - overhead_len
            content_ids = tokenizer(input_text, add_special_tokens=False).input_ids[:content_budget]
            truncated_text = tokenizer.decode(content_ids, skip_special_tokens=False)
            messages = [{"role": "user", "content": truncated_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_logits = outputs.logits[0, -1, :]
        pass_logit = last_logits[pass_id].float()
        fail_logit = last_logits[fail_id].float()

        logits_pf = torch.stack([pass_logit, fail_logit])
        probs_pf = torch.softmax(logits_pf, dim=0)
        p_pass = probs_pf[0].item()
        p_fail = probs_pf[1].item()

        pred = "pass" if p_pass > p_fail else "fail"
        confidence = max(p_pass, p_fail)

        all_probs_pass.append(p_pass)
        all_labels_binary.append(1 if label == "pass" else 0)
        all_preds.append(pred)
        all_confidences.append(confidence)

        if (i + 1) % 200 == 0:
            correct_so_far = sum(
                1 for j in range(len(all_preds))
                if all_preds[j] == ("pass" if all_labels_binary[j] == 1 else "fail")
            )
            print(f"  [{i+1}/{len(test_data)}] acc: {correct_so_far/len(all_preds):.4f}", flush=True)

    probs = np.array(all_probs_pass)
    labels = np.array(all_labels_binary)
    confidences = np.array(all_confidences)
    correct = np.array([
        1 if all_preds[j] == ("pass" if all_labels_binary[j] == 1 else "fail") else 0
        for j in range(len(all_preds))
    ])

    accuracy = correct.mean()
    ece, mce, bins = compute_calibration(confidences, correct)
    brier = compute_brier(probs, labels)

    print(f"\n{name}: acc={accuracy:.4f}, ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier:.4f}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "accuracy": round(float(accuracy), 4),
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "brier_score": round(float(brier), 6),
        "mean_confidence": round(float(confidences.mean()), 4),
        "n_samples": len(test_data),
        "bins": bins,
    }


def generate_markdown(results):
    lines = ["# Cross-Model Calibration Report (plan_019)", ""]
    lines.append("## Condition: exception_only")
    lines.append(f"N = {results[0]['n_samples']} test samples")
    lines.append("")
    lines.append("| Model | Accuracy | ECE | MCE | Brier | Mean Conf |")
    lines.append("|-------|----------|-----|-----|-------|-----------|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['ece']:.4f} | {r['mce']:.4f} | {r['brier_score']:.4f} | {r['mean_confidence']:.4f} |"
        )
    lines.append("")
    lines.append("## Bin Details")
    for r in results:
        lines.append(f"\n### {r['name']}")
        lines.append("| Bin | N | Avg Conf | Avg Acc | Gap |")
        lines.append("|-----|---|----------|---------|-----|")
        for b in r["bins"]:
            if b["n"] > 0:
                lines.append(f"| {b['bin']} | {b['n']} | {b['avg_conf']:.4f} | {b['avg_acc']:.4f} | {b['gap']:.4f} |")
            else:
                lines.append(f"| {b['bin']} | 0 | - | - | - |")
    return "\n".join(lines)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")

    test_data = load_test_data(TEST_DATA)
    print(f"Loaded {len(test_data)} test samples")

    results = []
    for name, config in MODELS.items():
        result = evaluate_model(name, config, test_data, device)
        results.append(result)

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    output = {"condition": "exception_only", "models": results}
    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved to {JSON_OUT}")

    md = generate_markdown(results)
    with open(MD_OUT, "w") as f:
        f.write(md)
    print(f"Markdown saved to {MD_OUT}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<16} {'Acc':>8} {'ECE':>8} {'MCE':>8} {'Brier':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['name']:<16} {r['accuracy']:>8.4f} {r['ece']:>8.4f} {r['mce']:>8.4f} {r['brier_score']:>8.4f}")


if __name__ == "__main__":
    main()
