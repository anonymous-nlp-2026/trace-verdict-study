"""Calibration analysis: ECE and Brier Score for exception_only models."""

import json
import os
import gc

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    {
        "name": "Qwen2.5-7B",
        "base_model": "./models/Qwen2.5-Coder-7B-Instruct",
        "adapter": "./checkpoints/exception_only/final/",
    },
    {
        "name": "DeepSeek-Coder-7B",
        "base_model": "./models/deepseek-coder-6.7b-instruct",
        "adapter": "./checkpoints/deepseek_exception_only/final/",
    },
    {
        "name": "CodeLlama-13B",
        "base_model": "./models/CodeLlama-13b-Instruct-hf",
        "adapter": "./checkpoints/codellama13b_exception_only_seed42/final/",
    },
    {
        "name": "Qwen2.5-1.5B",
        "base_model": "./models/Qwen2.5-Coder-1.5B-Instruct",
        "adapter": "./checkpoints/qwen15b_exception_only_seed42/final/",
    },
]

DATA_PATH = "./data/exception_only_test.jsonl"
OUTPUT_PATH = "./outputs/calibration_results.json"
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
    pass_tokens = tokenizer.encode("pass", add_special_tokens=False)
    fail_tokens = tokenizer.encode("fail", add_special_tokens=False)
    return pass_tokens[0], fail_tokens[0]


def compute_ece(confidences, correctness, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        if i == 0:
            mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            ece += (n_bin / total) * abs(avg_conf - avg_acc)
    return ece


def compute_brier_score(probs_positive, labels_binary):
    return float(np.mean((probs_positive - labels_binary) ** 2))


def evaluate_model(model_config, test_data, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['name']}")
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pass_id, fail_id = get_label_token_ids(tokenizer)
    print(f"Token IDs - pass: {pass_id} ('{tokenizer.decode([pass_id])}')")
    print(f"Token IDs - fail: {fail_id} ('{tokenizer.decode([fail_id])}')", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(model, model_config["adapter"])
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
            content_ids = tokenizer(input_text, add_special_tokens=False).input_ids
            content_ids = content_ids[:content_budget]
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

        if (i + 1) % 100 == 0:
            correct_so_far = sum(
                1 for j in range(len(all_preds))
                if all_preds[j] == ("pass" if all_labels_binary[j] == 1 else "fail")
            )
            acc = correct_so_far / len(all_preds)
            print(f"  [{i+1}/{len(test_data)}] running accuracy: {acc:.4f}", flush=True)

    probs = np.array(all_probs_pass)
    labels = np.array(all_labels_binary)
    confidences = np.array(all_confidences)
    correct = np.array([
        1 if all_preds[j] == ("pass" if all_labels_binary[j] == 1 else "fail") else 0
        for j in range(len(all_preds))
    ])

    accuracy = correct.mean()
    ece = compute_ece(confidences, correct)
    brier = compute_brier_score(probs, labels)

    print(f"\nResults for {model_config['name']}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Mean confidence: {confidences.mean():.4f}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": model_config["name"],
        "accuracy": float(accuracy),
        "ece": float(ece),
        "brier_score": float(brier),
        "mean_confidence": float(confidences.mean()),
        "n_samples": len(test_data),
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0")

    test_data = load_test_data(DATA_PATH)
    print(f"Loaded {len(test_data)} test samples")

    results = []
    for model_config in MODELS:
        result = evaluate_model(model_config, test_data, device)
        results.append(result)

    output = {"condition": "exception_only", "models": results}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':>10} {'ECE':>10} {'Brier':>10} {'Mean Conf':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['ece']:>10.4f} {r['brier_score']:>10.4f} {r['mean_confidence']:>10.4f}")

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
