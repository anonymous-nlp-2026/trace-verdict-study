"""Cross-model calibration analysis (plan_019).

Runs 4 exception_only models, computes ECE/MCE/Brier, generates reliability diagrams.
"""

import json
import os
import gc
import sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

MODELS = [
    {
        "name": "Qwen2.5-7B",
        "base_model": "./models/Qwen2.5-Coder-7B-Instruct",
        "adapter": "./checkpoints/exception_only/final/",
    },
    {
        "name": "DeepSeek-6.7B",
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
ARTIFACT_DIR = "./artifacts"
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


def compute_calibration_metrics(confidences, correctness, n_bins=10):
    """Compute ECE, MCE, and per-bin reliability data. Bins span [0.5, 1.0]."""
    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(confidences)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(mask.sum())
        if count > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correctness[mask].mean())
            gap = abs(avg_acc - avg_conf)
            ece += (count / total) * gap
            mce = max(mce, gap)
        else:
            avg_conf = (lo + hi) / 2
            avg_acc = 0.0
            gap = 0.0

        bins.append({
            "bin_lower": round(lo, 4),
            "bin_upper": round(hi, 4),
            "count": count,
            "mean_confidence": round(avg_conf, 6),
            "accuracy": round(avg_acc, 6),
            "gap": round(gap, 6),
        })

    return float(ece), float(mce), bins


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
    print(f"Token IDs - pass: {pass_id}, fail: {fail_id}", flush=True)

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
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"[INST] {input_text} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if inputs["input_ids"].shape[1] > MAX_LENGTH:
            overhead_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True
            ) if hasattr(tokenizer, 'apply_chat_template') else "[INST]  [/INST]"
            overhead_len = len(tokenizer(overhead_prompt).input_ids)
            content_budget = MAX_LENGTH - overhead_len
            content_ids = tokenizer(input_text, add_special_tokens=False).input_ids
            content_ids = content_ids[:content_budget]
            truncated_text = tokenizer.decode(content_ids, skip_special_tokens=False)
            messages = [{"role": "user", "content": truncated_text}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"[INST] {truncated_text} [/INST]"
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
            acc = correct_so_far / len(all_preds)
            print(f"  [{i+1}/{len(test_data)}] acc: {acc:.4f}", flush=True)

    probs = np.array(all_probs_pass)
    labels = np.array(all_labels_binary)
    confidences = np.array(all_confidences)
    correct = np.array([
        1 if all_preds[j] == ("pass" if all_labels_binary[j] == 1 else "fail") else 0
        for j in range(len(all_preds))
    ])

    accuracy = correct.mean()
    ece, mce, bins = compute_calibration_metrics(confidences, correct)
    brier = compute_brier_score(probs, labels)

    print(f"\n{model_config['name']}: Acc={accuracy:.4f} ECE={ece:.6f} MCE={mce:.6f} Brier={brier:.6f}")
    sys.stdout.flush()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": model_config["name"],
        "accuracy": float(accuracy),
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "mean_confidence": float(confidences.mean()),
        "n_samples": len(test_data),
        "reliability_bins": bins,
        "confidence_histogram": {
            "min": float(confidences.min()),
            "max": float(confidences.max()),
            "median": float(np.median(confidences)),
            "p5": float(np.percentile(confidences, 5)),
            "p95": float(np.percentile(confidences, 95)),
        },
    }


def plot_reliability_diagrams(all_results, output_path):
    """4-panel reliability diagram + summary bar chart."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for idx, result in enumerate(all_results):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])

        bins = result["reliability_bins"]
        bin_mids = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        confs = [b["mean_confidence"] for b in bins]
        counts = [b["count"] for b in bins]

        bar_width = 0.05
        ax.bar(bin_mids, accs, width=bar_width, alpha=0.7, color=colors[idx],
               label="Accuracy", edgecolor="white", linewidth=0.5)
        ax.plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, linewidth=1, label="Perfect")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.05)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{result['name']}\nECE={result['ece']:.4f}  MCE={result['mce']:.4f}",
                      fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")

        ax2 = ax.twinx()
        ax2.bar(bin_mids, counts, width=bar_width, alpha=0.15, color="gray")
        ax2.set_ylabel("Count", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    ax_summary = fig.add_subplot(gs[:, 2])
    names = [r["name"] for r in all_results]
    eces = [r["ece"] for r in all_results]
    briers = [r["brier_score"] for r in all_results]

    x = np.arange(len(names))
    w = 0.35
    ax_summary.barh(x - w/2, eces, w, label="ECE", color="#2196F3", alpha=0.8)
    ax_summary.barh(x + w/2, briers, w, label="Brier", color="#FF9800", alpha=0.8)
    ax_summary.set_yticks(x)
    ax_summary.set_yticklabels(names, fontsize=10)
    ax_summary.set_xlabel("Score (lower = better)", fontsize=10)
    ax_summary.set_title("Cross-Model Comparison", fontsize=12, fontweight="bold")
    ax_summary.legend(fontsize=9)
    ax_summary.invert_yaxis()

    for i, (e, b) in enumerate(zip(eces, briers)):
        ax_summary.text(max(e, b) + 0.0003, i, f"ECE={e:.4f}\nBrier={b:.4f}", va="center", fontsize=8)

    fig.suptitle("Cross-Model Calibration: Exception-Only Condition", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved: {output_path}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0")

    test_data = load_test_data(DATA_PATH)
    print(f"Loaded {len(test_data)} test samples")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    all_results = []
    for model_config in MODELS:
        result = evaluate_model(model_config, test_data, device)
        all_results.append(result)

    output = {"condition": "exception_only", "plan": "plan_019", "models": all_results}

    json_path = os.path.join(ARTIFACT_DIR, "cross_model_calibration_plan019.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    png_path = os.path.join(ARTIFACT_DIR, "cross_model_calibration_plan019.png")
    plot_reliability_diagrams(all_results, png_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Acc':>8} {'ECE':>10} {'MCE':>10} {'Brier':>10} {'MeanConf':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['name']:<20} {r['accuracy']:>8.4f} {r['ece']:>10.6f} {r['mce']:>10.6f} "
              f"{r['brier_score']:>10.6f} {r['mean_confidence']:>10.6f}")


if __name__ == "__main__":
    main()
