"""Batch evaluate all intermediate checkpoints for training dynamics analysis.

Loads the base model once, swaps LoRA adapters per checkpoint, runs full
test-set evaluation, and writes results to a JSONL file.
"""

import argparse
import importlib.util
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

eval_spec = importlib.util.spec_from_file_location(
    "eval_module", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")
)
eval_module = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(eval_module)
load_test_data = eval_module.load_test_data
predict = eval_module.predict
classify_prediction = eval_module.classify_prediction
compute_metrics = eval_module.compute_metrics


def extract_train_loss(ckpt_path):
    """Extract the most recent train loss from trainer_state.json."""
    state_path = os.path.join(ckpt_path, "trainer_state.json")
    if not os.path.isfile(state_path):
        return None
    with open(state_path) as f:
        state = json.load(f)
    log_history = state.get("log_history", [])
    for entry in reversed(log_history):
        if "loss" in entry:
            return entry["loss"]
    return None


def find_checkpoints(root_dir):
    """Find all checkpoint-N directories, sorted by step number."""
    ckpts = []
    for d in os.listdir(root_dir):
        m = re.match(r"checkpoint-(\d+)$", d)
        if m:
            step = int(m.group(1))
            ckpts.append((step, os.path.join(root_dir, d)))
    ckpts.sort()
    final_dir = os.path.join(root_dir, "final")
    if os.path.isdir(final_dir):
        ckpts.append((-1, final_dir))
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model_path",
                        default="./models/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--data_dir",
                        default="./data/")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--truncation_side", default="right",
                        choices=["left", "right"])
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.checkpoint_root, "dynamics_log.jsonl")

    ckpts = find_checkpoints(args.checkpoint_root)
    if not ckpts:
        print("No checkpoints found!")
        return

    max_step = max(s for s, _ in ckpts if s > 0) if any(s > 0 for s, _ in ckpts) else 0
    ckpts = [(s if s > 0 else max_step, p) for s, p in ckpts]
    ckpts.sort()
    seen = set()
    deduped = []
    for s, p in ckpts:
        if s not in seen:
            seen.add(s)
            deduped.append((s, p))
    ckpts = deduped

    print(f"Found {len(ckpts)} checkpoints: {[s for s, _ in ckpts]}")

    test_data = load_test_data(args.data_dir, args.condition)
    print(f"Test samples: {len(test_data)}")

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )

    with open(args.output, "w") as fout:
        for step, ckpt_path in ckpts:
            print(f"\n=== Step {step}: {ckpt_path} ===")

            train_loss = extract_train_loss(ckpt_path)
            if train_loss is not None:
                print(f"  Train loss: {train_loss:.4f}")

            model = PeftModel.from_pretrained(base_model, ckpt_path)
            model.eval()

            labels, preds = [], []
            for i, item in enumerate(test_data):
                raw_pred = predict(model, tokenizer, item["input"], device,
                                   max_length=args.max_length,
                                   truncation_side=args.truncation_side)
                pred = classify_prediction(raw_pred)
                labels.append(item["output"])
                preds.append(pred)
                if (i + 1) % 200 == 0:
                    acc = sum(1 for l, p in zip(labels, preds) if l == p) / len(labels)
                    print(f"  [{i+1}/{len(test_data)}] running acc: {acc:.4f}")

            metrics = compute_metrics(labels, preds)
            entry = {
                "step": step,
                "checkpoint": ckpt_path,
                "train_loss": train_loss,
                "accuracy": metrics["accuracy"],
                "pass_f1": metrics["pass_f1"],
                "fail_f1": metrics["fail_f1"],
                "pass_precision": metrics["pass_precision"],
                "pass_recall": metrics["pass_recall"],
                "fail_precision": metrics["fail_precision"],
                "fail_recall": metrics["fail_recall"],
            }
            fout.write(json.dumps(entry) + "\n")
            fout.flush()

            print(f"  Accuracy: {metrics['accuracy']:.4f}  "
                  f"Pass F1: {metrics['pass_f1']:.4f}  "
                  f"Fail F1: {metrics['fail_f1']:.4f}")

            base_model = model.unload()
            del model
            torch.cuda.empty_cache()

    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
