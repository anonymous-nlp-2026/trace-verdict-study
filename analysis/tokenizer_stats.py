import json
import csv
import os
import numpy as np
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoTokenizer

DATA_DIR = Path("./data")
OUT_DIR = Path("./analysis")

CONDITIONS = ["vota", "no_trace", "exception_only", "full_trace", "ae_only", "loop_only"]
SPLITS = ["train", "test"]

MODELS = {
    "Qwen2.5-Coder-7B": "./models/Qwen2.5-Coder-7B-Instruct",
    "DeepSeek-Coder-6.7B": "~/.cache/huggingface/hub/deepseek-coder-6.7b-instruct",
}

def load_samples(condition, split):
    path = DATA_DIR / f"{condition}_{split}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def tokenize_with_chat_template(tokenizer, sample):
    messages = [
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer.encode(text)
    return len(tokens)

def compute_stats(token_counts, ctx_limit=2048):
    arr = np.array(token_counts)
    return {
        "mean": round(float(np.mean(arr)), 1),
        "median": round(float(np.median(arr)), 1),
        "max": int(np.max(arr)),
        "pct_over_2048": round(float(np.mean(arr > ctx_limit) * 100), 2),
        "n": len(arr),
    }

def main():
    print("Loading tokenizers (CPU only, local_files_only)...", flush=True)
    tokenizers = {}
    for name, path in MODELS.items():
        print(f"  Loading {name} from {path}...", flush=True)
        tokenizers[name] = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print("Tokenizers loaded.\n", flush=True)

    all_results = []

    for condition in CONDITIONS:
        for split in SPLITS:
            samples = load_samples(condition, split)
            for model_name, tok in tokenizers.items():
                counts = [tokenize_with_chat_template(tok, s) for s in samples]
                stats = compute_stats(counts)
                row = {
                    "condition": condition,
                    "split": split,
                    "model": model_name,
                    **stats,
                }
                all_results.append(row)
                print(f"  {condition}/{split}/{model_name}: mean={stats['mean']:.0f}, median={stats['median']:.0f}, max={stats['max']}, >2048={stats['pct_over_2048']:.1f}%", flush=True)

    print("\n" + "="*100, flush=True)
    print(f"{'Condition':<18} {'Split':<7} {'Model':<22} {'Mean':>7} {'Median':>7} {'Max':>7} {'>2048%':>8} {'N':>6}", flush=True)
    print("-"*100, flush=True)
    for r in all_results:
        print(f"{r['condition']:<18} {r['split']:<7} {r['model']:<22} {r['mean']:>7.0f} {r['median']:>7.0f} {r['max']:>7} {r['pct_over_2048']:>7.1f}% {r['n']:>6}", flush=True)
    print("="*100, flush=True)

    csv_path = OUT_DIR / "tokenizer_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "split", "model", "mean", "median", "max", "pct_over_2048", "n"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved to {csv_path}", flush=True)

    json_path = OUT_DIR / "tokenizer_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved to {json_path}", flush=True)

if __name__ == "__main__":
    main()
