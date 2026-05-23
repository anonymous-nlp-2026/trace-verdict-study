"""Pad exception_only data to 2048 tokens using .\n padding (1 token per unit, round-trip stable)."""

import json
from transformers import AutoTokenizer

MODEL_PATH = "./models/Qwen2.5-Coder-7B-Instruct"
DATA_DIR = "./data"
TARGET_TOKENS = 2048
PAD_UNIT = ".\n"  # 1 token, round-trip stable in Qwen tokenizer

def pad_to_target(text: str, tokenizer, target: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) >= target:
        return text
    pad_count = target - len(ids)
    padded = text + PAD_UNIT * pad_count
    # Verify and adjust
    padded_ids = tokenizer(padded, add_special_tokens=False).input_ids
    if len(padded_ids) > target:
        padded_ids = padded_ids[:target]
        padded = tokenizer.decode(padded_ids, skip_special_tokens=False)
    return padded

def process_file(in_path: str, out_path: str, tokenizer):
    items = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    orig_lens = []
    padded_lens = []
    for item in items:
        orig_ids = tokenizer(item["input"], add_special_tokens=False).input_ids
        orig_lens.append(len(orig_ids))
        item["input"] = pad_to_target(item["input"], tokenizer, TARGET_TOKENS)
        new_ids = tokenizer(item["input"], add_special_tokens=False).input_ids
        padded_lens.append(len(new_ids))

    with open(out_path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return orig_lens, padded_lens

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    for split in ["train", "test"]:
        in_path = f"{DATA_DIR}/exception_only_{split}.jsonl"
        out_path = f"{DATA_DIR}/exception_only_padded2048_{split}.jsonl"
        print(f"\nProcessing {split}...")
        orig, padded = process_file(in_path, out_path, tokenizer)
        print(f"  Count: {len(orig)}")
        print(f"  Original tokens: mean={sum(orig)/len(orig):.1f}, min={min(orig)}, max={max(orig)}")
        print(f"  Padded tokens:   mean={sum(padded)/len(padded):.1f}, min={min(padded)}, max={max(padded)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
