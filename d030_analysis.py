import json, numpy as np, sys, os
from transformers import AutoTokenizer

MODEL_PATH = "./models/deepseek-coder-6.7b-instruct"
DATA = "./data"
OUT = "./artifacts/d030_token_analysis.txt"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("Loading DeepSeek tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
log(f"Tokenizer: {MODEL_PATH}")
log(f"Vocab size: {tok.vocab_size}")
log()

def analyze(path, name):
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    overhead_msg = tok.apply_chat_template(
        [{"role": "user", "content": ""},
         {"role": "assistant", "content": "pass"}],
        tokenize=False, add_generation_prompt=False,
    )
    overhead_len = len(tok(overhead_msg, add_special_tokens=False).input_ids)
    
    raw_lengths = []
    input_lengths = []
    for item in data:
        messages = [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tok.encode(text)
        raw_lengths.append(len(tokens))
        input_ids = tok(item["input"], add_special_tokens=False).input_ids
        input_lengths.append(len(input_ids))
    
    raw_lengths = np.array(raw_lengths)
    input_lengths = np.array(input_lengths)
    
    log(f"=== {name} (DeepSeek tokenizer) ===")
    log(f"Count: {len(raw_lengths)}")
    log(f"--- Full sequence (chat format, no truncation) ---")
    log(f"Mean: {raw_lengths.mean():.0f}, Median: {np.median(raw_lengths):.0f}")
    log(f"Min: {raw_lengths.min()}, Max: {raw_lengths.max()}")
    log(f"P90: {np.percentile(raw_lengths, 90):.0f}, P95: {np.percentile(raw_lengths, 95):.0f}, P99: {np.percentile(raw_lengths, 99):.0f}")
    log(f">2048: {(raw_lengths > 2048).sum()}/{len(raw_lengths)} ({(raw_lengths > 2048).mean()*100:.1f}%)")
    log(f">1024: {(raw_lengths > 1024).sum()}/{len(raw_lengths)} ({(raw_lengths > 1024).mean()*100:.1f}%)")
    log(f">512:  {(raw_lengths > 512).sum()}/{len(raw_lengths)} ({(raw_lengths > 512).mean()*100:.1f}%)")
    log(f"--- Input only ---")
    log(f"Mean: {input_lengths.mean():.0f}, Median: {np.median(input_lengths):.0f}, Max: {input_lengths.max()}")
    log(f"Chat template overhead: {overhead_len} tokens")
    log()
    return raw_lengths

conditions = [
    (f"{DATA}/vota_train.jsonl", "VOTA"),
    (f"{DATA}/exception_only_train.jsonl", "exception_only"),
    (f"{DATA}/no_trace_train.jsonl", "no_trace"),
    (f"{DATA}/full_trace_train.jsonl", "full_trace"),
]

results = {}
for path, name in conditions:
    if os.path.exists(path):
        results[name] = analyze(path, name)
    else:
        log(f"SKIP {name}: {path} not found")

log("=== Cross-condition comparison ===")
for name, lengths in results.items():
    pct = (lengths > 2048).mean() * 100
    log(f"  {name:20s}: mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, max={lengths.max()}, >2048={pct:.1f}%")
log()

# Check to_chat_text status
log("=== to_chat_text() label-preserving truncation status ===")
log("train_sft.py line 30-55: to_chat_text() EXISTS with label-preserving truncation")
log("  - Truncates input content, preserves assistant label ('pass'/'fail')")
log("  - max_length=2048 hardcoded at call sites (lines 164-165)")
log("  - SFTConfig also sets max_length=2048 (line ~193)")
log()

# Conclusion
vota_l = results.get("VOTA")
eo_l = results.get("exception_only")
if vota_l is not None:
    vota_over = (vota_l > 2048).mean() * 100
    log("=== CONCLUSION ===")
    if vota_over > 50:
        log(f"CONFIRMED: {vota_over:.1f}% of VOTA samples exceed 2048 tokens.")
        log("Without to_chat_text() label-preserving truncation, SFTTrainer's right-truncation")
        log("would have cut off the assistant label for the majority of VOTA training samples.")
        log("This is the root cause of garbage predictions in deepseek_vota_seed42.")
        log()
        log("HOWEVER: Current train_sft.py has to_chat_text() which truncates input content")
        log("to preserve labels. If deepseek_vota_seed42 was trained with the current code,")
        log("labels should have been preserved. The issue would only occur if:")
        log("  1. The run used an older version without to_chat_text()")
        log("  2. The run used --label_only_loss (to_prompt_completion has no truncation)")
        log("  3. to_chat_text was added AFTER the deepseek_vota run as a fix")
    else:
        log(f"NOT confirmed as root cause: only {vota_over:.1f}% of VOTA exceed 2048.")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
log(f"\nSaved to {OUT}")
