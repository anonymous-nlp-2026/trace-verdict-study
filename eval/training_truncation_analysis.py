"""Analyze DeepSeek VOTA accuracy grouped by training truncation status."""

import json
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

MODEL_PATH = "./models/deepseek-coder-6.7b-instruct"
TEST_DATA = "./data/vota_test.jsonl"
PREDICTIONS = "./checkpoints/deepseek_vota/final/predictions.jsonl"
MAX_LENGTH = 2048

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading test data...")
with open(TEST_DATA) as f:
    test_data = [json.loads(line) for line in f if line.strip()]

print("Loading predictions...")
with open(PREDICTIONS) as f:
    preds = [json.loads(line) for line in f if line.strip()]

assert len(test_data) == len(preds) == 1190

# Compute token lengths using the same logic as to_chat_text
print("Computing token lengths...")
token_lengths = []
overhead_lengths = []
content_lengths = []
for item in test_data:
    overhead_msg = tok.apply_chat_template(
        [{"role": "user", "content": ""},
         {"role": "assistant", "content": item["output"]}],
        tokenize=False, add_generation_prompt=False,
    )
    overhead_len = len(tok(overhead_msg, add_special_tokens=False).input_ids)
    content_len = len(tok(item["input"], add_special_tokens=False).input_ids)
    total_len = overhead_len + content_len
    
    overhead_lengths.append(overhead_len)
    content_lengths.append(content_len)
    token_lengths.append(total_len)

token_lengths = np.array(token_lengths)
content_lengths = np.array(content_lengths)
overhead_lengths = np.array(overhead_lengths)

# Determine truncation groups
budgets = MAX_LENGTH - overhead_lengths
truncated_mask = content_lengths > budgets  # True = truncated during training
intact_mask = ~truncated_mask

n_truncated = truncated_mask.sum()
n_intact = intact_mask.sum()

print(f"\n=== Token Length Statistics ===")
print(f"Total samples: {len(token_lengths)}")
print(f"Mean total tokens: {token_lengths.mean():.0f}")
print(f"Median total tokens: {np.median(token_lengths):.0f}")
print(f"Min / Max: {token_lengths.min()} / {token_lengths.max()}")
print(f"Overhead (chat template): {overhead_lengths[0]} tokens")
print(f"Content budget at max_length={MAX_LENGTH}: {budgets[0]} tokens")
print(f"Train-truncated (content > budget): {n_truncated} ({n_truncated/len(token_lengths)*100:.1f}%)")
print(f"Train-intact (content <= budget): {n_intact} ({n_intact/len(token_lengths)*100:.1f}%)")

# Get correctness arrays
correct = np.array([p["label"] == p["pred"] for p in preds])
overall_acc = correct.mean()
print(f"\nOverall accuracy: {overall_acc*100:.2f}% ({correct.sum()}/{len(correct)})")

# Group accuracies
trunc_correct = correct[truncated_mask]
intact_correct = correct[intact_mask]

trunc_acc = trunc_correct.mean()
intact_acc = intact_correct.mean()

print(f"\n=== DeepSeek VOTA: Training Truncation Group Analysis ===")
print(f"(Inference without truncation, i.e. original {overall_acc*100:.2f}% eval)")

# Bootstrap CIs
def bootstrap_ci(arr, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    accs = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        accs[i] = sample.mean()
    return np.percentile(accs, [2.5, 97.5])

trunc_ci = bootstrap_ci(trunc_correct)
intact_ci = bootstrap_ci(intact_correct)

print(f"\nGroup A (train-truncated, content>{budgets[0]} tok): N={n_truncated}, "
      f"accuracy={trunc_acc*100:.2f}%, 95% CI [{trunc_ci[0]*100:.2f}%, {trunc_ci[1]*100:.2f}%]")
print(f"Group B (train-intact, content<={budgets[0]} tok):   N={n_intact}, "
      f"accuracy={intact_acc*100:.2f}%, 95% CI [{intact_ci[0]*100:.2f}%, {intact_ci[1]*100:.2f}%]")
print(f"Difference: {(intact_acc - trunc_acc)*100:.2f}pp")

# Fisher exact test
trunc_correct_n = trunc_correct.sum().astype(int)
trunc_incorrect_n = n_truncated - trunc_correct_n
intact_correct_n = intact_correct.sum().astype(int)
intact_incorrect_n = n_intact - intact_correct_n

contingency = [[trunc_correct_n, trunc_incorrect_n],
               [intact_correct_n, intact_incorrect_n]]
odds_ratio, p_value = stats.fisher_exact(contingency)

print(f"\nFisher exact test:")
print(f"  Contingency table:")
print(f"    Truncated:  {trunc_correct_n} correct, {trunc_incorrect_n} incorrect")
print(f"    Intact:     {intact_correct_n} correct, {intact_incorrect_n} incorrect")
print(f"  Odds ratio: {odds_ratio:.4f}")
print(f"  p-value: {p_value:.6f}")

# Additional: per-class breakdown
print(f"\n=== Per-class breakdown ===")
for group_name, mask in [("Truncated", truncated_mask), ("Intact", intact_mask)]:
    group_labels = np.array([preds[i]["label"] for i in range(len(preds)) if mask[i]])
    group_preds = np.array([preds[i]["pred"] for i in range(len(preds)) if mask[i]])
    group_correct_arr = np.array([preds[i]["label"] == preds[i]["pred"] for i in range(len(preds)) if mask[i]])
    
    for cls in ["pass", "fail"]:
        cls_mask = group_labels == cls
        if cls_mask.sum() == 0:
            print(f"  {group_name} - {cls}: N=0")
            continue
        cls_correct = group_correct_arr[cls_mask]
        cls_acc = cls_correct.mean()
        print(f"  {group_name} - {cls}: N={cls_mask.sum()}, accuracy={cls_acc*100:.2f}%")

# Output as JSON for registry update
result = {
    "train_truncated": {
        "n": int(n_truncated),
        "accuracy": round(float(trunc_acc), 4),
        "ci_95": [round(float(trunc_ci[0]), 4), round(float(trunc_ci[1]), 4)],
        "correct": int(trunc_correct_n),
        "incorrect": int(trunc_incorrect_n)
    },
    "train_intact": {
        "n": int(n_intact),
        "accuracy": round(float(intact_acc), 4),
        "ci_95": [round(float(intact_ci[0]), 4), round(float(intact_ci[1]), 4)],
        "correct": int(intact_correct_n),
        "incorrect": int(intact_incorrect_n)
    },
    "fisher_exact": {
        "odds_ratio": round(float(odds_ratio), 4),
        "p_value": round(float(p_value), 6)
    },
    "overall_accuracy": round(float(overall_acc), 4),
    "max_length": MAX_LENGTH,
    "note": "Grouped by whether training sample content exceeded token budget (DeepSeek tokenizer, max_length=2048). Inference used full-length input (no truncation)."
}
print(f"\n=== JSON for registry ===")
print(json.dumps(result, indent=2))
