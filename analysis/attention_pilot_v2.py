"""Attention Heatmap Pilot v2: improved segment detection using decoded text positions."""
import json
import re
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

OUTPUT_DIR = "./docs/paper/figures/attention_pilot"
DATA_PATH = "./data/exception_only_test.jsonl"
BASE_MODEL = "./models/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "./checkpoints/exception_only/final"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def select_samples(n=10):
    with open(DATA_PATH) as f:
        lines = f.readlines()
    
    by_type = defaultdict(list)
    for i, line in enumerate(lines):
        s = json.loads(line)
        if s['output'] == 'fail':
            m = re.search(r'\[ERROR\]\s+(\w+Error|\w+Exception)', s['input'])
            if m:
                etype = m.group(1)
                by_type[etype].append(s)
    
    selected = []
    targets = [
        ('AssertionError', 4),
        ('TypeError', 3),
        ('NameError', 2),
        ('AttributeError', 1),
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    for etype, count in targets:
        if etype not in by_type:
            continue
        samples = by_type[etype]
        samples_with_len = [(s, len(tokenizer.encode(s['input']))) for s in samples]
        samples_with_len.sort(key=lambda x: x[1])
        for s, tlen in samples_with_len[:count]:
            selected.append((s, etype, tlen))
    
    return selected

def get_chat_input(tokenizer, text):
    messages = [
        {"role": "system", "content": "You are a code testing assistant. Analyze the code and test execution, then predict the verdict."},
        {"role": "user", "content": text + "\n\nPredict the verdict (pass or fail):"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def segment_tokens_precise(token_ids, tokenizer, raw_input_text):
    """Precise segment detection by decoding cumulative token prefixes."""
    # Decode each token individually
    token_strs = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
    
    # Build cumulative decoded text to find char positions
    cumulative = []
    running = ""
    for ts in token_strs:
        running += ts
        cumulative.append(len(running))
    
    # Find key boundaries in the CHAT-FORMATTED text
    full_text = running
    
    # Key positions in full_text
    trace_marker = full_text.find("Execution trace:")
    error_bracket = full_text.find("[ERROR]")
    test_code_pos = full_text.find("Test code:")
    locals_pos = full_text.find("| locals:")
    predict_pos = full_text.find("Predict the verdict")
    
    # Find error type text position (e.g., "AssertionError at line")
    error_type_match = re.search(r'\[ERROR\]\s+(\w+(?:Error|Exception))\s+at\s+line\s+\d+', full_text)
    error_type_end = error_type_match.end() if error_type_match else (error_bracket + 30 if error_bracket > 0 else -1)
    error_type_start = error_bracket + len("[ERROR] ") if error_bracket > 0 else -1
    
    segments = []
    for i, (ts, cum_pos) in enumerate(zip(token_strs, cumulative)):
        char_start = cumulative[i-1] if i > 0 else 0
        char_end = cum_pos
        
        # Determine segment based on character position
        if predict_pos > 0 and char_start >= predict_pos:
            segments.append('prompt_suffix')
        elif locals_pos > 0 and char_start >= locals_pos:
            segments.append('locals_dump')
        elif error_type_start > 0 and char_start >= error_type_start and char_end <= error_type_end:
            segments.append('error_type_line')
        elif error_bracket > 0 and char_start >= error_bracket and char_end <= error_bracket + len("[ERROR]") + 2:
            segments.append('error_marker')
        elif trace_marker > 0 and char_start >= trace_marker and (error_bracket < 0 or char_end <= error_bracket):
            segments.append('trace_header')
        elif test_code_pos > 0 and char_start >= test_code_pos and (trace_marker < 0 or char_end <= trace_marker):
            segments.append('test_code')
        elif test_code_pos > 0 and char_end <= test_code_pos:
            segments.append('source_code')
        elif trace_marker > 0 and char_end <= trace_marker and (test_code_pos < 0 or char_start >= test_code_pos):
            segments.append('test_code')
        else:
            # Fallback based on position
            if error_bracket > 0 and char_start >= error_bracket:
                segments.append('error_context')
            else:
                segments.append('other')
    
    return segments, token_strs

def extract_attention(model, tokenizer, text):
    chat_input = get_chat_input(tokenizer, text)
    inputs = tokenizer(chat_input, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    last_layer_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
    avg_attn = last_layer_attn.mean(dim=0)  # [seq_len, seq_len]
    last_token_attn = avg_attn[-1, :].cpu().numpy()
    
    token_ids = inputs['input_ids'][0].cpu().tolist()
    segments, token_strs = segment_tokens_precise(token_ids, tokenizer, text)
    
    return last_token_attn, token_strs, segments

def compute_segment_attention(attn_weights, segments):
    seg_attn = defaultdict(float)
    seg_count = defaultdict(int)
    for seg, w in zip(segments, attn_weights):
        seg_attn[seg] += w
        seg_count[seg] += 1
    return dict(seg_attn), dict(seg_count)

def plot_detailed_heatmap(attn_weights, token_strs, segments, sample_info, idx):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    
    seg_color_map = {
        'source_code': '#4CAF50',
        'test_code': '#2196F3',
        'trace_header': '#FF9800',
        'error_marker': '#F44336',
        'error_type_line': '#E91E63',
        'error_context': '#AB47BC',
        'locals_dump': '#7E57C2',
        'prompt_suffix': '#90A4AE',
        'other': '#757575',
    }
    
    # Aggregate into chunks of 8 tokens
    chunk_size = 8
    n_tokens = len(attn_weights)
    n_chunks = (n_tokens + chunk_size - 1) // chunk_size
    
    chunk_attn = []
    chunk_colors = []
    
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n_tokens)
        chunk_attn.append(attn_weights[start:end].sum())
        
        seg_counts = defaultdict(int)
        for s in segments[start:end]:
            seg_counts[s] += 1
        dominant = max(seg_counts, key=seg_counts.get)
        chunk_colors.append(seg_color_map.get(dominant, '#757575'))
    
    chunk_attn = np.array(chunk_attn)
    
    bars = ax.bar(range(n_chunks), chunk_attn, color=chunk_colors, width=0.9, edgecolor='white', linewidth=0.2)
    
    ax.set_xlabel('Token position (chunks of 8)', fontsize=9)
    ax.set_ylabel('Attention weight', fontsize=9)
    ax.set_title(f'Last-token attention: {sample_info["etype"]} ({sample_info["problem_id"]})', fontsize=10)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Source code'),
        Patch(facecolor='#2196F3', label='Test code'),
        Patch(facecolor='#FF9800', label='Trace header'),
        Patch(facecolor='#F44336', label='[ERROR]'),
        Patch(facecolor='#E91E63', label='Error type+line'),
        Patch(facecolor='#7E57C2', label='Locals dump'),
        Patch(facecolor='#90A4AE', label='Prompt suffix'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right', ncol=2)
    ax.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"attention_detail_{idx}_{sample_info['etype']}.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

def plot_summary(all_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: stacked bar chart of attention proportions
    categories = ['source_code', 'test_code', 'trace_header', 'error_marker', 'error_type_line', 'locals_dump', 'prompt_suffix']
    cat_labels = ['Source code', 'Test code', 'Trace header', '[ERROR]', 'Error type+line', 'Locals dump', 'Prompt suffix']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#E91E63', '#7E57C2', '#90A4AE']
    
    x = np.arange(len(all_results))
    bottoms = np.zeros(len(all_results))
    
    for cat, label, color in zip(categories, cat_labels, colors):
        values = []
        for r in all_results:
            seg_attn = r['segment_attn']
            total = sum(seg_attn.values())
            values.append(seg_attn.get(cat, 0) / total * 100 if total > 0 else 0)
        values = np.array(values)
        ax1.bar(x, values, bottom=bottoms, label=label, color=color, width=0.7)
        bottoms += values
    
    ax1.set_xlabel('Sample', fontsize=9)
    ax1.set_ylabel('Attention share (%)', fontsize=9)
    ax1.set_title('Attention by segment (stacked)', fontsize=10)
    xlabels = [f"{r['etype'][:6]}\n{r['problem_id'].split('/')[1]}" for r in all_results]
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=7)
    ax1.legend(fontsize=6, loc='upper left', ncol=1)
    ax1.set_ylim(0, 100)
    
    # Right: attention per token (normalized by segment size)
    focus_cats = ['source_code', 'test_code', 'error_marker', 'error_type_line', 'locals_dump']
    focus_labels = ['Source', 'Test', '[ERROR]', 'Error type', 'Locals']
    focus_colors = ['#4CAF50', '#2196F3', '#F44336', '#E91E63', '#7E57C2']
    
    width = 0.15
    for i, (cat, label, color) in enumerate(zip(focus_cats, focus_labels, focus_colors)):
        values = []
        for r in all_results:
            seg_attn = r['segment_attn']
            seg_count = r['segment_count']
            if seg_count.get(cat, 0) > 0:
                values.append(seg_attn.get(cat, 0) / seg_count[cat] * 1000)  # x1000 for readability
            else:
                values.append(0)
        ax2.bar(x + i * width - width * 2, values, width, label=label, color=color)
    
    ax2.set_xlabel('Sample', fontsize=9)
    ax2.set_ylabel('Attn per token (x1000)', fontsize=9)
    ax2.set_title('Attention density per token by segment', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=7)
    ax2.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "attention_summary_v2.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")
    
    print("Selecting samples...")
    samples = select_samples(n=10)
    print(f"Selected {len(samples)} samples")
    
    all_results = []
    
    for idx, (sample, etype, tlen) in enumerate(samples):
        print(f"\n[{idx+1}/{len(samples)}] {etype} ({sample['problem_id']}, {tlen} tok)...")
        
        attn_weights, token_strs, segments = extract_attention(model, tokenizer, sample['input'])
        seg_attn, seg_count = compute_segment_attention(attn_weights, segments)
        
        result = {
            'idx': idx,
            'etype': etype,
            'problem_id': sample['problem_id'],
            'n_tokens': len(token_strs),
            'segment_attn': {k: float(v) for k, v in seg_attn.items()},
            'segment_count': seg_count,
        }
        all_results.append(result)
        
        # Print breakdown
        total_attn = sum(seg_attn.values())
        print(f"  Breakdown (% attention / # tokens / attn-per-token):")
        for seg in ['source_code', 'test_code', 'trace_header', 'error_marker', 'error_type_line', 'locals_dump', 'prompt_suffix']:
            pct = seg_attn.get(seg, 0) / total_attn * 100
            cnt = seg_count.get(seg, 0)
            per_tok = seg_attn.get(seg, 0) / cnt * 1000 if cnt > 0 else 0
            print(f"    {seg:16s}: {pct:5.1f}% | {cnt:4d} tok | {per_tok:.2f} per-tok(x1k)")
        
        if idx < 3:
            plot_detailed_heatmap(attn_weights, token_strs, segments, result, idx)
    
    print("\nGenerating summary...")
    plot_summary(all_results)
    
    results_path = os.path.join(OUTPUT_DIR, "attention_pilot_results_v2.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {results_path}")
    print("Done!")

if __name__ == "__main__":
    main()
