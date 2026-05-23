"""Attention Heatmap Pilot: extract and visualize attention from exception_only checkpoint."""
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
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",  # need full attention weights, not flash
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
    
    # Pick diverse samples, prefer shorter ones
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
        # Sort by token length
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
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted

def segment_tokens(tokens, input_text):
    """Identify segments: code, test, trace_header, error_type, error_msg, other."""
    segments = []
    text_so_far = ""
    
    # Key patterns to locate
    error_pattern = re.compile(r'\[ERROR\]')
    
    # Find positions in the original input
    error_pos = input_text.find('[ERROR]')
    trace_pos = input_text.find('Execution trace:')
    test_pos = input_text.find('Test code:')
    
    # Map token positions to segments
    current_pos = 0
    for i, tok in enumerate(tokens):
        tok_text = tok
        
        # Determine which segment this token belongs to
        # We'll use a simple heuristic based on accumulated text
        text_so_far += tok_text
        decoded_len = len(text_so_far)
        
        if error_pos > 0 and '[ERROR]' in tok_text:
            segments.append('error_marker')
        elif any(err in tok_text for err in ['AssertionError', 'TypeError', 'NameError', 'AttributeError', 'ValueError', 'IndexError', 'KeyError']):
            segments.append('error_type')
        elif trace_pos > 0 and decoded_len > trace_pos and error_pos > 0 and decoded_len > error_pos + 20:
            segments.append('error_detail')
        elif trace_pos > 0 and decoded_len > trace_pos:
            segments.append('trace')
        elif test_pos > 0 and decoded_len > test_pos:
            segments.append('test')
        else:
            segments.append('code')
    
    return segments

def extract_attention(model, tokenizer, text):
    """Run forward pass and extract last-layer attention from prediction position."""
    chat_input = get_chat_input(tokenizer, text)
    inputs = tokenizer(chat_input, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Last layer attention: shape [1, num_heads, seq_len, seq_len]
    last_layer_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
    
    # Average over heads
    avg_attn = last_layer_attn.mean(dim=0)  # [seq_len, seq_len]
    
    # Attention from last token (prediction position) to all other tokens
    last_token_attn = avg_attn[-1, :].cpu().numpy()  # [seq_len]
    
    # Get token strings
    token_ids = inputs['input_ids'][0].cpu().tolist()
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    
    return last_token_attn, token_strs, chat_input

def compute_segment_attention(attn_weights, token_strs, input_text):
    """Compute attention aggregated by segment."""
    segments = segment_tokens(token_strs, input_text)
    
    segment_attn = defaultdict(float)
    segment_count = defaultdict(int)
    
    for i, (seg, w) in enumerate(zip(segments, attn_weights)):
        segment_attn[seg] += w
        segment_count[seg] += 1
    
    return dict(segment_attn), dict(segment_count)

def plot_detailed_heatmap(attn_weights, token_strs, input_text, sample_info, idx):
    """Plot detailed attention heatmap for a single sample with segment coloring."""
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Aggregate tokens into chunks of 10 for readability
    chunk_size = 10
    n_tokens = len(attn_weights)
    n_chunks = (n_tokens + chunk_size - 1) // chunk_size
    
    chunk_attn = []
    chunk_labels = []
    chunk_colors = []
    
    segments = segment_tokens(token_strs, input_text)
    seg_color_map = {
        'code': '#4CAF50',
        'test': '#2196F3', 
        'trace': '#FF9800',
        'error_marker': '#F44336',
        'error_type': '#E91E63',
        'error_detail': '#9C27B0',
    }
    
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, n_tokens)
        chunk_w = attn_weights[start:end].sum()
        chunk_attn.append(chunk_w)
        
        # Label: first few chars of first token in chunk
        label_text = ''.join(token_strs[start:min(start+3, end)])
        label_text = label_text.replace('\n', '\\n')[:15]
        chunk_labels.append(label_text)
        
        # Color based on dominant segment in chunk
        seg_counts = defaultdict(int)
        for s in segments[start:end]:
            seg_counts[s] += 1
        dominant = max(seg_counts, key=seg_counts.get)
        chunk_colors.append(seg_color_map.get(dominant, '#757575'))
    
    chunk_attn = np.array(chunk_attn)
    
    # Plot as colored bar chart
    bars = ax.bar(range(n_chunks), chunk_attn, color=chunk_colors, width=0.9, edgecolor='white', linewidth=0.3)
    
    ax.set_xlabel('Token position (chunks of 10)', fontsize=9)
    ax.set_ylabel('Attention weight', fontsize=9)
    ax.set_title(f'Last-token attention distribution: {sample_info["etype"]} ({sample_info["problem_id"]})', fontsize=10)
    
    # Add segment legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Code'),
        Patch(facecolor='#2196F3', label='Test'),
        Patch(facecolor='#FF9800', label='Trace header'),
        Patch(facecolor='#F44336', label='[ERROR]'),
        Patch(facecolor='#E91E63', label='Error type'),
        Patch(facecolor='#9C27B0', label='Error detail'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right', ncol=2)
    
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=8)
    
    # Only show every 5th label
    ax.set_xticks(range(0, n_chunks, 5))
    ax.set_xticklabels([chunk_labels[i] if i < len(chunk_labels) else '' for i in range(0, n_chunks, 5)], rotation=45, ha='right', fontsize=6)
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"attention_heatmap_{idx}_{sample_info['etype']}.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

def plot_summary_bar(all_results):
    """Plot summary bar chart: attention proportion by segment across all samples."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    categories = ['error_marker', 'error_type', 'error_detail', 'trace', 'test', 'code']
    cat_labels = ['[ERROR] marker', 'Error type', 'Error detail', 'Trace header', 'Test code', 'Source code']
    colors = ['#F44336', '#E91E63', '#9C27B0', '#FF9800', '#2196F3', '#4CAF50']
    
    x = np.arange(len(all_results))
    width = 0.12
    
    for i, (cat, label, color) in enumerate(zip(categories, cat_labels, colors)):
        values = []
        for r in all_results:
            seg_attn = r['segment_attn']
            total = sum(seg_attn.values())
            values.append(seg_attn.get(cat, 0) / total * 100 if total > 0 else 0)
        ax.bar(x + i * width, values, width, label=label, color=color)
    
    ax.set_xlabel('Sample', fontsize=9)
    ax.set_ylabel('Attention share (%)', fontsize=9)
    ax.set_title('Attention distribution by segment type across samples', fontsize=10)
    ax.set_xticks(x + width * 2.5)
    xlabels = [f"{r['etype'][:8]}\n{r['problem_id'].split('/')[1]}" for r in all_results]
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "attention_summary_bar.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

def plot_attention_heatmap_matrix(attn_weights, token_strs, input_text, sample_info, idx):
    """Plot a 2D attention heatmap (subset of tokens around error region)."""
    # Find error-related tokens
    error_indices = []
    for i, t in enumerate(token_strs):
        if '[ERROR]' in t or 'Error' in t or 'Exception' in t:
            error_indices.append(i)
    
    if not error_indices:
        return
    
    # Take a window around the error region
    center = error_indices[0]
    window_start = max(0, center - 30)
    window_end = min(len(attn_weights), center + 50)
    
    window_attn = attn_weights[window_start:window_end]
    window_tokens = token_strs[window_start:window_end]
    
    fig, ax = plt.subplots(figsize=(12, 2.5))
    
    # Single row heatmap
    data = window_attn.reshape(1, -1)
    sns.heatmap(data, ax=ax, cmap='RdYlBu_r', xticklabels=False, yticklabels=['Last token\nattention'],
                cbar_kws={'shrink': 0.6, 'label': 'Weight'})
    
    # Mark error tokens
    for i, t in enumerate(window_tokens):
        if '[ERROR]' in t or 'Error' in t:
            ax.axvline(x=i, color='red', linewidth=0.8, alpha=0.7)
    
    # Add some token labels
    step = max(1, len(window_tokens) // 20)
    positions = list(range(0, len(window_tokens), step))
    labels = [window_tokens[p].replace('\n', '\\n')[:8] for p in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    
    ax.set_title(f'Attention heatmap (error region): {sample_info["etype"]} ({sample_info["problem_id"]})', fontsize=9)
    
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"attention_heatmap_matrix_{idx}_{sample_info['etype']}.pdf")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded.")
    
    print("Selecting samples...")
    samples = select_samples(n=10)
    print(f"Selected {len(samples)} samples:")
    for s, etype, tlen in samples:
        print(f"  {etype}: {s['problem_id']} ({tlen} tokens)")
    
    all_results = []
    
    for idx, (sample, etype, tlen) in enumerate(samples):
        print(f"\nProcessing sample {idx+1}/{len(samples)}: {etype} ({sample['problem_id']})...")
        
        attn_weights, token_strs, chat_input = extract_attention(model, tokenizer, sample['input'])
        seg_attn, seg_count = compute_segment_attention(attn_weights, token_strs, sample['input'])
        
        result = {
            'idx': idx,
            'etype': etype,
            'problem_id': sample['problem_id'],
            'n_tokens': len(token_strs),
            'segment_attn': seg_attn,
            'segment_count': seg_count,
        }
        all_results.append(result)
        
        # Print segment attention breakdown
        total_attn = sum(seg_attn.values())
        print(f"  Segment attention breakdown:")
        for seg in ['error_marker', 'error_type', 'error_detail', 'trace', 'test', 'code']:
            pct = seg_attn.get(seg, 0) / total_attn * 100
            print(f"    {seg}: {pct:.1f}% ({seg_count.get(seg, 0)} tokens)")
        
        # Detailed heatmap for first 3
        if idx < 3:
            plot_detailed_heatmap(attn_weights, token_strs, sample['input'], result, idx)
            plot_attention_heatmap_matrix(attn_weights, token_strs, sample['input'], result, idx)
    
    # Summary plot
    print("\nGenerating summary plot...")
    plot_summary_bar(all_results)
    
    # Save raw results
    results_path = os.path.join(OUTPUT_DIR, "attention_pilot_results.json")
    # Convert numpy types for JSON
    for r in all_results:
        r['segment_attn'] = {k: float(v) for k, v in r['segment_attn'].items()}
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved results: {results_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
