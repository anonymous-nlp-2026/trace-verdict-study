"""Attention & Gradient Analysis: Gradient Dilution in full_trace Model.

Demonstrates that execution trace tokens dominate input length but receive
disproportionately low attention/gradient signal at the verdict position.
"""
import json, os, re, sys, random, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda:0"

BASE_MODEL = "./models/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "./checkpoints/full_trace/final"
DATA_PATH = "./data/full_trace_test.jsonl"
OUTPUT_DIR = "./artifacts/attention_gradient_analysis"
N_SAMPLES = 200
MAX_SEQ_LEN = 2048
SEED = 42

CATS = ["source_code", "test_code", "execution_trace", "framing"]
CAT_COLORS = {
    "source_code":     "#2E7D32",
    "test_code":       "#1565C0",
    "execution_trace": "#E65100",
    "framing":         "#BDBDBD",
}
CAT_LABELS = {
    "source_code":     "Source Code",
    "test_code":       "Test Code",
    "execution_trace": "Execution Trace",
    "framing":         "Framing / Template",
}


def load_model():
    print("Loading tokenizer + base model + LoRA adapter ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()
    model.requires_grad_(False)
    print("Model ready.\n")
    return model, tok


def load_fail_samples(n):
    random.seed(SEED)
    with open(DATA_PATH) as f:
        samples = [json.loads(l) for l in f]
    pool = [s for s in samples
            if s["output"] == "fail" and "[No trace available]" not in s["input"]]
    if len(pool) > n:
        pool = random.sample(pool, n)
    print(f"Selected {len(pool)} fail samples with execution traces")
    return pool


def make_prompt(tok, input_text):
    msgs = [{"role": "user", "content": input_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def classify_tokens(token_ids, tok):
    """Classify each token into source_code / test_code / execution_trace / framing
    using character-position mapping in the decoded sequence."""
    strs = [tok.decode([tid], skip_special_tokens=False) for tid in token_ids]
    cum = []
    pos = 0
    for s in strs:
        pos += len(s)
        cum.append(pos)
    full = "".join(strs)

    # Section boundaries (search in order)
    code_mark  = full.find("Given the following code:")
    test_mark  = full.find("Test code:")
    trace_mark = full.find("Execution trace:")
    q_mark     = full.find("Does this code pass all tests?")

    # Content starts (after the marker line)
    def after_nl(pos, marker_len):
        nl = full.find("\n", pos)
        return nl + 1 if nl >= 0 and nl < pos + marker_len + 5 else pos + marker_len

    code_c  = after_nl(code_mark, len("Given the following code:")) if code_mark >= 0 else -1
    test_c  = after_nl(test_mark, len("Test code:")) if test_mark >= 0 else -1
    trace_c = after_nl(trace_mark, len("Execution trace:")) if trace_mark >= 0 else -1

    cats = []
    for i in range(len(token_ids)):
        cs = cum[i - 1] if i > 0 else 0
        ce = cum[i]
        mid = (cs + ce) // 2

        if code_c >= 0 and mid >= code_c and (test_mark < 0 or mid < test_mark):
            cats.append("source_code")
        elif test_c >= 0 and mid >= test_c and (trace_mark < 0 or mid < trace_mark):
            cats.append("test_code")
        elif trace_c >= 0 and mid >= trace_c and (q_mark < 0 or mid < q_mark):
            cats.append("execution_trace")
        else:
            cats.append("framing")
    return cats, strs


# ─── Part A: Attention ─────────────────────────────────────────
def attention_analysis(model, tok, samples):
    cat_attn = defaultdict(list)
    cat_cnt  = defaultdict(list)
    heatmaps = []
    t0 = time.time()

    for idx, sample in enumerate(samples):
        prompt = make_prompt(tok, sample["input"])
        inp = tok(prompt, return_tensors="pt",
                  max_length=MAX_SEQ_LEN, truncation=True).to(DEVICE)

        with torch.no_grad():
            out = model(**inp, output_attentions=True)

        last_layer = out.attentions[-1][0]        # [heads, seq, seq]
        avg = last_layer.mean(dim=0)              # [seq, seq]
        va = avg[-1, :].cpu().numpy()             # attention from last token

        ids = inp["input_ids"][0].cpu().tolist()
        cats, tstrs = classify_tokens(ids, tok)

        ca = defaultdict(float)
        cc = defaultdict(int)
        for c, w in zip(cats, va):
            ca[c] += float(w); cc[c] += 1
        for c in CATS:
            cat_attn[c].append(ca.get(c, 0.0))
            cat_cnt[c].append(cc.get(c, 0))

        if len(heatmaps) < 2:
            heatmaps.append(dict(attn=va, cats=cats, tstrs=tstrs,
                                 pid=sample.get("problem_id", f"s{idx}")))

        del out; torch.cuda.empty_cache()
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Attention: {idx+1}/{len(samples)}  ({elapsed:.0f}s)")

    return cat_attn, cat_cnt, heatmaps


# ─── Part B: Gradient ──────────────────────────────────────────
def gradient_analysis(model, tok, samples):
    cat_grad = defaultdict(list)
    cat_cnt  = defaultdict(list)
    # per-token gradient norms aggregated by category (normalized per sample)
    cat_grad_normed = defaultdict(list)
    t0 = time.time()
    embed_layer = model.get_input_embeddings()

    for idx, sample in enumerate(samples):
        prompt = make_prompt(tok, sample["input"])
        inp = tok(prompt, return_tensors="pt",
                  max_length=MAX_SEQ_LEN, truncation=True).to(DEVICE)

        embeds = embed_layer(inp["input_ids"]).detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, attention_mask=inp["attention_mask"])

        # Loss: CE at last position predicting the verdict
        last_logits = out.logits[0, -1, :]
        tgt_ids = tok.encode(sample["output"], add_special_tokens=False)
        tgt = torch.tensor([tgt_ids[0]], device=DEVICE)
        loss = F.cross_entropy(last_logits.unsqueeze(0), tgt)
        loss.backward()

        gn = embeds.grad[0].float().norm(dim=-1).cpu().numpy()
        gn_total = gn.sum() + 1e-12
        gn_share = gn / gn_total  # normalized to sum=1

        ids = inp["input_ids"][0].cpu().tolist()
        cats, _ = classify_tokens(ids, tok)

        cg  = defaultdict(float)
        cgn = defaultdict(float)
        cc  = defaultdict(int)
        for c, g, gs in zip(cats, gn, gn_share):
            cg[c]  += float(g)
            cgn[c] += float(gs)
            cc[c]  += 1
        for c in CATS:
            cat_grad[c].append(cg.get(c, 0.0))
            cat_grad_normed[c].append(cgn.get(c, 0.0))
            cat_cnt[c].append(cc.get(c, 0))

        del out, embeds; torch.cuda.empty_cache()
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Gradient: {idx+1}/{len(samples)}  ({elapsed:.0f}s)")

    return cat_grad, cat_cnt, cat_grad_normed


# ─── Part C: Visualization ─────────────────────────────────────
def plot_heatmaps(data, outdir):
    for i, d in enumerate(data):
        attn, cats = d["attn"], d["cats"]
        n = len(attn)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 3.5),
                                        gridspec_kw={"height_ratios": [4, 1]},
                                        sharex=True)
        ax1.fill_between(range(n), attn, alpha=0.3, color="#1565C0")
        ax1.plot(range(n), attn, lw=0.5, color="#0D47A1")
        ax1.set_ylabel("Attention Weight")
        ax1.set_title(f"Verdict-Position Attention — {d['pid']}")

        for j in range(n):
            ax2.axvspan(j - 0.5, j + 0.5,
                        color=CAT_COLORS.get(cats[j], "#E0E0E0"), alpha=0.85)
        ax2.set_yticks([]); ax2.set_xlabel("Token Position")

        patches = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c])
                   for c in CATS if c in set(cats)]
        ax1.legend(handles=patches, fontsize=7, ncol=4, loc="upper right")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{outdir}/attention_heatmap_{i}.{ext}",
                        dpi=200, bbox_inches="tight")
        plt.close()


def plot_gradient_bars(grad_raw, counts, outdir):
    means, stds = [], []
    for c in CATS:
        per = [g / n if n > 0 else 0
               for g, n in zip(grad_raw[c], counts[c])]
        means.append(np.mean(per) if per else 0)
        stds.append(np.std(per) if per else 0)

    fig, ax = plt.subplots(figsize=(7, 4))
    y = range(len(CATS))
    ax.barh(y, means, xerr=stds,
            color=[CAT_COLORS[c] for c in CATS],
            edgecolor="black", lw=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([CAT_LABELS[c] for c in CATS])
    ax.set_xlabel("Mean Gradient L2 Norm (per token)")
    ax.set_title("Gradient Attribution by Input Section")
    ax.invert_yaxis()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/gradient_norm_by_section.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()


def plot_dilution(attn_s, grad_s, outdir):
    cats3 = ["source_code", "test_code", "execution_trace"]
    labels = [CAT_LABELS[c] for c in cats3]
    colors = [CAT_COLORS[c] for c in cats3]

    tok_pcts  = [attn_s[c]["token_pct"] for c in cats3]
    attn_pcts = [attn_s[c]["attention_pct"] for c in cats3]
    grad_pcts = [grad_s[c]["gradient_share_pct"] for c in cats3]

    x = np.arange(len(cats3))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, tok_pcts,  w, label="Token Share %",
           color=colors, alpha=0.35, edgecolor="black", lw=0.5)
    ax.bar(x,     attn_pcts, w, label="Attention Share %",
           color=colors, alpha=0.7, edgecolor="black", lw=0.5)
    ax.bar(x + w, grad_pcts, w, label="Gradient Share %",
           color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Gradient Dilution: Token Share vs Signal Share")
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/gradient_dilution_comparison.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, tok = load_model()
    samples = load_fail_samples(N_SAMPLES)

    # Part A
    print("\n=== Part A: Attention Analysis ===")
    attn_data, attn_cnt, heatmaps = attention_analysis(model, tok, samples)

    # Part B
    print("\n=== Part B: Gradient Analysis ===")
    grad_raw, grad_cnt, grad_normed = gradient_analysis(model, tok, samples)

    # ── Summaries ──
    attn_s = {}
    tot_a = sum(np.mean(attn_data[c]) for c in CATS)
    tot_t = sum(np.mean(attn_cnt[c]) for c in CATS)
    for c in CATS:
        ma = np.mean(attn_data[c]); mc = np.mean(attn_cnt[c])
        attn_s[c] = {
            "attention_mass":   round(float(ma), 6),
            "attention_pct":    round(float(ma / tot_a * 100), 2) if tot_a else 0,
            "token_count":      round(float(mc), 1),
            "token_pct":        round(float(mc / tot_t * 100), 2) if tot_t else 0,
            "attention_density": round(float(ma / mc), 8) if mc else 0,
        }

    grad_s = {}
    tot_g = sum(np.mean(grad_raw[c]) for c in CATS)
    tot_tg = sum(np.mean(grad_cnt[c]) for c in CATS)
    for c in CATS:
        mg = np.mean(grad_raw[c]); mc = np.mean(grad_cnt[c])
        gs = np.mean(grad_normed[c])  # normalized share
        grad_s[c] = {
            "gradient_norm_total": round(float(mg), 4),
            "gradient_pct":       round(float(mg / tot_g * 100), 2) if tot_g else 0,
            "gradient_share_pct": round(float(gs * 100), 2),
            "token_count":        round(float(mc), 1),
            "token_pct":          round(float(mc / tot_tg * 100), 2) if tot_tg else 0,
            "gradient_density":   round(float(mg / mc), 6) if mc else 0,
        }

    # Dilution metrics
    tr_tok  = attn_s["execution_trace"]["token_pct"]
    tr_attn = attn_s["execution_trace"]["attention_pct"]
    tr_grad = grad_s["execution_trace"]["gradient_share_pct"]
    ct_tok  = sum(attn_s[c]["token_pct"] for c in ("source_code", "test_code"))
    ct_attn = sum(attn_s[c]["attention_pct"] for c in ("source_code", "test_code"))
    ct_grad = sum(grad_s[c]["gradient_share_pct"] for c in ("source_code", "test_code"))

    dilution = {
        "execution_trace": {"token_pct": tr_tok, "attention_pct": tr_attn,
                            "gradient_share_pct": tr_grad},
        "code_and_test":   {"token_pct": ct_tok, "attention_pct": ct_attn,
                            "gradient_share_pct": ct_grad},
        "attention_dilution_ratio": round(tr_tok / tr_attn, 2) if tr_attn else None,
        "gradient_dilution_ratio":  round(tr_tok / tr_grad, 2) if tr_grad else None,
    }

    results = {
        "n_samples": len(samples),
        "max_seq_len": MAX_SEQ_LEN,
        "attention": attn_s,
        "gradient": grad_s,
        "gradient_dilution": dilution,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples: {len(samples)}")
    print(f"\nExecution Trace:  {tr_tok:.1f}% tokens | {tr_attn:.1f}% attention | {tr_grad:.1f}% gradient")
    print(f"Code + Test:      {ct_tok:.1f}% tokens | {ct_attn:.1f}% attention | {ct_grad:.1f}% gradient")
    if dilution["attention_dilution_ratio"]:
        print(f"\nAttention dilution ratio: {dilution['attention_dilution_ratio']:.2f}x")
    if dilution["gradient_dilution_ratio"]:
        print(f"Gradient dilution ratio:  {dilution['gradient_dilution_ratio']:.2f}x")
    print(f"\nPer-section:")
    for c in CATS:
        a, g = attn_s[c], grad_s[c]
        print(f"  {CAT_LABELS[c]:20s}: tok={a['token_pct']:.1f}%  "
              f"attn={a['attention_pct']:.1f}%  grad={g['gradient_share_pct']:.1f}%  "
              f"attn_density={a['attention_density']:.6f}  "
              f"grad_density={g['gradient_density']:.4f}")

    # Part C: Plots
    print("\n=== Part C: Visualization ===")
    plot_heatmaps(heatmaps, OUTPUT_DIR)
    print("  Saved attention heatmaps")
    plot_gradient_bars(grad_raw, grad_cnt, OUTPUT_DIR)
    print("  Saved gradient bar chart")
    plot_dilution(attn_s, grad_s, OUTPUT_DIR)
    print("  Saved dilution comparison chart")

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
