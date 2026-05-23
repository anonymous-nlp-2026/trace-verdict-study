import json, re

DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"

formats = {
    "exception_only": (f"{DATA_DIR}/exception_only_test.jsonl", f"{CKPT_DIR}/exception_only/final/predictions.jsonl"),
    "vota": (f"{DATA_DIR}/vota_test.jsonl", f"{CKPT_DIR}/vota/final/predictions.jsonl"),
    "full_trace": (f"{DATA_DIR}/full_trace_test.jsonl", f"{CKPT_DIR}/full_trace/final/predictions.jsonl"),
    "no_trace": (f"{DATA_DIR}/no_trace_test.jsonl", f"{CKPT_DIR}/no_trace/final/predictions.jsonl"),
}

def extract_trace(inp):
    ts = inp.find("Execution trace:\n")
    te = inp.find("\n\nDoes this code pass all tests?")
    return inp[ts + len("Execution trace:\n"):te] if ts >= 0 and te >= 0 else ""

def acc(y_true, y_pred):
    return sum(1 for a,b in zip(y_true,y_pred) if a==b) / len(y_true)

baselines = {
    "ERROR_tag": lambda t: "fail" if "[ERROR]" in t else "pass",
    "Traceback": lambda t: "fail" if "Traceback" in t else "pass",
    "any_keyword": lambda t: "fail" if any(kw in t for kw in ["[ERROR]", "Traceback", "Exception"]) else "pass",
    "exception_regex": lambda t: "fail" if re.search(r'(AssertionError|ValueError|TypeError|IndexError|KeyError|AttributeError|RuntimeError|ZeroDivisionError|Exception|Error)', t) else "pass",
    "no_exception_heuristic": lambda t: "pass" if t == "[No exception data in trace]" else "fail",
    "no_trace_marker": lambda t: "pass" if "[No trace available]" not in t and "[No exception data in trace]" not in t and "[ERROR]" not in t else ("pass" if t == "[No exception data in trace]" else "fail"),
}

for fmt, (data_path, pred_path) in formats.items():
    try:
        samples = []
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                trace = extract_trace(d["input"])
                samples.append({"trace": trace, "label": d["output"]})
        
        labels = [s["label"] for s in samples]
        
        sft_preds = []
        with open(pred_path) as f:
            for line in f:
                sft_preds.append(json.loads(line)["pred"])
        
        sft_acc = acc(labels, sft_preds)
        
        print(f"\n{'='*60}")
        print(f"Format: {fmt} (n={len(samples)}, pass={labels.count('pass')}, fail={labels.count('fail')})")
        print(f"SFT accuracy: {sft_acc:.4f}")
        print(f"{'Baseline':<25} {'Accuracy':>10} {'Gap vs SFT':>12}")
        print("-" * 50)
        
        for bname, bfn in baselines.items():
            preds = [bfn(s["trace"]) for s in samples]
            a = acc(labels, preds)
            gap = sft_acc - a
            print(f"{bname:<25} {a:>10.4f} {gap:>+12.4f}")
    except Exception as e:
        print(f"\n{fmt}: Error - {e}")
