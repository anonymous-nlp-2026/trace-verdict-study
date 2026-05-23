# Bootstrap comparison of trace-verdict conditions with MVP verdict determination.
# Computes pairwise accuracy differences with 95% CI (N=10000, seed=42)
# and outputs PASS / WEAK PASS / FAIL per ROADMAP D001 criteria.

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CONDITIONS = ["no_trace", "vota", "full_trace"]
PAIRS = [("no_trace", "vota"), ("no_trace", "full_trace"), ("vota", "full_trace")]
DEFAULT_N_BOOTSTRAP = 10000
SEED = 42


def load_test_data(data_dir, condition):
    path = os.path.join(data_dir, f"{condition}_test.jsonl")
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_predictions(pred_dir, condition):
    path = os.path.join(pred_dir, f"{condition}_predictions.json")
    with open(path) as f:
        return json.load(f)["predictions"]


def run_condition_inference(checkpoint_dir, model_path, data_dir, condition, device):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_data = load_test_data(data_dir, condition)
    ckpt_path = os.path.join(checkpoint_dir, condition, "final")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": device},
    )
    model = PeftModel.from_pretrained(model, ckpt_path)
    model.eval()

    predictions = []
    for i, item in enumerate(test_data):
        messages = [{"role": "user", "content": item["input"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, temperature=None, top_p=None,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

        pred = "pass" if "pass" in raw else ("fail" if "fail" in raw else raw)
        label = item["output"]

        predictions.append({
            "idx": i,
            "problem_id": item.get("problem_id", f"sample_{i}"),
            "label": label,
            "pred": pred,
            "correct": label == pred,
        })

        if (i + 1) % 100 == 0:
            acc = sum(p["correct"] for p in predictions) / len(predictions)
            print(f"  [{condition}] {i+1}/{len(test_data)} acc={acc:.4f}")

    del model
    torch.cuda.empty_cache()
    return predictions


def compute_metrics(labels, preds):
    n = len(labels)
    correct = sum(l == p for l, p in zip(labels, preds))
    acc = correct / n if n else 0.0

    count_pass = sum(l == "pass" for l in labels)
    count_fail = n - count_pass
    majority_baseline = max(count_pass, count_fail) / n if n else 0.0
    majority_class = "pass" if count_pass >= count_fail else "fail"

    metrics = {
        "accuracy": acc, "majority_baseline": majority_baseline,
        "majority_class": majority_class, "n": n,
    }
    for cls in ["pass", "fail"]:
        tp = sum(l == cls and p == cls for l, p in zip(labels, preds))
        fp = sum(l != cls and p == cls for l, p in zip(labels, preds))
        fn = sum(l == cls and p != cls for l, p in zip(labels, preds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[f"{cls}_precision"] = prec
        metrics[f"{cls}_recall"] = rec
        metrics[f"{cls}_f1"] = f1
    return metrics


def bootstrap_pairwise(correct_a, correct_b, n_bootstrap, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(correct_a)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_contains_zero": bool(lo <= 0 <= hi),
    }


def bootstrap_single_accuracy(correct, n_bootstrap, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(correct)
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        accs[i] = correct[idx].mean()
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def determine_verdict(comparisons, per_condition):
    nt_acc = per_condition["no_trace"]["accuracy"]
    vota_acc = per_condition["vota"]["accuracy"]
    ft_acc = per_condition["full_trace"]["accuracy"]

    vota_diff = vota_acc - nt_acc
    ft_diff = ft_acc - nt_acc

    vota_vs_nt = comparisons[("vota", "no_trace")]
    ft_vs_nt = comparisons[("full_trace", "no_trace")]

    if vota_diff >= 0.02 and not vota_vs_nt["ci_contains_zero"]:
        return {
            "verdict": "PASS",
            "reason": (
                f"VOTA {vota_acc*100:.1f}% >= no_trace {nt_acc*100:.1f}% + 2pp "
                f"(diff={vota_diff*100:+.1f}pp, 95% CI=[{vota_vs_nt['ci_lower']*100:+.1f}, "
                f"{vota_vs_nt['ci_upper']*100:+.1f}]pp, excludes 0)"
            ),
        }

    any_above_1pp = vota_diff >= 0.01 or ft_diff >= 0.01
    if any_above_1pp:
        parts = []
        if vota_diff >= 0.01:
            parts.append(
                f"VOTA diff={vota_diff*100:+.1f}pp, CI contains 0={vota_vs_nt['ci_contains_zero']}"
            )
        if ft_diff >= 0.01:
            parts.append(
                f"full_trace diff={ft_diff*100:+.1f}pp, CI contains 0={ft_vs_nt['ci_contains_zero']}"
            )
        return {"verdict": "WEAK PASS", "reason": "; ".join(parts)}

    return {
        "verdict": "FAIL",
        "reason": (
            f"No trace condition >= no_trace + 1pp "
            f"(vota={vota_diff*100:+.1f}pp, full_trace={ft_diff*100:+.1f}pp)"
        ),
    }


def print_results(per_condition, pairwise, verdict_info, correct_arrays, n_bootstrap):
    w = 80
    print("\n" + "=" * w)
    print("TRACE-VERDICT STUDY: Bootstrap Comparison Results")
    print("=" * w)

    mb = per_condition["no_trace"]["majority_baseline"]
    mc = per_condition["no_trace"]["majority_class"]
    n = per_condition["no_trace"]["n"]

    print(f"\nSamples: {n} | Majority baseline: {mb*100:.1f}% (always '{mc}')")
    print(f"Bootstrap: N={n_bootstrap}, seed={SEED}\n")

    print(f"{'Condition':<14} {'Accuracy':>10} {'95% CI':>20} {'Lift':>10} {'Pass-F1':>10} {'Fail-F1':>10}")
    print("-" * 76)
    for cond in CONDITIONS:
        m = per_condition[cond]
        lo, hi = bootstrap_single_accuracy(correct_arrays[cond], n_bootstrap)
        lift = m["accuracy"] - mb
        print(
            f"{cond:<14} {m['accuracy']*100:>9.1f}% "
            f"[{lo*100:>5.1f}, {hi*100:>5.1f}]%  "
            f"{lift*100:>+9.1f}pp {m['pass_f1']*100:>9.1f}% {m['fail_f1']*100:>9.1f}%"
        )
    print(f"{'majority':<14} {mb*100:>9.1f}%")

    print(f"\n{'Comparison':<28} {'Diff':>10} {'95% CI':>24} {'Sig':>6}")
    print("-" * 70)
    for a, b in PAIRS:
        c = pairwise[(a, b)]
        sig = "***" if not c["ci_contains_zero"] else "n.s."
        print(
            f"{a} vs {b:<16} {c['mean_diff']*100:>+9.1f}pp "
            f"[{c['ci_lower']*100:>+7.1f}, {c['ci_upper']*100:>+7.1f}]pp {sig:>6}"
        )

    print(f"\n{'=' * w}")
    print(f"MVP VERDICT: {verdict_info['verdict']}")
    print(f"Reason: {verdict_info['reason']}")
    print("=" * w)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap comparison of trace-verdict conditions with MVP verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry-run: validate paths
  python bootstrap_compare.py --dry-run

  # Run full inference + comparison
  python bootstrap_compare.py --run-inference

  # From saved per-sample predictions
  python bootstrap_compare.py --predictions-dir ./predictions/
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions-dir", help="Dir with {condition}_predictions.json")
    mode.add_argument("--run-inference", action="store_true", help="Run inference from checkpoints")
    mode.add_argument("--dry-run", action="store_true", help="Validate setup only")

    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Base dir with {condition}/final/ subdirs",
    )
    parser.add_argument(
        "--model-path",
        default="./models/Qwen2.5-Coder-7B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Dir with {condition}_test.jsonl files",
    )
    parser.add_argument(
        "--output",
        default="./results/bootstrap_compare.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="Dir to save per-sample predictions for re-use",
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # --- Dry-run ---
    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Checkpoint dir: {args.checkpoint_dir}")
        print(f"Data dir:       {args.data_dir}")
        print(f"Model path:     {args.model_path}")
        print(f"Output:         {args.output}")
        print(f"N bootstrap:    {args.n_bootstrap}")
        print()
        all_ok = True
        for cond in CONDITIONS:
            ckpt = os.path.join(args.checkpoint_dir, cond, "final")
            data = os.path.join(args.data_dir, f"{cond}_test.jsonl")
            ckpt_ok = os.path.isdir(ckpt)
            data_ok = os.path.isfile(data)
            n = 0
            if data_ok:
                with open(data) as f:
                    n = sum(1 for l in f if l.strip())
            status = "OK" if (ckpt_ok and data_ok) else "MISSING"
            if not (ckpt_ok and data_ok):
                all_ok = False
            print(f"  {cond}: ckpt={'OK' if ckpt_ok else 'MISSING'}, data={'OK' if data_ok else 'MISSING'} ({n} samples)")
        model_ok = os.path.isdir(args.model_path)
        if not model_ok:
            all_ok = False
        print(f"\n  Base model: {'OK' if model_ok else 'MISSING'} ({args.model_path})")

        # Check numpy
        try:
            import numpy
            print(f"  numpy: {numpy.__version__}")
        except ImportError:
            print("  numpy: MISSING")
            all_ok = False

        if args.predictions_dir is None:
            try:
                import torch, transformers, peft
                print(f"  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
                print(f"  transformers: {transformers.__version__}")
                print(f"  peft: {peft.__version__}")
            except ImportError as e:
                print(f"  Missing dependency: {e}")
                all_ok = False

        print(f"\nAll checks {'PASSED' if all_ok else 'FAILED'}.")
        sys.exit(0 if all_ok else 1)

    # --- Load or generate predictions ---
    all_predictions = {}

    if args.predictions_dir:
        print(f"Loading predictions from {args.predictions_dir}")
        for cond in CONDITIONS:
            all_predictions[cond] = load_predictions(args.predictions_dir, cond)
            print(f"  {cond}: {len(all_predictions[cond])} samples")
    else:
        import torch
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0")
        print(f"Running inference on GPU {args.gpu}")

        for cond in CONDITIONS:
            print(f"\n--- {cond} ---")
            all_predictions[cond] = run_condition_inference(
                args.checkpoint_dir, args.model_path, args.data_dir, cond, device,
            )
            acc = sum(p["correct"] for p in all_predictions[cond]) / len(all_predictions[cond])
            print(f"  Done: {len(all_predictions[cond])} samples, acc={acc:.4f}")

        if args.save_predictions:
            os.makedirs(args.save_predictions, exist_ok=True)
            for cond in CONDITIONS:
                path = os.path.join(args.save_predictions, f"{cond}_predictions.json")
                with open(path, "w") as f:
                    json.dump({"condition": cond, "predictions": all_predictions[cond]}, f, indent=2)
                print(f"Saved: {path}")

    # --- Verify alignment ---
    sizes = {c: len(all_predictions[c]) for c in CONDITIONS}
    if len(set(sizes.values())) != 1:
        print(f"ERROR: mismatched sample counts: {sizes}", file=sys.stderr)
        sys.exit(1)

    # --- Per-condition metrics ---
    per_condition = {}
    correct_arrays = {}
    for cond in CONDITIONS:
        preds_list = all_predictions[cond]
        labels = [p["label"] for p in preds_list]
        pred_labels = [p["pred"] for p in preds_list]
        per_condition[cond] = compute_metrics(labels, pred_labels)
        correct_arrays[cond] = np.array([p["correct"] for p in preds_list], dtype=np.float64)

    # --- Pairwise bootstrap ---
    print(f"\nBootstrap (N={args.n_bootstrap}, seed={SEED})...")
    pairwise = {}
    for a, b in PAIRS:
        pairwise[(a, b)] = bootstrap_pairwise(
            correct_arrays[a], correct_arrays[b], args.n_bootstrap,
        )

    # Reverse lookups for verdict logic
    comparisons_all = dict(pairwise)
    for a, b in PAIRS:
        c = pairwise[(a, b)]
        comparisons_all[(b, a)] = {
            "mean_diff": -c["mean_diff"],
            "ci_lower": -c["ci_upper"],
            "ci_upper": -c["ci_lower"],
            "ci_contains_zero": c["ci_contains_zero"],
        }

    # --- Verdict ---
    verdict_info = determine_verdict(comparisons_all, per_condition)

    # --- Output ---
    print_results(per_condition, pairwise, verdict_info, correct_arrays, args.n_bootstrap)

    result = {
        "per_condition": {c: per_condition[c] for c in CONDITIONS},
        "pairwise_comparisons": {f"{a}_vs_{b}": pairwise[(a, b)] for a, b in PAIRS},
        "verdict": verdict_info,
        "config": {"n_bootstrap": args.n_bootstrap, "seed": SEED, "conditions": CONDITIONS},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
