"""LoRA SFT training script for trace-verdict classification using Qwen2.5-Coder-7B-Instruct."""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from gradient_mask_trainer import GradientMaskTrainer


def load_data(data_dir: str, condition: str):
    """Load train and test JSONL files for a given condition."""
    def read_jsonl(path):
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    train_path = os.path.join(data_dir, f"{condition}_train.jsonl")
    test_path = os.path.join(data_dir, f"{condition}_test.jsonl")
    return read_jsonl(train_path), read_jsonl(test_path)


def to_chat_text(items: list[dict], tokenizer, max_length: int = 2048, truncation_side: str = "right") -> list[dict]:
    """Convert data items to chat-formatted text with label-preserving truncation.

    Truncates the input (trace) content instead of the right side of the sequence,
    so the assistant label ("pass"/"fail") is never lost.
    """
    texts = []
    for item in items:
        # Measure token overhead from chat template + label
        overhead_msg = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""},
             {"role": "assistant", "content": item["output"]}],
            tokenize=False, add_generation_prompt=False,
        )
        overhead_len = len(tokenizer(overhead_msg, add_special_tokens=False).input_ids)
        # Truncate input content to fit within budget, preserving label
        budget = max(0, max_length - overhead_len)
        content_ids = tokenizer(item["input"], add_special_tokens=False).input_ids
        if truncation_side == "left":
            content_ids = content_ids[-budget:]
        else:
            content_ids = content_ids[:budget]
        truncated_input = tokenizer.decode(content_ids, skip_special_tokens=False)
        messages = [
            {"role": "user", "content": truncated_input},
            {"role": "assistant", "content": item["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append({"text": text})
    return texts


def to_prompt_completion(items: list[dict], tokenizer) -> list[dict]:
    """Convert data items to prompt-completion format for completion-only loss."""
    results = []
    for item in items:
        messages = [
            {"role": "user", "content": item["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        results.append({"prompt": prompt, "completion": item["output"]})
    return results


def truncate_traces(items, tokenizer, max_trace_tokens, truncation_side="right"):
    """Truncate execution trace to first max_trace_tokens tokens (in-place)."""
    TRACE_MARKER = "Execution trace:\n"
    SUFFIX = "\n\nDoes this code pass all tests? Answer: "
    truncated_count = 0
    orig_lengths = []
    for item in items:
        inp = item["input"]
        t_start = inp.find(TRACE_MARKER)
        s_start = inp.rfind(SUFFIX)
        if t_start == -1 or s_start == -1 or t_start >= s_start:
            continue
        t_begin = t_start + len(TRACE_MARKER)
        trace_text = inp[t_begin:s_start]
        trace_ids = tokenizer(trace_text, add_special_tokens=False).input_ids
        orig_lengths.append(len(trace_ids))
        if len(trace_ids) <= max_trace_tokens:
            continue
        if truncation_side == "left":
            trunc_ids = trace_ids[-max_trace_tokens:]
        else:
            trunc_ids = trace_ids[:max_trace_tokens]
        trunc_text = tokenizer.decode(trunc_ids, skip_special_tokens=False)
        item["input"] = inp[:t_begin] + trunc_text + inp[s_start:]
        truncated_count += 1
    if orig_lengths:
        import statistics
        print(f"Trace truncation: {truncated_count}/{len(orig_lengths)} truncated to {max_trace_tokens} tokens")
        print(f"  Original trace lengths: mean={statistics.mean(orig_lengths):.0f}, "
              f"median={statistics.median(orig_lengths):.0f}, "
              f"max={max(orig_lengths)}, min={min(orig_lengths)}")


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT training for trace-verdict classification.")
    parser.add_argument("--condition", required=True, choices=["no_trace", "vota", "full_trace", "vota_no_ae", "ae_only", "exception_only", "exception_only_nomarker", "loop_only", "debugbench_no_trace", "debugbench_vota", "debugbench_full_trace", "debugbench_ae_only", "debugbench_exception_only", "full_trace_tail", "full_trace_tail_512", "full_trace_tail_256", "full_trace_tail_1024", "full_trace_tail_384", "full_trace_head", "exception_only_padded2048", "cross_he2mbpp_exception_only", "cross_he2mbpp_exception_only_nomarker", "cross_he2mbpp_no_trace", "cross_mbpp2he_exception_only", "cross_mbpp2he_no_trace", "exc_type_only", "exc_stacktrace_only", "exc_message_only"],
                        help="Training condition: no_trace, vota, or full_trace")
    parser.add_argument("--data_dir", default="./data/",
                        help="Directory containing formatted training data")
    parser.add_argument("--model_path", default="./models/Qwen2.5-Coder-7B-Instruct",
                        help="Path to base model")
    parser.add_argument("--output_dir", default=None,
                        help="Checkpoint output dir (default: ./checkpoints/{condition})")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", default="trace-verdict-study", help="W&B project name")
    parser.add_argument("--wandb_run_name", default=None,
                        help="W&B run name (default: {condition}_seed{seed})")
    parser.add_argument("--label_only_loss", action="store_true",
                        help="Only compute loss on completion tokens (mask prompt/trace tokens)")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for full training)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")
    parser.add_argument("--max_trace_tokens", type=int, default=None,
                        help="Truncate trace to first N tokens (None=no truncation)")
    parser.add_argument("--truncation_side", default="right", choices=["left", "right"],
                        help="Truncation side: right=keep head (default), left=keep tail")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="Save checkpoint every N steps (switches save/eval strategy to 'steps')")
    parser.add_argument("--gradient_mask_ratio", type=float, default=0.0,
                        help="Fraction of valid token gradients to mask (0=disabled, 0.998=keep 0.2%%)")
    args = parser.parse_args()

    suffix = "_label_only" if args.label_only_loss else ""
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{args.condition}{suffix}"
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.condition}{suffix}_seed{args.seed}"

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"Condition: {args.condition}")
    print(f"Label-only loss: {args.label_only_loss}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")

    train_data, eval_data = load_data(args.data_dir, args.condition)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_trace_tokens is not None:
        print(f"Truncating traces to {args.max_trace_tokens} tokens...")
        truncate_traces(train_data, tokenizer, args.max_trace_tokens, args.truncation_side)
        truncate_traces(eval_data, tokenizer, args.max_trace_tokens, args.truncation_side)

    if args.label_only_loss:
        train_dataset = Dataset.from_list(to_prompt_completion(train_data, tokenizer))
        eval_dataset = Dataset.from_list(to_prompt_completion(eval_data, tokenizer))
        print(f"Using prompt-completion format with completion_only_loss=True")
        print(f"  Sample prompt (last 80 chars): ...{train_dataset[0]['prompt'][-80:]}")
        print(f"  Sample completion: {train_dataset[0]['completion']}")
    else:
        train_dataset = Dataset.from_list(to_chat_text(train_data, tokenizer, max_length=2048, truncation_side=args.truncation_side))
        eval_dataset = Dataset.from_list(to_chat_text(eval_data, tokenizer, max_length=2048, truncation_side=args.truncation_side))

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        max_length=2048,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "tensorboard",
        run_name=args.wandb_run_name,
        seed=args.seed,
# TEMP_DISABLED:         save_total_limit=3,
# TEMP_DISABLED:         load_best_model_at_end=True,
# TEMP_DISABLED:         metric_for_best_model="eval_loss",
# TEMP_DISABLED:         greater_is_better=False,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.label_only_loss:
        sft_kwargs["completion_only_loss"] = True
    else:
        sft_kwargs["dataset_text_field"] = "text"

    if args.save_steps is not None:
        sft_kwargs["save_strategy"] = "steps"
        sft_kwargs["save_steps"] = args.save_steps
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = args.save_steps

    training_args = SFTConfig(**sft_kwargs)

    trainer_cls = SFTTrainer
    trainer_extra = {}
    if args.gradient_mask_ratio > 0:
        trainer_cls = GradientMaskTrainer
        trainer_extra["gradient_mask_ratio"] = args.gradient_mask_ratio
        print(f"Gradient masking: ratio={args.gradient_mask_ratio}, keep={1 - args.gradient_mask_ratio:.4f}")

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        **trainer_extra,
    )

    print("Starting training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
