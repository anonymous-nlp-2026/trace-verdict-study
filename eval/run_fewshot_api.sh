#!/bin/bash
# Few-shot API evaluation for verdict prediction
# Requires: OPENAI_API_KEY and/or ANTHROPIC_API_KEY

DATA=./data/exception_only_test.jsonl
TRAIN=./data/exception_only_train.jsonl
BASE_DIR=./checkpoints

# GPT-4o 0-shot
echo "=== GPT-4o 0-shot ==="
python ./eval/fewshot_api_eval.py \
    --data_path $DATA --train_path $TRAIN \
    --output_dir $BASE_DIR/fewshot_api_gpt4o_0shot \
    --provider openai --model gpt-4o --n_shot 0 --max_workers 20

# GPT-4o 3-shot
echo "=== GPT-4o 3-shot ==="
python ./eval/fewshot_api_eval.py \
    --data_path $DATA --train_path $TRAIN \
    --output_dir $BASE_DIR/fewshot_api_gpt4o_3shot \
    --provider openai --model gpt-4o --n_shot 3 --max_workers 20

# Claude 0-shot
echo "=== Claude Sonnet 0-shot ==="
python ./eval/fewshot_api_eval.py \
    --data_path $DATA --train_path $TRAIN \
    --output_dir $BASE_DIR/fewshot_api_claude_0shot \
    --provider anthropic --model claude-sonnet-4-20250514 --n_shot 0 --max_workers 20

# Claude 3-shot
echo "=== Claude Sonnet 3-shot ==="
python ./eval/fewshot_api_eval.py \
    --data_path $DATA --train_path $TRAIN \
    --output_dir $BASE_DIR/fewshot_api_claude_3shot \
    --provider anthropic --model claude-sonnet-4-20250514 --n_shot 3 --max_workers 20

echo "=== All done ==="
