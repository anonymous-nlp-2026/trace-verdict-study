#!/bin/bash
export PYTHONUNBUFFERED=1
# Activate your Python environment
conda activate base
cd .

python eval/fewshot_baseline.py --model gpt4o --condition exception_only --shots 0 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt4o exception_only 0shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt4o --condition exception_only --shots 3 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt4o exception_only 3shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt4o --condition no_trace --shots 0 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt4o no_trace 0shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt41 --condition exception_only --shots 0 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt41 exception_only 0shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt41 --condition exception_only --shots 3 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt41 exception_only 3shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt41 --condition no_trace --shots 0 --delay 0.3 >> /tmp/fewshot_all.log 2>&1
echo "=== gpt41 no_trace 0shot DONE ===" >> /tmp/fewshot_all.log

python eval/fewshot_baseline.py --model gpt4o --condition exception_only --shots 0 --summary-only >> /tmp/fewshot_all.log 2>&1
echo "=== ALL DONE ===" >> /tmp/fewshot_all.log
