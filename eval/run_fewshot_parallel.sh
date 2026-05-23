#!/bin/bash
export PYTHONUNBUFFERED=1
# Activate your Python environment
conda activate base
cd .

run_one() {
    local model=$1 cond=$2 shots=$3
    local tag="${model}_${cond}_${shots}shot"
    echo "[$(date +%H:%M:%S)] START $tag" >> /tmp/fewshot_all.log
    python eval/fewshot_baseline.py --model "$model" --condition "$cond" --shots "$shots" --delay 0.1 > "/tmp/fewshot_${tag}.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE $tag" >> /tmp/fewshot_all.log
}

> /tmp/fewshot_all.log

run_one gpt4o exception_only 0 &
run_one gpt4o exception_only 3 &
run_one gpt4o no_trace 0 &
run_one gpt41 exception_only 0 &
run_one gpt41 exception_only 3 &
run_one gpt41 no_trace 0 &

wait

echo "[$(date +%H:%M:%S)] ALL 6 DONE, generating summary" >> /tmp/fewshot_all.log
python eval/fewshot_baseline.py --model gpt4o --condition exception_only --shots 0 --summary-only >> /tmp/fewshot_all.log 2>&1
echo "[$(date +%H:%M:%S)] === ALL COMPLETE ===" >> /tmp/fewshot_all.log
