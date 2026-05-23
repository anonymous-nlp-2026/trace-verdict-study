import json

# Load exception_only test data to identify which samples have exceptions
exc_data = []
with open('./data/exception_only_test.jsonl') as f:
    for line in f:
        exc_data.append(json.loads(line))

# Load full_trace test data
ft_data = []
with open('./data/full_trace_test.jsonl') as f:
    for line in f:
        ft_data.append(json.loads(line))

total = len(ft_data)
assert total == len(exc_data)

# Verify alignment
for i in range(min(10, total)):
    assert ft_data[i]['problem_id'] == exc_data[i]['problem_id']
    assert ft_data[i]['output'] == exc_data[i]['output']

# Categorize samples
fail_with_exc = []
fail_silent = []
fail_no_trace = []
pass_samples = []

for i in range(total):
    label = ft_data[i]['output'].lower()
    exc_inp = exc_data[i]['input']
    exc_trace = exc_inp.split('Execution trace:')[1] if 'Execution trace:' in exc_inp else ''

    if label == 'pass':
        pass_samples.append(i)
        continue

    if '[No trace available]' in exc_trace:
        fail_no_trace.append(i)
    elif '[ERROR]' in exc_trace:
        fail_with_exc.append(i)
    elif '[No exception data in trace]' in exc_trace:
        fail_silent.append(i)
    else:
        fail_silent.append(i)

fail_total = len(fail_with_exc) + len(fail_silent) + len(fail_no_trace)

print("=" * 60)
print("SILENT LOGIC ERROR ANALYSIS")
print("=" * 60)
print(f"Total test samples: {total}")
print(f"  Pass: {len(pass_samples)} ({len(pass_samples)/total*100:.1f}%)")
print(f"  Fail: {fail_total} ({fail_total/total*100:.1f}%)")
print()
print("Fail samples breakdown:")
print(f"  With exception (crash/error): {len(fail_with_exc)} ({len(fail_with_exc)/fail_total*100:.1f}% of fails)")
print(f"  Silent logic errors (wrong output): {len(fail_silent)} ({len(fail_silent)/fail_total*100:.1f}% of fails)")
print(f"  No trace available: {len(fail_no_trace)} ({len(fail_no_trace)/fail_total*100:.1f}% of fails)")
print()
print(f"Silent error rate (of total): {len(fail_silent)/total*100:.1f}%")

# Debug: show what the fail samples look like in exception_only
print()
print("=" * 60)
print("DEBUG: What do fail samples look like in exception_only?")
print("=" * 60)
fail_exc_content_types = {}
for i in range(total):
    if ft_data[i]['output'].lower() != 'fail':
        continue
    exc_inp = exc_data[i]['input']
    exc_trace = exc_inp.split('Execution trace:')[1] if 'Execution trace:' in exc_inp else ''
    exc_trace = exc_trace.split('Does this code pass all tests?')[0].strip()
    if '[No trace available]' in exc_trace:
        key = '[No trace available]'
    elif '[No exception data in trace]' in exc_trace:
        key = '[No exception data in trace]'
    elif '[ERROR]' in exc_trace:
        key = '[ERROR] ...'
    else:
        key = f'OTHER: {exc_trace[:100]}'
    fail_exc_content_types[key] = fail_exc_content_types.get(key, 0) + 1

for k, v in sorted(fail_exc_content_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Also check full_trace data for fail samples
print()
print("=" * 60)
print("DEBUG: Full trace content for fail samples")
print("=" * 60)
ft_content_types = {}
for i in range(total):
    if ft_data[i]['output'].lower() != 'fail':
        continue
    ft_inp = ft_data[i]['input']
    ft_trace = ft_inp.split('Execution trace:')[1] if 'Execution trace:' in ft_inp else ''
    ft_trace = ft_trace.split('Does this code pass all tests?')[0].strip()
    if '[No trace available]' in ft_trace:
        key = '[No trace available]'
    elif '[ERROR]' in ft_trace:
        key = 'has [ERROR] in full trace'
    elif '[Line' in ft_trace:
        key = 'has [Line] trace (no ERROR)'
    else:
        key = f'OTHER: {ft_trace[:80]}'
    ft_content_types[key] = ft_content_types.get(key, 0) + 1

for k, v in sorted(ft_content_types.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Load predictions
def load_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            preds[d['id']] = d['pred']
    return preds

conditions = {
    'exception_only': './checkpoints/exception_only/final/predictions.jsonl',
    'full_trace':     './checkpoints/full_trace/final/predictions.jsonl',
    'vota':           './checkpoints/vota/final/predictions.jsonl',
    'no_trace':       './checkpoints/no_trace/final/predictions.jsonl',
    'ae_only':        './checkpoints/ae_only/final/predictions.jsonl',
    'loop_only':      './checkpoints/loop_only/final/predictions.jsonl',
}

all_preds = {name: load_preds(path) for name, path in conditions.items()}

# Compute recall on each fail subcategory
print()
print("=" * 60)
print("FAIL RECALL BY ERROR TYPE (correctly predicting fail)")
print("=" * 60)

header = f"{'Condition':<20} {'Exc-fails':>12} {'Silent fails':>14} {'No-trace fails':>16}"
print(header)
print("-" * len(header))

for name in ['no_trace', 'exception_only', 'ae_only', 'loop_only', 'full_trace', 'vota']:
    preds = all_preds[name]
    
    n_exc = len(fail_with_exc)
    c_exc = sum(1 for i in fail_with_exc if preds.get(i, '') == 'fail') if n_exc > 0 else 0
    r_exc = f"{c_exc}/{n_exc} {c_exc/n_exc*100:.1f}%" if n_exc > 0 else "N/A"
    
    n_sil = len(fail_silent)
    c_sil = sum(1 for i in fail_silent if preds.get(i, '') == 'fail') if n_sil > 0 else 0
    r_sil = f"{c_sil}/{n_sil} {c_sil/n_sil*100:.1f}%" if n_sil > 0 else "N/A"
    
    n_nt = len(fail_no_trace)
    c_nt = sum(1 for i in fail_no_trace if preds.get(i, '') == 'fail') if n_nt > 0 else 0
    r_nt = f"{c_nt}/{n_nt} {c_nt/n_nt*100:.1f}%" if n_nt > 0 else "N/A"
    
    print(f"{name:<20} {r_exc:>12} {r_sil:>14} {r_nt:>16}")

# Also overall accuracy
print()
print("=" * 60)
print("OVERALL ACCURACY PER CONDITION")
print("=" * 60)
for name in ['no_trace', 'exception_only', 'ae_only', 'loop_only', 'full_trace', 'vota']:
    preds = all_preds[name]
    correct = sum(1 for i in range(total) if preds.get(i, '') == ft_data[i]['output'].lower())
    print(f"  {name:<20} {correct}/{total} = {correct/total*100:.1f}%")
