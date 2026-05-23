import json
from collections import Counter

# Load exception_only test data
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

# Categorize error types in exception_only fail samples
error_types = Counter()
assertion_fails = []  # AssertionError from test check() - these ARE silent logic errors
code_crashes = []     # TypeError, ValueError, etc. from candidate code
no_trace_fails = []

for i in range(total):
    if ft_data[i]['output'].lower() != 'fail':
        continue
    
    exc_inp = exc_data[i]['input']
    exc_trace = exc_inp.split('Execution trace:')[1] if 'Execution trace:' in exc_inp else ''
    exc_trace = exc_trace.split('Does this code pass all tests?')[0].strip()
    
    if '[No trace available]' in exc_trace:
        no_trace_fails.append(i)
        error_types['[No trace available]'] += 1
        continue
    
    if '[ERROR]' not in exc_trace:
        error_types['[No ERROR tag]'] += 1
        continue
    
    # Extract error type from [ERROR] line
    error_line = exc_trace.split('[ERROR]')[1].split('|')[0].strip()
    
    # Classify
    if 'AssertionError' in error_line or 'AssertionError' in error_line:
        assertion_fails.append(i)
        error_types['AssertionError (test assertion)'] += 1
    elif 'TypeError' in error_line:
        code_crashes.append(i)
        error_types['TypeError'] += 1
    elif 'ValueError' in error_line:
        code_crashes.append(i)
        error_types['ValueError'] += 1
    elif 'IndexError' in error_line:
        code_crashes.append(i)
        error_types['IndexError'] += 1
    elif 'KeyError' in error_line:
        code_crashes.append(i)
        error_types['KeyError'] += 1
    elif 'AttributeError' in error_line:
        code_crashes.append(i)
        error_types['AttributeError'] += 1
    elif 'NameError' in error_line:
        code_crashes.append(i)
        error_types['NameError'] += 1
    elif 'RecursionError' in error_line:
        code_crashes.append(i)
        error_types['RecursionError'] += 1
    elif 'ZeroDivisionError' in error_line:
        code_crashes.append(i)
        error_types['ZeroDivisionError'] += 1
    elif 'RuntimeError' in error_line:
        code_crashes.append(i)
        error_types['RuntimeError'] += 1
    elif 'StopIteration' in error_line:
        code_crashes.append(i)
        error_types['StopIteration'] += 1
    elif 'TimeoutError' in error_line or 'Timeout' in error_line:
        code_crashes.append(i)
        error_types['TimeoutError'] += 1
    else:
        # Show what it is
        error_types[f'Other: {error_line[:60]}'] += 1
        code_crashes.append(i)

fail_total = len(assertion_fails) + len(code_crashes) + len(no_trace_fails)

print("=" * 60)
print("SILENT LOGIC ERROR ANALYSIS (v2)")
print("=" * 60)
print(f"Total test: {total}, Fail: {fail_total}")
print()
print("Error type distribution:")
for k, v in error_types.most_common():
    print(f"  {k}: {v} ({v/fail_total*100:.1f}%)")

print()
print("=" * 60)
print("RECLASSIFIED FAIL BREAKDOWN")
print("=" * 60)
print(f"  AssertionError (silent logic errors): {len(assertion_fails)} ({len(assertion_fails)/fail_total*100:.1f}%)")
print(f"    = code runs fine but produces wrong output")
print(f"    = test assert fails, NOT a code crash")
print(f"  Code crashes (real exceptions): {len(code_crashes)} ({len(code_crashes)/fail_total*100:.1f}%)")
print(f"    = code itself throws TypeError/ValueError/etc.")
print(f"  No trace available: {len(no_trace_fails)} ({len(no_trace_fails)/fail_total*100:.1f}%)")

# Now load predictions and compare
def load_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            preds[d['id']] = d['pred']
    return preds

conditions = {
    'no_trace':       './checkpoints/no_trace/final/predictions.jsonl',
    'exception_only': './checkpoints/exception_only/final/predictions.jsonl',
    'ae_only':        './checkpoints/ae_only/final/predictions.jsonl',
    'loop_only':      './checkpoints/loop_only/final/predictions.jsonl',
    'full_trace':     './checkpoints/full_trace/final/predictions.jsonl',
    'vota':           './checkpoints/vota/final/predictions.jsonl',
}
all_preds = {name: load_preds(path) for name, path in conditions.items()}

print()
print("=" * 60)
print("FAIL RECALL: AssertionError (silent) vs Code Crash")
print("=" * 60)

header = f"{'Condition':<20} {'Assert (silent)':>18} {'Code crash':>14} {'No trace':>12} {'Gap(crash-sil)':>15}"
print(header)
print("-" * len(header))

for name in ['no_trace', 'exception_only', 'ae_only', 'loop_only', 'full_trace', 'vota']:
    preds = all_preds[name]
    
    n_a = len(assertion_fails)
    c_a = sum(1 for i in assertion_fails if preds.get(i, '') == 'fail')
    r_a = c_a/n_a*100 if n_a > 0 else 0
    
    n_c = len(code_crashes)
    c_c = sum(1 for i in code_crashes if preds.get(i, '') == 'fail')
    r_c = c_c/n_c*100 if n_c > 0 else 0
    
    n_nt = len(no_trace_fails)
    c_nt = sum(1 for i in no_trace_fails if preds.get(i, '') == 'fail')
    r_nt = c_nt/n_nt*100 if n_nt > 0 else 0
    
    gap = r_c - r_a
    
    print(f"{name:<20} {c_a:>3}/{n_a} {r_a:5.1f}%   {c_c:>3}/{n_c} {r_c:5.1f}%   {c_nt:>2}/{n_nt} {r_nt:5.1f}%   {gap:>+6.1f}pp")

# Show some AssertionError examples (silent logic errors)
print()
print("=" * 60)
print("EXAMPLES: AssertionError samples (first 3)")
print("=" * 60)
for idx in assertion_fails[:3]:
    ft_inp = ft_data[idx]['input']
    ft_trace = ft_inp.split('Execution trace:')[1] if 'Execution trace:' in ft_inp else ''
    ft_trace = ft_trace.split('Does this code pass all tests?')[0].strip()
    print(f"\n--- Sample {idx} (problem={ft_data[idx]['problem_id']}) ---")
    print(f"Full trace (last 200 chars): ...{ft_trace[-200:]}")
    
    exc_inp = exc_data[idx]['input']
    exc_trace = exc_inp.split('Execution trace:')[1] if 'Execution trace:' in exc_inp else ''
    exc_trace = exc_trace.split('Does this code pass all tests?')[0].strip()
    print(f"Exception_only trace: {exc_trace[:200]}")

# Show code crash examples
print()
print("=" * 60)
print("EXAMPLES: Code crash samples (first 3)")
print("=" * 60)
for idx in code_crashes[:3]:
    exc_inp = exc_data[idx]['input']
    exc_trace = exc_inp.split('Execution trace:')[1] if 'Execution trace:' in exc_inp else ''
    exc_trace = exc_trace.split('Does this code pass all tests?')[0].strip()
    print(f"\n--- Sample {idx} (problem={ft_data[idx]['problem_id']}) ---")
    print(f"Exception_only trace: {exc_trace[:300]}")
