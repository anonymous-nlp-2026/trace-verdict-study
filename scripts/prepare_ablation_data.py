"""Remove actual/expected values from VOTA traces for plan_001 ablation.

Input:  vota_train.jsonl, vota_test.jsonl
Output: vota_no_actexp_train.jsonl, vota_no_actexp_test.jsonl

Processing: strips runtime actual=X, expected=Y snapshots and [fallback: full locals]
dumps from [ASSERT] lines. Keeps assertion text, [ERROR], [LOOP], and all non-trace content.
"""

import json
import sys
import os

DATA_DIR = "./data"

SPLITS = [
    ("vota_train.jsonl", "vota_no_actexp_train.jsonl"),
    ("vota_test.jsonl", "vota_no_actexp_test.jsonl"),
]


def strip_actual_expected(input_text: str) -> tuple[str, int, int]:
    """Remove actual/expected values from VOTA trace lines.

    Returns: (processed_text, num_actual_expected_removed, num_fallback_removed)
    """
    lines = input_text.split("\n")
    out_lines = []
    ae_removed = 0
    fb_removed = 0

    for line in lines:
        if not line.startswith("[ASSERT]"):
            out_lines.append(line)
            continue

        # [ASSERT] ... | actual=X, expected=Y  ->  [ASSERT] ...
        idx_ae = line.find(" | actual=")
        if idx_ae != -1:
            out_lines.append(line[:idx_ae])
            ae_removed += 1
            continue

        # [ASSERT] ... | [fallback: full locals] ...  ->  drop entire line
        idx_fb = line.find(" | [fallback:")
        if idx_fb != -1:
            fb_removed += 1
            continue

        # Other [ASSERT] lines (bare tags, etc.) - keep as-is
        out_lines.append(line)

    return "\n".join(out_lines), ae_removed, fb_removed


def process_split(in_name: str, out_name: str):
    in_path = os.path.join(DATA_DIR, in_name)
    out_path = os.path.join(DATA_DIR, out_name)

    if not os.path.exists(in_path):
        print(f"  SKIP: {in_path} not found")
        return

    total = 0
    total_ae = 0
    total_fb = 0

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            processed, ae, fb = strip_actual_expected(obj["input"])
            obj["input"] = processed
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1
            total_ae += ae
            total_fb += fb

    print(f"  {in_name} -> {out_name}")
    print(f"    rows: {total}")
    print(f"    actual/expected removed: {total_ae}")
    print(f"    fallback locals removed: {total_fb}")


if __name__ == "__main__":
    print("=== plan_001 ablation: remove actual/expected from VOTA ===")
    for in_name, out_name in SPLITS:
        process_split(in_name, out_name)
    print("Done.")
