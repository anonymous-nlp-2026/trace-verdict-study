#!/usr/bin/env python3
"""VOTA verification: pick 10 samples (5 pass + 5 fail), compress, compare."""

import json
import sys
import os

sys.path.insert(0, "./src")
from vota import VOTACompressor, FullTraceFormatter

DATA_PATH = "./data/raw_traces.jsonl"

def main():
    compressor = VOTACompressor()
    formatter = FullTraceFormatter()
    
    pass_samples = []
    fail_samples = []
    
    with open(DATA_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if not rec["trace_raw"]:
                continue
            if rec["verdict"] and len(pass_samples) < 5:
                pass_samples.append(rec)
            elif not rec["verdict"] and len(fail_samples) < 5:
                fail_samples.append(rec)
            if len(pass_samples) >= 5 and len(fail_samples) >= 5:
                break
    
    samples = pass_samples + fail_samples
    print(f"Selected {len(pass_samples)} pass + {len(fail_samples)} fail samples\n")
    
    total_full = 0
    total_vota = 0
    
    for i, rec in enumerate(samples):
        verdict = "PASS" if rec["verdict"] else "FAIL"
        trace = rec["trace_raw"]
        test_code = rec.get("test_code", "")
        
        # Full trace
        full_text = formatter.format(trace)
        full_len = len(full_text)
        
        # VOTA compress
        vota_text = compressor.compress(trace, test_code)
        vota_len = len(vota_text)
        
        total_full += full_len
        total_vota += vota_len
        
        ratio = vota_len / full_len if full_len > 0 else 0
        
        print(f"--- Sample {i+1} [{verdict}] {rec['problem_id']} ---")
        print(f"Trace events: {len(trace)}, Full: {full_len} chars, VOTA: {vota_len} chars, Ratio: {ratio:.2%}")
        
        # Check assertion snapshots
        assert_lines = [l for l in vota_text.split("\n") if l.startswith("[ASSERT]")]
        has_actual_expected = any("actual=" in l and "expected=" in l for l in assert_lines)
        has_fallback = any("[fallback" in l for l in assert_lines)
        print(f"  Assertions: {len(assert_lines)}, has actual/expected: {has_actual_expected}, has fallback: {has_fallback}")
        
        # Check exception summaries
        error_lines = [l for l in vota_text.split("\n") if l.startswith("[ERROR]")]
        print(f"  Exceptions: {len(error_lines)}")
        
        # Check loop compression
        loop_lines = [l for l in vota_text.split("\n") if l.startswith("[LOOP]")]
        print(f"  Loops: {len(loop_lines)}")
        
        # Show VOTA output (truncated)
        print(f"  VOTA output preview:")
        for line in vota_text.split("\n")[:5]:
            print(f"    {line[:150]}")
        if len(vota_text.split("\n")) > 5:
            print(f"    ... ({len(vota_text.split(chr(10)))} total lines)")
        
        if error_lines:
            print(f"  Error lines:")
            for el in error_lines[:2]:
                print(f"    {el[:150]}")
        print()
    
    avg_ratio = total_vota / total_full if total_full > 0 else 0
    print(f"{'='*60}")
    print(f"Overall: Full={total_full} chars, VOTA={total_vota} chars")
    print(f"Average compression ratio: {avg_ratio:.2%}")
    print(f"Compression factor: {total_full/total_vota:.1f}x" if total_vota > 0 else "N/A")

if __name__ == "__main__":
    main()
