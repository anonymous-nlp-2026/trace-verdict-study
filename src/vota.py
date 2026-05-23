"""VOTA (Verdict-Optimized Trace Abstraction) — compact trace representation for code verdict tasks.

Input: list of trace events from sys.settrace (each a dict with line/event/locals/globals_subset)
       + test source code string.
Output: compressed text retaining only verdict-relevant information.

Three compression components:
  1. Assertion-Point Snapshots — variable state at assert/comparison points
  2. Exception Summaries — error type, message, location, and local vars
  3. Loop Compression — first/last iteration + count for repeated line sequences
"""

from __future__ import annotations

import ast
import re
import json
from collections import defaultdict
from typing import Any

from src.assert_decompose import AssertDecomposer


def _repr_value(v: Any, max_len: int = 80) -> str:
    """Compact repr of a value, truncated if needed."""
    s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _format_locals(local_vars: dict, max_len: int = 80) -> str:
    """Format a locals dict as 'k1=v1, k2=v2'."""
    if not local_vars:
        return ""
    parts = []
    for k, v in local_vars.items():
        if k.startswith("__"):
            continue
        parts.append(f"{k}={_repr_value(v, max_len)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Assertion-point detection via AST
# ---------------------------------------------------------------------------

class _AssertVisitor(ast.NodeVisitor):
    """Find assert statements and bare comparisons that act as test checks."""

    def __init__(self):
        self.assert_lines: set[int] = set()

    def visit_Assert(self, node: ast.Assert):
        self.assert_lines.add(node.lineno)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        # Bare `==` at statement level (e.g. `func(x) == 3`) used as implicit assert
        if hasattr(node, "_stmt_lineno"):
            self.assert_lines.add(node._stmt_lineno)
        self.generic_visit(node)


def _find_assert_lines(test_code: str) -> set[int]:
    """Return line numbers that contain assert statements or top-level comparisons."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return set()

    visitor = _AssertVisitor()

    # Tag top-level Expr nodes that are bare comparisons
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Compare):
            node.value._stmt_lineno = node.lineno

    visitor.visit(tree)
    return visitor.assert_lines


def _extract_assert_info(test_code: str, lineno: int) -> dict | None:
    """Parse a single assert line to extract expected value and function call pattern."""
    lines = test_code.splitlines()
    if lineno < 1 or lineno > len(lines):
        return None
    line = lines[lineno - 1].strip()
    return {"raw": line, "lineno": lineno}


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _detect_loops(trace_events: list[dict]) -> list[dict]:
    """Detect loops as repeated sequences of the same line numbers.

    Algorithm: scan line numbers for contiguous runs where a subsequence
    repeats. A 'loop' is identified when the same line number appears
    3+ times, and the block of lines between repetitions forms a cycle.
    We group consecutive events sharing the same repeating line-set
    into one loop record.
    """
    if not trace_events:
        return []

    loops = []
    line_positions: dict[int, list[int]] = defaultdict(list)
    for i, ev in enumerate(trace_events):
        line_positions[ev.get("line", -1)].append(i)

    # Find line numbers that appear >= 3 times (likely loop bodies)
    hot_lines = {ln for ln, positions in line_positions.items() if len(positions) >= 3 and ln != -1}
    if not hot_lines:
        return []

    # Group contiguous runs of events that involve hot_lines
    visited = set()
    i = 0
    n = len(trace_events)
    while i < n:
        ev = trace_events[i]
        ln = ev.get("line", -1)
        if ln in hot_lines and i not in visited:
            # Find the extent of this loop block
            start = i
            # Collect the repeating line pattern
            pattern_lines = set()
            j = i
            while j < n and trace_events[j].get("line", -1) in hot_lines:
                pattern_lines.add(trace_events[j]["line"])
                visited.add(j)
                j += 1
            end = j  # exclusive

            iterations = end - start
            # Count iterations as repetitions of the smallest repeating unit
            line_seq = [trace_events[k]["line"] for k in range(start, end)]
            cycle_len = _find_cycle_length(line_seq)
            iter_count = len(line_seq) // cycle_len if cycle_len > 0 else 1

            first_locals = trace_events[start].get("locals", {})
            last_locals = trace_events[end - 1].get("locals", {})

            loops.append({
                "start_idx": start,
                "end_idx": end,
                "lines": sorted(pattern_lines),
                "iterations": iter_count,
                "first_locals": first_locals,
                "last_locals": last_locals,
            })
            i = end
        else:
            i += 1
    return loops


def _find_cycle_length(seq: list[int]) -> int:
    """Find the length of the shortest repeating cycle in a sequence of line numbers."""
    n = len(seq)
    for length in range(1, n // 2 + 1):
        if n % length != 0:
            continue
        pattern = seq[:length]
        if all(seq[i:i+length] == pattern for i in range(0, n, length)):
            return length
    return n  # no cycle found, treat entire sequence as one "iteration"


# ---------------------------------------------------------------------------
# VOTACompressor
# ---------------------------------------------------------------------------

class VOTACompressor:
    """Compress raw sys.settrace events into a verdict-focused VOTA summary."""

    def __init__(self):
        self._decomposer = AssertDecomposer()

    def compress(
        self,
        trace_events: list[dict],
        test_code: str,
        source_lines: dict[int, str] | None = None,
    ) -> str:
        """Compress raw trace into VOTA format text.

        Args:
            trace_events: list of dicts from sys.settrace, each with
                          keys: line, event, locals, globals_subset
            test_code: source code of the test (used to locate asserts)
            source_lines: optional mapping lineno -> source text, for
                          real-time AST decomposition when trace events
                          lack pre-decomposed actual/expected fields.

        Returns:
            Multi-line VOTA text with [ASSERT], [ERROR], and [LOOP] sections.
        """
        parts: list[str] = []

        # 1. Assertion-Point Snapshots
        assert_lines = _find_assert_lines(test_code)
        parts.extend(
            self._extract_assertions(trace_events, assert_lines, test_code, source_lines)
        )

        # 2. Exception Summaries
        parts.extend(self._extract_exceptions(trace_events))

        # 3. Loop Compression
        parts.extend(self._extract_loops(trace_events))

        if not parts:
            parts.append("[INFO] No assertions, exceptions, or loops detected in trace.")

        return "\n".join(parts)

    # -- assertion snapshots --

    def _extract_assertions(
        self,
        trace_events: list[dict],
        assert_lines: set[int],
        test_code: str,
        source_lines: dict[int, str] | None = None,
    ) -> list[str]:
        results = []
        code_lines = test_code.splitlines()

        for ev in trace_events:
            if ev.get("source") != "test":
                continue
            ln = ev.get("line", -1)
            if ln not in assert_lines:
                continue

            raw_line = code_lines[ln - 1].strip() if 1 <= ln <= len(code_lines) else ""
            local_vars = ev.get("locals", {})
            globals_subset = ev.get("globals_subset", {})

            # Strategy 0: pre-decomposed actual/expected directly in trace event
            if "actual" in ev and "expected" in ev:
                results.append(
                    f"[ASSERT] {raw_line} | actual={_repr_value(ev['actual'])}, "
                    f"expected={_repr_value(ev['expected'])}"
                )
                continue

            # Strategy 1: pre-decomposed actual/expected in locals
            actual = local_vars.get("actual", local_vars.get("result", None))
            expected = local_vars.get("expected", None)

            if actual is not None and expected is not None:
                results.append(
                    f"[ASSERT] {raw_line} | actual={_repr_value(actual)}, "
                    f"expected={_repr_value(expected)}"
                )
                continue

            # Strategy 2: AST decomposition on the source line
            src = None
            if source_lines and ln in source_lines:
                src = source_lines[ln]
            elif raw_line:
                src = raw_line

            if src:
                decomposed = self._decomposer.decompose(src, local_vars, globals_subset)
                if decomposed is not None:
                    results.append(
                        f"[ASSERT] {raw_line} | actual={_repr_value(decomposed['actual'])}, "
                        f"expected={_repr_value(decomposed['expected'])}"
                    )
                    continue

            # Strategy 3: fallback — show all locals, marked as fallback
            var_str = _format_locals(local_vars)
            suffix = f" | [fallback: full locals] {var_str}" if var_str else ""
            results.append(f"[ASSERT] {raw_line}{suffix}")

        return results

    # -- exception summaries --

    def _extract_exceptions(self, trace_events: list[dict]) -> list[str]:
        results = []
        for ev in trace_events:
            if ev.get("event") == "exception":
                exc_info = ev.get("exception", {})
                exc_type = exc_info.get("type", ev.get("exc_type", "UnknownError"))
                exc_msg = exc_info.get("message", ev.get("exc_message", ""))
                ln = ev.get("line", "?")
                local_vars = ev.get("locals", {})
                var_str = _format_locals(local_vars)

                line = f"[ERROR] {exc_type} at line {ln}: {exc_msg}"
                if var_str:
                    line += f" | locals: {var_str}"
                results.append(line)
        return results

    # -- loop compression --

    def _extract_loops(self, trace_events: list[dict]) -> list[str]:
        results = []
        loops = _detect_loops(trace_events)
        for loop in loops:
            lines_range = f"{loop['lines'][0]}-{loop['lines'][-1]}" if len(loop['lines']) > 1 else str(loop['lines'][0])
            first_str = _format_locals(loop["first_locals"])
            last_str = _format_locals(loop["last_locals"])
            results.append(
                f"[LOOP] lines {lines_range}, {loop['iterations']} iterations: "
                f"first={{{first_str}}} → last={{{last_str}}}"
            )
        return results


# ---------------------------------------------------------------------------
# FullTraceFormatter — baseline format (scratchpad-style, every line)
# ---------------------------------------------------------------------------

class FullTraceFormatter:
    """Format raw trace events into verbose line-by-line scratchpad text.

    Used as the full-trace baseline condition for training data.
    Format: [Line X] var1=val1, var2=val2
    """

    def format(self, trace_events: list[dict]) -> str:
        """Format all trace events into scratchpad text.

        Args:
            trace_events: list of dicts from sys.settrace

        Returns:
            Multi-line text, one line per trace event.
        """
        lines = []
        for ev in trace_events:
            ln = ev.get("line", "?")
            local_vars = ev.get("locals", {})
            var_str = _format_locals(local_vars)
            lines.append(f"[Line {ln}] {var_str}")
        return "\n".join(lines)
