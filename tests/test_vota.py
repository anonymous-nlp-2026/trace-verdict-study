"""Tests for VOTA compression, FullTraceFormatter, and AST assertion decomposition.

Covers:
  - Assertion-point snapshot extraction (pre-decomposed + AST fallback)
  - AST decomposition of HumanEval/MBPP assert patterns
  - Exception summary extraction
  - Loop compression with iteration counting
  - Compression ratio target (< 0.3)
  - Key information retention (actual/expected values preserved)
  - Graceful fallback on eval failure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vota import VOTACompressor, FullTraceFormatter
from src.assert_decompose import AssertDecomposer


# ---------------------------------------------------------------------------
# Helper: build trace events for a simple function execution
# ---------------------------------------------------------------------------

def _make_trace_sum_1_to_n(n: int) -> list[dict]:
    """Simulate trace for: def f(n): s=0; for i in range(n): s+=i; return s"""
    events = []
    # line 1: s = 0
    events.append({"line": 1, "event": "line", "locals": {"n": n}, "globals_subset": {}, "source": "test"})
    events.append({"line": 2, "event": "line", "locals": {"n": n, "s": 0}, "globals_subset": {}, "source": "test"})
    # loop body: lines 3-4
    s = 0
    for i in range(n):
        events.append({"line": 3, "event": "line", "locals": {"n": n, "s": s, "i": i}, "globals_subset": {}, "source": "test"})
        s += i
        events.append({"line": 4, "event": "line", "locals": {"n": n, "s": s, "i": i}, "globals_subset": {}, "source": "test"})
    # line 5: return
    events.append({"line": 5, "event": "line", "locals": {"n": n, "s": s}, "globals_subset": {}, "source": "test"})
    return events


def _make_trace_with_assert(actual, expected) -> tuple[list[dict], str]:
    """Simulate trace for a test that does: assert func(x) == expected."""
    test_code = "result = func(x)\nassert result == expected"
    events = [
        {"line": 1, "event": "line", "locals": {"x": 5}, "globals_subset": {}, "source": "test"},
        {"line": 2, "event": "line", "locals": {"result": actual, "expected": expected, "actual": actual}, "globals_subset": {}, "source": "test"},
    ]
    return events, test_code


def _make_trace_with_exception() -> list[dict]:
    """Simulate trace that hits a TypeError."""
    return [
        {"line": 10, "event": "line", "locals": {"x": 1, "y": "hello"}, "globals_subset": {}, "source": "test"},
        {"line": 11, "event": "line", "locals": {"x": 1, "y": "hello"}, "globals_subset": {}, "source": "test"},
        {
            "line": 12, "event": "exception",
            "locals": {"x": 1, "y": "hello"},
            "exception": {"type": "TypeError", "message": "unsupported operand type(s) for +: 'int' and 'str'"},
            "globals_subset": {}, "source": "test",
        },
    ]


# ---------------------------------------------------------------------------
# Tests: AST AssertDecomposer
# ---------------------------------------------------------------------------

class TestAssertDecomposer:
    """Test AST-level assert decomposition — the core fix for HumanEval traces."""

    def test_humaneval_basic_eq(self):
        """HumanEval pattern: assert candidate([1,2,3]) == 6"""
        decomp = AssertDecomposer()
        def candidate(lst):
            return sum(lst)
        result = decomp.decompose(
            "assert candidate([1, 2, 3]) == 6",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == 6
        assert result["expected"] == 6
        assert result["match"] is True

    def test_humaneval_string(self):
        """assert candidate("abc") == "cba" """
        decomp = AssertDecomposer()
        def candidate(s):
            return s[::-1]
        result = decomp.decompose(
            'assert candidate("abc") == "cba"',
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == "cba"
        assert result["expected"] == "cba"
        assert result["match"] is True

    def test_humaneval_mismatch(self):
        decomp = AssertDecomposer()
        def candidate(x):
            return x + 1
        result = decomp.decompose(
            "assert candidate(5) == 100",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == 6
        assert result["expected"] == 100
        assert result["match"] is False

    def test_in_operator(self):
        """assert candidate(5) in [1, 2, 5]"""
        decomp = AssertDecomposer()
        def candidate(x):
            return x
        result = decomp.decompose(
            "assert candidate(5) in [1, 2, 5]",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == 5
        assert result["expected"] == [1, 2, 5]
        assert result["match"] is True

    def test_not_operator(self):
        """assert not candidate(0) → actual=0, expected=False"""
        decomp = AssertDecomposer()
        def candidate(x):
            return x
        result = decomp.decompose(
            "assert not candidate(0)",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == 0
        assert result["expected"] is False
        assert result["match"] is True

    def test_approx_compare(self):
        """assert abs(candidate(3.14) - 6.28) < 1e-6"""
        decomp = AssertDecomposer()
        def candidate(x):
            return x * 2
        result = decomp.decompose(
            "assert abs(candidate(3.14) - 6.28) < 1e-6",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert abs(result["actual"] - 6.28) < 1e-6
        assert result["expected"] == 6.28
        assert result["match"] is True

    def test_mbpp_pattern(self):
        """MBPP often uses: assert func(args) == expected"""
        decomp = AssertDecomposer()
        def min_cost(arr, n):
            return 10
        result = decomp.decompose(
            "assert min_cost([1, 2, 3], 3) == 10",
            {"min_cost": min_cost},
            {},
        )
        assert result is not None
        assert result["actual"] == 10
        assert result["expected"] == 10
        assert result["match"] is True

    def test_eval_failure_returns_none(self):
        """If eval fails (e.g., undefined function), return None gracefully."""
        decomp = AssertDecomposer()
        result = decomp.decompose(
            "assert undefined_func(1) == 2",
            {},
            {},
        )
        assert result is None

    def test_not_assert_returns_none(self):
        decomp = AssertDecomposer()
        result = decomp.decompose("x = 1 + 2", {}, {})
        assert result is None

    def test_syntax_error_returns_none(self):
        decomp = AssertDecomposer()
        result = decomp.decompose("assert (((", {}, {})
        assert result is None

    def test_empty_string_returns_none(self):
        decomp = AssertDecomposer()
        result = decomp.decompose("", {}, {})
        assert result is None

    def test_bare_assert_truthy(self):
        """assert candidate(5) with truthy result"""
        decomp = AssertDecomposer()
        def candidate(x):
            return [x]
        result = decomp.decompose(
            "assert candidate(5)",
            {"candidate": candidate},
            {},
        )
        assert result is not None
        assert result["actual"] == [5]
        assert result["expected"] is True
        assert result["match"] is True

    def test_globals_accessible(self):
        """Functions in frame_globals should be callable."""
        decomp = AssertDecomposer()
        def my_func(x):
            return x * 2
        result = decomp.decompose(
            "assert my_func(3) == 6",
            {},
            {"my_func": my_func},
        )
        assert result is not None
        assert result["actual"] == 6
        assert result["match"] is True


# ---------------------------------------------------------------------------
# Tests: Assertion-Point Snapshots (pre-decomposed path)
# ---------------------------------------------------------------------------

class TestAssertionSnapshots:
    def test_assert_with_actual_expected(self):
        comp = VOTACompressor()
        events, test_code = _make_trace_with_assert(actual=42, expected=42)
        result = comp.compress(events, test_code)

        assert "[ASSERT]" in result
        assert "actual=42" in result
        assert "expected=42" in result
        assert "match=" not in result

    def test_assert_mismatch(self):
        comp = VOTACompressor()
        events, test_code = _make_trace_with_assert(actual=41, expected=42)
        result = comp.compress(events, test_code)

        assert "match=" not in result
        assert "actual=41" in result
        assert "expected=42" in result

    def test_assert_with_complex_values(self):
        comp = VOTACompressor()
        events, test_code = _make_trace_with_assert(actual=[1, 2, 3], expected=[1, 2, 3])
        result = comp.compress(events, test_code)

        assert "[ASSERT]" in result
        assert "match=" not in result
        assert "[1, 2, 3]" in result

    def test_no_asserts_in_code(self):
        comp = VOTACompressor()
        events = [{"line": 1, "event": "line", "locals": {"x": 1}, "globals_subset": {}, "source": "test"}]
        test_code = "x = 1\ny = 2"
        result = comp.compress(events, test_code)
        # Should not crash; no [ASSERT] sections


# ---------------------------------------------------------------------------
# Tests: VOTA with AST decomposition (HumanEval-style traces)
# ---------------------------------------------------------------------------

class TestVOTAWithASTDecomposition:
    """End-to-end: simulate HumanEval-style traces where locals only have `candidate`."""

    def test_humaneval_e2e(self):
        """Simulate real HumanEval trace: locals={candidate: <func>}, assert on line 2."""
        def candidate(lst):
            return sum(lst)

        test_code = "# test\nassert candidate([1, 2, 3]) == 6"
        events = [
            {"line": 2, "event": "line",
             "locals": {"candidate": candidate},
             "globals_subset": {}, "source": "test"},
        ]
        comp = VOTACompressor()
        result = comp.compress(events, test_code)

        assert "[ASSERT]" in result
        assert "actual=6" in result
        assert "expected=6" in result
        assert "match=" not in result

    def test_humaneval_e2e_fail(self):
        """Candidate returns wrong answer — VOTA should show match=False."""
        def candidate(lst):
            return 999

        test_code = "# test\nassert candidate([1, 2, 3]) == 6"
        events = [
            {"line": 2, "event": "line",
             "locals": {"candidate": candidate},
             "globals_subset": {}, "source": "test"},
        ]
        comp = VOTACompressor()
        result = comp.compress(events, test_code)

        assert "match=" not in result
        assert "actual=999" in result
        assert "expected=6" in result

    def test_source_lines_override(self):
        """source_lines parameter should override raw_line from test_code."""
        def candidate(x):
            return x + 1

        test_code = "assert candidate(5) == 6"
        events = [
            {"line": 1, "event": "line",
             "locals": {"candidate": candidate},
             "globals_subset": {}, "source": "test"},
        ]
        comp = VOTACompressor()
        result = comp.compress(
            events, test_code,
            source_lines={1: "assert candidate(5) == 6"},
        )
        assert "actual=6" in result
        assert "expected=6" in result
        assert "match=" not in result

    def test_fallback_on_decompose_failure(self):
        """When AST decomposition fails, should fall back with [fallback: full locals] marker."""
        test_code = "assert unknown_var == 42"
        events = [
            {"line": 1, "event": "line",
             "locals": {"x": 10, "y": 20},
             "globals_subset": {}, "source": "test"},
        ]
        comp = VOTACompressor()
        result = comp.compress(events, test_code)

        assert "[ASSERT]" in result
        assert "[fallback: full locals]" in result
        assert "x=10" in result

    def test_multiple_asserts(self):
        """Multiple assert lines in one test — all should be decomposed."""
        def candidate(x):
            return x * 2

        test_code = "assert candidate(1) == 2\nassert candidate(5) == 10\nassert candidate(0) == 0"
        events = [
            {"line": 1, "event": "line", "locals": {"candidate": candidate}, "globals_subset": {}, "source": "test"},
            {"line": 2, "event": "line", "locals": {"candidate": candidate}, "globals_subset": {}, "source": "test"},
            {"line": 3, "event": "line", "locals": {"candidate": candidate}, "globals_subset": {}, "source": "test"},
        ]
        comp = VOTACompressor()
        result = comp.compress(events, test_code)

        assert result.count("[ASSERT]") == 3
        assert "actual=2" in result
        assert "actual=10" in result
        assert "actual=0" in result


# ---------------------------------------------------------------------------
# Tests: Exception Summaries
# ---------------------------------------------------------------------------

class TestExceptionSummaries:
    def test_exception_output(self):
        comp = VOTACompressor()
        events = _make_trace_with_exception()
        result = comp.compress(events, "")

        assert "[ERROR]" in result
        assert "TypeError" in result
        assert "unsupported operand" in result
        assert "line 12" in result

    def test_exception_with_locals(self):
        comp = VOTACompressor()
        events = _make_trace_with_exception()
        result = comp.compress(events, "")
        assert "x=1" in result
        assert "y='hello'" in result

    def test_no_exceptions(self):
        comp = VOTACompressor()
        events = [{"line": 1, "event": "line", "locals": {}, "globals_subset": {}, "source": "test"}]
        result = comp.compress(events, "")
        assert "[ERROR]" not in result


# ---------------------------------------------------------------------------
# Tests: Loop Compression
# ---------------------------------------------------------------------------

class TestLoopCompression:
    def test_loop_detected(self):
        comp = VOTACompressor()
        events = _make_trace_sum_1_to_n(10)
        result = comp.compress(events, "")

        assert "[LOOP]" in result
        assert "iterations" in result

    def test_first_last_locals(self):
        comp = VOTACompressor()
        events = _make_trace_sum_1_to_n(10)
        result = comp.compress(events, "")

        # First iteration has s=0, i=0; last has accumulated sum
        assert "s=0" in result

    def test_no_loop_short_trace(self):
        comp = VOTACompressor()
        events = [
            {"line": 1, "event": "line", "locals": {"x": 1}, "globals_subset": {}, "source": "test"},
            {"line": 2, "event": "line", "locals": {"x": 2}, "globals_subset": {}, "source": "test"},
            {"line": 3, "event": "line", "locals": {"x": 3}, "globals_subset": {}, "source": "test"},
        ]
        result = comp.compress(events, "")
        assert "[LOOP]" not in result


# ---------------------------------------------------------------------------
# Tests: Compression Ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_ratio_under_threshold(self):
        """VOTA output should be < 30% the size of full trace for loop-heavy traces."""
        comp = VOTACompressor()
        fmt = FullTraceFormatter()

        events = _make_trace_sum_1_to_n(100)
        test_code = ""

        vota_text = comp.compress(events, test_code)
        full_text = fmt.format(events)

        ratio = len(vota_text) / len(full_text)
        assert ratio < 0.3, f"Compression ratio {ratio:.3f} exceeds 0.3 target"

    def test_ratio_with_mixed_content(self):
        """Mixed trace (loop + assert + exception) should still compress well."""
        comp = VOTACompressor()
        fmt = FullTraceFormatter()

        events = _make_trace_sum_1_to_n(50)
        # Add assertion event
        events.append({"line": 10, "event": "line",
                        "locals": {"result": 1225, "expected": 1225, "actual": 1225},
                        "globals_subset": {}, "source": "test"})
        # Add exception event
        events.append({
            "line": 15, "event": "exception",
            "locals": {"err": "boom"},
            "exception": {"type": "RuntimeError", "message": "boom"},
            "globals_subset": {}, "source": "test",
        })

        test_code = "s = sum_func(50)\nassert s == 1225\n# ...\nraise RuntimeError\n# padding\n# more\n# lines\n# here\n# to\nassert result == expected"
        vota_text = comp.compress(events, test_code)
        full_text = fmt.format(events)

        ratio = len(vota_text) / len(full_text)
        assert ratio < 0.3, f"Compression ratio {ratio:.3f} exceeds 0.3 target"


# ---------------------------------------------------------------------------
# Tests: Key Information Retention
# ---------------------------------------------------------------------------

class TestKeyInfoRetention:
    def test_actual_expected_preserved(self):
        """Actual and expected values from assert points must appear in VOTA output."""
        comp = VOTACompressor()
        events, test_code = _make_trace_with_assert(actual="hello world", expected="hello world")
        result = comp.compress(events, test_code)

        assert "'hello world'" in result
        assert "match=" not in result

    def test_exception_info_preserved(self):
        """Exception type and message must appear in VOTA output."""
        comp = VOTACompressor()
        events = _make_trace_with_exception()
        result = comp.compress(events, "")

        assert "TypeError" in result
        assert "unsupported operand" in result


# ---------------------------------------------------------------------------
# Tests: FullTraceFormatter
# ---------------------------------------------------------------------------

class TestFullTraceFormatter:
    def test_basic_format(self):
        fmt = FullTraceFormatter()
        events = [
            {"line": 1, "event": "line", "locals": {"x": 1}, "globals_subset": {}, "source": "test"},
            {"line": 2, "event": "line", "locals": {"x": 2, "y": 3}, "globals_subset": {}, "source": "test"},
        ]
        result = fmt.format(events)
        assert "[Line 1] x=1" in result
        assert "[Line 2] x=2, y=3" in result

    def test_empty_trace(self):
        fmt = FullTraceFormatter()
        result = fmt.format([])
        assert result == ""

    def test_line_count_matches_events(self):
        fmt = FullTraceFormatter()
        events = _make_trace_sum_1_to_n(5)
        result = fmt.format(events)
        assert result.count("[Line") == len(events)
