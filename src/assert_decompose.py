"""AST-based assertion decomposer for extracting actual/expected values from assert statements.

Solves the core problem: HumanEval tests use `assert candidate([1,2,3]) == 6`,
where sys.settrace locals only contain `candidate=<function>`. This module parses
the assert statement via AST and evaluates sub-expressions against the frame's
namespace to recover actual and expected values.
"""

from __future__ import annotations

import ast
import threading
from typing import Any

_EVAL_TIMEOUT = 1  # seconds


def _repr_value(v: Any, max_len: int = 120) -> str:
    s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _safe_eval(node: ast.expr, env: dict, timeout: float = _EVAL_TIMEOUT) -> Any:
    """Eval a single AST expression node against env with timeout protection.

    Uses threading to enforce timeout since signal.alarm doesn't work in
    non-main threads. Returns _SENTINEL on failure.
    """
    expr = ast.Expression(body=node)
    ast.fix_missing_locations(expr)
    code = compile(expr, "<assert_decompose>", "eval")
    result = _SENTINEL
    exc_holder = [None]

    def _run():
        nonlocal result
        try:
            result = eval(code, {"__builtins__": __builtins__}, env)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive() or exc_holder[0] is not None:
        return _SENTINEL
    return result


class _Sentinel:
    """Unique sentinel to distinguish eval failure from None results."""
    def __repr__(self):
        return "<EVAL_FAILED>"

_SENTINEL = _Sentinel()


class AssertDecomposer:
    """Decompose assert statements via AST to extract actual and expected values."""

    def decompose(
        self,
        source_line: str,
        frame_locals: dict,
        frame_globals: dict,
    ) -> dict | None:
        """Parse an assert statement and evaluate its sub-expressions.

        Args:
            source_line: Source code line (e.g. "assert candidate([1,2,3]) == 6")
            frame_locals: frame.f_locals from sys.settrace callback
            frame_globals: frame.f_globals from sys.settrace callback

        Returns:
            {"actual": val, "expected": val, "expression": str, "match": bool}
            None if not an assert or parsing fails.
        """
        source_line = source_line.strip()
        if not source_line:
            return None

        try:
            tree = ast.parse(source_line, mode="exec")
        except SyntaxError:
            return None

        if not tree.body or not isinstance(tree.body[0], ast.Assert):
            return None

        test_node = tree.body[0].test
        env = {**frame_globals, **frame_locals}

        # Check approx pattern first — it's also a Compare node, so must precede generic compare
        if self._is_approx_compare(test_node):
            return self._handle_approx(test_node, env, source_line)

        # Generic compare: ==, !=, in, not in, <, <=, >, >=
        if isinstance(test_node, ast.Compare) and len(test_node.ops) == 1:
            return self._handle_compare(test_node, env, source_line)

        # `assert not expr` → actual=expr, expected=False
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            return self._handle_not(test_node, env, source_line)

        # Bare `assert expr` → actual=expr, expected=True
        return self._handle_bare(test_node, env, source_line)

    def _handle_compare(
        self, node: ast.Compare, env: dict, source_line: str
    ) -> dict | None:
        left = _safe_eval(node.left, env)
        right = _safe_eval(node.comparators[0], env)
        if left is _SENTINEL or right is _SENTINEL:
            return None

        op = node.ops[0]
        if isinstance(op, (ast.Eq, ast.NotEq)):
            match = (left == right) if isinstance(op, ast.Eq) else (left != right)
        elif isinstance(op, ast.In):
            match = left in right
        elif isinstance(op, ast.NotIn):
            match = left not in right
        elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            try:
                match = eval(f"left {_op_symbol(op)} right", {"left": left, "right": right})
            except Exception:
                return None
        else:
            return None

        return {
            "actual": left,
            "expected": right,
            "expression": source_line,
            "match": bool(match),
        }

    def _handle_not(
        self, node: ast.UnaryOp, env: dict, source_line: str
    ) -> dict | None:
        val = _safe_eval(node.operand, env)
        if val is _SENTINEL:
            return None
        return {
            "actual": val,
            "expected": False,
            "expression": source_line,
            "match": not val,
        }

    def _is_approx_compare(self, node: ast.expr) -> bool:
        """Detect pattern: abs(X - Y) < eps."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.LtE)):
            return False
        left = node.left
        if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                and left.func.id == "abs" and len(left.args) == 1):
            return False
        inner = left.args[0]
        return isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub)

    def _handle_approx(
        self, node: ast.Compare, env: dict, source_line: str
    ) -> dict | None:
        """Handle `assert abs(actual_expr - expected_val) < eps`."""
        abs_call = node.left  # abs(X - Y)
        sub_node = abs_call.args[0]  # X - Y

        actual = _safe_eval(sub_node.left, env)
        expected = _safe_eval(sub_node.right, env)
        eps = _safe_eval(node.comparators[0], env)

        if any(v is _SENTINEL for v in (actual, expected, eps)):
            return None

        try:
            match = abs(actual - expected) < eps
        except Exception:
            return None

        return {
            "actual": actual,
            "expected": expected,
            "expression": source_line,
            "match": bool(match),
        }

    def _handle_bare(
        self, node: ast.expr, env: dict, source_line: str
    ) -> dict | None:
        val = _safe_eval(node, env)
        if val is _SENTINEL:
            return None
        return {
            "actual": val,
            "expected": True,
            "expression": source_line,
            "match": bool(val),
        }


def _op_symbol(op: ast.cmpop) -> str:
    return {
        ast.Lt: "<", ast.LtE: "<=",
        ast.Gt: ">", ast.GtE: ">=",
    }.get(type(op), "==")
