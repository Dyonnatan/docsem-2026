"""Safe arithmetic and number formatting shared by the C2 and C3 runtimes."""

from __future__ import annotations

import ast
import math
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Pow: operator.pow,
}
_FUNCS = {
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
}
_NUM = re.compile(r"\d+(?:\.\d+)?")
_LITERAL = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?")


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression without executing arbitrary code.

    Only numeric literals, + - * / ** (``^`` is read as power), unary signs and
    the functions floor, ceil, round, abs, min and max are allowed.
    """
    tree = ast.parse(str(expr).replace("^", "**"), mode="eval")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](walk(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](walk(node.left), walk(node.right)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return float(_FUNCS[node.func.id](*[walk(a) for a in node.args]))
        raise ValueError(f"unsafe or invalid expression: {expr!r}")

    return walk(tree)


def nearly(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-6, 1e-6 * abs(b))


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text or "")]


def format_answer(value: float) -> str:
    """Integers without a decimal point; other values with 12 significant digits."""
    return str(int(value)) if nearly(value, round(value)) else format(value, ".12g")


def fmt_num(value: object) -> str:
    """Render an observed quantity so it can be substituted into an expression."""
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if _LITERAL.fullmatch(text):
            return f"({text})" if "/" in text else text
        return text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if nearly(number, round(number)):
        return str(int(round(number)))
    return format(number, ".12g")


def substitute(expr: str, env: dict[str, str], *, rounds: int = 12) -> str:
    """Replace bound names in ``expr`` by their (parenthesised) definitions."""
    current = str(expr)
    names = sorted(env, key=len, reverse=True)
    for _ in range(rounds):
        nxt = current
        for name in names:
            nxt = re.sub(rf"\b{re.escape(name)}\b", f"({env[name]})", nxt)
        if nxt == current:
            return current
        current = nxt
    return current
