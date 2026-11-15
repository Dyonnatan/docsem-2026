"""C3 runtime: compile step-level derivations into evaluable expressions.

In C3 each reading is a short list of steps instead of a single expression::

    {"op": "obs",       "id": "a", "value": 12}
    {"op": "assume",    "text": "the discount applies once"}
    {"op": "therefore", "id": "q", "expr": "a * 3"}
    {"op": "return",    "expr": "q"}

The vocabulary is open. Verbs are grouped into four families (observe,
assume, assign, return) and unknown verbs still compile when they carry a value
or an expression; a reading is never rejected because the model wrote
``hence`` instead of ``therefore``. Compilation substitutes bound names until
the returned expression contains only numbers, which the C2 runtime
(:mod:`dfm.readings`) then evaluates and validates.

This module also adds, without a model call, the missing variants of any
reading shaped as ``base * fraction`` (used by C3C, see
:func:`infer_base_plus_added`).
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

from .arith import fmt_num, nearly, numbers_in, safe_eval, substitute

_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUM = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?")

OBSERVE = {"obs", "observe", "from", "take", "read", "bind", "given_number", "const"}
ASSUME = {
    "assume", "assuming", "considering", "consider", "suppose", "supposing", "if",
    "given", "forall", "for_all", "each", "note", "warrant", "because", "since",
    "hypothesis",
}
ASSIGN = {
    "therefore", "thus", "hence", "so", "then", "let", "define", "set", "compute",
    "calculate", "deduce", "derive", "isolate", "solve", "rewrite", "simplify",
    "expand", "factor", "apply", "implies", "yields", "gives",
}
RETURN = {"return", "conclude", "answer", "result", "qeds", "qed", "submit"}

# Small constants that commonly appear as unit or multiplicity factors and are
# not expected to be bound by an ``obs`` step.
_UNIT_CONSTANTS = (1.0, 2.0, 60.0, 100.0, 24.0, 7.0, 12.0)


def norm_op(raw: object) -> str:
    return str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")


def family(op: str) -> str:
    if op in OBSERVE:
        return "obs"
    if op in RETURN:
        return "return"
    if op in ASSUME:
        return "assume"
    if op in ASSIGN:
        return "assign"
    return "open"


def step_list(reading: dict) -> list:
    for key in ("steps", "derivation", "proof", "chain", "argument", "trace"):
        raw = reading.get(key)
        if isinstance(raw, list) and raw:
            return raw
    return []


def _as_step(raw: object, index: int) -> dict:
    if isinstance(raw, str):
        text = raw.strip()
        if ":" in text:
            head, rest = text.split(":", 1)
            op = norm_op(head)
            rest = rest.strip()
            if family(op) == "obs" or _NUM.fullmatch(rest.replace(" ", "")):
                return {"op": op or "obs", "id": f"s{index}", "value": rest}
            return {"op": op or "therefore", "id": f"s{index}", "expr": rest}
        return {"op": "therefore", "id": f"s{index}", "expr": text}
    if not isinstance(raw, dict):
        return {"op": "note", "id": f"s{index}"}
    step = dict(raw)
    step["op"] = norm_op(
        step.get("op") or step.get("kind") or step.get("do") or step.get("type") or "therefore"
    )
    if not step.get("id"):
        step["id"] = str(step.get("name") or step.get("as") or f"s{index}").strip() or f"s{index}"
    if not step.get("expr"):
        for key in ("expression", "formula", "rhs", "as", "equals", "to"):
            if step.get(key):
                step["expr"] = step[key]
                break
    if step.get("ref") and not step.get("expr"):
        step["expr"] = str(step["ref"]).strip()
    return step


def compile_reading(reading: dict) -> dict[str, Any]:
    """Compile one reading's steps into a single expression.

    Never raises. Problems are recorded as soft notes; if the steps do not
    compile, the reading's own ``expression`` field (if any) is used instead.
    """
    notes: list[str] = []
    authored = str(reading.get("expression") or reading.get("expr") or "").strip()
    steps = [_as_step(row, i + 1) for i, row in enumerate(step_list(reading))]
    env: dict[str, str] = {}
    last: str | None = None
    returned: str | None = None
    assume_n = 0
    obs_n = 0

    for step in steps:
        op = step.get("op") or "therefore"
        fam = family(op)
        sid = str(step.get("id") or "").strip()
        if sid and not _ID.fullmatch(sid):
            sid = re.sub(r"[^A-Za-z0-9_]", "_", sid) or f"s{len(env) + 1}"

        if fam == "obs" or (fam == "open" and step.get("value") is not None and not step.get("expr")):
            raw = step.get("value") if step.get("value") is not None else step.get("span")
            if raw is None and step.get("expr") and _NUM.fullmatch(str(step["expr"]).strip()):
                raw = step["expr"]
            if raw is None:
                notes.append(f"{op} without a bound value")
                continue
            key = sid or f"n{obs_n + 1}"
            env[key] = fmt_num(raw)
            last = env[key]
            obs_n += 1
            continue

        if fam == "assume":
            assume_n += 1
            raw = step.get("value") if step.get("value") is not None else step.get("span")
            expr = str(step.get("expr") or "").strip()
            if raw is not None:
                key = sid or f"a{assume_n}"
                env[key] = fmt_num(raw)
                last = env[key]
            elif expr:
                resolved = substitute(expr, env) if env else expr
                if sid:
                    env[sid] = resolved
                last = resolved
            continue

        expr = str(step.get("expr") or "").strip()
        if fam == "return":
            if expr:
                returned = substitute(expr, env) if env else expr
            elif last:
                returned = last
            else:
                notes.append("return with no expression")
            continue

        if not expr:
            if fam == "open":
                notes.append(f"open op {op!r} ignored (no expr/value)")
            continue
        resolved = substitute(expr, env) if env else expr
        if sid:
            env[sid] = resolved
        last = resolved

    compiled = returned or last or ""
    if compiled:
        try:
            safe_eval(compiled)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"compiled expr failed eval: {exc}")
            if authored:
                compiled = authored
                notes.append("fell back to authored expression")
    elif authored:
        compiled = authored
        if steps:
            notes.append("steps did not yield an expr; used authored expression")
    else:
        notes.append("no compiled or authored expression")

    notes.extend(_binding_notes(compiled, env, steps))
    return {
        "expression": compiled,
        "notes": notes,
        "n_steps": len(steps),
        "n_obs": obs_n,
        "n_assume": assume_n + (1 if isinstance(reading.get("assume"), (dict, str)) else 0),
        "compiled_from_steps": bool(steps) and bool(compiled) and compiled != authored,
        "steps": steps,
    }


def _binding_notes(expr: str, env: dict[str, str], steps: list[dict]) -> list[str]:
    """Flag numbers in the final expression that no step observed (soft)."""
    if not expr:
        return []
    used = numbers_in(expr)
    if not used:
        return []
    bound: list[float] = []
    for val in env.values():
        try:
            bound.append(float(safe_eval(val)))
        except Exception:  # noqa: BLE001
            continue
    for step in steps:
        raw = step.get("value")
        if raw is None:
            continue
        try:
            bound.append(float(str(raw).replace(",", "")))
        except (TypeError, ValueError):
            continue
    unbound = [
        num
        for num in used
        if not any(nearly(num, b) for b in bound)
        and not any(nearly(num, k) for k in _UNIT_CONSTANTS)
    ]
    if unbound and not env:
        return ["literals in expr are not named in steps (soft)"]
    if unbound:
        return [f"unbound literals {unbound} (soft)"]
    return []


# --- base * fraction completion (used by C3C) --------------------------------


def _fraction_leaf(node: ast.AST) -> float | None:
    """Value of a bare literal or literal division that lies strictly in (0, 1).

    Differences such as ``0.5 - 0.25`` are excluded on purpose: they appear in
    percentage-difference readings, where a value below one is a computed gap
    and not a fraction of a base quantity.
    """
    if not isinstance(node, (ast.Constant, ast.BinOp)):
        return None
    if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Div):
        return None
    try:
        value = safe_eval(ast.unparse(node))
    except Exception:  # noqa: BLE001
        return None
    return value if 0 < value < 1 else None


def decompose_base_fraction(expr: str) -> tuple[str, str, float] | None:
    """Split ``base * fraction`` into ``(base_text, fraction_text, fraction)``.

    Only a top-level product of exactly two factors qualifies, where one factor
    is a plain fraction in (0, 1) and the other is compound (not a bare
    constant, which would rather be a unit conversion such as ``* 100``).
    """
    try:
        tree = ast.parse(str(expr), mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Mult)):
        return None
    left = _fraction_leaf(tree.left) if not isinstance(tree.right, ast.Constant) else None
    right = _fraction_leaf(tree.right) if not isinstance(tree.left, ast.Constant) else None
    if left is not None and right is None:
        return ast.unparse(tree.right), ast.unparse(tree.left), left
    if right is not None and left is None:
        return ast.unparse(tree.left), ast.unparse(tree.right), right
    return None


def infer_base_plus_added(readings: list[dict]) -> list[dict]:
    """Add the ``base + fraction*base`` variants of every ``base * fraction`` reading.

    For each such reading, the "added once" and "added on each of two named
    occasions" variants are created when their values are not already among
    the readings. The new readings carry their notebook tags directly, so the
    C3C closing rule can use them without relying on the reviewer.
    """
    existing: list[float] = []
    for r in readings:
        if not isinstance(r, dict):
            continue
        v = r.get("value")
        if v is None and r.get("expression"):
            try:
                v = safe_eval(str(r["expression"]))
            except Exception:  # noqa: BLE001
                v = None
        if v is not None:
            existing.append(float(v))

    def present(value: float) -> bool:
        return any(nearly(value, v) for v in existing)

    added: list[dict] = []
    seen: set[str] = set()
    for reading in readings:
        if not isinstance(reading, dict) or reading.get("fails"):
            continue
        parts = decompose_base_fraction(str(reading.get("expression") or ""))
        if not parts:
            continue
        base_text, frac_text, frac = parts
        key = re.sub(r"\s+", "", base_text)
        if key in seen:
            continue
        try:
            base = safe_eval(base_text)
        except Exception:  # noqa: BLE001
            continue
        rid = str(reading.get("id") or "r")
        variants = [
            ("i1", base + frac * base, f"(({base_text}) + (({frac_text})*({base_text})))",
             "base plus the fraction added once, combined (S-incl base_plus_added, S-quant once)",
             "once"),
            ("i2", base + 2 * frac * base, f"(({base_text}) + ((2)*({frac_text})*({base_text})))",
             "base plus the fraction added once per named occasion "
             "(S-incl base_plus_added, S-quant times_k)",
             "times_k"),
        ]
        for suffix, value, expression, gloss, quant in variants:
            if present(value):
                continue
            seen.add(key)
            added.append(
                {
                    "id": f"{rid}{suffix}",
                    "expression": expression,
                    "steps": [{"op": "assume", "gloss": gloss, "side": "base_plus_added", "text": gloss}],
                    "inferred": "infer_base_plus_added",
                    "side_by_schema": {"S-incl": "base_plus_added", "S-quant": quant},
                }
            )
    return added


def compile_state(state: dict) -> tuple[dict, dict]:
    """Return a copy of ``state`` whose readings carry compiled expressions."""
    out = copy.deepcopy(state)
    reports = []
    has_steps = False
    for reading in out.get("readings") or []:
        if not isinstance(reading, dict):
            continue
        compiled = compile_reading(reading)
        if compiled["n_steps"]:
            has_steps = True
        if compiled["expression"]:
            reading["expression"] = compiled["expression"]
            if reading.get("result") in (0, 0.0, "0"):
                try:
                    if not nearly(float(safe_eval(compiled["expression"])), 0.0):
                        reading.pop("result", None)
                        compiled["notes"].append("ignored placeholder result 0")
                except Exception:  # noqa: BLE001
                    pass
        reading["c3"] = {
            k: compiled[k] for k in ("notes", "n_steps", "n_obs", "n_assume", "compiled_from_steps")
        }
        reports.append({"id": reading.get("id"), **reading["c3"]})
    if isinstance(out.get("assume"), (dict, str)):
        has_steps = True

    inferred = infer_base_plus_added(out.get("readings") or [])
    if inferred:
        out["readings"] = list(out.get("readings") or []) + inferred
        reports.append({"id": None, "notes": [f"infer_base_plus_added added {[m['id'] for m in inferred]}"]})

    return out, {"has_steps": has_steps, "readings": reports}
