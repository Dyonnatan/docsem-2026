"""C2 runtime: evaluate rival readings, apply validators, select one.

A *reading* is one interpretation of the passage, written as an arithmetic
expression over the quantities the model observed. The model writes two of
them; this module evaluates each one in Python and rejects readings with three
validators (numbered as in the paper):

* **V5** the expression does not evaluate, or disagrees with a declared result;
* **V4** a narrow check for tiered rates: when the evidence says a rate
  *drops*, reject a reading that treats the later rate as an increase;
* **V2** among the survivors, reject a reading that uses fewer of the
  evidence block's numbers than its rival.

Selection: a unique survivor becomes the answer; with several survivors the
reading the model committed to is kept; with none, the caller may ask the
model for one repair.

V2 and V4 read the evidence text through :func:`block_map`, which recognises
the ``b01:`` block labels of the train and validation splits. On documents with
opaque identifiers (the test split and the replica) the evidence text is empty,
so only V5 can fire there. This is the behaviour behind every result reported
in the paper and is kept unchanged for reproducibility.
"""

from __future__ import annotations

import re
from typing import Any

from .arith import format_answer, nearly, numbers_in, safe_eval

_BLOCK = re.compile(r"(?m)^(b\d{2,}):\s*(.*?)(?=^(?:b\d{2,}):|^# |\Z)", re.S)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def block_map(document: str) -> dict[str, str]:
    """Map ``bNN`` block labels to their text (train/validation format)."""
    return {m.group(1): m.group(2) for m in _BLOCK.finditer(document.replace("\r\n", "\n"))}


def evidence_text(document: str, evidence_ids: list[str]) -> str:
    blocks = block_map(document)
    return "\n".join(blocks.get(eid, "") for eid in evidence_ids)


def evaluate_reading(reading: dict) -> tuple[float | None, str | None]:
    """V5: the reading's expression must evaluate (and match a declared result)."""
    expr = str(reading.get("expression") or "").strip()
    if not expr:
        return None, "V5 missing expression"
    try:
        value = safe_eval(expr)
    except Exception as exc:  # noqa: BLE001
        return None, f"V5 cannot eval: {exc}"
    declared = reading.get("result")
    if declared is not None:
        try:
            if not nearly(value, float(declared)):
                return value, f"V5 result {declared} != eval {value}"
        except (TypeError, ValueError):
            pass
    return value, None


def rate_drop_check(evidence: str, value: float) -> str | None:
    """V4: "pay X for the first N units, then the rate drops to Y" must not
    produce the flat reading when Y is in fact higher than X/N."""
    if not re.search(r"\bdrop(?:s|ped)?\b", evidence, re.I):
        return None
    nums = numbers_in(evidence)
    if len(nums) < 4:
        return None
    x, n, y, t = nums[0], nums[1], nums[2], nums[3]
    if n == 0:
        return None
    extra = t - n
    flat = x + extra * y
    per = n * x + extra * y
    if (x / n) < y and nearly(value, flat) and not nearly(flat, per):
        return (
            f"V4 'rate drops' but {x}/{n}={x / n:.4g} < {y}, so flat {flat:g} "
            "treats the later rate as an increase"
        )
    return None


def coverage(evidence: str, expr: str) -> int:
    """Number of evidence numbers that the expression uses (input to V2)."""
    used = numbers_in(expr)
    return sum(1 for num in numbers_in(evidence) if any(nearly(num, u) for u in used))


def number_pool(reading: dict) -> str:
    """Text whose numbers count towards V2 coverage.

    A multi-step derivation can pass through a step whose expression is
    already a bare computed literal, which hides the observed quantities from
    the final flattened expression. Only in that case are the observed values
    credited as well; otherwise a reading that merely observed a number
    without using it would tie with one that used it.
    """
    steps = reading.get("steps") or []
    collapsed = any(
        isinstance(step, dict)
        and step.get("op") in ("therefore", "compute")
        and isinstance(step.get("expr"), str)
        and _NUM.fullmatch(step["expr"].strip())
        for step in steps
    )
    parts = [str(reading.get("expression") or "")]
    if collapsed:
        for step in steps:
            if isinstance(step, dict) and step.get("op") == "obs" and step.get("value") is not None:
                parts.append(str(step["value"]))
    return " ".join(parts)


def select_reading(state: dict, document: str, *, v2: bool = True) -> dict[str, Any]:
    """Evaluate and validate every reading in ``state`` and pick one.

    Returns a report with the per-reading rows, the chosen reading id, the path
    that decided it (``reason``), the formatted answer and whether a repair
    call is needed.
    """
    evidence_ids = [str(x) for x in (state.get("evidence") or [])]
    text = evidence_text(document, evidence_ids)
    rows = []
    for raw in state.get("readings") or []:
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("id") or f"r{len(rows) + 1}")
        value, v5 = evaluate_reading(raw)
        fails = [v5] if v5 else []
        if value is not None:
            v4 = rate_drop_check(text, value)
            if v4:
                fails.append(v4)
        rows.append(
            {
                "id": rid,
                "value": value,
                "coverage": coverage(text, number_pool(raw)),
                "fails": fails,
                "expression": raw.get("expression"),
                "sides": raw.get("sides"),
                "side_by_schema": raw.get("side_by_schema"),
            }
        )

    live = [row for row in rows if row["value"] is not None and not row["fails"]]
    if v2 and len(live) >= 2:
        best = max(row["coverage"] for row in live)
        for row in live:
            if best - row["coverage"] >= 1:
                row["fails"].append(
                    f"V2 omits a number that a sibling reading uses (coverage {row['coverage']}<{best})"
                )
        live = [row for row in live if not row["fails"]]

    chosen = None
    reason = "keep_model_commit"
    if len(live) == 1:
        chosen, reason = live[0], "unique_survivor"
    elif len(live) > 1:
        commit = str(state.get("commit") or "")
        chosen = next((row for row in live if row["id"] == commit), None)
        if chosen is not None:
            reason = "commit_among_survivors"
        else:
            chosen = max(live, key=lambda r: r["coverage"])
            reason = "max_coverage_among_survivors"

    answer = None
    if chosen is not None and chosen["value"] is not None:
        answer = format_answer(chosen["value"])

    return {
        "evidence": evidence_ids,
        "readings": rows,
        "chosen": None if chosen is None else chosen["id"],
        "reason": reason,
        "override": reason
        in {"unique_survivor", "max_coverage_among_survivors", "commit_among_survivors"},
        "answer": answer,
        "need_repair": not live,
    }
