"""Scoring and paired statistics.

Answers are compared by normalised exact match (whitespace, case, a leading
"final answer:" marker and equivalent decimal forms are ignored), as in the
official scorer. Evidence is correct only when it is exactly the gold block
list, which always holds one block.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from math import comb
from pathlib import Path

_MARKER = re.compile(r"^\s*(final\s*answer\s*[:\-]\s*)", re.I)


def norm_answer(raw: object) -> str:
    s = _MARKER.sub("", str(raw)).strip().lower()
    try:
        n = float(s.replace(",", ""))
    except ValueError:
        return s
    return str(int(n)) if n.is_integer() else format(n, ".12g")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["instance_id"]: r for r in rows}


@dataclass
class Outcome:
    """Per-item correctness of one arm on the items it completed."""

    answer: dict[str, bool]
    evidence: dict[str, bool]

    @property
    def joint(self) -> dict[str, bool]:
        return {i: self.answer[i] and self.evidence[i] for i in self.answer}

    @property
    def n(self) -> int:
        return len(self.answer)

    def rate(self, metric: str) -> float:
        values = getattr(self, metric)
        return 100.0 * sum(values.values()) / len(values) if values else 0.0

    def count(self, metric: str) -> int:
        return sum(getattr(self, metric).values())


def evaluate(predictions: list[dict], gold: dict[str, dict], *, over: list[str] | None = None) -> Outcome:
    """Score predictions against gold.

    By default only completed items count (the replica convention). Pass
    ``over`` to score a fixed id list, counting missing predictions as wrong
    (the dev200 convention).
    """
    preds = by_id(predictions)
    ids = over if over is not None else [i for i in preds if i in gold]
    answer, evidence = {}, {}
    for iid in ids:
        p, g = preds.get(iid), gold[iid]
        if p is None or p.get("answer") is None:
            answer[iid] = evidence[iid] = False
            continue
        answer[iid] = norm_answer(p["answer"]) == norm_answer(g["answer"])
        evidence[iid] = sorted(map(str, p.get("evidence") or [])) == sorted(map(str, g["evidence"]))
    return Outcome(answer, evidence)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, in percent."""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int, float]:
    """Exact two-sided McNemar test on the items both arms completed.

    Returns ``(n, a_only, b_only, p)``.
    """
    ids = set(a) & set(b)
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n, k = a_only + b_only, min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return len(ids), a_only, b_only, p
