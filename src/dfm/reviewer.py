"""C3C: notebook reviewer and deterministic closing.

C3C runs only when two or more C3 readings survive validation. A reviewer
call sees the query, the evidence, the notebook rules and the compiled
readings, and labels each reading with a ``{schema: side}`` tag from a closed
set. When the labels separate the readings cleanly, the notebook's preferred
side decides (:func:`dfm.notebook.mechanical_close`). Otherwise the analyst's
own preferred surviving reading is kept, exactly as in C3.

The reviewer may also propose a missing reading. Additions are applied only
when ``reviewer_adds_readings`` is set in the arm configuration; see
``docs/REPRODUCING.md`` for which runs used them.
"""

from __future__ import annotations

import json
from typing import Callable

from .arith import format_answer
from .derivation import compile_state, family, norm_op, step_list
from .notebook import collect_tags, mechanical_close, prompt_snippet
from .readings import block_map, select_reading

# ask(step_name, system_prompt, user_message, required_key) -> parsed JSON or None
AskFn = Callable[[str, str, str, str], dict | None]


def evidence_span(document: str, evidence: list[str]) -> str:
    blocks = block_map(document)
    return "\n".join(f"{eid}: {blocks.get(eid, '').strip()}" for eid in evidence)


def assume_gloss(reading: dict) -> str:
    """The reading's first stated assumption, used as its short description."""
    for raw in step_list(reading):
        if isinstance(raw, str):
            head, _, rest = raw.partition(":")
            if family(norm_op(head)) == "assume" and rest.strip():
                return rest.strip()
            continue
        if not isinstance(raw, dict):
            continue
        if family(norm_op(raw.get("op") or raw.get("kind") or raw.get("type"))) != "assume":
            continue
        text = str(raw.get("text") or raw.get("warrant") or raw.get("note") or "").strip()
        if text:
            return text
    for key in ("gloss", "warrant", "side"):
        text = str(reading.get(key) or "").strip()
        if text:
            return text
    return str(reading.get("id") or "reading")


def reading_views(state: dict, report: dict) -> list[dict]:
    """Join the compiled report rows with the authored readings."""
    authored = {str(r.get("id")): r for r in (state or {}).get("readings") or [] if isinstance(r, dict)}
    views = []
    for row in report.get("readings") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        src = authored.get(str(row["id"])) or {}
        views.append(
            {
                "id": str(row["id"]),
                "value": row.get("value"),
                "fails": list(row.get("fails") or []),
                "coverage": row.get("coverage") or 0,
                "expr": str(row.get("expression") or src.get("expression") or ""),
                "gloss": assume_gloss(src),
                "side_by_schema": dict(row.get("side_by_schema") or src.get("side_by_schema") or {}),
            }
        )
    return views


def _is_live(view: dict) -> bool:
    return view.get("value") is not None and not view.get("fails")


def merge_readings(state: dict, incoming: list) -> dict:
    out = dict(state or {})
    readings = [dict(r) for r in out.get("readings") or [] if isinstance(r, dict)]
    index = {str(r.get("id")): i for i, r in enumerate(readings) if r.get("id")}
    for raw in incoming or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        rid = str(row.get("id") or "").strip() or f"r{len(readings) + 1}"
        row["id"] = rid
        if rid in index:
            readings[index[rid]] = row
        else:
            readings.append(row)
            index[rid] = len(readings) - 1
    out["readings"] = readings
    return out


def recompile(state: dict, document: str) -> tuple[dict, dict]:
    visible = set(block_map(document))
    cleaned = dict(state)
    cleaned["evidence"] = [e for e in state.get("evidence") or [] if e in visible] or list(
        state.get("evidence") or []
    )
    compiled, derivation = compile_state(cleaned)
    report = select_reading(compiled, document)
    report["derivation"] = derivation
    return compiled, report


def review_user_message(*, state: dict, report: dict, document: str, query: str, snippet: str) -> str:
    evidence = list(report.get("evidence") or state.get("evidence") or [])
    shown = [
        {"id": v["id"], "gloss": v["gloss"], "expr": v["expr"], "value": v["value"], "fails": v["fails"]}
        for v in reading_views(state, report)
    ]
    return (
        f"QUERY:\n{query}\n\n"
        f"EVIDENCE:\n{evidence_span(document, evidence)}\n\n"
        f"{snippet}\n\n"
        f"CURRENT READINGS (analyst DSL already compiled):\n"
        f"{json.dumps(shown, ensure_ascii=False, indent=2)}\n\n"
        f"Check the notebook. Add a missing licensed side only if the wording opens it."
    )


def apply_reviewer(
    report: dict,
    *,
    state: dict,
    document: str,
    query: str,
    prediction: dict | None,
    ask: AskFn,
    reviewer_prompt: str,
    notebook: dict,
    adds_readings: bool = False,
) -> tuple[dict | None, dict]:
    """Run the C3C stage on a compiled C3 state.

    Returns the (possibly changed) prediction and a report whose ``c3c`` entry
    records the reviewer's verdict, the tags and the closing decision.
    """
    try:
        review = ask(
            "reviewer",
            reviewer_prompt,
            review_user_message(
                state=state, report=report, document=document, query=query,
                snippet=prompt_snippet(notebook, max_cards=8),
            ),
            "verdict",
        )
    except Exception:  # noqa: BLE001  a failed reviewer call leaves the C3 answer in place
        review = None
    if not isinstance(review, dict):
        review = {"verdict": "unavailable", "readings": [], "notation": {}}

    additions = [r for r in review.get("readings") or [] if isinstance(r, dict)]
    if additions and adds_readings:
        state, report = recompile(merge_readings(state, additions), document)

    views = reading_views(state, report)
    tags, conflicts = collect_tags(review)
    for v in views:
        # Tags inferred during compilation stay; reviewer tags are added on top.
        v["side_by_schema"] = {**v["side_by_schema"], **tags.get(v["id"], {})}
    live = [v for v in views if _is_live(v)]
    evidence = list(report.get("evidence") or state.get("evidence") or [])

    trace = {
        "verdict": review.get("verdict"),
        "missing": review.get("missing") or [],
        "n_proposed": len(additions),
        "tags": tags,
        "tag_conflicts": conflicts,
        "closing": None,
        "decision": None,
    }
    report = dict(report)
    report["c3c"] = trace

    def commit(reading_id: str, reason: str) -> tuple[dict | None, dict]:
        view = next(v for v in views if v["id"] == reading_id)
        answer = format_answer(float(view["value"]))
        report.update({"chosen": reading_id, "reason": reason, "answer": answer, "override": True})
        out = dict(prediction or {"evidence": evidence})
        out["answer"] = answer
        if evidence:
            out["evidence"] = evidence
        return out, report

    if len(live) == 1:
        trace["decision"] = "unique_survivor"
        return commit(live[0]["id"], "unique_survivor")
    if len(live) < 1:
        trace["decision"] = "no_survivor"
        return prediction, report

    closing = mechanical_close(live, notebook)
    if closing:
        trace["closing"] = closing
        trace["decision"] = "notebook_rule"
        return commit(closing["reading"], "notebook_rule")

    # Labels did not separate the readings: keep the analyst's C3 choice.
    trace["decision"] = "analyst_preference"
    return prediction, report
