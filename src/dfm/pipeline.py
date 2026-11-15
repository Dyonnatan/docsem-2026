"""The four pipeline variants of the paper.

==========  ==========================================================  =====
variant     what changes                                                calls
==========  ==========================================================  =====
``A0``      query + document -> ``{answer, evidence}``                  1
``C2``      two rival readings as expressions, evaluated and validated  1-2
``C3``      the readings become step-level derivations (DSL)            1-2
``C3C``     C3 + notebook reviewer and deterministic closing            2-3
==========  ==========================================================  =====

The selection card is not a variant of its own: it is one paragraph added to
the C3 prompt (``prompts/c3-selection-card.txt``).

Every model call is separated from the deterministic part. :func:`solve`
calls the model and records each completion; :func:`derive` turns recorded
completions into a prediction without any network access. ``dfm replay``
uses :func:`derive` to rebuild the released predictions from the released
completions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .derivation import compile_state
from .notebook import load_notebook
from .parsing import extract_prediction, extract_state, first_json
from .paths import PROMPTS_DIR
from .readings import select_reading
from .reviewer import apply_reviewer

USER_TEMPLATE = """instance_id: {instance_id}

QUERY:
{query}

DOCUMENT:
{document}
"""

METHODS = ("A0", "C2", "C3", "C3C")


@dataclass
class Arm:
    """One variant run with one model (an *arm*), loaded from ``configs/*.json``."""

    name: str
    method: str
    prompt: str
    model: dict
    split: str = "replica"
    reviewer_prompt: str | None = None
    reviewer_model: dict | None = None
    notebook: str = "notebook.json"
    reviewer_adds_readings: bool = False
    max_repairs: int = 1
    concurrency: int = 5
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Arm":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        if cfg["method"] not in METHODS:
            raise ValueError(f"unknown method {cfg['method']!r}; expected one of {METHODS}")
        known = {f for f in cls.__dataclass_fields__} - {"extra"}
        return cls(**{k: v for k, v in cfg.items() if k in known},
                   extra={k: v for k, v in cfg.items() if k not in known})

    def system_prompt(self) -> str:
        return (PROMPTS_DIR / self.prompt).read_text(encoding="utf-8")


def repair_message(method: str, report: dict) -> str:
    notes = json.dumps(report.get("readings") or report, ensure_ascii=False)[:1500]
    name = "repair-c3.txt" if method in ("C3", "C3C") else "repair-c2.txt"
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip("\n") + "\n" + notes


# --- deterministic part ---------------------------------------------------------


def _apply_readings(method: str, document: str, text: str, prediction: dict | None) -> tuple[dict | None, dict]:
    """Run the C2 or C3 runtime on one completion."""
    state = extract_state(text)
    if not state:
        return prediction, {"reason": "no_state", "override": False, "need_repair": True}
    derivation = None
    if method in ("C3", "C3C"):
        state, derivation = compile_state(state)
    report = select_reading(state, document)
    if derivation is not None:
        report["derivation"] = derivation
    if report.get("answer") is not None and report.get("override"):
        prediction = dict(prediction or {"evidence": report.get("evidence") or []})
        prediction["answer"] = report["answer"]
        if report.get("evidence"):
            prediction["evidence"] = report["evidence"]
    return prediction, report


def derive(method: str, document: str, analyst: str, repair: str | None = None) -> tuple[dict | None, dict]:
    """Prediction and runtime report from the analyst completion(s).

    ``analyst`` is the first completion; ``repair`` the completion of the
    repair call, if one was made. A0 only parses the answer.
    """
    prediction = extract_prediction(analyst)
    if method == "A0":
        return prediction, {"reason": "direct"}

    prediction, report = _apply_readings(method, document, analyst, prediction)
    if repair is not None:
        parsed = extract_prediction(repair) or extract_prediction(analyst + "\n\nREPAIR\n" + repair)
        if parsed:
            prediction = parsed
        prediction, report = _apply_readings(method, document, repair, prediction)
    if prediction and report.get("answer") and report.get("override"):
        prediction["answer"] = report["answer"]
        if report.get("evidence"):
            prediction["evidence"] = report["evidence"]
    return prediction, report


def c3_state(document: str, analyst: str, repair: str | None = None) -> tuple[dict, dict]:
    """The compiled C3 state the reviewer sees: the last completion's state."""
    state = extract_state(repair if repair is not None else analyst) or {}
    compiled, derivation = compile_state(dict(state))
    report = select_reading(compiled, document)
    report["derivation"] = derivation
    return compiled, report


def _live(report: dict) -> int:
    return sum(1 for r in report.get("readings") or [] if r.get("value") is not None and not r.get("fails"))


def derive_c3c(
    document: str,
    query: str,
    analyst: str,
    repair: str | None,
    reviewer: str | None,
    *,
    reviewer_prompt: str,
    notebook: dict,
    adds_readings: bool = False,
) -> tuple[dict | None, dict]:
    """C3C prediction from recorded analyst and reviewer completions."""
    state, report = c3_state(document, analyst, repair)
    prediction = {"answer": report.get("answer"), "evidence": report.get("evidence") or []}
    if reviewer is None and not adds_readings and _live(report) < 2:
        return prediction, report

    def ask(step, system, user, must_have):  # replays the recorded reviewer completion
        return first_json(reviewer or "", must_have)

    return apply_reviewer(
        report, state=state, document=document, query=query, prediction=prediction,
        ask=ask, reviewer_prompt=reviewer_prompt, notebook=notebook, adds_readings=adds_readings,
    )


# --- model calls ----------------------------------------------------------------


Chat = Callable[[list[dict]], "object"]


def solve(arm: Arm, task: dict, document: str, chat: Chat, reviewer_chat: Chat | None = None) -> dict:
    """Run one task through an arm. Returns a JSON-serialisable record."""
    iid = task["instance_id"]
    system = arm.system_prompt()
    user = USER_TEMPLATE.format(instance_id=iid, query=task["user_query"], document=document)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls: list[dict] = []

    def call(step: str, msgs: list[dict], client: Chat) -> str:
        reply = client(msgs)
        calls.append({"step": step, "content": reply.content, "provider": reply.provider, "usage": reply.usage})
        return reply.content

    analyst = call("analyst", messages, chat)
    repair = None
    prediction, report = derive(arm.method, document, analyst)
    if arm.method != "A0" and report.get("need_repair") and arm.max_repairs:
        msgs = messages + [
            {"role": "assistant", "content": analyst},
            {"role": "user", "content": repair_message(arm.method, report)},
        ]
        repair = call("repair", msgs, chat)
        prediction, report = derive(arm.method, document, analyst, repair)

    if arm.method == "C3C":
        notebook = load_notebook(PROMPTS_DIR / arm.notebook)
        reviewer_prompt = (PROMPTS_DIR / arm.reviewer_prompt).read_text(encoding="utf-8")
        state, c3_report = c3_state(document, analyst, repair)
        if _live(c3_report) >= 2 or arm.reviewer_adds_readings:
            def ask(step, system_prompt, user_message, must_have):
                content = call(step, [{"role": "system", "content": system_prompt},
                                      {"role": "user", "content": user_message}], reviewer_chat or chat)
                return first_json(content, must_have)

            native = {"answer": c3_report.get("answer"), "evidence": c3_report.get("evidence") or []}
            prediction, report = apply_reviewer(
                c3_report, state=state, document=document, query=task["user_query"],
                prediction=native, ask=ask, reviewer_prompt=reviewer_prompt,
                notebook=notebook, adds_readings=arm.reviewer_adds_readings,
            )

    return record(iid, prediction, report, calls)


def record(iid: str, prediction: dict | None, report: dict, calls: list[dict]) -> dict:
    out = {
        "instance_id": iid,
        "answer": (prediction or {}).get("answer"),
        "evidence": (prediction or {}).get("evidence") or [],
        "decision": report.get("reason"),
        "calls": calls,
    }
    if "c3c" in report:
        out["c3c"] = {k: report["c3c"].get(k) for k in ("verdict", "tags", "closing", "decision")}
    return out
