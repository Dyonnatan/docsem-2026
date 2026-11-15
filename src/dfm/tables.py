"""Recompute the paper's tables from the released predictions.

    python -m dfm tables            # print everything
    python -m dfm tables --json     # machine-readable

Nothing here calls a model: every number is computed from ``results/`` and the
labels in ``data/``.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from .paths import DATA_DIR, RESULTS_DIR
from .scoring import Outcome, by_id, evaluate, mcnemar, read_jsonl, wilson

REPLICA_ARMS = [
    # (label, results folder)
    ("DeepSeek A0", "a0-deepseek"),
    ("DeepSeek C2", "c2-deepseek"),
    ("DeepSeek C3", "c3-deepseek"),
    ("DeepSeek C3+card", "c3-card-deepseek"),
    ("GLM A0", "a0-glm"),
    ("GLM C2", "c2-glm"),
    ("GLM C3", "c3-glm"),
    ("GLM C3+card", "c3-card-glm"),
    ("Claude Code (Sonnet 5)", "claude-code-sonnet5"),
]

PAIRED = [
    ("DeepSeek C2", "DeepSeek A0", "C2 vs A0"),
    ("DeepSeek C3", "DeepSeek C2", "C3 vs C2"),
    ("DeepSeek C3+card", "DeepSeek C3", "card vs C3"),
    ("DeepSeek C3+card", "DeepSeek A0", "best vs A0"),
    ("GLM C2", "GLM A0", "C2 vs A0"),
    ("GLM C3", "GLM C2", "C3 vs C2"),
    ("GLM C3+card", "GLM C3", "card vs C3"),
    ("GLM C3+card", "GLM A0", "best vs A0"),
]

# The decoy template planted in test documents (and reproduced in the replica).
DECOY = re.compile(r"In a separate concerning\s+and\s+\w+\s+and\s+\w+\s+record", re.I)


def _gold(path) -> dict[str, dict]:
    return by_id(read_jsonl(path))


def _preds(group: str, arm: str) -> list[dict]:
    return read_jsonl(RESULTS_DIR / group / arm / "predictions.jsonl")


def _fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"{p:.3f}".lstrip("0") if p < 1 else "1.0"
    if p >= 0.001:
        return f"{p:.4f}".lstrip("0")
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def replica_table() -> dict:
    gold = _gold(DATA_DIR / "replica" / "labels.jsonl")
    arms = {label: evaluate(_preds("replica", folder), gold) for label, folder in REPLICA_ARMS}
    rows = []
    for label, out in arms.items():
        k, n = out.count("answer"), out.n
        rows.append({
            "arm": label, "n": n,
            "answer": round(out.rate("answer"), 1), "answer_ci95": [round(x, 1) for x in wilson(k, n)],
            "evidence": round(out.rate("evidence"), 1), "joint": round(out.rate("joint"), 1),
        })
    tests = []
    for a, b, what in PAIRED + [
        ("GLM A0", "DeepSeek C3", "GLM A0 vs DeepSeek C3"),
        ("Claude Code (Sonnet 5)", "GLM C3+card", "closed agent vs best open"),
    ]:
        _, a1, b1, p1 = mcnemar(arms[a].answer, arms[b].answer)
        _, a2, b2, p2 = mcnemar(arms[a].joint, arms[b].joint)
        tests.append({"model": a.split()[0], "comparison": what,
                      "answer": [a1, b1, p1], "joint": [a2, b2, p2]})

    full = {m: evaluate(_preds("replica", f"c3-card-full-{m.lower()}"), gold) for m in ("DeepSeek", "GLM")}
    _, x, y, p = mcnemar(full["DeepSeek"].answer, arms["DeepSeek C3+card"].answer)
    card_variant = {
        "full_card_answer": {m: round(o.rate("answer"), 1) for m, o in full.items()},
        "full_vs_decoy_only_deepseek": [x, y, p],
    }
    return {"rows": rows, "paired": tests, "card_variant": card_variant}


def decomposition() -> dict:
    """Selection vs resolution errors on the replica (paper, Section 6)."""
    gold = _gold(DATA_DIR / "replica" / "labels.jsonl")
    docs = DATA_DIR / "replica" / "documents"
    from .blocks import parse_blocks

    def run(preds: list[dict], skip: set[str]) -> dict:
        counts = Counter()
        decoy = Counter()
        for p in preds:
            iid = p["instance_id"]
            if iid in skip or p.get("answer") is None:
                continue
            g = gold[iid]
            ev = [str(x) for x in p.get("evidence") or []]
            ev_ok = len(ev) == 1 and ev[0] == g["evidence"][0]
            ans_ok = evaluate([p], gold).answer[iid]
            key = "correct" if ev_ok and ans_ok else "resolution" if ev_ok else (
                "selection, right number" if ans_ok else "selection")
            counts[key] += 1
            if key.startswith("selection") and ev:
                blocks = parse_blocks((docs / f"{iid}.md").read_text(encoding="utf-8"))
                decoy[key] += bool(DECOY.search(blocks.get(ev[0], "")))
        n = sum(counts.values())
        sel = counts["selection"] + counts["selection, right number"]
        return {
            "n": n, "counts": dict(counts),
            "answer_accuracy": round(100 * (counts["correct"] + counts["selection, right number"]) / n, 1),
            "joint_accuracy": round(100 * counts["correct"] / n, 1),
            "selection_errors": [sel, round(100 * sel / n, 1), [round(v, 1) for v in wilson(sel, n)]],
            "resolution_errors": [counts["resolution"], round(100 * counts["resolution"] / n, 1),
                                  [round(v, 1) for v in wilson(counts["resolution"], n)]],
            "decoy_cited": {"selection": [decoy["selection"], counts["selection"]],
                            "selection, right number": [decoy["selection, right number"],
                                                        counts["selection, right number"]]},
        }

    first_pass = set(json.loads((RESULTS_DIR / "replica/c3-deepseek/rerun-after-first-pass.json")
                                .read_text())["instance_ids"])
    return {
        "C3C, all tasks (paper)": run(_preds("replica", "c3c-deepseek"), set()),
        "C3 analyst outputs, first pass (as presented at the workshop)": run(_preds("replica", "c3-deepseek"), first_pass),
    }


def dev200_table() -> dict:
    gold = _gold(DATA_DIR / "docsem" / "train" / "labels.jsonl")
    ids = json.loads((DATA_DIR / "splits" / "dev200.json").read_text())["instance_ids"]
    preds = {
        "A0": _preds("dev200", "a0-deepseek"),
        "C2": _preds("dev200", "c2-deepseek"),
        # The paper's C3 count is the compiled answer that C3C starts from,
        # without the C3 runtime's fallback to the answer the model stated.
        "C3": _compiled_c3_answers("dev200", "c3-deepseek", DATA_DIR / "extraction" / "train"),
        "C3 (with stated-answer fallback)": _preds("dev200", "c3-deepseek"),
        "C3C": _preds("dev200", "c3c-deepseek"),
        "C2 (GPT-5.6 Sol)": _preds("dev200", "c2-gpt5.6-sol"),
    }
    out: dict[str, Outcome] = {k: evaluate(v, gold, over=ids) for k, v in preds.items()}
    completed = {k: {p["instance_id"] for p in v if p.get("answer") is not None} for k, v in preds.items()}

    def paired(a: str, b: str, *, all_items: bool = False) -> list:
        both = set(ids) if all_items else completed[a] & completed[b]
        _, x, y, p = mcnemar({i: out[a].answer[i] for i in both}, {i: out[b].answer[i] for i in both})
        return [x, y, p, len(both)]

    paths = Counter(p.get("decision") for p in preds["C3C"])
    return {
        "correct": {k: o.count("answer") for k, o in out.items()},
        "paired": {
            "C2 vs A0": paired("C2", "A0"),
            "C3 vs C2": paired("C3", "C2"),
            "C3 vs C2, missing answers counted as wrong": paired("C3", "C2", all_items=True),
            "C3C vs C3": paired("C3C", "C3"),
        },
        "c3c_decision_paths": dict(paths),
    }


def _compiled_c3_answers(group: str, arm: str, docs_dir) -> list[dict]:
    """C3 answers recompiled from the released completions, without fallback."""
    import gzip

    from .pipeline import c3_state

    out = []
    with gzip.open(RESULTS_DIR / group / arm / "completions.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            calls = {c["step"]: c["content"] for c in rec["calls"]}
            if "analyst" not in calls:
                continue
            document = (docs_dir / f"{rec['instance_id']}.md").read_text(encoding="utf-8")
            _, report = c3_state(document, calls["analyst"], calls.get("repair"))
            out.append({"instance_id": rec["instance_id"], "answer": report.get("answer"),
                        "evidence": report.get("evidence") or []})
    return out


def c3c_replica_paths() -> dict:
    return dict(Counter(p.get("decision") for p in _preds("replica", "c3c-deepseek")))


def main(as_json: bool = False) -> None:
    rep, dec, dev = replica_table(), decomposition(), dev200_table()
    paths = c3c_replica_paths()
    if as_json:
        print(json.dumps({"replica": rep, "decomposition": dec, "dev200": dev,
                          "c3c_replica_paths": paths}, indent=2))
        return

    print("Table 5 - replica (% over completed items)")
    print(f"  {'arm':26s} {'n':>4s} {'answer':>7s} {'95% CI':>13s} {'evid.':>6s} {'joint':>6s}")
    for r in rep["rows"]:
        lo, hi = r["answer_ci95"]
        print(f"  {r['arm']:26s} {r['n']:4d} {r['answer']:7.1f} [{lo:4.1f}, {hi:4.1f}] {r['evidence']:6.1f} {r['joint']:6.1f}")

    print("\nAppendix C - paired exact McNemar tests on the replica (a:b, p)")
    for t in rep["paired"]:
        a1, b1, p1 = t["answer"]
        a2, b2, p2 = t["joint"]
        print(f"  {t['model']:9s} {t['comparison']:26s} answer {a1:3d}:{b1:<3d} p={_fmt_p(p1):8s}"
              f" joint {a2:3d}:{b2:<3d} p={_fmt_p(p2)}")

    cv = rep["card_variant"]
    x, y, p = cv["full_vs_decoy_only_deepseek"]
    print(f"\nAppendix E - card with the outcome-record clause: answer "
          f"{cv['full_card_answer']['DeepSeek']} (DeepSeek), {cv['full_card_answer']['GLM']} (GLM); "
          f"vs decoy-only card on DeepSeek {x}:{y}, p={_fmt_p(p)}")

    print("\nSection 6 - selection vs resolution errors on the replica")
    for name, d in dec.items():
        s, r = d["selection_errors"], d["resolution_errors"]
        dc, dn = d["decoy_cited"]["selection"]
        rc, right = d["decoy_cited"]["selection, right number"]
        print(f"  {name}: n={d['n']}, answer {d['answer_accuracy']}, joint {d['joint_accuracy']}")
        print(f"    selection {s[0]} ({s[1]}% [{s[2][0]}, {s[2][1]}]), {right} of them with the right number;"
              f" resolution {r[0]} ({r[1]}% [{r[2][0]}, {r[2][1]}])")
        print(f"    a decoy is the cited block in {dc} of the {dn} selection errors with a wrong number"
              f" (and in {rc} of the {right} with the right number)")

    print("\nTable 4 - dev200, correct answers out of 200 (DeepSeek-V4-Flash unless noted)")
    for k, v in dev["correct"].items():
        print(f"  {k:34s} {v}")
    for k, (x, y, p, n) in dev["paired"].items():
        print(f"  {k:44s} {x}:{y} on {n} items, p={_fmt_p(p)}")

    print("\nSection 6 - which path decided each C3C answer")
    print(f"  dev200 : {dev['c3c_decision_paths']}")
    print(f"  replica: {paths}")
