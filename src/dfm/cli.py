"""Command-line interface: ``python -m dfm <command>``.

    tables                      recompute the paper's tables from results/
    replay  CONFIG              rebuild an arm's predictions from its released completions
    run     CONFIG              run an arm through the API (needs OPENROUTER_API_KEY)
    score   PREDICTIONS SPLIT   score a predictions.jsonl file
    replica build --out DIR     regenerate the replica
    replica gate                lexical validity gate of the replica
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import tables
from .data import load_split
from .notebook import load_notebook
from .paths import PROMPTS_DIR, RESULTS_DIR
from .pipeline import Arm, derive, derive_c3c, solve
from .scoring import evaluate, read_jsonl, wilson


def _completions(arm: Arm) -> list[dict]:
    path = RESULTS_DIR / arm.name / "completions.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def cmd_replay(args: argparse.Namespace) -> int:
    arm = Arm.load(args.config)
    split = load_split(arm.split)
    queries = {t["instance_id"]: t["user_query"] for t in split.tasks}
    released = {p["instance_id"]: p for p in read_jsonl(RESULTS_DIR / arm.name / "predictions.jsonl")}
    notebook = load_notebook(PROMPTS_DIR / arm.notebook) if arm.method == "C3C" else None
    reviewer_prompt = (PROMPTS_DIR / arm.reviewer_prompt).read_text() if arm.reviewer_prompt else ""

    same, total, mismatches = 0, 0, []
    for rec in _completions(arm):
        iid = rec["instance_id"]
        calls = {c["step"]: c.get("content") for c in rec["calls"]}
        if "analyst" not in calls:
            prediction = None
        elif arm.method == "C3C":
            prediction, _ = derive_c3c(
                split.document(iid), queries[iid], calls["analyst"], calls.get("repair"), calls.get("reviewer"),
                reviewer_prompt=reviewer_prompt, notebook=notebook, adds_readings=arm.reviewer_adds_readings,
            )
        else:
            prediction, _ = derive(arm.method, split.document(iid), calls["analyst"], calls.get("repair"))
        total += 1
        answer = (prediction or {}).get("answer")
        ours = (answer, (prediction or {}).get("evidence") or [] if answer is not None else [])
        ref = released.get(iid)
        theirs = (ref["answer"], ref["evidence"]) if ref else (None, [])
        if ours == theirs:
            same += 1
        else:
            mismatches.append((iid, theirs, ours))
    print(f"{arm.name}: {same}/{total} predictions rebuilt identically from the released completions")
    for m in mismatches[:10]:
        print("  mismatch", m)
    return 0 if not mismatches else 1


def cmd_run(args: argparse.Namespace) -> int:
    from .client import OpenRouter

    arm = Arm.load(args.config)
    split = load_split(args.split or arm.split)
    tasks = split.tasks[: args.limit] if args.limit else split.tasks
    out_dir = Path(args.out or f"runs/{arm.name.replace('/', '-')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "records.jsonl"
    done = {r["instance_id"] for r in read_jsonl(out_file)} if out_file.is_file() else set()
    todo = [t for t in tasks if t["instance_id"] not in done]
    print(f"{arm.name}: {len(todo)} tasks to run ({len(done)} already in {out_file})")

    chat = OpenRouter(arm.model).chat
    reviewer = OpenRouter(arm.reviewer_model).chat if arm.reviewer_model else None
    lock = threading.Lock()

    def one(task: dict) -> dict:
        return solve(arm, task, split.document(task["instance_id"]), chat, reviewer)

    with ThreadPoolExecutor(max_workers=args.concurrency or arm.concurrency) as pool:
        futures = {pool.submit(one, t): t["instance_id"] for t in todo}
        for fut in as_completed(futures):
            iid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001  keep going; the task can be re-run later
                print(f"  {iid} failed: {exc}", file=sys.stderr)
                continue
            with lock, out_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {iid} {rec['answer']} {rec['evidence']} ({rec['decision']})")

    preds = [{k: r[k] for k in ("instance_id", "answer", "evidence", "decision")}
             for r in read_jsonl(out_file) if r.get("answer") is not None]
    (out_dir / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    if split.gold:
        _print_score(evaluate(preds, split.gold))
    return 0


def _print_score(out) -> None:
    k, n = out.count("answer"), out.n
    lo, hi = wilson(k, n)
    print(f"n={n}  answer {out.rate('answer'):.1f} [{lo:.1f}, {hi:.1f}]  "
          f"evidence {out.rate('evidence'):.1f}  joint {out.rate('joint'):.1f}")


def cmd_score(args: argparse.Namespace) -> int:
    split = load_split(args.split)
    if not split.gold:
        sys.exit(f"split {args.split!r} has no public labels")
    over = [t["instance_id"] for t in split.tasks] if args.all_items else None
    _print_score(evaluate(read_jsonl(args.predictions), split.gold, over=over))
    return 0


def cmd_replica(args: argparse.Namespace) -> int:
    from .replica import build, validity_gate

    if args.action == "build":
        print(json.dumps(build(Path(args.out), n=args.n, seed=args.seed), indent=2))
    else:
        print(json.dumps(validity_gate(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dfm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tables", help="recompute the paper's tables")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("replay", help="rebuild predictions from released completions (no API)")
    p.add_argument("config", type=Path, nargs="+")

    p = sub.add_parser("run", help="run an arm through OpenRouter")
    p.add_argument("config", type=Path)
    p.add_argument("--split", help="override the split in the config")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--out")

    p = sub.add_parser("score", help="score a predictions file")
    p.add_argument("predictions", type=Path)
    p.add_argument("split")
    p.add_argument("--all-items", action="store_true", help="count missing predictions as wrong")

    p = sub.add_parser("replica", help="build the replica or run its validity gate")
    p.add_argument("action", choices=["build", "gate"])
    p.add_argument("--out", default="replica-build")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args(argv)
    if args.command == "tables":
        tables.main(as_json=args.json)
        return 0
    if args.command == "replay":
        status = 0
        for config in args.config:
            ns = argparse.Namespace(config=config)
            status |= cmd_replay(ns)
        return status
    return {"run": cmd_run, "score": cmd_score, "replica": cmd_replica}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
