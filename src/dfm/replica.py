"""Build the test-like replica from the training split.

Train and validation documents pose one quantitative question each, so they
cannot measure passage *selection*. The test split poses about two dozen per
document, but its labels are withheld. The replica closes that gap: it takes
training target blocks, whose answer and evidence are public, and embeds each
one in a document generated from the test template, so that the gold label is
known by construction.

Each generated document contains (paper, Appendix A):

* the target block, with the training template's framing sentence removed
  (that sentence lists the query's anchor words verbatim);
* about 22 competing question blocks drawn from other training targets;
* in 68% of documents, 4 to 12 decoys that repeat the query's anchors inside a
  counting template, at least as often as the target does;
* filler prose taken from non-target training blocks;
* opaque block ids in the eight shapes and three separators seen in test;
* "outcome records" for about 70% of the non-target cases and never for the
  target.

``python -m dfm replica build --out <dir>`` regenerates ``data/replica``
byte for byte with the default seed.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .blocks import parse_blocks
from .paths import DATA_DIR

QUERY_TEMPLATES = [
    "According to the brief's quantitative scenario concerning {x} and {y}, what final value is required?",
    "Use the relevant quantitative passage in this document to determine the outcome concerning {x} and {y}.",
    "What is the requested result for the report's scenario about {x} and {y}?",
]
REVISED_PREFIX = "For this revised scenario, "
ANCHOR_RE = re.compile(r"concerning\s+(\w+)\s+and\s+(\w+)|about\s+(\w+)\s+and\s+(\w+)", re.I)

CASE_ALPHA = "0123456789ABCDEF"
ID_ALPHA = "abcdefghijkmnpqrstuvwxyz"
ID_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"

_ECHO = [
    "The {x} and {y} entries were cross-referenced during the same review.",
    "A related note filed under {x} and {y} was retained with this record.",
    "Earlier correspondence about {x} and {y} is appended for context.",
    "The desk log lists {x} and {y} among the items reviewed that morning.",
    "Subsequent updates concerning {x} and {y} did not alter this entry.",
]
_SENT = re.compile(r"(?<=[.!?])\s+")


def block_id(rng: random.Random) -> str:
    """An opaque id in one of the eight shapes seen in test (D#73, RB2-5E9, z-713K, ...)."""
    shape = rng.randrange(8)

    def digits(n: int) -> str:
        return "".join(rng.choice("0123456789") for _ in range(n))

    if shape == 0:
        return f"{rng.choice(ID_UPPER)}#{digits(2)}"
    if shape == 1:
        return f"R{rng.choice(CASE_ALPHA)}{digits(1)}-{digits(1)}{rng.choice(CASE_ALPHA)}{digits(1)}"
    if shape == 2:
        return f"{rng.choice(ID_ALPHA)}-{digits(3)}{rng.choice(ID_UPPER)}"
    if shape == 3:
        return f"{digits(2)}{rng.choice(ID_UPPER)}{digits(3)}"
    if shape == 4:
        return f"{digits(1)}{rng.choice(ID_ALPHA)}{digits(2)}"
    if shape == 5:
        return f"{rng.choice(ID_UPPER)}&{digits(2)}"
    if shape == 6:
        return f"{digits(2)}/{rng.choice(ID_UPPER)}{digits(1)}"
    return digits(rng.choice([3, 4]))


def case_id(rng: random.Random) -> str:
    return f"{rng.choice(CASE_ALPHA)}{rng.choice(CASE_ALPHA)}-{''.join(rng.choice('0123456789') for _ in range(3))}"


def anchors(query: str) -> tuple[str, str]:
    """The two anchor words of a training query ("concerning X and Y")."""
    m = ANCHOR_RE.search(query)
    if not m:
        return ("scenario", "record")
    g = [x for x in m.groups() if x]
    return (g[0], g[1]) if len(g) >= 2 else ("scenario", "record")


def strip_framing(text: str) -> str:
    """Drop the leading framing sentences of a training block.

    Training blocks open with sentences such as "During the planning review,
    several teams discussed school, hours, and long.", which contain the query
    anchors verbatim. In test it is the decoy, not the target, that repeats the
    query's words, so these sentences are removed: everything before the first
    sentence with a digit or a question mark is dropped.
    """
    sents = _SENT.split(text.strip())
    for i, s in enumerate(sents):
        if re.search(r"\d", s) or "?" in s:
            return " ".join(sents[i:]).strip()
    return text.strip()


def load_pool(data_dir: Path = DATA_DIR) -> tuple[list[dict], list[str]]:
    """Training targets with gold, and filler prose from non-target blocks."""
    def rows(path: Path) -> dict[str, dict]:
        return {json.loads(l)["instance_id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}

    labels = rows(data_dir / "docsem/train/labels.jsonl")
    tasks = rows(data_dir / "docsem/train/tasks.jsonl")
    pool, filler = [], []
    for iid, gold in labels.items():
        path = data_dir / "extraction/train" / f"{iid}.md"
        if not path.is_file():
            continue
        blocks = parse_blocks(path.read_text(encoding="utf-8"))
        ev = gold["evidence"][0]
        if ev not in blocks:
            continue
        x, y = anchors(tasks[iid]["user_query"])
        pool.append({"src": iid, "text": strip_framing(blocks[ev]), "answer": gold["answer"], "x": x, "y": y})
        for bid, txt in blocks.items():
            if bid != ev and 120 < len(txt) < 700 and "?" not in txt:
                filler.append(txt.strip())
    return pool, filler


def decoy(rng: random.Random, x: str, y: str, case: str, min_echo: int = 1) -> tuple[str, int]:
    """A keyword-stuffed counting problem that repeats the anchors ``min_echo`` times.

    Matching the target's anchor density matters: with a single mention per
    anchor, counting anchor occurrences found the target 60.6% of the time.
    """
    a, b = rng.randrange(1, 30), rng.randrange(1, 30)
    c = rng.randrange(5, 60)
    who = rng.choice(["the case owner", "the reviewer", "the clerk", "the auditor"])
    text = (f"Record {case}. The file preserves the following case entry. In a separate concerning "
            f"and {x} and {y} record, {who} counted ({a} + {b}) items in the first review and {c} "
            f"in the second review. How many items did {who} count altogether?")
    for _ in range(1, min_echo):
        text += " " + rng.choice(_ECHO).format(x=x, y=y)
    return text + " The entry was retained with the case materials for review.", a + b + c


def build_document(rng: random.Random, target: dict, pool: list[dict], filler: list[str],
                   n_questions: int, n_filler: int) -> tuple[str, str]:
    """Return ``(markdown, target_block_id)``."""
    others = rng.sample([p for p in pool if p["src"] != target["src"]], k=max(4, n_questions - 1 - 6))
    entries = []  # (text, is_target, (case, value) for an outcome record or None)

    entries.append((f"Record {case_id(rng)}. The file preserves the following case entry. {target['text']} "
                    f"The entry was retained with the case materials for review.", True, None))
    for p in others:
        c = case_id(rng)
        entries.append((f"Record {c}. The file preserves the following case entry. {p['text']} "
                        f"The entry was retained with the case materials for review.", False, (c, p["answer"])))

    # About a third of test documents have no competitor for the query's anchors.
    n_decoy = 0 if rng.random() < 0.32 else rng.randrange(4, 13)
    tl = target["text"].lower()
    echo = max(tl.count(target["x"].lower()), tl.count(target["y"].lower()), 1)
    for _ in range(n_decoy):
        c = case_id(rng)
        txt, val = decoy(rng, target["x"], target["y"], c, min_echo=echo + rng.randrange(0, 2))
        entries.append((txt, False, (c, str(val))))

    rng.shuffle(entries)

    lines = ["# Quantitative Case File", ""]
    used, outcomes = set(), []
    target_bid = None
    pending = list(entries)
    filler_pool = rng.sample(filler, k=min(n_filler, len(filler)))
    fi = 0

    def fresh_id() -> str:
        bid = block_id(rng)
        while bid in used:
            bid = block_id(rng)
        used.add(bid)
        return bid

    while pending or fi < len(filler_pool):
        take_question = pending and (fi >= len(filler_pool) or rng.random() < 0.22)
        bid = fresh_id()
        sep = rng.choice([": ", ": ", ": ", ". ", " "])
        if take_question:
            txt, is_target, outcome = pending.pop()
            if is_target:
                target_bid = bid
            elif outcome and rng.random() < 0.70:
                outcomes.append(outcome)
            lines.append(f"{bid}{sep}{txt}")
        else:
            lines.append(f"{bid}{sep}{filler_pool[fi]}")
            fi += 1
        lines.append("")

    rng.shuffle(outcomes)
    for c, v in outcomes:
        lines.append(f"{fresh_id()}: Outcome record {c}: the final recorded value for case {c} was {v}.")
        lines.append("")
    return "\n".join(lines), target_bid


def build(out_dir: Path, *, n: int = 300, seed: int = 2026, n_questions: int = 24, n_filler: int = 92,
          data_dir: Path = DATA_DIR) -> dict:
    """Write ``tasks.jsonl``, ``labels.jsonl`` and ``documents/*.md`` to ``out_dir``."""
    rng = random.Random(seed)
    pool, filler = load_pool(data_dir)
    targets = rng.sample(pool, k=min(n, len(pool)))

    docs = out_dir / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    tasks, labels, sizes = [], [], []
    for k, target in enumerate(targets, 1):
        iid = f"synth_{k:06d}"
        doc, bid = build_document(rng, target, pool, filler, n_questions, n_filler)
        if bid is None:
            continue
        (docs / f"{iid}.md").write_text(doc, encoding="utf-8")
        sizes.append(len(doc))
        q = rng.choice(QUERY_TEMPLATES).format(x=target["x"], y=target["y"])
        if rng.random() < 0.66:
            q = REVISED_PREFIX + q[0].lower() + q[1:]
        tasks.append({"instance_id": iid, "user_query": q, "document_pdf": f"synth/{iid}.pdf"})
        labels.append({"instance_id": iid, "answer": target["answer"], "evidence": [bid],
                       "source_task": target["src"]})

    for name, rows in (("tasks.jsonl", tasks), ("labels.jsonl", labels)):
        (out_dir / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                    encoding="utf-8")
    sizes.sort()
    return {"documents": len(tasks), "pool": len(pool), "filler_blocks": len(filler),
            "median_chars": sizes[len(sizes) // 2] if sizes else 0}


def validity_gate(replica_dir: Path = DATA_DIR / "replica") -> dict:
    """Check that lexical heuristics do not find the target better than chance.

    Candidates are the question blocks (containing "?", excluding outcome
    records) that mention at least one of the query's two anchor words. The
    target counts only when it is itself a candidate. Heuristics:

    * keyword: most anchor-word occurrences (ties split evenly);
    * longest: the longest candidate;
    * first: the first candidate in reading order;
    * random: expected accuracy of a uniform choice, ``1 / |candidates|``.

    In test the decoys are the closer lexical match, so a faithful replica
    must keep the keyword heuristic at or below chance.
    """
    def word_count(word: str, text: str) -> int:
        return len(re.findall(r"\b" + re.escape(word.lower()) + r"\b", text.lower()))

    tasks = [json.loads(l) for l in (replica_dir / "tasks.jsonl").read_text().splitlines() if l.strip()]
    gold = {json.loads(l)["instance_id"]: json.loads(l)
            for l in (replica_dir / "labels.jsonl").read_text().splitlines() if l.strip()}
    hits = {"keyword": 0.0, "longest": 0.0, "first": 0.0, "random": 0.0}
    n = 0
    for task in tasks:
        iid = task["instance_id"]
        blocks = parse_blocks((replica_dir / "documents" / f"{iid}.md").read_text(encoding="utf-8"))
        x, y = anchors(task["user_query"])
        target = gold[iid]["evidence"][0]
        cands = [k for k, v in blocks.items()
                 if "?" in v and "Outcome record" not in v and (word_count(x, v) or word_count(y, v))]
        if target not in cands:
            continue
        n += 1
        score = {k: word_count(x, blocks[k]) + word_count(y, blocks[k]) for k in cands}
        top = [k for k in cands if score[k] == max(score.values())]
        hits["keyword"] += (target in top) / len(top)
        hits["longest"] += max(cands, key=lambda k: len(blocks[k])) == target
        hits["first"] += cands[0] == target
        hits["random"] += 1 / len(cands)
    return {"n": n, **{k: round(100 * v / n, 1) for k, v in hits.items()}}
