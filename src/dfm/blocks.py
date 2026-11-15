"""Block parsing and evidence post-processing.

Each document block starts with a printed identifier. Train and validation
use ``b01:``-style labels; the test split (and the replica) use opaque ids
with varying separators: ``F#72:``, ``U&64.``, ``2032 Record ...``. Ids are
never reformatted or filtered by shape here; they are only checked against the
document.

The functions below implement the deterministic evidence post-processing of
the submitted test system (paper, Appendix B):

* :func:`scope_evidence` keeps a single block when the committed reading's
  steps name exactly one candidate, or when the numbers it observes appear in
  exactly one candidate block; otherwise the list is left unchanged;
* :func:`repair_unknown_ids` maps ids that name no block (for example a case
  number cited instead of the block id) back to a block of the document;
* :func:`relocate_to_question_block` moves a citation off a block that poses no
  question when the reading's observed numbers identify a unique question
  block.
"""

from __future__ import annotations

import re

# An id at the start of a line, optionally after a markdown heading marker,
# followed by ":" / "." (any token) or by a space (then the id must contain a
# digit or symbol, so that ordinary words and titles are not taken as ids).
_BLOCK_LINE = re.compile(
    r"(?m)^(?:#{1,6}\s+)?(?:"
    r"([A-Za-z0-9][^\s]{0,31}?)[:.]\s+"
    r"|"
    r"([A-Za-z0-9][^\s]{0,31}?)\s+(?=[A-Z(])"
    r")"
)
_HAS_DIGIT_OR_SYMBOL = re.compile(r"[^A-Za-z]")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

# The decoy template of the test split: the query's own words inside a
# counting template ("In a separate concerning and opal and spends record,
# Given counted (10 + 13) items in the first review ...").
_DECOY = re.compile(r"In a separate\s+.{0,80}?record,|counted\s+.{0,40}?items in the first review", re.I)
_RECORD = re.compile(r"Record\s+(\S+?)[.\s]")
_QUESTION = re.compile(r"How (?:many|much)[^?]*\?|Determine[^?.]*[.?]|What (?:is|was)[^?]*\?", re.I)
_QUERY_KEYWORDS = re.compile(r"(?:about|concerning|regarding)\s+(\w+)\s+and\s+(\w+)", re.I)


def parse_blocks(document: str) -> dict[str, str]:
    """Split a stage-1 markdown document into ``{block_id: block_text}``."""
    out: dict[str, str] = {}
    spans: list[tuple[str, int]] = []
    for m in _BLOCK_LINE.finditer(document):
        punctuated, spaced = m.group(1), m.group(2)
        bid = punctuated if punctuated is not None else spaced
        if punctuated is None and not _HAS_DIGIT_OR_SYMBOL.search(bid):
            continue
        spans.append((bid, m.end()))
    for i, (bid, start) in enumerate(spans):
        end = len(document)
        if i + 1 < len(spans):
            nxt = spans[i + 1][1]
            line_start = document.rfind("\n", start, nxt)
            end = line_start if line_start != -1 else nxt
        out[bid] = document[start:end]
    return out


def looks_like_decoy(text: str) -> bool:
    return bool(_DECOY.search(text or ""))


def _committed_reading(state: dict) -> dict | None:
    readings = [r for r in (state.get("readings") or []) if isinstance(r, dict)]
    if not readings:
        return None
    commit = str(state.get("commit") or "")
    for r in readings:
        if str(r.get("id") or "") == commit:
            return r
    return readings[0] if len(readings) == 1 else None


def _reading_text(reading: dict) -> str:
    parts = [v for step in reading.get("steps") or [] if isinstance(step, dict)
             for v in step.values() if isinstance(v, str)]
    parts += [reading[k] for k in ("expression", "gloss", "note") if isinstance(reading.get(k), str)]
    return "\n".join(parts)


def _canon(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))


def _observed_values(reading: dict) -> list[str]:
    """Numeric literals the reading observes from the document (not derived)."""
    return [
        _canon(step["value"])
        for step in reading.get("steps") or []
        if isinstance(step, dict)
        and str(step.get("op") or "").lower() in {"obs", "given", "observe"}
        and isinstance(step.get("value"), (int, float))
    ]


def _numbers_in(text: str) -> set[str]:
    return {_canon(float(m.group(0))) for m in _NUM.finditer(text)}


def scope_evidence(evidence: list[str], state: dict, document: str) -> tuple[list[str], str]:
    """Reduce a multi-block citation to the block the committed reading used."""
    ev = [str(x) for x in (evidence or []) if str(x).strip()]
    if len(ev) <= 1:
        return ev, "single"
    reading = _committed_reading(state)
    if reading is None:
        return ev, "no_committed_reading"

    named = [b for b in ev if b and b in _reading_text(reading)]
    if len(named) == 1:
        return named, "named_in_steps"

    observed = _observed_values(reading)
    if observed:
        blocks = parse_blocks(document)
        scores = sorted(
            ((b, -1 if b not in blocks else sum(v in _numbers_in(blocks[b]) for v in observed)) for b in ev),
            key=lambda x: -x[1],
        )
        if scores and scores[0][1] > 0:
            winners = [b for b, s in scores if s == scores[0][1]]
            if len(winners) == 1:
                return winners, "obs_numbers_match"
    return ev, "ambiguous"


def _best_block_for_query(blocks: dict[str, str], state: dict, query: str) -> str | None:
    kw = _QUERY_KEYWORDS.search(query or "")
    keywords = [kw.group(1).lower(), kw.group(2).lower()] if kw else []
    reading = _committed_reading(state)
    observed = _observed_values(reading) if reading else []
    best, best_score = None, 0
    for bid, body in blocks.items():
        if not _QUESTION.search(body) or looks_like_decoy(body):
            continue
        low = body.lower()
        score = 3 * sum(1 for k in keywords if k in low)
        if observed:
            nums = _numbers_in(body)
            score += sum(1 for v in observed if v in nums)
        if score > best_score:
            best, best_score = bid, score
    return best if best_score > 0 else None


def repair_unknown_ids(evidence: list[str], state: dict, document: str, query: str) -> tuple[list[str], str]:
    """Replace evidence ids that name no block of the document."""
    blocks = parse_blocks(document)
    ev = [str(x) for x in (evidence or []) if str(x).strip()]
    if all(b in blocks for b in ev):
        return ev, "all_ids_real"
    out: list[str] = []
    reasons: set[str] = set()
    for b in ev:
        if b in blocks:
            out.append(b)
            continue
        owner = [bid for bid, body in blocks.items()
                 if _RECORD.search(body) and re.search(r"Record\s+" + re.escape(b) + r"[.\s]", body)]
        if len(owner) == 1:
            out.append(owner[0])
            reasons.add("case_id_to_block")
            continue
        guess = _best_block_for_query(blocks, state, query)
        if guess:
            out.append(guess)
            reasons.add("relocated_by_query")
        else:
            reasons.add("unrepairable")
    deduped = list(dict.fromkeys(out))
    if not deduped:
        return ev, "unrepairable"
    return deduped, "+".join(sorted(reasons)) or "all_ids_real"


def relocate_to_question_block(evidence: list[str], state: dict, document: str) -> tuple[list[str], str]:
    """Move a citation off a block that poses no question, using observed numbers."""
    ev = [str(x) for x in (evidence or []) if str(x).strip()]
    blocks = parse_blocks(document)
    if _QUESTION.search(" ".join(blocks.get(b, "") for b in ev)):
        return ev, "cites_a_question"
    reading = _committed_reading(state)
    observed = _observed_values(reading) if reading else []
    if not observed:
        return ev, "no_observed_values"
    scored = []
    for bid, body in blocks.items():
        if not _QUESTION.search(body):
            continue
        hits = sum(1 for v in observed if v in _numbers_in(body))
        if hits:
            scored.append((bid, hits))
    if not scored:
        return ev, "no_candidate"
    best = max(h for _, h in scored)
    if best < len(observed):
        return ev, "partial_match_only"
    winners = [b for b, h in scored if h == best]
    if len(winners) != 1:
        return ev, "ambiguous_candidates"
    return winners, "relocated_to_question"
