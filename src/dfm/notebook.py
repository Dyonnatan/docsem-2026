"""The C3C notebook: recurring interpretive ambiguities and their preferred side.

Each card names a *schema* (an ambiguity family) and, for some of them, the
side the notebook prefers. Three schemas carry a preferred side in the
released notebook:

* ``S-incl``  whether an added amount includes its base (``base_plus_added``);
* ``S-quant`` whether a fraction applies once or on each of k named occasions
  (``times_k``);
* ``S-pct``   whether a percentage difference is in points or relative terms
  (``points``).

The rules were written while analysing training cases. The reviewer model
labels each surviving reading with ``{schema: side}`` tags; when the tags
separate the readings cleanly, :func:`mechanical_close` picks the reading on
the preferred side. No model call takes part in that decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import PROMPTS_DIR

NOTEBOOK_PATH = PROMPTS_DIR / "notebook.json"

_SCHEMA_ALIASES = {
    "s-incl": "S-incl", "s_incl": "S-incl",
    "s-pct": "S-pct", "s_pct": "S-pct",
    "s-dist": "S-dist", "s_dist": "S-dist",
    "s-quant": "S-quant", "s_quant": "S-quant", "times_k": "S-quant",
    "s-time": "S-quant", "s_time": "S-quant",
    "s-rate": "S-rate", "s_rate": "S-rate",
    "s-space": "S-space", "s_space": "S-space",
}

_SIDE_ALIASES = {
    "base_plus_added": {"base_plus_added", "base+added", "plus_base", "base_and_added"},
    "added_only": {"added_only", "added", "new_only"},
    "times_k": {"times_k", "k_added", "k*added"},
    "once": {"once"},
    "points": {"points"},
    "relative": {"relative"},
    "group": {"group"},
    "group_times_n": {"group_times_n", "each"},
    "gap_only": {"gap_only", "gap", "length_over_gap", "no_width"},
    "gap_plus_width": {"gap_plus_width", "fencepost", "width_included", "with_width"},
}


def load_notebook(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or NOTEBOOK_PATH).read_text(encoding="utf-8"))


def canon_side(raw: object) -> str:
    key = str(raw or "").strip().lower().replace(" ", "")
    if not key:
        return ""
    for name, aliases in _SIDE_ALIASES.items():
        if key == name or key in aliases:
            return name
    return str(raw or "").strip().lower()


def canon_schema(raw: object) -> str:
    key = str(raw or "").strip().lower().replace(" ", "")
    return _SCHEMA_ALIASES.get(key, str(raw or "").strip())


def live_cards(notebook: dict[str, Any]) -> list[dict]:
    return [
        card
        for card in notebook.get("cards") or []
        if isinstance(card, dict) and card.get("status") not in ("downvoted", "rejected")
    ]


def preferred_sides(notebook: dict[str, Any], schemas: list[str]) -> list[dict]:
    """``[{schema, side}]`` for the given schemas whose card has a preferred side."""
    declared = {canon_schema(s) for s in schemas}
    prefers: list[dict] = []
    seen: set[str] = set()
    for card in live_cards(notebook):
        schema = canon_schema(card.get("schema"))
        if not (schema in declared or card.get("always")):
            continue
        side = canon_side(card.get("prefer_side"))
        if not schema or not side or schema not in declared or schema in seen:
            continue
        seen.add(schema)
        prefers.append({"schema": schema, "side": side})
    return prefers


def prompt_snippet(notebook: dict[str, Any], *, max_cards: int = 8) -> str:
    """The notebook rules as shown to the reviewer (no instance numbers)."""
    lines = []
    for card in live_cards(notebook)[:max_cards]:
        rule = str(card.get("rule") or "").strip()
        if rule:
            lines.append(f"- {card.get('schema')}/{card.get('fault')}: {rule}")
    if not lines:
        return "Notebook: no live cards."
    return "Notebook (generic; do not copy numbers from it):\n" + "\n".join(lines)


def applicable_prefers(live: list[dict], prefers: list[dict]) -> list[dict]:
    schemas: set[str] = set()
    for row in live:
        schemas.update(row.get("side_by_schema") or {})
    return [p for p in prefers or [] if canon_schema(p.get("schema")) in schemas and p.get("schema")]


def match_prefer(live: list[dict], prefers: list[dict]) -> list[dict]:
    """Readings consistent with every preferred side they are tagged on."""
    if not prefers:
        return list(live)
    matched = []
    for row in live:
        sides = row.get("side_by_schema") or {}
        ok = True
        for item in prefers:
            schema = canon_schema(item.get("schema"))
            want = canon_side(item.get("side"))
            if not schema or not want or schema not in sides:
                continue
            if canon_side(sides.get(schema)) != want:
                ok = False
                break
        if ok:
            matched.append(row)
    return matched


def collect_tags(review: dict) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Turn the reviewer's tags into ``{reading_id: {schema: side}}``.

    A reading tagged with two different sides on the same schema keeps no tag
    for that schema; the conflict is reported instead.
    """
    raw: dict[str, dict[str, set[str]]] = {}
    for tag in (review or {}).get("tags") or []:
        if not isinstance(tag, dict):
            continue
        rid, schema, side = tag.get("id"), tag.get("schema"), tag.get("side")
        if not rid or not schema or not side:
            continue
        schema_c, side_c = canon_schema(schema), canon_side(side)
        if schema_c and side_c:
            raw.setdefault(str(rid), {}).setdefault(schema_c, set()).add(side_c)

    tags: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for rid, schemas in raw.items():
        clean: dict[str, str] = {}
        for schema, sides in schemas.items():
            if len(sides) == 1:
                clean[schema] = next(iter(sides))
            else:
                conflicts.append(f"{rid}/{schema}: {sorted(sides)}")
        if clean:
            tags[rid] = clean
    return tags, conflicts


def mechanical_close(live: list[dict], notebook: dict) -> dict | None:
    """Pick one surviving reading from its notebook tags, or return ``None``.

    Two paths:

    * ``multi_schema``: at least two tagged schemas have a preferred side and
      exactly one reading is consistent with all of them;
    * ``single_axis``: exactly two readings, tagged on the same single schema
      with opposite sides, one of which is preferred. Cards can switch this
      path off with ``"single_axis_ok": false``.

    Cards marked ``"mechanical_ok": false`` never take part.
    """
    tagged = sorted({s for v in live for s in (v.get("side_by_schema") or {})})
    if not tagged:
        return None
    cards = notebook.get("cards") or []
    disabled = {canon_schema(c.get("schema")) for c in cards if c.get("mechanical_ok") is False}
    prefers = [p for p in preferred_sides(notebook, tagged) if canon_schema(p["schema"]) not in disabled]
    if not prefers:
        return None

    active = applicable_prefers(live, prefers)
    if len(active) >= 2:
        picked = match_prefer(live, active)
        if len(picked) == 1:
            return {"reading": picked[0]["id"], "path": "multi_schema", "prefers": active}

    single_off = {canon_schema(c.get("schema")) for c in cards if c.get("single_axis_ok") is False}
    if len(live) == 2 and len(tagged) == 1 and canon_schema(tagged[0]) not in single_off:
        schema = tagged[0]
        want = next((canon_side(p["side"]) for p in prefers if canon_schema(p["schema"]) == schema), None)
        if want:
            sides = [canon_side((v.get("side_by_schema") or {}).get(schema)) for v in live]
            if sides[0] and sides[1] and sides[0] != sides[1] and want in sides:
                chosen = live[0] if sides[0] == want else live[1]
                return {
                    "reading": chosen["id"],
                    "path": "single_axis",
                    "prefers": [{"schema": schema, "side": want}],
                }
    return None
