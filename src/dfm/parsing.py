"""Extract JSON objects from model completions."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
# Block identifiers are validated against the document, never by format: the
# train and validation splits use ``b01``-style ids, the test split uses opaque
# ids such as ``D#73`` or ``RB2-5E9``.
_ID_OK = re.compile(r"^\S+$")


def json_objects(text: str) -> Iterator[dict]:
    """Yield every top-level JSON object found in ``text``, in order."""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        index = start + end


def _valid_prediction(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    evidence = payload.get("evidence")
    if not isinstance(answer, str) or not answer.strip():
        if isinstance(answer, (int, float)):
            answer = str(answer)
        else:
            return None
    if not isinstance(evidence, list) or not evidence:
        return None
    ids: list[str] = []
    for item in evidence:
        if not isinstance(item, str) or not _ID_OK.match(item.strip()):
            return None
        ids.append(item.strip())
    instance_id = payload.get("instance_id")
    return {
        "instance_id": instance_id if isinstance(instance_id, str) else "",
        "answer": answer.strip(),
        "evidence": ids,
    }


def extract_prediction(text: str) -> dict | None:
    """Return the last ``{answer, evidence}`` object in a completion.

    Fenced JSON blocks take precedence over bare objects.
    """
    if not text or not text.strip():
        return None
    for match in reversed(list(_FENCE.finditer(text))):
        try:
            parsed = _valid_prediction(json.loads(match.group(1)))
        except json.JSONDecodeError:
            parsed = None
        if parsed:
            return parsed
    candidates = [p for p in map(_valid_prediction, json_objects(text)) if p]
    return candidates[-1] if candidates else None


def extract_state(text: str) -> dict | None:
    """Return the first object that carries a non-empty ``readings`` list."""
    if not text:
        return None
    for obj in json_objects(text):
        if isinstance(obj.get("readings"), list) and obj["readings"]:
            return obj
    return None


def first_json(text: str, must_have: str) -> dict | None:
    """Return the first object that contains the key ``must_have``."""
    if not text:
        return None
    for obj in json_objects(text):
        if must_have in obj:
            return obj
    return None
