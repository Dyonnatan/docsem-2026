"""Evaluation sets: tasks, gold labels and stage-1 documents.

==========  =====================================================  ===========
split       tasks and documents                                    gold
==========  =====================================================  ===========
``replica`` ``data/replica`` (300 test-like documents)             by construction
``dev200``  200 training tasks (``data/splits/dev200.json``)       public
``hard18``  18 training tasks (``data/splits/hard18.json``)        public
``train``   908 training tasks                                     public
``val``     217 validation tasks                                   portal only
``test``    1730 test tasks (documents not bundled, see data/README) portal only
==========  =====================================================  ===========
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR

SPLITS = ("replica", "dev200", "hard18", "train", "val", "test")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@dataclass
class Split:
    name: str
    tasks: list[dict]
    documents: Path
    gold: dict[str, dict] | None

    def document(self, instance_id: str) -> str:
        return (self.documents / f"{instance_id}.md").read_text(encoding="utf-8")


def load_split(name: str, data_dir: Path = DATA_DIR) -> Split:
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    if name == "replica":
        root = data_dir / "replica"
        gold = {r["instance_id"]: r for r in _jsonl(root / "labels.jsonl")}
        return Split(name, _jsonl(root / "tasks.jsonl"), root / "documents", gold)

    source = "train" if name in ("dev200", "hard18", "train") else name
    tasks = _jsonl(data_dir / "docsem" / source / "tasks.jsonl")
    labels_path = data_dir / "docsem" / source / "labels.jsonl"
    gold = {r["instance_id"]: r for r in _jsonl(labels_path)} if labels_path.is_file() else None
    if name in ("dev200", "hard18"):
        keep = set(json.loads((data_dir / "splits" / f"{name}.json").read_text())["instance_ids"])
        tasks = [t for t in tasks if t["instance_id"] in keep]
    return Split(name, tasks, data_dir / "extraction" / source, gold)
