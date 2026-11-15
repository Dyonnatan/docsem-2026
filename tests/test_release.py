"""End-to-end checks on the released data and outputs (no network)."""
import filecmp
import json
from pathlib import Path

import pytest

from dfm import tables
from dfm.cli import main
from dfm.paths import CONFIGS_DIR, DATA_DIR
from dfm.replica import build


@pytest.mark.parametrize("config", sorted(CONFIGS_DIR.glob("*/*.json")), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_replay_rebuilds_released_predictions(config):
    assert main(["replay", str(config)]) == 0


def test_replica_is_regenerated_byte_for_byte(tmp_path):
    build(tmp_path)
    ref = DATA_DIR / "replica"
    for name in ("tasks.jsonl", "labels.jsonl"):
        assert filecmp.cmp(tmp_path / name, ref / name, shallow=False)
    match, mismatch, errors = filecmp.cmpfiles(
        tmp_path / "documents", ref / "documents",
        [p.name for p in (ref / "documents").iterdir()], shallow=False)
    assert not mismatch and not errors and len(match) == 300


def test_replica_table_matches_paper():
    rows = {r["arm"]: (r["answer"], r["evidence"], r["joint"]) for r in tables.replica_table()["rows"]}
    assert rows["DeepSeek A0"] == (46.3, 46.0, 40.3)
    assert rows["DeepSeek C3+card"] == (68.3, 76.0, 63.7)
    assert rows["GLM C3+card"] == (77.7, 80.3, 67.0)
    assert rows["Claude Code (Sonnet 5)"] == (82.0, 94.0, 82.0)


def test_decomposition_matches_paper():
    dec = tables.decomposition()
    d = dec["C3C, all tasks (paper)"]
    assert d["n"] == 300 and d["answer_accuracy"] == 49.0 and d["joint_accuracy"] == 38.7
    assert d["selection_errors"][:2] == [162, 54.0] and d["resolution_errors"][:2] == [22, 7.3]
    assert sum(d["decoy_cited"]["selection"][:1] + d["decoy_cited"]["selection, right number"][:1]) == 88
    d = dec["C3 analyst outputs, first pass (as presented at the workshop)"]
    assert d["n"] == 291 and d["answer_accuracy"] == 50.5
    assert d["selection_errors"][:2] == [154, 52.9] and d["resolution_errors"][:2] == [22, 7.6]
