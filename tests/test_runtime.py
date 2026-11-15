from dfm.derivation import compile_reading, compile_state, decompose_base_fraction
from dfm.readings import rate_drop_check, select_reading

DOC = "b01: Title\n\nb02: Ana buys 3 boxes of 12 eggs and uses 5. How many eggs are left?\n"


def test_dsl_compiles_open_vocabulary():
    reading = {"id": "r1", "steps": [
        {"op": "obs", "id": "boxes", "value": 3},
        {"op": "take", "id": "per", "value": 12},
        {"op": "assume", "text": "every box is full"},
        {"op": "hence", "id": "total", "expr": "boxes * per"},
        {"op": "return", "expr": "total - 5"},
    ]}
    out = compile_reading(reading)
    assert out["expression"] == "((3) * (12)) - 5"
    assert out["n_obs"] == 2 and out["n_assume"] == 1


def test_unique_survivor_and_v2():
    state = {"evidence": ["b02"], "commit": "r2", "readings": [
        {"id": "r1", "expression": "3*12-5"},
        {"id": "r2", "expression": "12-5"},  # omits the 3 that r1 uses -> V2
    ]}
    report = select_reading(state, DOC)
    assert report["reason"] == "unique_survivor" and report["answer"] == "31"
    assert any(f.startswith("V2") for f in report["readings"][1]["fails"])


def test_commit_among_survivors_and_v5():
    state = {"evidence": ["b02"], "commit": "r2", "readings": [
        {"id": "r1", "expression": "3*12-5"},
        {"id": "r2", "expression": "3*(12-5)"},
        {"id": "r3", "expression": "import os"},
    ]}
    report = select_reading(state, DOC)
    assert report["reason"] == "commit_among_survivors" and report["answer"] == "21"
    assert report["readings"][2]["fails"][0].startswith("V5")


def test_v4_rate_drop():
    text = "It costs 10 for the first 5 minutes, then the rate drops to 3 per minute. A call lasts 20 minutes."
    assert rate_drop_check(text, 10 + 15 * 3) is not None
    assert rate_drop_check(text, 5 * 10 + 15 * 3) is None


def test_base_fraction_completion():
    assert decompose_base_fraction("(3*40)*0.25") == ("3 * 40", "0.25", 0.25)
    state, _ = compile_state({"readings": [{"id": "r1", "expression": "(3*40)*0.25"}]})
    added = {r["id"]: r["side_by_schema"] for r in state["readings"] if r.get("inferred")}
    assert added == {"r1i1": {"S-incl": "base_plus_added", "S-quant": "once"},
                     "r1i2": {"S-incl": "base_plus_added", "S-quant": "times_k"}}
