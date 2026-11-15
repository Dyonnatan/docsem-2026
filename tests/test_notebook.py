from dfm.notebook import collect_tags, load_notebook, mechanical_close


def test_multi_schema_close_prefers_notebook_side():
    live = [
        {"id": "r1", "side_by_schema": {"S-incl": "added_only", "S-quant": "once"}},
        {"id": "r2", "side_by_schema": {"S-incl": "base_plus_added", "S-quant": "times_k"}},
        {"id": "r3", "side_by_schema": {"S-incl": "base_plus_added", "S-quant": "once"}},
    ]
    out = mechanical_close(live, load_notebook())
    assert out["reading"] == "r2" and out["path"] == "multi_schema"


def test_single_axis_disabled_for_s_incl():
    live = [{"id": "r1", "side_by_schema": {"S-incl": "added_only"}},
            {"id": "r2", "side_by_schema": {"S-incl": "base_plus_added"}}]
    assert mechanical_close(live, load_notebook()) is None


def test_conflicting_tags_are_dropped():
    tags, conflicts = collect_tags({"tags": [
        {"id": "r1", "schema": "S-quant", "side": "once"},
        {"id": "r1", "schema": "S-quant", "side": "times_k"},
        {"id": "r2", "schema": "s_pct", "side": "points"},
    ]})
    assert tags == {"r2": {"S-pct": "points"}} and conflicts
