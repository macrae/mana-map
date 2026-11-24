"""A newer goldfish model is an older STAMP on an engine record, never a red gate
(2026-11-23): every deck's preflight failed on `_meta.model_version` alone after a
Meren-only simulator change. Any difference of substance still fails."""
from manamap.pilot import validate_engine as ve


def _rec(model="m1", nodes=("a", "b")):
    return {"_meta": {"v": 2, "model_version": model, "forge_digest": "f"},
            "graph": {"nodes": list(nodes)}}


def test_a_record_that_differs_only_in_the_model_stamp_has_no_substantive_difference():
    diffs = ve._differences(_rec("m2"), _rec("m1"))
    assert diffs == ["_meta.model_version differs"]
    assert ve.substantive(diffs) == []


def test_a_record_whose_content_moved_still_differs_with_or_without_the_stamp():
    diffs = ve.substantive(ve._differences(_rec("m2", ("a", "b", "c")), _rec("m1")))
    assert diffs and all("model_version" not in d for d in diffs)
