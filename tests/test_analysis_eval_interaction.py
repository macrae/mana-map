"""The interaction space's evaluation (2026-11-22). The record is tracked; the
regression test reruns it and holds the committed file to the fresh one, shape and
figures, so a parser change that moves the result cannot pass unrecorded."""
import json

import pytest

from manamap.analysis import eval_interaction as ev


@pytest.mark.regression
def test_the_committed_evaluation_matches_a_fresh_run():
    from manamap import spaces
    if not ev.EVAL_PATH.exists() or not spaces.get(spaces.INTERACTION).exists():
        pytest.skip("evaluation or interaction space not on disk")
    committed = json.loads(ev.EVAL_PATH.read_text())
    fresh = ev.run()
    assert committed["verified_pairs"]["recall_at_10"] == fresh["verified_pairs"]["recall_at_10"]
    assert committed["golden"] == fresh["golden"]
    s = fresh["verified_pairs"]["recall_at_10"]
    assert s["queries"] >= 100 and {"feeds", "interaction", "function", "random"} <= set(s)


def test_the_default_space_does_not_move_without_the_evaluation_saying_so():
    """`spaces.DEFAULT` flips to interaction only when the committed evaluation has
    it beating the incumbent on the verified pairs. On 2026-11-22 it did not."""
    from manamap import spaces
    if not ev.EVAL_PATH.exists():
        pytest.skip("evaluation not on disk")
    s = json.loads(ev.EVAL_PATH.read_text())["verified_pairs"]["recall_at_10"]
    if spaces.DEFAULT == spaces.INTERACTION:
        assert s["interaction"] > s["function"], "the default moved against the evaluation"
