"""The computed engine against three witnesses, over the tracked fleet (memo §6.1).

Regression tier, and asserted on SHAPE only: the figures are the finding, written
into `docs/research/engine-model.md` §6.1, and a test that pinned them would be a
test of the memo. `engine_graph.agreement` is the one producer.
"""
import pytest

from manamap.pilot import engine_graph as eg

pytestmark = [pytest.mark.regression,
              pytest.mark.skipif(not __import__("manamap.sim.forge_cards", fromlist=["x"]).installed(),
                                 reason="the record is read from Forge's card scripts")]


def test_agreement_is_measured_for_every_deck_with_a_record():
    from manamap.config import DECKS_DIR
    checked = 0
    for path in sorted(DECKS_DIR.glob("*/engine.json")):
        a = eg.agreement(path.parent.name)
        checked += 1
        assert a["engine_cards"] >= 1
        d = a["declared"]
        if d is not None:
            assert 0.0 <= d["jaccard"] <= 1.0 and d["groups"] >= 1
        f = a["forge"]
        if f is not None:
            assert f["n"] >= 0 and f["games"] >= 1
            assert f["spearman_degree_vs_plays"] is None or -1.0 <= f["spearman_degree_vs_plays"] <= 1.0
        lr = a["log_reads"]
        if lr is not None:
            assert lr["n"] == sum(v for row in lr["by_position"].values() for v in row.values())
            assert set(lr["by_position"]) <= {"hub", "leaf", "outside"}
    assert checked >= 1


def test_spearman_is_one_on_a_monotone_pair_and_handles_ties():
    assert eg.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert eg.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert eg.spearman([1, 1, 1], [1, 2, 3]) is None
    assert eg.spearman([1, 2], [1, 2]) is None
