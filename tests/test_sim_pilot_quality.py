"""Was the AI playing the deck, or holding it?

A Forge run carries Forge's own caveat that its AI "is not trained" — but a
caveat is a warning, not a measurement, and it cannot say whether THIS run was
piloted well enough for its outcome to mean anything. Every run contains its own
control: the other seats, played by the same AI, in the same games.
"""

import glob
import json

import pytest

from manamap.sim import pilot_quality as pq


def _runs():
    return glob.glob("data/decks/**/sim/*.json", recursive=True)


def test_it_reads_the_record_not_the_logs():
    """The logs are gitignored and only exist where the run was made. A reading
    that needed them could never be taken from a checkout."""
    import inspect
    src = inspect.getsource(pq)
    assert ".log" not in src and "logs" not in src.replace("the logs", "")


@pytest.mark.parametrize("path", _runs() or ["none"])
def test_a_tracked_run_yields_a_piloting_reading(path):
    if path == "none":
        pytest.skip("no sim runs on this machine")
    rec = json.load(open(path))
    q = pq.from_record(rec)
    if q is None:
        pytest.skip(f"{path}: too few seats or games to compare")
    for metric in (pq.LANDS, pq.CASTS):
        m = q[metric]
        assert m["ours"] >= 0 and m["pod_mean"] > 0
        assert 0 < m["ratio"] < 5, f"{path}: implausible ratio {m['ratio']}"
    assert isinstance(q["comparable"], bool)
    assert q["reading"]


def test_the_verdict_follows_the_ratio_and_says_which():
    """A verdict a reader has to take on trust is worse than a number."""
    rec = {
        "seats": [{"slug": "ours"}, {"slug": "them"}],
        "games": [{"round": 10,
                   "per_seat": {"ours": {"lands": 2, "casts": 5},
                                "them": {"lands": 8, "casts": 10}}}],
    }
    q = pq.from_record(rec)
    assert q["comparable"] is False
    assert "HANDLED WORSE" in q["reading"]
    assert q[pq.LANDS]["ratio"] < pq.COMPARABLE
    # And the numbers are present either way, so the verdict is checkable.
    assert q[pq.LANDS]["ours"] == 0.2 and q[pq.LANDS]["pod_mean"] == 0.8

    even = {
        "seats": [{"slug": "ours"}, {"slug": "them"}],
        "games": [{"round": 10,
                   "per_seat": {"ours": {"lands": 7, "casts": 10},
                                "them": {"lands": 7, "casts": 10}}}],
    }
    assert pq.from_record(even)["comparable"] is True
