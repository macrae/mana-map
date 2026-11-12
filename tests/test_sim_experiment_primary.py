"""`experiment --primary`: a pre-registered verdict figure other than the win rate (2026-11-11).

Edgar's threat-v1 A/B pre-registers forge.life_removed_total. The reading must speak to the
chosen figure and call every other row descriptive; a sequential design (which tests a
proportion at each boundary) refuses a non-proportion primary. Proved by re-introducing the
bug: make `_reading` ignore `primary` and the first test reads the win-rate sentence.
"""
import pytest

from manamap.sim import experiment


def _out():
    return {"win_rate": {"ci95_diff": [-0.05, 0.08], "excludes_zero": False},
            "life_removed_total": {"ci95_diff": [3.0, 11.5], "excludes_zero": True,
                                   "median_ci95_diff": [2.0, 9.0]},
            "power": {}}


def test_the_reading_speaks_to_the_preregistered_primary():
    r = experiment._reading(_out(), "life_removed_total")
    assert r.startswith("PRIMARY life_removed_total (pre-registered)")
    assert "EXCLUDES zero" in r and "descriptive" in r and "MEDIAN" in r


def test_the_default_primary_is_still_the_win_rate():
    assert "win-rate difference" in experiment._reading(_out())


def test_a_sequential_design_refuses_a_continuous_primary():
    with pytest.raises(SystemExit, match="--looks"):
        experiment.run("edgar-vampires", "working", "working", ["krenko-tokens"],
                       looks=2, aa=True, dry_run=True, primary="life_removed_total")


def test_a_win_rate_boundary_never_stops_a_continuous_primary():
    """The 2026-11-12 A/A read STOPPED (efficacy) off the win rate it had not
    pre-registered. Proved by re-introducing the bug: drop _decide's primary guard."""
    look = {"excludes_zero": True, "futility_excluded": False}
    assert experiment._decide(look, 0, 1, "life_removed_total") == ("complete", "complete")
    assert experiment._decide(look, 0, 1) == ("stop: efficacy", "stopped_efficacy")


def test_an_unknown_primary_is_refused():
    with pytest.raises(SystemExit, match="--primary"):
        experiment.run("edgar-vampires", "working", "working", ["krenko-tokens"],
                       dry_run=True, primary="vibes")
