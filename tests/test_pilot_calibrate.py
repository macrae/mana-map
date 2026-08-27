"""`calibrate` — and the refusal that is its main output today."""

import pytest

from conftest import requires_deck
from manamap.pilot import calibrate


def test_it_refuses_a_verdict_below_a_usable_sample(monkeypatch):
    """THE POINT OF THE COMMAND.

    A rank correlation on five points has a CI spanning almost the whole range.
    Printing `rho = -0.10` without that is how a null gets read as a finding —
    the failure this repo has rejected in three other places. So below the
    threshold it reports the GAP and what would close it, and no coefficient.
    """
    monkeypatch.setattr(calibrate, "forge_record", lambda *a, **k: ({}, [], []))
    got = calibrate.calibrate(iterations=50)
    assert got["verdict"] == "NOT ANSWERABLE"
    assert "spearman" not in got
    assert "more deck(s)" in got["what_it_would_take"]


def test_a_deck_below_the_game_threshold_is_excluded_not_downweighted():
    """heliod's 8 games is a shuffle, not a sample: its win rate reads 0.250
    against a fleet that spans 0.00-0.21, purely on n. Including it flipped the
    sign of the only rank correlation available."""
    got = calibrate.calibrate(iterations=50)
    for row in got["decks"]:
        assert row["games"] >= calibrate.MIN_GAMES, row


@requires_deck
def test_it_correlates_our_decks_and_never_the_pod():
    """An opponent seat is a fetched EDHREC average list with no goldfish
    figures of its own — giada-angels wins 41% and correlating it against
    nothing would be the easiest way to manufacture a relationship."""
    record, _pod, _dropped = calibrate.forge_record()
    assert record, "no tracked sim runs — this test cannot see the bug"
    assert not any(s.startswith(("giada", "vito", "baylen")) for s in record), record


def test_the_combat_family_is_represented_once():
    """Correlating power, damage and kill would be one test counted three
    times — the axis collapse in `candidates.AXES`, one module over."""
    combat = [k for k, v in calibrate.MEASURES.items() if v[0] == "combat"]
    assert len(combat) <= 2, combat


def test_spearman_handles_ties():
    """Two decks at 0.000 is the common case on this fleet, and a naive rank
    that breaks ties by position invents an ordering the data does not have."""
    assert calibrate._spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert calibrate._spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert calibrate._spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)


def test_runs_against_a_different_pod_are_dropped_not_pooled():
    """A WIN RATE IS AGAINST SOMEBODY.

    The first cut summed every tracked run regardless of who was at the table.
    Measured on what exists: kianne's 24 games are 12 against the standard pod
    and 12 in a 1v1 against giada alone — a different game, no politics and no
    second threat — while radagast's 28 are 20 standard and 8 against a pod of
    our OWN decks. Pooling those gives a number that is not a win rate against
    anything.
    """
    record, pod, dropped = calibrate.forge_record()
    assert pod, "no pod chosen"
    assert dropped, "nothing was dropped — this fleet has mixed pods"
    # the 1v1 must not be in the pooled figure
    assert all(len(d["pod"]) >= 1 for d in dropped)
    assert not any(set(d["pod"]) == set(pod) for d in dropped)


def test_the_pod_is_chosen_by_games_not_by_run_count():
    """One 100-game run is better evidence than three 8-game ones, and counting
    RUNS would prefer the noise."""
    rows = [("a", frozenset({"x"}), 1, 100), ("b", frozenset({"y"}), 0, 8),
            ("c", frozenset({"y"}), 0, 8), ("d", frozenset({"y"}), 0, 8)]
    import unittest.mock as mock
    with mock.patch.object(calibrate, "_seat_rows", lambda: rows):
        _, pod, _ = calibrate.forge_record()
    assert pod == ["x"], pod


def test_a_correlation_with_the_wrong_sign_is_a_confound_not_a_result():
    """THE FIRST REAL RUN'S STRONGEST NUMBER WAS ITS WORST.

    With 11 decks the table led with `cmdr_turn` at **+0.760** — decks whose
    commander casts LATER win MORE. As a validation that reads "cast your
    commander later"; honestly it is an expensive commander standing in for an
    expensive DECK (gishath 7.8, ur-dragon 8.0, edgar 6.5 are three of the four
    best win rates; hapatra 2.1, sisay 3.2, radagast 3.9 three of the four
    worst). In a bare table of coefficients a confound and a finding look
    identical, so every measure declares which way it should point.
    """
    assert calibrate.EXPECTED["cmdr_turn"] < 0, "casting sooner is better"
    assert calibrate.EXPECTED["kill_by_8"] > 0
    got = calibrate.calibrate(iterations=200)
    if got["verdict"] != "measured":
        pytest.skip("not enough Forge data on this checkout")
    for name, cell in got["spearman"].items():
        assert set(cell) >= {"rho", "sign_agrees", "significant", "reading"}
        wrong = not cell["sign_agrees"]
        assert wrong == ("WRONG SIGN" in cell["reading"]), (name, cell)


def test_significance_is_reported_against_the_sample_size():
    """0.585 at n=11 is not a weak finding, it is no finding — the critical
    value is 0.618. Printing a coefficient bare is how a null gets read as one.
    """
    assert calibrate.critical_rho(11) == pytest.approx(0.618)
    assert calibrate.critical_rho(30) < calibrate.critical_rho(11)
    # A sample too small for any honest verdict must demand a perfect rank.
    assert calibrate.critical_rho(3) == 1.0
