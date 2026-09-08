"""The arithmetic that was available all along and never ran before a run.

`stats.power_for`, `mde_proportion` and `games_for_difference` have existed
since the statistics went in. Nothing called them before launching, so a
100-per-arm A/B went out against a 0.244 baseline with a **0.34** probability of
detecting a real ten-point improvement — four hours to be more likely to miss
than to find, and knowable in a millisecond.

It PRINTS and never refuses. An A/B that cannot resolve the effect the pilot
cares about is still legitimate — as a noise floor, a smoke test, or the first
half of a bigger sample — and a gate that blocked it would be a validator firing
on correct use, which this repo has rejected six times.
"""

import json

import pytest

from manamap.sim import power


def test_no_baseline_is_absent_rather_than_assumed():
    """A preflight computed from a default rate would be a number about nothing,
    and it would look exactly like a number about something."""
    lines = power.preflight(None, 100, detect=0.10)
    assert len(lines) == 1 and "no measured baseline" in lines[0]


def test_the_underpowered_case_is_named_as_such():
    """THE CASE THAT COST FOUR HOURS. 0.244 baseline, 100 per arm, looking for
    +0.10 — the exact configuration that was launched."""
    lines = "\n".join(power.preflight(0.244, 100, detect=0.10))
    assert "UNDERPOWERED" in lines
    assert "34%" in lines
    assert "330" in lines or "324" in lines, "it must say how many games WOULD do it"


def test_an_adequate_run_is_not_scolded():
    """A validator that fires on correct use is worse than none. Detecting a
    +0.20 change at 100 per arm is a properly powered experiment and must read
    as one."""
    lines = "\n".join(power.preflight(0.244, 100, detect=0.20))
    assert "adequate" in lines and "UNDERPOWERED" not in lines


def test_the_thin_middle_is_distinguished_from_the_hopeless(monkeypatch):
    """Three verdicts, not two: 0.63 power is a real experiment with a real risk
    of missing, and calling it UNDERPOWERED alongside 0.12 would flatten a
    distinction the pilot needs."""
    lines = "\n".join(power.preflight(0.244, 100, detect=0.15))
    assert "THIN" in lines and "UNDERPOWERED" not in lines


def test_the_baseline_comes_from_a_run_against_THIS_table(tmp_path, monkeypatch):
    """A win rate is relative to the pod. Reading the deck's best-known rate off
    a run against a DIFFERENT table would anchor the whole preflight to a
    population the experiment will never face."""
    decks = tmp_path / "decks"
    sim = decks / "heliod" / "sim"
    sim.mkdir(parents=True)

    def rec(opps, rate, n, rid):
        return {"run_id": rid, "games_completed": n,
                "seats": [{"slug": "heliod"}] + [{"slug": o} for o in opps],
                "analysis": {"seats": {"heliod": {"win_rate": rate}}}}
    (sim / "wrong.json").write_text(json.dumps(rec(["vito", "x", "y"], 0.90, 400, "wrong")))
    (sim / "right.json").write_text(json.dumps(rec(["a", "b", "c"], 0.25, 120, "right")))
    (sim / "right-small.json").write_text(json.dumps(rec(["a", "b", "c"], 0.10, 20, "small")))
    monkeypatch.setattr(power, "DECKS_DIR", decks)

    rate, rid = power.baseline_rate("heliod", ["c", "b", "a"])
    assert rid == "right", "it took a run against another table"
    assert rate == 0.25
    # and the LARGEST sample against that table wins, not the newest filename
    assert rid != "small"


def test_an_unmeasured_table_reports_absent():
    rate, rid = power.baseline_rate("heliod", ["nobody-has-played-this"])
    assert rate is None and rid is None
