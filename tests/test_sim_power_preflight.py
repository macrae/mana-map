"""The arithmetic that was available all along and never ran before a run.

`stats.power_for`, `mde_proportion` and `games_for_difference` have existed
since the statistics went in. Nothing called them before launching, so a
100-per-arm A/B went out against a 0.244 baseline with a **0.34** probability of
detecting a real ten-point improvement — four hours to be more likely to miss
than to find, and knowable in a millisecond.

It PRINTS and does not refuse — unless the pilot ASKED a question the run cannot
answer. `--detect X` is that question; a run whose power for X is under 0.8 is
refused with the arithmetic, and `--anyway` runs it on the record as a screen.
Without `--detect` nothing is refused: a noise floor, a smoke test, the first
half of a bigger sample are all legitimate, and a gate that blocked them would
be a validator firing on correct use, which this repo has rejected six times.
"""

import json

import pytest

from manamap import config
from manamap.sim import power


def test_no_baseline_is_absent_rather_than_assumed():
    """A preflight computed from a default rate would be a number about nothing,
    and it would look exactly like a number about something."""
    lines = power.preflight(None, 100, detect=0.10)
    assert len(lines) == 1 and "no measured baseline" in lines[0]


@pytest.mark.slow
def test_the_underpowered_case_is_named_as_such():
    """THE CASE THAT COST FOUR HOURS. 0.244 baseline, 100 per arm, looking for
    +0.10 — the exact configuration that was launched."""
    lines = "\n".join(power.preflight(0.244, 100, detect=0.10))
    assert "UNDERPOWERED" in lines
    assert "34%" in lines
    assert "330" in lines or "324" in lines, "it must say how many games WOULD do it"


@pytest.mark.slow
def test_an_adequate_run_is_not_scolded():
    """A validator that fires on correct use is worse than none. Detecting a
    +0.20 change at 100 per arm is a properly powered experiment and must read
    as one."""
    lines = "\n".join(power.preflight(0.244, 100, detect=0.20))
    assert "adequate" in lines and "UNDERPOWERED" not in lines


@pytest.mark.slow
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
    monkeypatch.setattr(config, "DECKS_DIR", decks)

    rate, rid = power.baseline_rate("heliod", ["c", "b", "a"])
    assert rid == "right", "it took a run against another table"
    assert rate == 0.25
    # and the LARGEST sample against that table wins, not the newest filename
    assert rid != "small"


def test_an_unmeasured_table_reports_absent():
    rate, rid = power.baseline_rate("heliod", ["nobody-has-played-this"])
    assert rate is None and rid is None


def test_a_detect_the_run_cannot_see_is_refused_unless_anyway():
    """THE QUESTION AND THE ANSWER MUST MATCH. 20 games against a 0.233
    baseline cannot see +0.05 at any useful power; `--detect 0.05` on such a
    run is refused with the games that would do it, `--anyway` runs it, and
    no `--detect` at all refuses nothing (re-introduce by making the refusal
    unconditional and the last assertion fails)."""
    with pytest.raises(SystemExit) as e:
        power.refuse_if_underpowered(0.233, 20, 0.05, anyway=False)
    msg = str(e.value)
    assert "UNDERPOWERED" in msg and "per arm" in msg and "--anyway" in msg
    assert power.refuse_if_underpowered(0.233, 20, 0.05, anyway=True) is None
    assert power.refuse_if_underpowered(0.233, 20, None) is None
    assert power.refuse_if_underpowered(None, 20, 0.05) is None, "no baseline: nothing to refuse on"
    # A properly powered question is let through and its power returned.
    assert power.refuse_if_underpowered(0.244, 100, 0.20) >= 0.8


@pytest.mark.slow
def test_a_one_arm_preflight_quotes_one_arm_of_hours():
    """`simulate` plays ONE arm — its comparison is the pod's null, already
    paid for. The hours line hardcoded `2 * games` for every caller, so a
    100-game simulate read as a 4.8-hour job. (Bug: put the `2 *` back.)"""
    two = power.preflight(0.233, 100, arms=2)[0]
    one = power.preflight(0.233, 100, arms=1)[0]
    assert "games/arm" in two and "about 4.8 h" in two
    assert "games/arm" not in one and "about 2.4 h" in one


@pytest.mark.slow
def test_the_comparison_arm_can_be_the_null_with_its_own_size():
    """A run compared against a 400-game null has more power than one against
    an equal 100-game arm, and the preflight must compute the test that will
    actually be run."""
    equal = power.preflight(0.233, 100, detect=0.15)
    vs_null = power.preflight(0.233, 100, detect=0.15, arms=1, n_a=400)
    assert "against 400 on the other side" in "\n".join(vs_null)
    def pw(lines):
        return float([l for l in lines if "chance of seeing" in l][0].split("%")[0].split()[-1])
    assert pw(vs_null) > pw(equal)


def test_the_null_is_absent_not_defaulted(monkeypatch):
    """(rate, games) from the pod's calibration, or (None, None) — never a
    quarter, never a zero."""
    assert power.null_rate(None) == (None, None)
    assert power.null_rate("") == (None, None)
    assert power.null_rate("a-table-that-does-not-exist") == (None, None)
