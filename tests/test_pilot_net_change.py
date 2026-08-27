"""`net-change` — the report a spending decision rests on.

It was assembled by hand once: eight commands and a page of HTML, to decide
whether to buy 21 cards for the Ur-Dragon treasure refactor. The answer was no.
Doing that by hand again is how the next one gets skipped.
"""

import json

import pytest

from conftest import requires_deck
from manamap.pilot import net_change, validate_net_change

SLUG, BRANCH = "ur-dragon", "treasure-v2"


def _doc():
    from manamap.pilot.common import deck_dir
    path = deck_dir(SLUG, BRANCH) / net_change.ARTIFACT
    if not path.exists():
        pytest.skip("no net_change.json on this checkout")
    return json.loads(path.read_text())


@requires_deck
def test_it_reproduces_the_report_the_decision_was_made_on():
    """THE FIXTURE IS THE DECISION. These are the figures that stopped a
    purchase; a rewrite that moves one is a regression, not an improvement.

    Signs and magnitudes, not exact values — the harness is fixed but the
    goldfish is a simulation and this must not become a brittle golden file.
    """
    doc = _doc()
    by = {r["measure"]: r for r in doc["table"]}
    hoard = by["hoard @T10"]
    assert hoard["delta"] > 4.0 and hoard["verdict"] == "better", hoard
    kill6 = by["killed by T6"]
    assert kill6["delta"] < -0.03 and kill6["verdict"] == "worse", kill6
    damage = by["damage @T10"]
    assert damage["delta"] < 0 and damage["verdict"] == "worse", damage


@requires_deck
def test_the_engine_lift_is_the_measurement_that_decided_it():
    """The champion's declared engine makes it win; the branch's makes it win
    LESS, and both intervals exclude zero. Nothing else in the suite says this,
    which is the whole reason this figure is computed here."""
    doc = _doc()
    a = doc["engine_lift"]["champion"]
    b = doc["engine_lift"]["branch"]
    assert a["available"] and b["available"]
    assert a["lift"] > 0 and a["excludes_zero"], a
    assert b["lift"] < 0, b
    assert "win LESS" in b["reading"]


@requires_deck
def test_an_underpowered_forge_run_says_so():
    """12 v 11 wins over 201 games cannot resolve a 1-point difference. A delta
    printed without its MDE reads as 'no difference' when it means 'we could not
    have seen one'."""
    doc = _doc()
    f = doc["forge"]
    if not f.get("available"):
        pytest.skip("no Forge runs on this checkout")
    assert f["mde"] is not None
    assert abs(f["delta"]) < f["mde"], "this fixture is meant to be underpowered"
    assert f["excludes_zero"] is False


# ── the validator ────────────────────────────────────────────────────────

def _minimal(**over):
    doc = {"slug": "x", "branch": "b", "harness": {}, "limits": [],
           "table": [{"measure": "m", "champion": 1.0, "branch": 1.0,
                      "delta": 0.001, "mde": 0.05, "verdict": "noise"}],
           "engine_lift": {}, "forge": {"available": False, "why": "none"}}
    doc.update(over)
    return doc


def test_a_row_under_the_mde_may_not_be_ranked():
    """A report that ranks noise is how a spending decision gets made on a coin
    flip."""
    assert not validate_net_change.validate(_minimal())
    bad = _minimal(table=[{"measure": "m", "champion": 1.0, "branch": 1.0,
                           "delta": 0.001, "mde": 0.05, "verdict": "better"}])
    assert any("must be marked noise" in e for e in validate_net_change.validate(bad))


def test_a_row_that_clears_the_mde_may_not_be_called_noise():
    bad = _minimal(table=[{"measure": "m", "champion": 1.0, "branch": 2.0,
                           "delta": 1.0, "mde": 0.05, "verdict": "noise"}])
    assert any("reported as noise" in e for e in validate_net_change.validate(bad))


def test_an_engine_lift_must_state_whether_its_interval_excludes_zero():
    bad = _minimal(engine_lift={"branch": {"available": True, "lift": -0.03,
                                           "ci95": [-0.05, -0.01],
                                           "reading": "…"}})
    assert any("excludes_zero" in e for e in validate_net_change.validate(bad))


def test_an_unavailable_block_owes_a_reason():
    bad = _minimal(engine_lift={"branch": {"available": False}})
    assert any("no reason given" in e for e in validate_net_change.validate(bad))


def test_an_objective_stated_and_never_graded_is_an_error():
    bad = _minimal(objective={"axis": "kill_by_8", "op": ">=", "value": 0.3})
    assert any("never graded" in e for e in validate_net_change.validate(bad))
    worse = _minimal(objective={"axis": "kill_by_8", "op": ">=", "value": 0.3},
                     objective_grade={"state": "probably fine"})
    assert any("is not one of" in e for e in validate_net_change.validate(worse))


def test_an_empty_table_claims_nothing():
    assert any("claims nothing" in e
               for e in validate_net_change.validate(_minimal(table=[])))
