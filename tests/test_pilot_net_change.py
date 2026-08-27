"""`net-change` — the report a spending decision rests on.

It was assembled by hand once: eight commands and a page of HTML, to decide
whether to buy 21 cards for the Ur-Dragon treasure refactor. The answer was no.
Doing that by hand again is how the next one gets skipped.
"""

import json

import pytest

from conftest import A_BRANCH, requires_branch, requires_deck
from manamap.pilot import net_change, validate_net_change

SLUG, BRANCH = "ur-dragon", A_BRANCH


def _doc():
    from manamap.pilot.common import deck_dir
    path = deck_dir(SLUG, BRANCH) / net_change.ARTIFACT
    if not path.exists():
        pytest.skip("no net_change.json on this checkout")
    return json.loads(path.read_text())


@requires_branch
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


@requires_branch
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


@requires_branch
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


# ── the recommendation: a ledger plus a rule you can argue with ──────────

def _ledger(grade_state, verdicts, objective=True):
    """A net-change document with the rows a rule reads and nothing else.

    Drives `net_change.recommend` — the production function — rather than
    re-deriving the rule, which is the test this repo has shipped four times.
    """
    table = [{"measure": f"m{i}", "champion": 1.0, "branch": 1.0,
              "delta": 0.5, "mde": 0.1, "verdict": v}
             for i, v in enumerate(verdicts)]
    return {
        "table": table,
        "objective": ({"axis": "hoard_8", "op": ">=", "value": 6.0}
                      if objective else None),
        "objective_grade": ({"state": grade_state, "why": "because"}
                            if objective else None),
        "bill": {"counts": {"buy": 21, "box": 7}},
    }


def test_the_ledger_sorts_every_row_and_loses_none():
    got = net_change.recommend(_ledger("met", ["better", "worse", "noise", "better"]))
    assert got["rose"] == ["m0", "m3"]
    assert got["fell"] == ["m1"]
    assert got["no_call"] == ["m2"]


def test_objective_met_and_nothing_lost_is_a_merge():
    got = net_change.recommend(_ledger("met", ["better", "noise"]))
    assert got["state"] == "merge"
    assert "nothing measured here got worse" in got["because"]


def test_objective_met_with_something_lost_is_a_trade_that_names_both_sides():
    """A metric falling is not a veto — it is a price. The report's job is to
    put both halves in one sentence so the pilot can decide."""
    got = net_change.recommend(_ledger("met", ["better", "worse"]))
    assert got["state"] == "a trade"
    assert "buy" in got["because"] and "pay" in got["because"]
    assert "m0" in got["because"] and "m1" in got["because"]
    assert got["bill"] == {"buy": 21, "box": 7}


def test_objective_not_met_is_a_refusal_even_when_other_things_improved():
    """THE UR-DRAGON FAILURE, as a rule. That branch hit its stated engine
    figure 4.4x over while getting worse at the thing it was for."""
    got = net_change.recommend(_ledger("not met", ["better", "better"]))
    assert got["state"] == "do not merge"
    assert "different branch's case" in got["because"]


def test_a_miss_the_run_cannot_see_is_inconclusive_not_a_failure():
    got = net_change.recommend(_ledger("not resolvable", ["noise"]))
    assert got["state"] == "inconclusive"
    assert "larger N" in got["because"]


def test_an_unreadable_objective_is_inconclusive_and_names_the_axis():
    """DIFFERENT FROM HAVING NO OBJECTIVE, and one state in the first draft.
    A branch that stated a goal the run could not read has been falsifiable all
    along; collapsing the two lets a branch that stated nothing borrow the
    credibility of one that did."""
    got = net_change.recommend(_ledger("not measured", ["better"]))
    assert got["state"] == "inconclusive"
    assert "hoard_8" in got["because"]
    assert got["state"] != net_change.recommend(
        _ledger(None, ["better"], objective=False))["state"]


def test_no_objective_says_the_ledger_still_stands():
    got = net_change.recommend(_ledger(None, ["better", "worse"], objective=False))
    assert got["state"] == "no objective"
    assert "only what changed" in got["because"]
    assert got["rose"] and got["fell"], "the ledger is still reported"


def test_every_state_is_declared():
    seen = {net_change.recommend(_ledger(g, ["better", "worse"]))["state"]
            for g in ("met", "not met", "not resolvable", "not measured")}
    seen.add(net_change.recommend(_ledger("met", ["better"]))["state"])
    seen.add(net_change.recommend(_ledger(None, ["better"], objective=False))["state"])
    assert seen == set(net_change.STATES)


def test_the_real_table_is_named_beside_the_verdict_and_never_folded_into_it():
    """Forge is evidence the rule does not use. Hiding it because the rule
    ignores it would be the worse error."""
    doc = _ledger("met", ["better"])
    doc["forge"] = {"available": True, "delta": -0.011, "ci95": [-0.102, 0.080]}
    got = net_change.recommend(doc)
    assert got["state"] == "merge", "the note must not change the verdict"
    assert any("cannot separate" in n for n in got["notes"])


@requires_branch
@requires_deck
def test_the_report_that_stopped_a_purchase_still_stops_it():
    """THE ACCEPTANCE CASE, and it is a decision already made by hand.

    ur-dragon's treasure branch: hoard @T10 +5.09, and damage, board power,
    turn-6 kill and stall all worse, against a bill of 21 cards to buy. The
    pilot read that report and did not spend the money. Reproducing the
    conclusion from the artifact that produced it is the proof.
    """
    from manamap.pilot.common import deck_dir
    path = (deck_dir(SLUG, A_BRANCH) / net_change.ARTIFACT)
    if not path.exists():
        pytest.skip("no tracked net_change.json")
    got = net_change.recommend(json.loads(path.read_text()))
    assert got["state"] != "merge"
    assert got["state"] == "no objective"
    assert len(got["rose"]) == 3 and len(got["fell"]) == 4
    assert got["bill"]["buy"] == 21


def test_a_table_where_nothing_moved_says_so():
    """NINE BLANK ROWS IS A FINDING. Measured on a one-swap branch of ur-dragon:
    every measure came back inside the MDE, which is the correct answer and
    reads as a broken tool unless the report says it out loud."""
    got = net_change.recommend(_ledger("not met", ["noise", "noise", "noise"]))
    assert any("Nothing moved" in n for n in got["notes"])
    assert any("minimum detectable difference" in n for n in got["notes"])
    # THE MEASUREMENT IS DECK-LEVEL. On a barely-changed branch the blank table
    # is arithmetic, not a verdict on the swaps — and the note says "unless it
    # is a Game Changer or a table-warper", because some single cards do move a
    # number and claiming otherwise would be a law where there is a tendency.
    thin = dict(_ledger("not met", ["noise", "noise"]), staged=1)
    note = " ".join(net_change.recommend(thin)["notes"])
    assert "not a verdict on the swap" in note
    assert "table-warper" in note, "the exception is stated, not hidden"
    fat = dict(_ledger("not met", ["noise", "noise"]), staged=22)
    assert "not a verdict on the swap" not in " ".join(
        net_change.recommend(fat)["notes"])
    # ...and it does not fire when something did move.
    quiet = net_change.recommend(_ledger("met", ["better", "noise"]))
    assert not any("Nothing moved" in n for n in quiet["notes"])
