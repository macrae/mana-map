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
def test_every_row_is_consistent_with_its_own_minimum_detectable_difference():
    """THE CONTRACT, NOT ONE EXPERIMENT'S FIGURES.

    This asserted the treasure branch's exact numbers — hoard @T10 delta > 4.0,
    killed by T6 worse, damage @T10 negative — as "the decision fixture". That
    branch was measured, found worse and deleted, and a test may not pin an
    experimental deck (PLAN.md, the 2026-08-27 issue). What must hold for ANY
    report is the thing a spending decision rests on: a row is ranked only when
    the run could actually see it.
    """
    doc = _doc()
    checked = 0
    for r in doc["table"]:
        assert r["mde"] is not None, r
        under = abs(r["delta"]) <= r["mde"]
        assert (r["verdict"] == "noise") == under, (
            f"{r['measure']}: delta {r['delta']} against MDE {r['mde']} is "
            f"reported {r['verdict']!r}")
        checked += 1
    assert checked >= 5, "a report with almost no rows decides nothing"


@requires_branch
@requires_deck
def test_the_engine_lift_states_whether_its_interval_excludes_zero():
    """The lift is the one measurement computed here rather than composed.
    Published without saying whether its interval excludes zero it is a number
    with no claim attached — so both arms carry that, or say why they cannot.

    The SIGN is not asserted. Which way a branch's engine moves is a fact about
    that branch, and the previous version required it to be negative because
    the treasure refactor's was.
    """
    doc = _doc()
    for who in ("champion", "branch"):
        e = doc["engine_lift"][who]
        if not e.get("available"):
            assert e.get("why"), f"{who}: unavailable with no reason"
            continue
        for key in ("lift", "ci95", "excludes_zero", "reading"):
            assert key in e, f"{who}: no {key!r}"
        assert isinstance(e["excludes_zero"], bool)
        lo, hi = e["ci95"]
        assert (lo > 0 or hi < 0) == e["excludes_zero"], e
        assert ("win LESS" in e["reading"]) == (e["lift"] < 0), e


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
    """A REAL REPORT, HELD TO THE RULE. This named the treasure branch's own
    figures; that branch is deleted, so what survives is the rule applied to
    whatever report is on disk — the ledger accounts for every row, and the
    verdict follows the OBJECTIVE rather than the count of improvements.
    """
    doc = _doc()
    got = doc.get("recommendation") or net_change.recommend(doc)
    assert got["state"] in net_change.STATES
    measures = {r["measure"] for r in doc["table"]}
    named = set(got["rose"]) | set(got["fell"]) | set(got["no_call"])
    assert named == measures, "the ledger must account for every measured row"
    grade = (doc.get("objective_grade") or {}).get("state")
    if grade == "not met":
        # THE RULE THAT MATTERS: improvements elsewhere do not buy a merge.
        assert got["state"] == "do not merge", got
    if got["state"] == "merge":
        assert not got["fell"], "a merge with something worse is a trade"


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


@requires_branch
@requires_deck
def test_the_report_carries_the_mana_half_the_goldfish_rows_cannot_see():
    """THE REPORT DECIDED A PURCHASE WITHOUT IT.

    `ROWS` is derived from the goldfish, which measures development and not
    castability by colour. So a branch that cut three counterspells (blue pips)
    and added six dorks changed its entire pip distribution and the report said
    nothing at all — the pilot had to be told separately that the nine rows do
    not cover it. On the real branch this immediately showed the dork swaps
    costing a white and a blue source, because three of the five dorks added
    contribute nothing to a STATIC count (two scale with the board, one is
    restricted mana).
    """
    doc = _doc()
    m = doc.get("mana")
    assert m is not None, "no mana block at all"
    if not m.get("available"):
        assert m.get("why")
        return
    seen = set()
    for r in m["colours"]:
        seen.add(r["colour"])
        # THE GAP IS THE FIGURE, not the count: a target moves when the pips
        # move, which is the whole reason this runs after a spell change.
        for key in ("target", "have", "gap"):
            assert len(r[key]) == 2, r
        assert r["gap"][0] == r["have"][0] - r["target"][0], r
        assert r["gap"][1] == r["have"][1] - r["target"][1], r
        assert r["delta"] == r["gap"][1] - r["gap"][0], r
    assert seen == set("WUBRG")
    assert len(m["lands"]) == 2 and len(m["enters_tapped_always"]) == 2
    # NOT A `table` ROW, and deliberately: those carry a Newcombe interval on
    # the difference, and a source count is deterministic with no sampling
    # error. Giving it a verdict beside them would make a different KIND of
    # number look like the same kind.
    assert "verdict" not in m
    assert not any(r["measure"].lower().startswith(("w ", "colour"))
                   for r in doc["table"])
    assert "no sampling error" in m["note"]


@requires_branch
@requires_deck
def test_the_mana_block_agrees_with_mana_analysis_rather_than_recomputing():
    """One owner per figure. `mana_fit` learned this the expensive way — its
    first cut recomputed and reported 53 red sources against mana-analysis's
    27 — and this composes the same module for the same reason."""
    from manamap.pilot import mana_analysis
    doc = _doc()
    m = doc.get("mana") or {}
    if not m.get("available"):
        pytest.skip("no mana block on this checkout")
    theirs = mana_analysis.analyze(SLUG)
    checked = 0
    for r in m["colours"]:
        assert r["have"][0] == theirs["sources"]["total"][r["colour"]], r
        assert r["target"][0] == theirs["source_targets"][r["colour"]], r
        checked += 1
    assert checked == 5
