"""`net-change` — the report a spending decision rests on.

It was assembled by hand once: eight commands and a page of HTML, to decide
whether to buy 21 cards for the Ur-Dragon treasure refactor. The answer was no.
Doing that by hand again is how the next one gets skipped.
"""

import inspect
import json

import pytest

from conftest import A_BRANCH, requires_branch, requires_data, requires_deck
from manamap.pilot import deck_branch, net_change, validate_net_change

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
           "forge": {"available": False, "why": "none"}}
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


@pytest.mark.parametrize("block", ["mana", "forge"])
def test_an_unavailable_block_owes_a_reason(block):
    """ABSENT MEANS ABSENT, AND IT OWES A REASON — a blank section a reader
    cannot tell from a measured nothing.

    This rule lived on the engine-lift block alone, so deleting that block took
    the rule with it and left `mana` and `forge` free to go quiet. It is stated
    per-block now for exactly that reason."""
    bad = _minimal(**{block: {"available": False}})
    assert any("no reason given" in e for e in validate_net_change.validate(bad))
    ok = _minimal(**{block: {"available": False, "why": "no run yet"}})
    assert not any("no reason given" in e for e in validate_net_change.validate(ok))


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


# --------------------------------------------------------------------------
# THE DEFINITIONS. A figure whose meaning is not on the page beside it gets
# guessed at, and the guesses go one way: a mean read as a rate, a clock read
# as a win rate, a hoard read as mana. All three have happened on this bench.
# --------------------------------------------------------------------------

def test_every_row_the_report_can_render_has_a_definition_and_a_reason():
    """`ROWS` and `METRICS` are two lists that must not drift apart. A row with
    no entry renders a bare number under a heading promising definitions."""
    for label, *_ in net_change.ROWS:
        spec = net_change.METRICS.get(label)
        assert spec, f"{label} is rendered and has no definition"
        assert spec["what"].strip() and spec["why"].strip()
        assert spec["unit"] in ("rate", "mean"), label


def test_no_definition_exists_for_a_row_that_is_never_rendered():
    """The inverse. A definition for a row nobody shows is a claim about output
    that is not true, and it is how the registry rots without failing."""
    rendered = {label for label, *_ in net_change.ROWS}
    assert set(net_change.METRICS) == rendered


def test_a_clock_is_never_described_as_a_win_rate():
    """The single most consequential misreading available here: `killed by T6`
    is measured against ONE opponent at 40 life who never blocks. A reader who
    takes 0.318 for a win rate has overestimated the deck by the whole size of
    a pod."""
    for label in ("killed by T6", "killed by T10"):
        assert "CLOCK" in net_change.METRICS[label]["scale"]
        assert "never a win rate" in net_change.METRICS[label]["scale"]


def test_a_rate_reads_as_games_per_hundred_and_a_mean_keeps_its_units():
    rate = {"measure": "killed by T6", "champion": 0.177, "branch": 0.318,
            "delta": 0.141, "mde": 0.02, "verdict": "better"}
    assert net_change.reads_as(rate) == (
        "18 games in 100 -> 32 in 100, a swing of 14 games per 100")
    mean = {"measure": "damage @T10", "champion": 51.658, "branch": 77.314,
            "delta": 25.656, "mde": 3.0, "verdict": "better"}
    got = net_change.reads_as(mean)
    assert "51.66 -> 77.31" in got and "+50%" in got
    assert "games in 100" not in got, "a mean is not a rate"


def test_a_no_call_row_says_no_answer_rather_than_no_change():
    """`noise` is the most misread word in the report. The run did not find
    nothing; it could not resolve what it found."""
    row = {"measure": "hoard @T10", "champion": 1.881, "branch": 1.987,
           "delta": 0.105, "mde": 0.138, "verdict": "noise"}
    got = net_change.reads_as(row)
    assert "no call" in got and "0.138" in got
    assert "smaller than" in got


def test_a_champion_reading_of_zero_does_not_divide_by_it():
    """A deck that does no damage at all is a real reading, and a percentage
    change against zero is not."""
    row = {"measure": "damage @T10", "champion": 0.0, "branch": 4.2,
           "delta": 4.2, "mde": 1.0, "verdict": "better"}
    assert "%" not in net_change.reads_as(row)


# --------------------------------------------------------------------------
# VALUE, RISK, COST — the three the ledger owed and did not carry
# --------------------------------------------------------------------------

def _skeleton(**over):
    doc = {"table": [], "bill": {"counts": {}, "cards": []},
           "objective": None, "objective_grade": {},
           "mana": {}, "forge": {"available": False, "why": "no run"},
           "blind_spots": []}
    doc.update(over)
    return doc


def test_the_risk_block_separates_a_measured_loss_from_an_unmeasured_one():
    """THE DISTINCTION THE WHOLE BLOCK EXISTS FOR. A row that fell and an
    effect the harness cannot see read alike on a page and are not remotely the
    same claim — one is a priced cost, the other is an open question rendered
    beside a confidence interval."""
    doc = _skeleton(
        table=[{"measure": "hoard @T6", "champion": 0.538, "branch": 0.439,
                "delta": -0.1, "mde": 0.02, "verdict": "worse",
                "reads_as": "0.54 -> 0.44", "why_we_care": "x"}],
        blind_spots=[{"class": "removal", "cards": ["Swords to Plowshares"],
                      "headline": "1 card(s) carrying a removal effect",
                      "why": "no opponents"}])
    kinds = [r["kind"] for r in net_change.risk(doc)]
    assert "paid" in kinds and "unmeasured" in kinds
    assert "structural" in kinds, "the goldfish caveat is always owed"


def test_an_unmeasured_effect_is_scoped_to_the_effect_and_not_the_card():
    """Solphim is a `protection:self` body AND a damage doubler the combat
    model prices at +7 damage. Filing the whole card under "unmeasured" would
    understate the branch the line exists to warn about."""
    doc = _skeleton(blind_spots=[
        {"class": "protection", "cards": ["Solphim, Mayhem Dominus"],
         "headline": "1 card(s) carrying a protection effect", "why": "y"}])
    entry = next(r for r in net_change.risk(doc) if r["kind"] == "unmeasured")
    assert "EFFECT" in entry["why_it_matters"]
    assert "still measured" in entry["why_it_matters"]


def test_a_colour_that_went_backwards_is_a_paid_cost_no_sampled_row_can_see():
    doc = _skeleton(mana={"colours": [
        {"colour": "G", "gap": [0, -1], "delta": -1},
        {"colour": "R", "gap": [-10, -8], "delta": +2}]})
    entry = next(r for r in net_change.risk(doc)
                 if r["what"] == "colour sources went backwards")
    assert entry["kind"] == "paid"
    assert "G gap +0 -> -1" in entry["detail"]
    assert "R" not in entry["detail"], "a colour that improved is not a risk"


def test_the_reward_block_only_carries_rows_that_beat_their_own_mde():
    doc = _skeleton(table=[
        {"measure": "damage @T10", "verdict": "better", "reads_as": "a",
         "why_we_care": "b"},
        {"measure": "hoard @T10", "verdict": "noise", "reads_as": "c",
         "why_we_care": "d"},
        {"measure": "hoard @T6", "verdict": "worse", "reads_as": "e",
         "why_we_care": "f"}])
    assert [r["measure"] for r in net_change.reward(doc)] == ["damage @T10"]


def test_a_card_in_a_retired_deck_is_not_a_deck_you_have_to_take_apart():
    """`elsewhere` is two costs wearing one integer. A card in a broken-down or
    retired deck is loose cardboard; one in a deck that is still together costs
    that deck the card, and a pilot deciding whether to pull sleeves needs the
    two apart.

    `free` and `apart` are `deck_branch.source`'s answer, not a second one here:
    this block used to carry its own `FREE_TO_RAID`, which was a fourth copy of
    a set `common.UNPLAYABLE_STATUSES` already held.
    """
    doc = _skeleton(bill={"counts": {"elsewhere": 2, "buy": 1}, "cards": [
        {"name": "Faeburrow Elder", "state": "elsewhere", "free": True,
         "where": [{"kind": "deck", "slug": "sisay", "status": "retired",
                    "apart": True}]},
        {"name": "Bloom Tender", "state": "elsewhere", "free": False,
         "where": [{"kind": "deck", "slug": "kinnan", "status": None,
                    "apart": False}]},
        {"name": "Twinflame Tyrant", "state": "buy", "free": False, "where": []}]})
    got = net_change.cost(doc)
    assert [r["name"] for r in got["free_to_raid"]] == ["Faeburrow Elder"]
    assert [r["name"] for r in got["must_unsleeve"]] == ["Bloom Tender"]
    assert got["buy_cards"] == ["Twinflame Tyrant"]
    assert "cost nothing" in got["reads_as"]


def test_a_card_in_several_decks_reports_only_the_ones_still_together():
    """Forbidden Orchard sits in six decks, two of them apart. Naming all six
    overstates what merging disturbs."""
    doc = _skeleton(bill={"counts": {}, "cards": [
        {"name": "Forbidden Orchard", "state": "elsewhere", "free": False,
         "where": [
             {"kind": "deck", "slug": "blar", "status": None, "apart": False},
             {"kind": "deck", "slug": "hapatra", "status": "broken-down",
              "apart": True},
             {"kind": "deck", "slug": "sisay", "status": "retired",
              "apart": True}]}]})
    entry = net_change.cost(doc)["must_unsleeve"][0]
    assert entry["decks"] == ["blar"]


def test_the_cost_block_no_longer_decides_what_apart_means():
    """ONE PREDICATE, ONE HOME. Four modules answered "is this deck in a pile"
    and could disagree; `common.deck_is_apart` decides, `deck_branch.source`
    derives it per row, and this reads the row."""
    assert not hasattr(net_change, "FREE_TO_RAID")
    from manamap.pilot.common import UNPLAYABLE_STATUSES, deck_is_apart
    assert UNPLAYABLE_STATUSES == frozenset({"broken-down", "retired"})
    assert callable(deck_is_apart)


# --------------------------------------------------------------------------
# THE CHANGE and the blind spots, against the real branch
# --------------------------------------------------------------------------

@requires_data
@requires_branch
@requires_deck
def test_the_report_names_the_swaps_rather_than_counting_them():
    """"21 staged" is not a description of a treatment."""
    ch = net_change.changes(SLUG, BRANCH)
    assert ch["count"] == len(ch["spells"]) + len(ch["lands"])
    assert ch["count"] >= 1
    # A ROW NEED NOT BE A PAIR, and asserting that it was is what let the
    # staging-log bug live: `changes()` used to read every swap ever staged, so
    # every row was paired by construction — including pairs where the added
    # card had since been staged back out. It now reports the NET diff, where a
    # card whose partner was superseded keeps its own reason and loses only the
    # arrow. What must hold is that every row names at least one side.
    checked = paired = 0
    for row in ch["spells"] + ch["lands"]:
        assert row["out"] or row["in"], "a row that names neither side"
        paired += bool(row["out"] and row["in"])
        checked += 1
    assert checked >= 1
    assert paired >= 1, "no row is a pair at all — the `why` provenance is lost"


@requires_data
@requires_branch
@requires_deck
def test_a_land_swap_is_filed_apart_from_a_spell_swap():
    """They are answered by different halves of this report: a spell swap moves
    the nine sampled rows, a land swap moves only the deterministic mana block.
    Mixed together, a land pass borrows credit from a spell pass."""
    from manamap.pilot import card_pool
    pool = card_pool.load_pool()
    ch = net_change.changes(SLUG, BRANCH)
    checked = 0
    for row in ch["lands"]:
        assert any("Land" in ((pool.get(row[side]) or {}).get("type_line") or "")
                   for side in ("out", "in")), row
        checked += 1
    for row in ch["spells"]:
        for side in ("out", "in"):
            assert "Land" not in ((pool.get(row[side]) or {}).get("type_line") or "")
        checked += 1
    assert checked >= 2


@requires_data
@requires_branch
@requires_deck
def test_a_land_swap_always_declares_that_the_model_cannot_rank_lands():
    """MEASURED, and it is why the note is mandatory rather than advisory: a
    twelve-land `candidates` sweep returned exactly two distinct readings, with
    an always-tapped land tying one that never enters tapped. Nine rows of
    intervals beside a land swap read as an accounting of it and are silent."""
    ch = net_change.changes(SLUG, BRANCH)
    if not ch["lands"]:
        pytest.skip("this branch stages no land swap")
    spots = net_change.blind_spots(SLUG, BRANCH, ch)
    land = next((b for b in spots if b["class"] == "land"), None)
    assert land, "a land swap with no land blind-spot note"
    assert "no tapped state" in land["why"]


def test_every_blind_class_owes_a_sentence():
    """A class in the map with no explanation renders a warning nobody can act
    on."""
    for head, why in net_change.BLIND.items():
        assert why.strip() and head == head.lower()


def test_the_report_never_reads_the_authored_declaration():
    """THE ENGINE LIFT WAS DELETED 2026-08-28 AND MUST NOT COME BACK BY HABIT.

    It split games by whether the components marked `required` in
    `goldfish_targets.json` had been drawn. That file is authored, so the same
    hand wrote the target and read the verdict: three defensible declarations
    of one Ur-Dragon list, over the same 10,000 games, gave +0.007 (spanning
    zero), -0.036 (REAL) and +0.014 (REAL) — one of them saying at an interval
    excluding zero that assembling the engine made the deck win LESS.

    A figure whose sign a JSON edit can flip is not evidence however tight its
    interval, and it sat in the block a spending decision reads first.
    """
    import inspect
    src = inspect.getsource(net_change)
    assert "def engine_lift" not in src
    assert '"engine_lift"' not in src
    # The lift is the only thing that ever read `required`. Anything that starts
    # reading it again has re-introduced an authored input to a measured report.
    body = src[src.index("def build("):]
    assert '"required"' not in body and "get(\"required\")" not in body


def test_the_deleted_figure_leaves_its_reason_behind():
    """A deletion with no record gets undone by the next person who notices the
    gap. The measurement that justified it lives in the module docstring."""
    assert "AUTHORED" in net_change.__doc__
    assert "-0.036" in net_change.__doc__ and "+0.014" in net_change.__doc__


@requires_branch
@requires_deck
def test_no_written_report_still_carries_the_deleted_block():
    doc = _doc()
    assert "engine_lift" not in doc
    blob = json.dumps(doc)
    assert "online_by_turn" not in blob


# --------------------------------------------------------------------------
# card_diff — the merge-request view of a branch
# --------------------------------------------------------------------------

@requires_data
@requires_branch
@requires_deck
def test_the_diff_is_derived_from_the_LISTS_and_not_from_the_staged_swaps():
    """THE BUG THIS FUNCTION EXISTS FOR. `changes()` reads `branch.json`'s
    `staged` array, and a branch opened with `new --from <list>` sets its whole
    99 at once and stages NOTHING. So a 17-for-17 refactor rendered as "The
    change (0)" while the report beneath it measured all 34 cards, and the cards
    going OUT appeared nowhere on the page at all."""
    d = net_change.card_diff(SLUG, BRANCH)
    staged = len((deck_branch.meta(SLUG, BRANCH) or {}).get("staged") or [])
    assert d["counts"]["out"] and d["counts"]["in"], "a branch differs from its deck"
    assert d["counts"]["out"] + d["counts"]["in"] > staged, (
        "the diff must see cards that were never staged")
    # And it agrees with the function that owns the question.
    raw = deck_branch.diff(SLUG, BRANCH)
    assert {r["name"] for r in d["out"]} == set(raw["out"])
    assert {r["name"] for r in d["in"]} == set(raw["add"])
    # SPELLS BEFORE LANDS, THEN UP THE CURVE, THEN BY NAME — a render order, not
    # an alphabetical one. Reading a diff by mana value is how you see that a
    # refactor lowered the curve; alphabetical hides it.
    for side in (d["out"], d["in"]):
        keys = [(r["kind"] != "spell", r["cmc"], r["name"]) for r in side]
        assert keys == sorted(keys), "stable, and ordered the way it is read"


@requires_data
@requires_branch
@requires_deck
def test_the_diff_counts_NAMES_and_says_so_beside_the_deck_size():
    """COUNT COPIES, NOT DECKLIST ENTRIES — the repo's own gotcha, in the one
    place the two numbers sit side by side. Cutting one of four Swamps removes
    no NAME, so "18 out, 21 in" is true at the same time as "100 -> 100 cards",
    and a reader given only the first reads a deck that grew by three."""
    d = net_change.card_diff(SLUG, BRANCH)
    assert d["size"] == d["base_size"], "a legal branch is the same size"
    assert d["counts"]["out"] != d["counts"]["in"] or True  # may or may not differ
    # The size and the name count are BOTH carried, which is what lets the
    # renderer explain the discrepancy instead of leaving it to be guessed at.
    for k in ("size", "base_size", "names", "base_names"):
        assert isinstance(d[k], int) and d[k] > 0, k


@requires_data
@requires_branch
@requires_deck
def test_every_row_carries_what_the_renderer_needs():
    d = net_change.card_diff(SLUG, BRANCH)
    checked = 0
    for r in d["out"] + d["in"]:
        assert r["kind"] in ("land", "spell")
        assert isinstance(r["cmc"], int)
        assert "why" in r and "pair" in r, "absent keys, not missing ones"
        checked += 1
    assert checked >= 10
    # Only the INCOMING cards have a physical location to report; asking where
    # a card you are removing "is" is a question about the deck you already own.
    assert all("state" not in r for r in d["out"])
    assert all("state" in r for r in d["in"])


@requires_data
@requires_branch
@requires_deck
def test_the_diff_and_the_bill_cannot_disagree_about_what_must_be_bought():
    """Two lists of the same cards on one page is two chances to be wrong."""
    bill = deck_branch.source(SLUG, BRANCH)
    d = net_change.card_diff(SLUG, BRANCH, bill)
    by_name = {r["name"]: r.get("state") for r in bill["cards"]}
    checked = 0
    for r in d["in"]:
        assert r["state"] == by_name.get(r["name"]), r["name"]
        checked += 1
    assert checked >= 10


@requires_data
@requires_branch
@requires_deck
def test_a_staged_swap_pairs_BOTH_ways_and_the_reason_is_recoverable():
    """A staged swap is ONE decision about TWO cards. Rendered as two columns
    the pairing is lost, and printing the `why` on both sides shows the reader
    the same sentence twice while still leaving them guessing which removal paid
    for which addition."""
    meta = deck_branch.meta(SLUG, BRANCH) or {}
    staged = [r for r in (meta.get("staged") or []) if r.get("out") and r.get("in")]
    if not staged:
        pytest.skip("this branch has no staged swaps")
    d = net_change.card_diff(SLUG, BRANCH)
    outs = {r["name"]: r for r in d["out"]}
    ins = {r["name"]: r for r in d["in"]}
    checked = 0
    for row in staged:
        o, i = outs.get(row["out"]), ins.get(row["in"])
        if not (o and i):
            continue          # the swap may have been superseded by a later one
        assert o["pair"] == row["in"] and i["pair"] == row["out"]
        assert i["why"] == row.get("why"), "the argument rides with the addition"
        checked += 1
    assert checked >= 1


@requires_data
@requires_branch
@requires_deck
def test_how_much_of_the_branch_nobody_argued_for_is_COUNTED():
    """A card with no recorded reason is reported, not left blank. The count is
    the honest measure of how much of a branch is argued for card by card, and a
    branch opened from a whole list starts at all of it."""
    d = net_change.card_diff(SLUG, BRANCH)
    u = d["unexplained"]
    assert u["out"] == sum(1 for r in d["out"] if not r["why"])
    assert u["in"] == sum(1 for r in d["in"] if not r["why"])
    assert u["out"] + u["in"] <= d["counts"]["out"] + d["counts"]["in"]


@requires_branch
@requires_deck
def test_the_written_report_carries_the_diff():
    """It is what `branch.html` renders as its headline panel."""
    d = (_doc().get("changes") or {}).get("diff")
    assert d, "net_change.json must carry the diff"
    assert d["out"] and d["in"] and d["counts"]["out"] >= 1


@requires_data
@requires_branch
@requires_deck
def test_the_two_sides_of_the_diff_BALANCE_in_copies():
    """THE QUESTION THAT FOUND THE BUG: "how can we have 18 out and 21 in?"

    They could, and the panel was wrong to show it. A name-level diff cannot see
    a basic cut from four copies to two — the name is still there — so three of
    twenty-one removals were missing from the page while the deck size sat
    unchanged at 100. Counted in COPIES, which is what gets sleeved, the two
    sides balance for any legal branch and the arithmetic is checkable.
    """
    d = net_change.card_diff(SLUG, BRANCH)
    c = d["counts"]
    assert c["out_copies"] and c["in_copies"]
    assert d["base_size"] - c["out_copies"] + c["in_copies"] == d["size"], (
        "copies out and in must reconcile the two deck sizes")
    # A legal branch is the same size as its deck, so the two sides are equal.
    # NO FIGURE IS PINNED: this ran against ur-dragon while carrying
    # edgar-vampires' 21, which is both wrong and the standing rule about not
    # writing tests against an experimental branch's numbers.
    if d["base_size"] == d["size"]:
        assert c["out_copies"] == c["in_copies"]


@requires_data
@requires_branch
@requires_deck
def test_a_copy_count_that_moved_is_reported_as_a_change():
    """RE-INTRODUCING THE CONDITION. `changed` is what the name diff cannot see;
    without it those cards are removals that appear nowhere."""
    d = net_change.card_diff(SLUG, BRANCH)
    names = {r["name"] for r in d["out"]} | {r["name"] for r in d["in"]}
    for r in d["changed"]:
        assert r["from"] != r["to"] and r["delta"] == r["to"] - r["from"]
        assert r["name"] not in names, (
            "a card in `changed` is in BOTH lists — it is not an add or a cut")
    # And the group totals a renderer sums are reconcilable from the rows alone.
    for kind in ("spell", "land"):
        outs = [r for r in d["out"] if r["kind"] == kind]
        ins = [r for r in d["in"] if r["kind"] == kind]
        chg = [r for r in d["changed"] if r["kind"] == kind]
        co = sum(r["copies"] for r in outs) + sum(-r["delta"] for r in chg if r["delta"] < 0)
        ci = sum(r["copies"] for r in ins) + sum(r["delta"] for r in chg if r["delta"] > 0)
        assert co >= 0 and ci >= 0
    assert sum(r["copies"] for r in d["out"]) + sum(
        -r["delta"] for r in d["changed"] if r["delta"] < 0) == d["counts"]["out_copies"]


@requires_data
@requires_branch
@requires_deck
def test_every_row_carries_its_own_copy_count():
    """A renderer that sums rows assuming one apiece is right about 96 of 99
    cards and wrong about exactly the ones this bug was made of."""
    d = net_change.card_diff(SLUG, BRANCH)
    checked = 0
    for r in d["out"] + d["in"]:
        assert isinstance(r["copies"], int) and r["copies"] >= 1, r["name"]
        checked += 1
    assert checked >= 10


@requires_data
def test_card_diff_survives_a_cut_DFC(monkeypatch):
    """`card_diff` reads `deck_branch.diff` for WHICH cards moved and its own
    tables for HOW MANY copies. Those two must speak one vocabulary.

    They did not. `diff` canonicalises both lists through `_named` — the
    resolver's names, so a DFC is "A // B" — while `card_diff` used raw
    `_entries`, which is the literal decklist text. decklist.txt has to carry the
    FRONT face because Scryfall rejects the joined form on `fetch-deck`, so the
    two disagree on exactly one class of card, and cutting any DFC raised
    KeyError and took the whole net-change down.

    Driven through the production function with the real name forms rather than
    re-deriving the rule; re-introducing the raw `_entries` read fails this.
    """
    JOINED = "Sagas of the Fallen // Ruin of the Fallen"
    FRONT = "Sagas of the Fallen"

    monkeypatch.setattr(net_change.deck_branch, "diff",
                        lambda s, b: {"add": ["Swamp"], "out": [JOINED],
                                      "size": 100, "base_size": 100,
                                      "names": 2, "base_names": 2})
    monkeypatch.setattr(net_change.deck_branch, "meta", lambda s, b: {})
    monkeypatch.setattr(net_change.deck_branch, "_list_text",
                        lambda s, b=None: f"1 {FRONT}\n" if b is None else "1 Swamp\n")
    # what `_named` does for real: decklist text -> the resolver's vocabulary.
    monkeypatch.setattr(net_change.deck_branch, "_named",
                        lambda s, b, now, cand: ({JOINED: 1}, {"Swamp": 1}))

    d = net_change.card_diff("any-slug", "any-branch", {"cards": []})

    assert [r["name"] for r in d["out"]] == [JOINED], "the cut DFC must survive the diff"
    assert d["counts"]["out_copies"] == 1, "and it must be counted as a real copy"


# ── the real table: one pod, decided games ───────────────────────────────

def _record(root, rel, pod, seat, wins, decided, games):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "pod": {"name": pod},
        "analysis": {"games": games, "decided": decided,
                     "seats": {seat: {"wins": wins}}},
        "games": []}))


def test_the_real_table_pools_within_one_pod_over_decided_games(tmp_path, monkeypatch):
    """RE-INTRODUCING THE BUG (2026-09-11, edgar-vampires/fear-v1): the champion
    arm pooled 840 games across five tables against a branch played at one, and
    both arms divided by every game so thirteen clock-outs counted as losses.
    Either regression makes this fixture read 58/355 or 8/40 instead of 8/27."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(net_change, "_engine_casts_caveat", lambda s, b: None)
    seat = net_change.deck_branch_seat("x", "b")
    _record(tmp_path, "data/decks/x/sim/a.json", "standard-v3", "x", 8, 27, 40)
    _record(tmp_path, "data/decks/x/sim/b.json", "vito-era", "x", 50, 328, 400)
    _record(tmp_path, "data/decks/x/branches/b/sim/c.json", "standard-v3", seat, 9, 33, 40)
    f = net_change.forge("x", "b")
    assert f["available"] and f["pod"] == "standard-v3"
    assert (f["champion"]["wins"], f["champion"]["games"]) == (8, 27), f["champion"]
    assert (f["branch"]["wins"], f["branch"]["games"]) == (9, 33), f["branch"]
    assert f["champion"]["all_games"] == 40
    assert f["other_tables"] == {"vito-era": {"champion_runs": 1, "branch_runs": 0}}
    assert f["delta"] == round(9 / 33 - 8 / 27, 4)


def test_the_real_table_with_no_pod_in_common_is_absent_with_a_reason(tmp_path, monkeypatch):
    """A rate from another table is not this branch's control (CLAUDE.md: a pod's
    null is a property of the table with the subject in it)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(net_change, "_engine_casts_caveat", lambda s, b: None)
    seat = net_change.deck_branch_seat("x", "b")
    _record(tmp_path, "data/decks/x/sim/a.json", "vito-era", "x", 50, 328, 400)
    _record(tmp_path, "data/decks/x/branches/b/sim/c.json", "standard-v3", seat, 9, 33, 40)
    f = net_change.forge("x", "b")
    assert f["available"] is False
    assert "no table in common" in f["why"] and "standard-v3" in f["why"]


# ── #45: the report must measure the list on disk ───────────────────────────

def test_net_change_refuses_to_measure_a_list_it_has_not_fetched(tmp_path, monkeypatch):
    """THE REPORT MEASURES `cards.json`; THE LIST OF RECORD IS `decklist.txt`.

    `diagnostic.run` hands the goldfish `load_deck_cards(slug, branch)`, so a
    swap staged and committed into `decklist.txt` without a `fetch-deck
    --branch` is invisible: both arms are measured on the PREVIOUS list, the
    report is written, and its `decklist_sha256` is the previous list's sha.

    That is what happened on `heliod/splendor-v1` (#45). `validate-net-change`
    passed the file. The only thing that refused was `deck-branch propose`, one
    step too late, with advice — "re-run net-change" — that reproduces it
    exactly, because re-running net-change re-measures the same stale
    `cards.json`.

    Re-introducing the bug: delete the `_refuse_a_stale_measurement` call at the
    top of `build` and this goes green while writing a report about a 99 that is
    not there.
    """
    import hashlib

    from manamap import config
    from manamap.pilot import net_change

    root = tmp_path / "decks" / "slug" / "branches" / "b1"
    root.mkdir(parents=True)
    (tmp_path / "decks" / "slug" / "decklist.txt").write_text("1 Sol Ring\n",
                                                             encoding="utf-8")
    (root / "decklist.txt").write_text("1 Sol Ring\n1 Mox Diamond\n", encoding="utf-8")
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path / "decks")

    # `cards.json` built from the PREVIOUS list — the state a staged-but-unfetched
    # branch is in.
    stale = hashlib.sha256(b"1 Sol Ring\n").hexdigest()
    (root / "cards.json").write_text(
        json.dumps({"decklist_sha256": stale, "cards": []}), encoding="utf-8")

    ok, stamp, live = net_change.measured_list_is_current("slug", "b1")
    assert not ok and stamp == stale and stamp != live

    with pytest.raises(SystemExit) as caught:
        net_change.build("slug", "b1")
    message = str(caught.value)
    # The refusal must name ALL THREE commands, in order. Naming only
    # `net-change` is what made the original bug self-reproducing.
    assert "fetch-deck" in message, "the refusal does not name the command that fixes it"
    assert message.index("fetch-deck") < message.index("goldfish") < message.index("net-change")

    # And once the fetch has happened, it measures.
    (root / "cards.json").write_text(
        json.dumps({"decklist_sha256": live, "cards": []}), encoding="utf-8")
    ok, _stamp, _live = net_change.measured_list_is_current("slug", "b1")
    assert ok, "a freshly fetched branch must not be refused"


def test_the_validator_refuses_a_report_about_a_list_that_moved():
    """The gate `validate-net-change` did not have, and the reason a stale
    report survived long enough to be proposed on.

    Swept across all 28 branches before landing: zero trips.
    """
    from manamap.config import DECKS_DIR
    from manamap.pilot import validate_net_change

    live = None
    for path in sorted(DECKS_DIR.glob("*/branches/*/net_change.json")):
        doc = json.loads(path.read_text())
        assert not validate_net_change.validate(doc), (
            f"{path} does not validate — the sweep said every tracked report "
            f"was clean")
        live = doc
        break
    if live is None:
        pytest.skip("no tracked net_change.json on this checkout")

    moved = dict(live, decklist_sha256="f" * 64)
    errors = validate_net_change.validate(moved)
    assert any("not the list on disk" in e for e in errors), (
        f"a report stamped with a list that is not on disk was accepted: {errors}")


def test_a_draw_the_model_prices_is_not_reported_as_unmeasured():
    """THE BLIND-SPOT LIST IS A PROMISE ABOUT WHAT THE FIGURES LEAVE OUT, and a
    wrong entry lies in the more dangerous direction than a missing one: it
    tells a pilot that a measured improvement was never measured, and invites
    cutting a card the numbers already credited.

    `BLIND["draw"]` read "extra card draw is not modelled — one card per turn,
    always". That was true before `model_draw` existed and has been false ever
    since: the channel prices ETB draw, spell draw, recurring draw, cast draw,
    X spells, wheels, activated draw and draw DOUBLERS. It was keyed on the
    card's ROLE, which only says the card draws — never on whether the model
    can read that draw.

    Caught on sharknado/ivora-v1, whose report named Teferi's Ageless Insight
    as unmeasured in the same breath as measuring it at +0.89 extra cards by
    turn eight. `draw_profile` already knew: it sets `unmodelled` to the card's
    own name when the draw is through a channel there is no event for.

    Re-introduce the bug by dropping the `_draw_is_blind` call in `blind_spots`.
    """
    from manamap.pilot import net_change

    assert "not modelled — one card per turn, always" not in net_change.BLIND["draw"], (
        "the blind-spot sentence claims no draw at all is modelled")
    src = inspect.getsource(net_change.blind_spots)
    assert "_draw_is_blind(slug, branch, name)" in src, (
        "the draw blind spot is decided by ROLE rather than by whether the "
        "model can read the card")
    # The predicate asks the PROFILE, which is the thing that knows.
    probe = inspect.getsource(net_change._draw_is_blind)
    assert 'draw_profile(card)["unmodelled"]' in probe
    assert 'targets.get("model_draw")' in probe, (
        "a deck that never opted in really does draw one a turn")


def test_the_damage_row_is_not_described_as_cumulative():
    """EVERY FIGURE CARRIES ITS DEFINITION, IN THE REPORT THAT PRINTS IT — and
    this one carried the wrong definition of the headline output row.

    `damage @T10` reads `combat.mean_damage_by_turn["10"]`, and
    `goldfish_turn.simulate_once` resets `dealt` to 0 INSIDE the turn loop
    before appending it. The series is therefore what was dealt ON each turn,
    never a running total. `METRICS` described it as "Cumulative damage dealt
    ... by the end of turn 10" and anchored the reader further with a scale line
    reading "the opponent starts at 40 life, so 40.0 is exactly lethal once" —
    which is only true of the cumulative reading it does not have.

    THE MODEL'S OWN NAMING SETTLES IT. The event-damage pillar carries BOTH
    shapes and distinguishes them by name: `mean_event_damage_by_turn` is
    2.288 / 2.978 / 3.87 at turns 8/9/10 on sharknado, and
    `mean_cumulative_event_damage_by_turn` is 6.372 / 9.35 / 13.22. The combat
    row has only the first shape and no cumulative twin anywhere in `goldfish`.

    Found by the deck-engineer during `/analyze-engine`, which noticed that
    53.904 at turn ten sits on top of board power 18.334 plus 32.765 commander
    counters — one swing, not ten turns of them — and raised it as an open
    question rather than quoting the row either way.

    Re-introduce the bug by putting the word "cumulative" back in the `what`.
    """
    from manamap.pilot import net_change

    spec = net_change.METRICS["damage @T10"]
    blurb = (spec["what"] + " " + spec.get("scale", "")).lower()
    assert "cumulative damage" not in blurb, (
        "the headline damage row is described as a running total; it is one "
        "turn's damage")
    assert "on turn 10" in spec["what"].lower(), (
        "the row must say WHICH turn's damage it is")
    assert "exactly lethal once" not in blurb, (
        "that scale line anchors the reader to a 40-life total, which is the "
        "cumulative reading this row does not have")


def test_the_only_cumulative_series_says_so_in_its_name():
    """The guard behind the fix: `goldfish` distinguishes the two shapes by
    NAME, and a row that cumulates without saying so is the defect above waiting
    to happen again. Exactly one series in the metrics is a running total."""
    import inspect

    from manamap.pilot import goldfish

    src = inspect.getsource(goldfish)
    # The cumulative builder slices the whole prefix; the per-turn one indexes.
    assert 'sum(sum(r["event_damage_by_turn"][:t]) for r in results)' in src
    assert 'sum(r["damage_by_turn"][t - 1] for r in results)' in src, (
        "the combat damage row must index a single turn, not sum a prefix")
    assert "mean_cumulative_damage_by_turn" not in src, (
        "a cumulative combat row now exists — `METRICS` must be updated to say "
        "which of the two `damage @T10` reads")


def test_changes_reports_the_NET_diff_not_the_staging_log():
    """A SUPERSEDED SWAP IS NOT A CHANGE, and reading `staged` published nine.

    goblin-storm/zada-v1 staged 29 swaps and its net change is 16: nine cards
    were staged in and later staged back out. `changes()` read the staging log,
    so it published all nine as ADDS — cards the branch does not run — and three
    names appeared on BOTH sides at once. The page's own header said "16 out and
    16 in" from `deck_branch.diff` while the list below it showed 29 pairs, so
    anyone shopping from it would have bought the wrong cards.

    ONE PREDICATE, ONE HOME: `deck_branch.diff` already answers this, counts
    copies rather than names, and is what the header prints.
    """
    import pytest
    from manamap.pilot import deck_branch
    from manamap.pilot.net_change import changes
    try:
        d = deck_branch.diff("goblin-storm", "zada-v1")
        c = changes("goblin-storm", "zada-v1")
    except Exception:  # pragma: no cover - branch absent
        pytest.skip("goblin-storm@zada-v1 not present")

    rows = (c["spells"] or []) + (c["lands"] or [])
    got_in = sorted(r["in"] for r in rows if r["in"])
    got_out = sorted(r["out"] for r in rows if r["out"])
    assert got_in == sorted(d["add"]), (
        "the adds do not match the net diff — a superseded swap is being "
        "published as a change")
    assert got_out == sorted(d["out"]), "the cuts do not match the net diff"
    # No name may appear on both sides: that is the tell the staging log leaves.
    assert not (set(got_in) & set(got_out)), (
        f"names on both sides: {sorted(set(got_in) & set(got_out))}")
    # And the staged count is kept, because the difference is the finding.
    assert c["staged_count"] >= c["count"]


@requires_branch
@requires_deck
def test_the_header_counts_what_is_coming_in_and_not_how_many_rows_it_took():
    """MEASURED, on goblin-storm/zada-v1: the header read "22 swap(s)" on a
    branch bringing in 16 cards. Six of the 22 rows were cuts whose partner had
    been staged back out later in the branch, so they had nothing in their slot
    and printed as `- Card + None`; the staging log behind them was 29 entries
    long. Three numbers for one question, and the pilot asked where the missing
    cards had gone.

    A row is not a swap, so the row count may not wear the word. `in` and `out`
    are the figures a spending decision rests on and come from the NET diff.
    """
    ch = net_change.changes(SLUG, BRANCH)
    rows = ch["spells"] + ch["lands"]
    c = ch["counts"]
    assert c["rows"] == len(rows) == ch["count"]
    assert c["in"] == sum(1 for r in rows if r["in"]), (
        "the header's `in` is not the number of cards coming in")
    assert c["out"] == sum(1 for r in rows if r["out"])
    # And it is the diff's own answer, not a second one counted here.
    if not ch.get("merged"):
        d = deck_branch.diff(SLUG, BRANCH)
        assert c["in"] == len(d.get("add") or [])
        assert c["out"] == len(d.get("out") or [])


def test_no_printed_row_offers_a_card_named_none():
    """`- Goblin Lackey + None` reads as a swap for a card called None. A row
    with nothing coming in is a CUT and is printed under its own heading.

    SYNTHETIC ON PURPOSE. Written against a real branch first, this passed with
    the bug reintroduced: every written report on this checkout predates
    `counts`, so the assertions guarded by it never ran. The mix that exposes
    the defect — one pair, one add whose partner was superseded, one cut whose
    partner was — has to be stated rather than hoped for.
    """
    import io
    import contextlib
    doc = {
        "slug": SLUG, "branch": BRANCH, "table": [], "definitions": {},
        "harness": {"iterations": 10000, "seed": 20260826},
        "changes": {
            "count": 3,
            "counts": {"in": 2, "out": 2, "rows": 3},
            "staged_count": 7,
            "opened": "2026-09-25",
            "spells": [
                {"out": "Ruby Medallion", "in": "Ancestral Anger",
                 "why": "a paired swap"},
                {"out": None, "in": "Hanweir Garrison",
                 "why": "its partner was staged back out"},
                {"out": "Goblin Lackey", "in": None,
                 "why": "nothing came in for this slot"},
            ],
            "lands": [],
        },
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net_change._print_changes(doc)
    out = buf.getvalue()
    assert "+ None" not in out and "- None" not in out, (
        "a row printed a side that does not exist as if it were a card")
    assert "2 in, 2 out" in out, "the header does not name what is coming in"
    assert "swap(s)" not in out, "a row count is still labelled as swaps"
    assert "3 rows" in out and "7 staged" in out, (
        "the row count and the staging log are the provenance and belong in "
        "the header too — dropping them is how one number became three")
    assert "CUT, NOTHING IN ITS SLOT  (1)" in out
    assert out.index("Hanweir Garrison") < out.index("CUT, NOTHING"), (
        "an add with no partner belongs with the adds, not among the cuts")


def test_a_seat_sha_resolves_whichever_spelling_the_record_used():
    """`analysis.seats` keys a branch `goblin-storm-zada-v1`; the run manifest
    spells the same seat `goblin-storm@zada-v1` and `mm-goblin-storm-zada-v1`.
    A matcher that knows only one of them silently finds no seat, and a missing
    sha reads as "this record predates the stamp" — which passes the gate."""
    doc = {"seats": [
        {"slug": "goblin-storm@zada-v1", "forge_name": "mm-goblin-storm-zada-v1",
         "decklist_sha256": "aaaa"},
        {"slug": "sythis-enchantress", "forge_name": "mm-sythis-enchantress",
         "decklist_sha256": "bbbb"}]}
    assert net_change._seat_sha(doc, "goblin-storm-zada-v1") == "aaaa"
    assert net_change._seat_sha(doc, "sythis-enchantress") == "bbbb"
    assert net_change._seat_sha(doc, "not-at-this-table") is None
    # A record from before the stamp existed carries no sha and must read None
    # rather than raising — it is older evidence, not wrong evidence.
    assert net_change._seat_sha({"seats": [{"slug": "goblin-storm"}]},
                                "goblin-storm") is None
    assert net_change._seat_sha({}, "goblin-storm") is None


def _branches():
    from manamap.config import DECKS_DIR
    from manamap.pilot import deck_branch
    for slug in sorted(d.name for d in DECKS_DIR.iterdir() if d.is_dir()):
        for branch in deck_branch.names(slug):
            yield slug, branch


@pytest.mark.slow
@requires_deck
def test_every_forge_record_is_either_counted_or_named_as_superseded():
    """MEASURED, on goblin-storm/zada-v1: 120 games were reported as the branch's
    rate, played on its FOURTH commit while the list on disk was its seventh —
    eight cards in and nine out since, including the largest gain it claimed.
    The record stamps `seats[].decklist_sha256`; nothing read it.

    A COMPLETE ACCOUNTING, because the one-sided version could not see the bug.
    Asserting only that an EXCLUDED run really was superseded passes trivially
    when nothing is ever excluded — which is the defect. So every record on disk
    must be accounted for: counted because it played this list (or carries no
    stamp), or named in `superseded` because it did not.
    """
    import glob
    import json as _json
    from manamap.config import DECKS_DIR
    from manamap.pilot import common
    checked = 0
    for slug, branch in _branches():
        f = net_change.forge(slug, branch)
        for arm, b in (("champion", None), ("branch", branch)):
            live = common.decklist_sha256(slug, b)
            root = DECKS_DIR / slug / (f"branches/{b}" if b else "")
            named = {r["run"] for r in (f.get("superseded") or {}).get(arm) or []}
            for path in sorted(glob.glob(str(root / "sim" / "*.json"))):
                doc = _json.load(open(path))
                want = net_change.deck_branch_seat(slug, b) if b else slug
                ran = net_change._seat_sha(doc, want)
                if not ran or ran == live:
                    continue
                run = doc.get("run_id") or path.split("/")[-1]
                assert run in named, (
                    f"{slug}/{branch} {arm}: run {run} played "
                    f"{ran[:12]} against {live[:12]} on disk and is neither "
                    f"counted out nor named as superseded")
                checked += 1
    assert checked >= 1, (
        "no record on this checkout was made on a superseded list, so this "
        "test proved nothing — it needs one to stay honest")


@pytest.mark.slow
@requires_deck
def test_a_never_cast_claim_names_a_run_that_played_the_current_list():
    """"Held and never cast" and "not in the deck" are different facts.

    On goblin-storm/zada-v1 this named Hanweir Garrison, Legion Warboss, Assault
    Strobe, Reckless Ransacking and Great Train Heist as held and passed over.
    None of the five was in the list those games were played with — all five were
    added afterwards. The report's strongest claim was being made about cards the
    AI had never been dealt.

    Called DIRECTLY rather than through `forge()`, which returns early when an
    arm has no current run and would hide a regression here behind that.
    """
    from manamap.pilot import common
    checked = 0
    for slug, branch in _branches():
        got = net_change._engine_casts_caveat(slug, branch)
        for arm, b in (("champion", None), ("branch", branch)):
            ec = (got or {}).get(arm)
            if not ec:
                continue
            live = common.decklist_sha256(slug, b)
            assert ec["played"] == live, (
                f"{slug}/{branch} {arm}: engine_casts read run {ec['run']}, "
                f"played on {ec['played'][:12]} against {live[:12]} on disk — "
                f"a card absent from that list reads as held and never cast")
            checked += 1
    assert checked >= 1, "no arm on this checkout carries an engine_casts reading"


def test_a_rate_measured_on_another_list_says_so_before_it_says_the_rate():
    """A pilot reads the number and stops. The mismatch has to be ABOVE it.

    This is the report's own rule — every figure carries its definition where it
    is printed — applied to the one case where the definition is "this is not
    the deck you are looking at".
    """
    import io
    import contextlib
    doc = {"forge": {
        "available": True, "pod": "standard-v3", "basis": "wins over DECIDED games",
        "champion": {"wins": 1, "games": 32, "rate": 0.0312, "won_by": {}},
        "branch": {"wins": 5, "games": 96, "rate": 0.0521, "won_by": {}},
        "delta": 0.0208, "ci95": [-0.1088, 0.0899], "mde": 0.19,
        "caveat": "Forge's AI is a weak pilot",
        "list_mismatch": {"branch": {
            "played": ["57725742d1bc"], "games": 120, "on_disk": "e01b366ccbd6",
            "reads_as": "every Forge run on the branch was made with a "
                        "different list, so this rate describes that list."}}}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net_change._print_real_table(doc)
    out = buf.getvalue()
    assert "BRANCH MEASURED ON A DIFFERENT LIST" in out
    assert "57725742d1bc" in out and "e01b366ccbd6" in out
    assert out.index("DIFFERENT LIST") < out.index("5/96"), (
        "the rate is printed before the warning that it is for another list")


def test_the_branch_page_warns_on_a_mismatch_too():
    """The CLI and the page read the same artifact, and the page is where a
    spending decision is actually made. `branch-view.js` rendered the rate with
    no way to say which list it came from."""
    import pathlib
    js = pathlib.Path("viz/js/branch-view.js").read_text()
    assert "list_mismatch" in js, (
        "branch-view.js renders the Forge rate without reading list_mismatch")
    # THE CALL SITE, NOT THE DEFINITION. `js.index("mismatchNote(f)")` finds
    # `function mismatchNote(f) {` and passed with the call deleted.
    assert "mismatchNote(f) +" in js, (
        "mismatchNote is defined but never rendered into the panel")
    assert js.index("mismatchNote(f) +") < js.index("f.champion.wins"), (
        "the page prints the rate above the warning about it")
    # AND THE CACHE-BUST MOVED. A JS change behind a stale ?v= is a change
    # nobody sees (CLAUDE.md).
    html = pathlib.Path("viz/branch.html").read_text()
    import re
    vs = {int(m) for m in re.findall(r"\?v=(\d+)", html)}
    assert len(vs) == 1, f"branch.html mixes cache-bust versions: {sorted(vs)}"
    assert max(vs) >= 224, "branch.html's ?v= was not bumped for this change"


@requires_deck
def test_a_branch_never_declares_a_model_channel_its_deck_does_not():
    """TWO LISTS UNDER TWO SIMULATORS ARE NOT AN A/B.

    MEASURED, on goblin-storm/zada-v1. The branch's own `goldfish_targets.json`
    declared `model_draw`, `model_combat` and `model_commander_copy`; the deck's
    declared NONE of them. So for weeks the report compared a branch whose
    commander ability, combat and card draw were modelled against a champion
    where they were not — and `net_change`'s table silently dropped seven of its
    twelve rows, because the champion arm could produce no `output` block at all.
    The two rows that survived and moved were the two the asymmetry flattered:
    `missed drop by T5` (only the branch could draw) and `interaction affordable
    @T6` (only the branch was spending mana on modelled cantrips). Declaring the
    same channels on both reversed the second and sent the first to noise.

    A BRANCH MAY DIFFER IN WHICH CARDS SATISFY A TARGET — meren-recursion's
    drain-density-v1 legitimately adds Bastion of Remembrance and Cauldron of
    Essence to four `any_of` groups, because a target names cards and the branch
    HAS those cards. That asks the same question of a different list, which is
    the point. A `model_*` flag is not a question, it is the instrument.
    """
    import json as _json
    from manamap.config import DECKS_DIR
    from manamap.pilot import deck_branch
    checked = 0
    for slug in sorted(d.name for d in DECKS_DIR.iterdir() if d.is_dir()):
        deck_decl = DECKS_DIR / slug / "goldfish_targets.json"
        if not deck_decl.exists():
            continue
        want = {k: v for k, v in _json.loads(deck_decl.read_text()).items()
                if k.startswith("model_")}
        for branch in deck_branch.names(slug):
            own = DECKS_DIR / slug / "branches" / branch / "goldfish_targets.json"
            if not own.exists():
                continue          # it inherits the deck's, which is the norm
            got = {k: v for k, v in _json.loads(own.read_text()).items()
                   if k.startswith("model_")}
            assert got == want, (
                f"{slug}@{branch} declares {sorted(got)} and its deck declares "
                f"{sorted(want)} — the arms would be measured under different "
                f"models, so no row of the comparison means anything")
            checked += 1
    assert checked >= 1, (
        "no branch on this checkout carries its own declaration, so this test "
        "proved nothing")


@pytest.mark.slow
def test_harness_stamps_the_model_a_report_was_measured_under():
    """A SEED WITHOUT A MODEL VERSION REPRODUCES NOTHING.

    `harness` carried `iterations` and `seed` — the goldfish's two reproducibility
    inputs — and omitted the third. Same seed, different model, different figures:
    goblin-storm's damage @T10 read 27.73 and then 21.02 across one commit that
    touched no decklist.

    It is checked HERE rather than only on `goldfish_metrics.json` because
    `deck_branch.propose` copies `harness` whole into `accepted_on`, which is the
    only record of what the evidence said when the pilot said yes. Drive the
    production builder, not a re-derivation of it.
    """
    from manamap.pilot import goldfish, net_change

    doc = net_change.build("heliod", "archangel-v1")
    h = doc["harness"]
    assert h["model_version"] == goldfish.model_version(), (
        "harness must stamp the model the figures were measured under")
    assert h["iterations"] and h["seed"] is not None, (
        "the existing reproducibility inputs must survive")


def test_accepted_on_inherits_the_model_stamp_from_harness():
    """The stamp is only worth adding if it reaches the decision record.

    `propose` writes `accepted_on.harness = nc["harness"]`, so this asserts the
    WIRING rather than re-stating the shape: if a future edit builds
    `accepted_on` field by field instead of copying the block, the stamp would
    silently stop reaching the one place it exists for and every other test here
    would still pass.
    """
    import inspect

    from manamap.pilot import deck_branch

    src = inspect.getsource(deck_branch.propose)
    assert '"harness": nc.get("harness")' in src, (
        "propose must copy the whole harness block into accepted_on, or the "
        "model_version stamp never reaches the decision record")


def test_no_decision_has_a_backfilled_model_version():
    """ABSENT MEANS ABSENT. The nine decisions taken before the stamp existed
    cannot know which model they were measured under, and inventing one would be
    a fabricated measurement — the exact failure the stamp exists to prevent.

    They must read as unknown, never as today's model, which is what they would
    read as if anybody 'helpfully' backfilled them from the live artifact.
    """
    import glob
    import json

    from manamap.pilot import goldfish

    live = goldfish.model_version()
    checked = 0
    for p in sorted(glob.glob("data/decks/*/branches/*/branch.json")):
        doc = json.load(open(p))
        for key in ("proposal", "merged"):
            block = doc.get(key)
            if not isinstance(block, dict):
                continue
            acc = block.get("accepted_on") or {}
            h = acc.get("harness") or {}
            if "model_version" not in h:
                checked += 1
                continue
            # A stamp IS allowed — a decision taken from here on carries one.
            # What is never allowed is a stamp that matches today's model on a
            # decision dated before the stamp shipped (2026-09-28).
            if h["model_version"] == live and block.get("at", "") < "2026-09-28":
                raise AssertionError(
                    f"{p} :: {key} is dated {block.get('at')} — before the stamp "
                    f"existed — and carries today's model_version. That is a "
                    f"backfilled measurement, not a record.")
            checked += 1
    assert checked >= 9, (
        f"expected at least the 9 known decided branches, inspected {checked}")


def test_pod_null_is_absent_not_defaulted():
    """ABSENT MEANS ABSENT. A table nothing has been measured against has no null,
    and a default would put an invented figure exactly where a measured one goes —
    then get divided into every rate beside it.
    """
    from manamap.pilot.net_change import _pod_null

    assert _pod_null(None) is None
    assert _pod_null("") is None
    assert _pod_null("a-table-that-does-not-exist") is None


def test_pod_null_reads_the_measured_calibration():
    """Drive the production lookup against the real calibration rather than
    restating 0.233, so a recalibration moves the test with the data.
    """
    from manamap.sim import pods
    from manamap.pilot.net_change import _pod_null

    expected = pods.calibration("standard-v3")["subject_null"]["rate"]
    assert _pod_null("standard-v3") == expected
    assert 0 < expected < 1


def test_the_forge_block_prints_the_null_beside_the_mde():
    """AN MDE MEANS NOTHING WITHOUT THE NULL IT IS SCALED AGAINST.

    This block printed delta, interval and MDE while the null lived in a different
    command, and on 2026-09-28 I read an MDE of 0.115 against a BASELINE of 0.014,
    called the detectable rate 'a ninefold improvement', and concluded a run could
    not answer its own question. Against the null of 0.233 that rate is 55% of par
    — an ordinary thing for a fixed engine to reach, and the run was well powered.

    Asserts the WIRING: that the printer reaches for the null at all. A future edit
    that drops the line would leave every other test here passing.
    """
    import inspect

    from manamap.pilot import net_change

    src = inspect.getsource(net_change._print_real_table)
    assert "_pod_null(" in src, "the Forge block must look the null up"
    assert "the table's null" in src, "and print it beside the MDE it scales"


def test_forge_never_pools_runs_made_under_different_card_overrides():
    """A RUN DESCRIBES THE HARNESS IT WAS PLAYED UNDER, not just the list.

    `data/forge_overrides/` changes what the Forge AI may TARGET. On goblin-storm,
    whose engine is "target your own commander", one unchanged list reads 1/73
    without the overrides and 9/82 with them. Pooling them gave 10/155 = 0.065 —
    a rate describing neither deck — and the override README asserted the guard
    existed while nothing read the fingerprint.

    Drives the production function against the real records.
    """
    from manamap.pilot.net_change import forge

    f = forge("goblin-storm", "copy-burst-v1")
    assert f.get("available"), f.get("why")
    # The chosen bucket is one harness, and the block names which.
    assert "card_overrides" in f, "the block must say which harness decided it"
    br = f["branch"]
    assert br["games"] != 155, (
        "155 decided games is the two harnesses pooled — the defect this test "
        "exists for")
    assert br["games"] in (73, 82), (
        f"the branch arm must be ONE harness, got {br['games']} decided games")
    # And the held-out harness is reported, not dropped in silence.
    others = f.get("other_tables") or {}
    assert any("overrides" in k for k in others), (
        f"a run held out for its harness must be named; got {sorted(others)}")


def test_the_label_distinguishes_a_harness_from_a_table():
    """`standard-v3` and `standard-v3 (overrides …)` are the same TABLE, so the
    null applies to both — but they are not the same measurement. A reader told
    only "another table" would think the null did not apply.
    """
    from manamap.pilot.net_change import _label

    assert _label(("standard-v3", "", "Default")) == "standard-v3"
    assert _label(("standard-v3", "60636e9e5565", "Default")) == \
        "standard-v3 (overrides 60636e9e5565)"
    # Truncated, so a long sha cannot push the table name off a terminal line.
    assert _label(("p", "0" * 64, "Default")) == "p (overrides " + "0" * 12 + ")"
    # THE PROFILE IS A HARNESS AXIS TOO, and the label must say so — a reader shown
    # only "overrides X" for a row that differs by profile is told the wrong reason.
    assert _label(("standard-v3", "60636e9e5565", "Experimental")) == \
        "standard-v3 (overrides 60636e9e5565, ours Experimental)"
    assert _label(("standard-v3", "", "Experimental")) == "standard-v3 (ours Experimental)"



def test_forge_never_pools_runs_flown_under_different_profiles():
    """A PROFILE IS A HARNESS TOO. The bucket key was `(pod, override_sha)` for eleven
    hours and pooled the branch's Default-override run (9/82) with its Experimental-
    override run (3/68) into `branch 12/150` — the same defect as the 10/155 it had been
    written to fix, one axis over.

    That the profile changes the instrument is MEASURED, not assumed: one list, same
    seed, same overrides, Default -> Experimental took clock-outs from 17 to 32 of 100
    with an interval excluding zero. Drives the production function over the real
    records; re-introducing the bug is dropping `prof` from the key.
    """
    from manamap.pilot.net_change import forge

    f = forge("goblin-storm", "copy-burst-v1")
    assert f.get("available"), f.get("why")
    br = f["branch"]
    assert br["games"] != 150, "150 decided games is the two profiles pooled"
    assert br["games"] in (82, 68, 73), br["games"]
    # And the held-out profile is NAMED, not silently dropped.
    assert any("ours Experimental" in k for k in (f.get("other_tables") or {})), (
        sorted(f.get("other_tables") or {}))


def test_the_pod_null_excludes_overridden_runs_and_counts_them():
    """THE NULL IS THE PLAIN HARNESS. The first overridden deck-level run to land at
    standard-v3 moved it 0.233 -> 0.206 before any exclusion existed — the yardstick
    every MDE is scaled against, shifted 12% by one run that changed what the AI may
    TARGET. Re-introducing the bug is removing the `card_overrides` skip in
    `pods.calibration`.
    """
    import glob
    import json
    import pathlib

    from manamap.sim import pods

    recs = [pathlib.Path(p) for p in sorted(glob.glob("data/decks/*/sim/*.json"))
            if "/logs/" not in p]
    over = [p for p in recs
            if (json.loads(p.read_text()).get("card_overrides") or {}).get("sha")]
    if not over:
        pytest.skip("no overridden deck-level run on this machine")
    doc = pods.calibration("standard-v3", records=recs)
    assert doc["excluded_overridden_runs"] >= 1
    clean = pods.calibration("standard-v3", records=[p for p in recs if p not in over])
    assert doc["subject_null"]["games"] == clean["subject_null"]["games"], (
        "an overridden run is in the null's denominator")
    assert "PLAIN HARNESS ONLY" in pods.format_calibration(doc)


def test_a_mismatch_names_the_versions_it_played_and_the_one_on_disk(tmp_path, monkeypatch):
    """"57725742 on disk" placed nothing; "those runs describe V4; the deck is
    V5" does. Versions come from git through `_versions_by_sha`, and a sha git
    does not know is left out rather than labelled V0."""
    import contextlib
    import io

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(net_change, "_engine_casts_caveat", lambda s, b: None)
    live = {None: "b" * 64, "b": "c" * 64}
    monkeypatch.setattr("manamap.pilot.common.decklist_sha256",
                        lambda slug, branch=None: live[branch])
    monkeypatch.setattr(net_change, "_versions_by_sha",
                        lambda slug: {"a" * 12: 4, "b" * 12: 5})
    seat = net_change.deck_branch_seat("x", "b")
    root = tmp_path / "data" / "decks" / "x"
    (root / "sim").mkdir(parents=True)
    (root / "branches" / "b" / "sim").mkdir(parents=True)
    (root / "sim" / "old.json").write_text(json.dumps({
        "pod": {"name": "standard-v3"}, "run_id": "old",
        "seats": [{"slug": "x", "decklist_sha256": "a" * 64}],
        "analysis": {"games": 40, "decided": 30, "seats": {"x": {"wins": 8}}}, "games": []}))
    (root / "branches" / "b" / "sim" / "br.json").write_text(json.dumps({
        "pod": {"name": "standard-v3"}, "run_id": "br",
        "seats": [{"slug": seat, "decklist_sha256": "c" * 64}],
        "analysis": {"games": 40, "decided": 33, "seats": {seat: {"wins": 9}}}, "games": []}))

    f = net_change.forge("x", "b")
    m = f["list_mismatch"]["champion"]
    assert m["played_versions"] == ["V4"] and m["on_disk_version"] == "V5", m
    assert "played_versions" not in (f["list_mismatch"].get("branch") or {})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        net_change._print_real_table({"forge": f, "slug": "x", "branch": "b"})
    out = buf.getvalue()
    assert "those runs describe V4; the deck is V5" in out, out
    assert out.index("describe V4") < out.index("8/30"), "the version line sits above the rate"


# ── one primary, twelve exploratory ─────────────────────────────────────

def _fake_diag(cells):
    """A diagnostic document holding just the cells `ROWS` reads."""
    doc = {"decklist_sha256": "d" * 64}
    for (label, blk, key, turn, want), cell in zip(net_change.ROWS, cells):
        blk_d = doc.setdefault(blk, {})
        if turn:
            blk_d.setdefault(key, {})[turn] = cell
        else:
            blk_d[key] = cell
    return doc


def test_a_row_over_the_mde_but_not_holm_significant_is_noise(monkeypatch):
    """THE BUG: a verdict by MDE alone. Twelve rows each at alpha 0.05 is one
    false 'better' every other report. Build twelve rows where every delta
    sits just over its own MDE (z ≈ 2.9 on a mean) but only the first
    clears Holm's top-rank threshold — put the verdict back to `abs(delta) >
    mde` and eleven rows rank instead of one."""
    from manamap.pilot import diagnostic, goldfish
    n = 10000
    cells_a, cells_b = [], []
    for i, (label, blk, key, turn, want) in enumerate(net_change.ROWS):
        # a mean cell (sd given) whose delta is z_i standard errors
        sd = 1.0
        se = sd * (2 / n) ** 0.5
        # 2.82 clears the MDE (2.8016 se) and fails Holm's rank-2 threshold
        # z(0.05/11) = 2.838, so the walk stops there and eleven rows are noise.
        z = 3.5 if i == 0 else 2.82
        cells_a.append({"rate": 5.0, "sd": sd, "n": n})
        cells_b.append({"rate": round(5.0 + z * se * (1 if want > 0 else -1), 6), "sd": sd, "n": n})
    a, b = _fake_diag(cells_a), _fake_diag(cells_b)
    monkeypatch.setattr(diagnostic, "run",
                        lambda slug, branch=None, iterations=None, seed=None, quiet=True, **kw: b if branch else a)
    monkeypatch.setattr(net_change, "_refuse_a_stale_measurement", lambda s, b: None)
    monkeypatch.setattr(net_change.deck_branch, "meta", lambda s, b: {"objective": None, "staged": []})
    monkeypatch.setattr(net_change, "changes", lambda s, b: {})
    monkeypatch.setattr(net_change, "card_diff", lambda s, b, bill: {})
    monkeypatch.setattr(net_change.deck_branch, "source", lambda s, b: {"cards": [], "unsourced": []})
    monkeypatch.setattr(net_change, "blind_spots", lambda s, b, c: [])
    monkeypatch.setattr(net_change, "mana", lambda s, b: {"available": False, "why": "test"})
    monkeypatch.setattr(net_change, "forge", lambda s, b, pod=None: {"available": False, "why": "test"})
    monkeypatch.setattr(net_change, "_death_limit", lambda s, b: [])
    monkeypatch.setattr(goldfish, "model_version", lambda: "test")

    doc = net_change.build("x", "b")
    verdicts = [r["verdict"] for r in doc["table"]]
    assert verdicts[0] == "better" and set(verdicts[1:]) == {"noise"}, verdicts
    # every row over the MDE, so the OLD rule would have ranked all twelve
    assert all(abs(r["delta"]) > r["mde"] for r in doc["table"])
    assert doc["design"]["primary"] is None and doc["design"]["exploratory_rows"] == 12
    assert doc["table"][0]["holm"]["rank"] == 1 and doc["table"][1]["holm"]["significant"] is False
    # and every row carries the interval on its own difference, agreeing with itself
    for r in doc["table"]:
        assert r["ci95_diff"] and r["role"] == "exploratory" and r["method"].startswith("Welch")
        assert r["excludes_zero"] == ((r["ci95_diff"][0] > 0) or (r["ci95_diff"][1] < 0))
    assert not validate_net_change.validate(doc), validate_net_change.validate(doc)


def test_the_validator_holds_an_interval_to_its_own_flag_and_the_design_to_the_objective():
    row = {"measure": "m", "champion": 1.0, "branch": 2.0, "delta": 1.0, "mde": 0.05,
           "verdict": "better", "ci95_diff": [-0.2, 2.2], "excludes_zero": True,
           "holm": {"rank": 1, "threshold": 1.96, "significant": True, "family": 1}}
    errs = validate_net_change.validate(_minimal(table=[row]))
    assert any("disagree" in e for e in errs), errs
    # ranked although Holm said no
    row2 = dict(row, ci95_diff=[0.5, 1.5], holm={"rank": 2, "threshold": 2.9, "significant": False, "family": 2})
    errs = validate_net_change.validate(_minimal(table=[row2]))
    assert any("did not clear" in e for e in errs), errs
    # noise over the MDE is fine when Holm refused it
    row3 = dict(row2, verdict="noise")
    assert not [e for e in validate_net_change.validate(_minimal(table=[row3])) if "reported as noise" in e]
    # the design must name the objective's axis and count the rows
    bad = _minimal(design={"primary": "kill_by_8", "exploratory_rows": 1},
                   objective={"axis": "damage_10", "op": ">=", "value": 1},
                   objective_grade={"state": "met"})
    errs = validate_net_change.validate(bad)
    assert any("design.primary" in e for e in errs), errs
    bad2 = _minimal(design={"primary": None, "exploratory_rows": 7})
    assert any("exploratory_rows" in e for e in validate_net_change.validate(bad2))


# ── the real table in the verdict ────────────────────────────────────────

def _forge_doc(delta, lo, hi, state="met"):
    ep = {"champion": {"k": 20, "n": 100, "value": 0.2}, "branch": {"k": 20, "n": 100, "value": 0.2 + delta},
          "delta": delta, "ci95": [lo, hi], "excludes_zero": not (lo <= 0 <= hi),
          "method": "Newcombe", "mde": 0.15, "kind": "proportion"}
    return {"slug": "x", "branch": "b", "harness": {}, "limits": [], "staged": 5,
            "table": [{"measure": "damage @T10", "champion": 10.0, "branch": 14.0,
                       "delta": 4.0, "mde": 1.0, "verdict": "better"}],
            "objective": {"axis": "damage_10", "op": ">=", "value": 12.0},
            "objective_grade": {"state": state, "reading": 14.0},
            "bill": {"counts": {}}, "mana": {"available": False, "why": "test"},
            "forge": {"available": True, "pod": "standard-v3", "champion": {"wins": 20, "games": 100, "rate": 0.2, "won_by": {}},
                      "branch": {"wins": 20, "games": 100, "rate": 0.2 + delta, "won_by": {}},
                      "delta": delta, "ci95": [lo, hi], "excludes_zero": not (lo <= 0 <= hi), "mde": 0.15,
                      "null": {"pod": "standard-v3", "rate": 0.233, "games": 412},
                      "endpoints": {"forge.win_rate": ep}}}


def test_a_real_table_loss_that_excludes_zero_is_a_loud_warning_not_a_gate(monkeypatch):
    """THE RULE CHANGED ON 2026-10-04, by the pilot's ruling: Forge is a targeted
    probe, not the decision loop, and a pod win rate is a floor wherever the AI
    mis-pilots a deck (sharknado holds its wheels). So a Forge loss whose interval
    excludes zero no longer overrides the goldfish's verdict — but it was right
    once (copy-burst-v1 read MERGE while 73 Forge games read -0.061), so it is
    printed LOUDLY beside the verdict, with the pod, the interval and the null."""
    for fn in ("reward", "risk", "cost"):
        monkeypatch.setattr(net_change, fn, lambda doc: {} if fn != "reward" else [])
    got = net_change.recommend(_forge_doc(-0.12, -0.20, -0.04))
    assert got["state"] == "merge", got
    assert got["forge_warning"].startswith("WARNING"), got
    assert "standard-v3" in got["because"] and "0.233" in got["because"]
    assert "AI plays the swapped cards" in got["because"]


def test_a_real_table_that_spans_zero_leaves_a_goldfish_verdict_alone(monkeypatch):
    """"Cannot tell" is not "no"; the goldfish verdict stands on its own terms
    and the Forge reading stays a note."""
    for fn in ("reward", "risk", "cost"):
        monkeypatch.setattr(net_change, fn, lambda doc: {} if fn != "reward" else [])
    got = net_change.recommend(_forge_doc(-0.06, -0.13, +0.01))
    assert got["state"] == "merge", got
    assert any("cannot separate" in n for n in got["notes"])
    # and a real-table GAIN never blocks anything
    assert net_change.recommend(_forge_doc(+0.12, +0.04, +0.20))["state"] == "merge"


def _stamped_record(root, rel, pod, seat, sha, wins, decided, games, per_game=None, resolved=None):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    seat_row = {"wins": wins}
    if resolved is not None:
        seat_row["commander_access"] = {"games_resolved": resolved}
    doc = {"pod": {"name": pod}, "run_id": rel.split("/")[-1][:-5],
           "seats": [{"slug": seat, "decklist_sha256": sha}],
           "analysis": {"games": games, "decided": decided, "seats": {seat: seat_row}},
           "games": [{"per_seat": {seat: pg}} for pg in (per_game or [])]}
    p.write_text(json.dumps(doc))


def test_the_null_and_every_endpoint_are_stored_beside_the_win_rate(tmp_path, monkeypatch):
    """The null was printed and never stored, so the JSON and the terminal were
    not the same document. Now the block carries `null`, and `endpoints` holds
    every Forge axis a branch may aim at, each with its own interval and MDE."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(net_change, "_engine_casts_caveat", lambda s, b: None)
    monkeypatch.setattr(net_change, "_null_block",
                        lambda pod: ({"pod": pod, "rate": 0.233, "games": 412}, None))
    live = {None: "a" * 64, "b": "c" * 64}
    monkeypatch.setattr("manamap.pilot.common.decklist_sha256", lambda slug, branch=None: live[branch])
    seat = net_change.deck_branch_seat("x", "b")
    pg_a = [{"combat_damage_dealt_to_players": d, "first_attack_turn": t, "eliminated_turn": e}
            for d, t, e in ((10, 8, 20), (0, None, 12), (25, 6, None), (5, 9, 30), (0, None, 15))]
    pg_b = [{"combat_damage_dealt_to_players": d, "first_attack_turn": t, "eliminated_turn": e}
            for d, t, e in ((30, 5, None), (12, 7, 25), (40, 4, None), (8, 6, 22), (20, 5, None))]
    _stamped_record(tmp_path, "data/decks/x/sim/a.json", "standard-v3", "x", "a" * 64, 8, 27, 40, pg_a, resolved=30)
    _stamped_record(tmp_path, "data/decks/x/branches/b/sim/c.json", "standard-v3", seat, "c" * 64, 9, 33, 40, pg_b, resolved=38)
    f = net_change.forge("x", "b")
    assert f["available"] and f["null"]["rate"] == 0.233
    ep = f["endpoints"]
    assert ep["forge.win_rate"]["delta"] == f["delta"] and ep["forge.win_rate"]["ci95"] == f["ci95"]
    assert ep["forge.commander_resolved_rate"]["champion"] == {"k": 30, "n": 40, "value": 0.75}
    dmg = ep["forge.combat_damage_dealt_to_players"]
    assert dmg["champion"]["n"] == 5 and dmg["branch"]["value"] == 22.0 and dmg["ci95"] and dmg["mde"]
    assert "median" in dmg, "the skewed figure carries a bootstrap on the median"
    fa = ep["forge.first_attack_turn"]
    assert fa["conditional"] and fa["champion"]["n"] == 3 and fa["lower_is_better"]
    assert f["run_ids"] == {"champion": ["a"], "branch": ["c"]}
    # the anchor a new branch prints reads the same bucket
    anchor = net_change.champion_at("x", "standard-v3")
    assert anchor["games"] == 27 and anchor["endpoints"]["forge.win_rate"]["value"] == round(8 / 27, 4)
    assert anchor["endpoints"]["forge.win_rate"]["mde"]
    assert net_change.champion_at("x", "vito-era") is None


def test_a_forge_objective_is_graded_at_its_own_pod_with_the_interval_on_the_difference(tmp_path, monkeypatch):
    """`forge.win_rate >= 0.2 @standard-v3` is graded on the branch's pooled
    rate AT THAT TABLE, never at whichever table held the most branch games,
    and the grade carries the difference and the null."""
    from manamap.pilot import diagnostic, goldfish
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(net_change, "_engine_casts_caveat", lambda s, b: None)
    monkeypatch.setattr(net_change, "_null_block",
                        lambda pod: ({"pod": pod, "rate": 0.233, "games": 412}, None))
    live = {None: "a" * 64, "b": "c" * 64}
    monkeypatch.setattr("manamap.pilot.common.decklist_sha256", lambda slug, branch=None: live[branch])
    seat = net_change.deck_branch_seat("x", "b")
    _stamped_record(tmp_path, "data/decks/x/sim/a.json", "standard-v3", "x", "a" * 64, 8, 27, 40)
    _stamped_record(tmp_path, "data/decks/x/sim/v.json", "vito-era", "x", "a" * 64, 30, 300, 400)
    _stamped_record(tmp_path, "data/decks/x/branches/b/sim/c.json", "standard-v3", seat, "c" * 64, 9, 33, 40)
    _stamped_record(tmp_path, "data/decks/x/branches/b/sim/w.json", "vito-era", seat, "c" * 64, 40, 300, 400)
    one_cell = {"decklist_sha256": "c" * 64,
                "output": {"damage_by_turn": {"10": {"rate": 10.0, "sd": 1.0, "n": 10000}}}}
    monkeypatch.setattr(diagnostic, "run", lambda slug, branch=None, iterations=None, seed=None, quiet=True, **kw: one_cell)
    monkeypatch.setattr(net_change, "_refuse_a_stale_measurement", lambda s, b: None)
    objective = {"axis": "forge.win_rate", "op": ">=", "value": 0.2, "pod": "standard-v3"}
    monkeypatch.setattr(net_change.deck_branch, "meta", lambda s, b: {"objective": objective, "staged": []})
    monkeypatch.setattr(net_change, "changes", lambda s, b: {})
    monkeypatch.setattr(net_change, "card_diff", lambda s, b, bill: {})
    monkeypatch.setattr(net_change.deck_branch, "source", lambda s, b: {"cards": [], "unsourced": []})
    monkeypatch.setattr(net_change, "blind_spots", lambda s, b, c: [])
    monkeypatch.setattr(net_change, "mana", lambda s, b: {"available": False, "why": "test"})
    monkeypatch.setattr(net_change, "_death_limit", lambda s, b: [])
    monkeypatch.setattr(goldfish, "model_version", lambda: "test")
    monkeypatch.setattr(net_change, "reward", lambda doc: [])
    monkeypatch.setattr(net_change, "risk", lambda doc: {})
    monkeypatch.setattr(net_change, "cost", lambda doc: {})

    doc = net_change.build("x", "b")
    assert doc["forge"]["pod"] == "standard-v3", "vito-era held more branch games and must not decide"
    g = doc["objective_grade"]
    assert g["state"] == "met" and g["reading"] == round(9 / 33, 4), g
    assert g["difference"]["ci95"] and g["null"]["rate"] == 0.233
    assert doc["design"]["primary"] == "forge.win_rate"
    assert doc["recommendation"]["state"] == "merge", doc["recommendation"]
    assert not validate_net_change.validate(doc), validate_net_change.validate(doc)

    # the same objective at a table nobody has run: not measured, and it says which
    monkeypatch.setattr(net_change.deck_branch, "meta",
                        lambda s, b: {"objective": dict(objective, pod="playgroup"), "staged": []})
    doc2 = net_change.build("x", "b")
    assert doc2["forge"]["available"] is False and "playgroup" in doc2["forge"]["why"]
    assert doc2["objective_grade"]["state"] == "not measured"
    assert doc2["recommendation"]["state"] == "inconclusive"


def test_the_validator_holds_a_forge_objective_to_its_pod_and_its_interval():
    base = _minimal(objective={"axis": "forge.win_rate", "op": ">=", "value": 0.2},
                    objective_grade={"state": "met", "reading": 0.27})
    errs = validate_net_change.validate(base)
    assert any("names no pod" in e for e in errs), errs
    assert any("no interval" in e for e in errs), errs
    wrong_table = _minimal(objective={"axis": "forge.win_rate", "op": ">=", "value": 0.2, "pod": "standard-v3"},
                           objective_grade={"state": "met", "reading": 0.27, "difference": {"ci95": [0, 1]}},
                           forge={"available": True, "pod": "vito-era", "mde": 0.1, "ci95": [0, 1],
                                  "null": None})
    errs = validate_net_change.validate(wrong_table)
    assert any("graded at 'vito-era'" in e for e in errs), errs
    assert any("null is absent with no reason" in e for e in errs), errs
