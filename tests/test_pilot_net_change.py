"""`net-change` — the report a spending decision rests on.

It was assembled by hand once: eight commands and a page of HTML, to decide
whether to buy 21 cards for the Ur-Dragon treasure refactor. The answer was no.
Doing that by hand again is how the next one gets skipped.
"""

import json

import pytest

from conftest import A_BRANCH, requires_branch, requires_deck
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

@requires_branch
@requires_deck
def test_the_report_names_the_swaps_rather_than_counting_them():
    """"21 staged" is not a description of a treatment."""
    ch = net_change.changes(SLUG, BRANCH)
    assert ch["count"] == len(ch["spells"]) + len(ch["lands"])
    assert ch["count"] >= 1
    checked = 0
    for row in ch["spells"] + ch["lands"]:
        assert row["out"] and row["in"]
        checked += 1
    assert checked >= 1


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
