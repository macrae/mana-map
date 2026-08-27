"""`upgrades` — the obsolescence index read deck-aware.

Every flag in this module was measured across all 20,827 pairs before it was
written, and two earlier cuts were rejected for firing on correct data. These
tests hold the survivors to the cards that killed the rejects.
"""

import pytest

from conftest import requires_data, requires_deck
from manamap.pilot import upgrades

SLUG, BRANCH = "ur-dragon", "treasure-v2"


# --------------------------------------------------------------------------
# role_heads — the check that replaced `assess.job_of`
# --------------------------------------------------------------------------

def test_a_role_head_ignores_the_flavour_of_the_job():
    """`ramp:dork` and `ramp:land` are one job. The index pairs cards that do the
    same thing, and two flavours of ramp do."""
    assert upgrades.role_heads(["ramp:dork", "ramp:land"]) == {"ramp"}
    assert upgrades.role_heads([]) == set()
    assert upgrades.role_heads(None) == set()


def test_being_a_creature_is_not_a_job():
    """`threat:body` is 62.4% of the classified corpus. Counting it makes any two
    creatures look related and drops the catch rate from 5.68% to 2.58% — the
    same reason `eval_obsolescence` excludes it from its own retrieval figure."""
    assert upgrades.role_heads(["threat:body"]) == set()
    assert upgrades.role_heads(["threat:body", "ramp:rock"]) == {"ramp"}


@pytest.mark.parametrize("out_roles,in_roles", [
    # THE TWO FALSE POSITIVES THAT KILLED `assess.job_of` AS THE FLAG.
    # job_of collapses the list by FIRST MATCH over an alphabetically sorted
    # array, so these read as `wincon` -> `protection` and `tutor` -> `ramp`.
    (["threat:body", "wincon:combat"],                       # Atarka, World Render
     ["protection:self", "threat:body", "wincon:combat"]),   # Super-Adaptoid
    (["tutor:narrow"],                                       # Skyshroud Claim
     ["ramp:land", "tutor:narrow"]),                         # Three Visits
    (["ramp:dork", "utility:activated"],                     # Nantuko Elder
     ["draw:engine", "ramp:rock", "utility:activated"]),     # Bugenhagen
])
def test_a_shared_job_is_not_a_disagreement(out_roles, in_roles):
    """Each of these is a real, defensible comparison that the rejected cut
    flagged. A check that fires on correct data is worse than no check."""
    a, b = upgrades.role_heads(out_roles), upgrades.role_heads(in_roles)
    assert a and b, "the fixture must exercise the both-classified branch"
    assert a & b, f"{sorted(a)} and {sorted(b)} share a job"


@pytest.mark.parametrize("out_roles,in_roles", [
    (["ramp:rock"], ["buff:counters"]),          # Phyrexian Broodstar -> Aerial Doombot
    (["protection:self"], ["value:etb"]),        # Soul of the Rapids -> Eon Frolicker
    (["wincon:drain"], ["buff:anthem"]),         # Cruel Celebrant -> Syr Vondam
])
def test_no_shared_job_means_the_search_failed(out_roles, in_roles):
    a, b = upgrades.role_heads(out_roles), upgrades.role_heads(in_roles)
    assert a and b and not (a & b)


def test_one_side_unclassified_is_silence_not_agreement():
    """14.3% of pairs have a card with no role but `threat:body` — Leaden Myr
    against H.E.R.B.I.E. among them. The question is unanswerable there, and a
    row that reports it as agreement carries one fewer guard while looking like
    it carries the same number."""
    assert upgrades.role_heads(["ramp:dork"]) and not upgrades.role_heads([])


# --------------------------------------------------------------------------
# the schema it reads
# --------------------------------------------------------------------------

def test_the_pre_repair_key_still_parses():
    """`obsoleted_by` shipped for months. A reader that silently answers zero
    against an old artifact is worse than one that fails."""
    assert upgrades._entries({"compare_with": [{"name": "A"}]})[0]["name"] == "A"
    assert upgrades._entries({"obsoleted_by": [{"name": "B"}]})[0]["name"] == "B"
    assert upgrades._entries(None) == []
    assert upgrades._entries([]) == []


# --------------------------------------------------------------------------
# against the real fleet
# --------------------------------------------------------------------------

@requires_data
@requires_deck
def test_every_row_carries_both_sides():
    """The pre-repair schema carried `advantages` alone, so a card that CHARGED
    you something rendered as pure upside. Anything that renders one of these
    must be able to render all of them."""
    doc = upgrades.propose(SLUG, limit=200)
    checked = 0
    for r in doc["swaps"]:
        for key in ("gains", "costs", "narrows", "also_differs",
                    "roles_out", "roles_in"):
            assert isinstance(r[key], list), f"{r['out']}->{r['in']} {key}"
        for key in ("roles_disjoint", "roles_unclassified", "newly_combat_gated"):
            assert isinstance(r[key], bool)
        assert r["gains"], "a pair with no gain is not a comparison"
        checked += 1
    assert checked >= 10, "the fixture deck stopped producing comparisons"


@requires_data
@requires_deck
def test_the_floor_is_a_floor_and_is_reported():
    doc = upgrades.propose(SLUG, min_strength=0.6, limit=200)
    assert doc["min_strength"] == 0.6
    assert doc["swaps"], "ur-dragon has strong comparisons"
    assert all(r["strength"] >= 0.6 for r in doc["swaps"])
    # And nothing in the shipped data can beat the index's own ceiling.
    assert all(r["strength"] <= upgrades.MAX_STRENGTH_IN_DATA for r in doc["swaps"])


@requires_data
@requires_deck
def test_a_replacement_is_never_a_card_already_in_the_list():
    doc = upgrades.propose(SLUG, limit=200)
    from manamap.pilot.common import load_deck_cards
    held = {c["name"] for c in load_deck_cards(SLUG)["cards"]}
    assert doc["swaps"]
    for r in doc["swaps"]:
        assert r["in"] not in held
        assert r["out"] in held, "the anchor is a card you actually run"


@requires_data
@requires_deck
def test_a_replacement_is_inside_the_commanders_identity():
    """The index gates legality; it knows nothing about THIS deck's colours."""
    doc = upgrades.propose(SLUG, limit=200)
    from manamap.pilot import card_pool
    corpus = card_pool.load_pool()
    identity = set(doc["identity"])
    checked = 0
    for r in doc["swaps"]:
        ci = corpus[r["in"]].get("color_identity") or set()
        assert ci <= identity, f"{r['in']} {sorted(ci)} outside {sorted(identity)}"
        checked += 1
    assert checked >= 10


@requires_data
@requires_deck
def test_owned_only_asks_the_boxes_and_not_the_decks():
    """Ownership is a BOX. Deck membership is a build plan, and counting it once
    made 99 unowned cards read as owned."""
    doc = upgrades.propose(SLUG, limit=200, owned_only=True)
    from manamap.pilot import collection
    owned = collection.owned_names()
    for r in doc["swaps"]:
        assert r["in"] in owned
        assert r["owned"] is True
    wide = upgrades.propose(SLUG, limit=200)
    assert len(doc["swaps"]) <= len(wide["swaps"])


@requires_data
@requires_deck
def test_a_pile_card_with_no_comparison_is_reported_not_dropped():
    """"The index has nothing to say about this" is a real answer. An empty
    report that looks like a clean bill is not."""
    # Two cards ur-dragon does not run. A pile card already IN the list is not
    # unmatched — it has no comparison to make because you already have it.
    doc = upgrades.propose(SLUG, pool=["Counterspell", "Grizzly Bears"], limit=200)
    assert doc["pool"] == ["Counterspell", "Grizzly Bears"]
    assert doc["pool_unmatched"] is not None
    assert any("no comparison" in n for n in doc["notes"])
    # With no pool at all the key is absent, not an empty list: nobody asked.
    assert upgrades.propose(SLUG, limit=5)["pool_unmatched"] is None


@requires_data
@requires_deck
def test_a_truncated_list_says_so():
    """A silently truncated list reads as the whole set."""
    doc = upgrades.propose(SLUG, limit=3)
    assert len(doc["swaps"]) == 3
    assert doc["not_considered"], "ur-dragon has more than three"
    assert all(r["strength"] >= doc["swaps"][-1]["strength"]
               for r in doc["swaps"])


@requires_data
@requires_deck
def test_the_notes_name_the_traps_and_refuse_to_be_read_as_verdicts():
    doc = upgrades.propose(SLUG, limit=200)
    blob = " ".join(doc["notes"])
    assert "NOT VERDICTS" in blob
    assert "NOTHING HERE IS MEASURED" in blob
    assert "candidates" in blob, "it must name what measures a swap"
    assert str(upgrades.MAX_STRENGTH_IN_DATA) in blob


@requires_data
@requires_deck
def test_the_flags_fire_on_the_fleet_at_the_rate_they_were_measured_at():
    """Not a re-derivation of the rule — a floor and a ceiling on how often it
    speaks. A flag that never fires is decoration; one that fires on every row
    is noise. Both rejected cuts failed exactly here."""
    seen, disjoint, combat = 0, 0, 0
    for slug in ("ur-dragon", "edgar-vampires", "gishath"):
        doc = upgrades.propose(slug, limit=200)
        for r in doc["swaps"]:
            seen += 1
            disjoint += r["roles_disjoint"]
            combat += r["newly_combat_gated"]
    assert seen >= 50, "not enough rows to say anything about a rate"
    assert 0 < disjoint < seen * 0.25, f"{disjoint}/{seen} share no job"
    assert 0 < combat < seen * 0.25, f"{combat}/{seen} newly combat-gated"


@requires_data
@requires_deck
def test_it_runs_against_a_branch_and_reads_the_branchs_own_list():
    from manamap.pilot.common import load_deck_cards
    doc = upgrades.propose(SLUG, branch=BRANCH, limit=200)
    assert doc["branch"] == BRANCH
    held = {c["name"] for c in load_deck_cards(SLUG, BRANCH)["cards"]}
    checked = 0
    for r in doc["swaps"]:
        assert r["out"] in held
        checked += 1
    assert checked >= 5


@requires_data
@requires_deck
def test_the_notes_say_this_is_efficiency_and_not_impact():
    """MEASURED, AND IT IS THE INDEX'S DEFINING LIMIT. It pairs cards that do the
    same job more cheaply, so it cannot propose the card that moves a number:
    across 149 rows on six decks it offered a Game Changer zero times. A reader
    who expects a net-change delta from these swaps is expecting the wrong
    thing, and the report has to say so rather than let them find out by
    spending a run."""
    blob = " ".join(upgrades.propose(SLUG, limit=200)["notes"])
    assert "EFFICIENCY, NOT IMPACT" in blob
    assert "Game Changer" in blob
    assert "DECK-LEVEL" in blob
    assert "table-warper" in blob, "a single card CAN register; some do"


@requires_data
@requires_deck
def test_the_index_does_not_propose_game_changers():
    """The claim above, held to the data rather than asserted in prose."""
    from manamap.pilot import card_pool
    corpus = card_pool.load_pool()
    checked = 0
    for slug in ("ur-dragon", "edgar-vampires", "gishath"):
        for r in upgrades.propose(slug, limit=200)["swaps"]:
            assert not (corpus.get(r["in"]) or {}).get("game_changer"), r["in"]
            checked += 1
    assert checked >= 50
