"""A stack scenario about a branch reads the branch's list (2026-11-12).

Edgar's stack 013 names three threat-v1 cards. Read against the sleeved list, the
brief every resolver is told to read first filed Pitiless Plunderer and Zulaport
Cutthroat as non-creatures (a body count of zero) and the gate warned that the line
used cards outside the 99. A top-level `branch` now points both at the branch's
`cards.json`. Proved by re-introducing the bug: make `stack_branch` return None and
both tests fail.
"""
import pytest

from conftest import requires_deck
from manamap.pilot import scenario_facts, validate_stack
from manamap.pilot.common import load_json, deck_dir

SLUG, SID = "edgar-vampires", "013"


def _stack():
    paths = sorted((deck_dir(SLUG) / "stacks").glob(f"{SID}-*.json"))
    if not paths or not (deck_dir(SLUG) / "branches" / "threat-v1" / "cards.json").is_file():
        pytest.skip("stack 013 or branch threat-v1 is not on disk")
    return load_json(paths[0])


def _branch_names():
    """The branch's own list. The board may legitimately name a card the branch has
    since cut (013 kept Reassembling Skeleton after study 016 and is withheld for
    it), so the tests assert what the bug got wrong — a card the BRANCH runs
    called outside the deck — never that the board names nothing outside it."""
    cards = load_json(deck_dir(SLUG) / "branches" / "threat-v1" / "cards.json")
    cards = cards.get("cards", cards) if isinstance(cards, dict) else cards
    return {c["name"] for c in cards}


@pytest.mark.regression
@requires_deck
def test_the_brief_counts_branch_creatures_as_bodies():
    doc = _stack()
    assert doc.get("branch") == "threat-v1"
    facts = scenario_facts.analyze(SLUG, SID, with_rulings=False)["stacks"][SID]
    assert facts["branch"] == "threat-v1"
    assert {"Pitiless Plunderer", "Zulaport Cutthroat"} <= set(facts["your_board"]["creature_bodies"])
    flagged = set(facts["card_membership"]["NOT_IN_THE_DECK"])
    assert not (flagged & _branch_names()), (
        f"the brief called branch cards outside the deck: {sorted(flagged & _branch_names())}")


@pytest.mark.regression
@requires_deck
def test_the_gate_does_not_call_branch_cards_outside_the_list():
    unknown, elsewhere = validate_stack.unknown_cards(_stack(), SLUG)
    assert unknown == []
    called_out = {m.split("'")[1] for m in elsewhere}
    assert not (called_out & _branch_names()), (
        f"the gate called branch cards outside the list: {sorted(called_out & _branch_names())}")


@pytest.mark.regression
@requires_deck
def test_a_back_face_names_the_card_the_deck_runs():
    """Heliod 007 names "Heliod, the Warped Eclipse"; the brief matched full names
    only and called the commander's back face outside the deck. Proved by
    re-introducing the bug: index `deck_names` by full name and this fails."""
    if not list((deck_dir("heliod") / "stacks").glob("007-*.json")):
        pytest.skip("heliod stack 007 is not on disk")
    facts = scenario_facts.analyze("heliod", "007", with_rulings=False)["stacks"]["007"]
    assert "Heliod, the Warped Eclipse" not in facts["card_membership"]["NOT_IN_THE_DECK"]
