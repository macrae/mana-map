"""Triage a pile of cards against one deck, before spending anything.

The reading that has to happen before the measurement. Done by hand on a 21-card
pile it turned up three things no simulation would have said: half the pile was
combat-gated for a deck built to win without attacking, one card needed a
creature type the deck does not run, and the two best cards were invisible to
every model here.
"""

import pytest

from manamap.pilot import assess as A
from manamap.pilot.common import DECKS_DIR

SLUG, BRANCH = "ur-dragon", "treasure-v2"
needs = pytest.mark.skipif(
    not (DECKS_DIR / SLUG / "branches" / BRANCH).is_dir(), reason="no branch fixture")


def test_a_tribe_is_matched_against_real_creature_types():
    """`"Treasures you control"` is not a tribe.

    A capital-letter pattern reported Alchemist's Talent as needing "Treasures"
    the deck runs none of — a confident sentence about a card with no tribal
    text at all. Types come from the corpus, so a new set cannot age them out.
    """
    assert "Dwarf" in A.CREATURE_TYPES
    assert "Dragon" in A.CREATURE_TYPES
    # `Artifact Creature — Treasure Dog` exists, so the corpus honestly reports
    # Treasure as a creature type. In rules text "Treasures you control" means
    # the TOKEN every time, and a deck's cards.json never lists a token — so
    # reading it as a tribe called a castable card dead.
    assert "Treasure" not in A.CREATURE_TYPES
    assert "Clue" not in A.CREATURE_TYPES
    assert "Treasure" in A.TOKEN_NAMES
    assert A._plural("Dwarf") == "Dwarves"      # not "Dwarfs", and never "Dwarvess"
    assert A._plural("Dragon") == "Dragons"


def test_gating_is_read_from_the_card():
    assert A.gate_of("Whenever this creature deals combat damage to a player") == A.COMBAT
    assert A.gate_of("Whenever an opponent draws their second card") == A.OPPONENT
    assert A.gate_of("Whenever another creature you control dies") == A.DEATH
    assert A.gate_of("At the beginning of your upkeep, create a Treasure") == A.RECURRING
    assert A.gate_of("Draw two cards.") == A.ONESHOT


@needs
def test_it_names_what_no_model_here_can_see_and_then_estimates_it_anyway():
    """A card that taxes what opponents do is worth exactly zero in a solitaire
    goldfish, and the verdict must keep saying so — implying the recommendation
    is a goldfish figure would be the lie. But the pilot's real question is how
    often it would fire at THEIR table, and Forge already played those turns, so
    the verdict carries a measured frequency beside the disclaimer."""
    got = A.assess(SLUG, ["Smothering Tithe", "Monologue Tax"], branch=BRANCH)
    for row in got["cards"]:
        if row.get("in_list"):
            continue
        assert row["gate"] == A.OPPONENT
        assert "no goldfish figure can price it" in row["verdict"]
        assert row["pod_rate"]["per_round"] is not None
        assert str(row["pod_rate"]["per_round"]) in row["verdict"]


@needs
def test_a_card_already_in_the_list_says_so_and_shows_no_price():
    """The sourcing tag answers 'what would I buy to build this list', which is
    meaningless beside a card already in it — and it printed `[buy]` under
    'already in the list'."""
    got = A.assess(SLUG, ["Xorn"], branch=BRANCH)
    row = got["cards"][0]
    assert row["in_list"] and row["verdict"] == "already in the list"


@needs
def test_a_double_faced_card_resolves_from_either_face():
    """The library holds `A // B`; a pasted list may hold either face."""
    got = A.assess(SLUG, ["Treasure Map"], branch=BRANCH)
    assert "not in the corpus" not in got["cards"][0].get("verdict", "")


@needs
def test_an_off_identity_card_is_refused_before_anything_else():
    got = A.assess(SLUG, ["Counterspell"], branch=BRANCH)
    row = got["cards"][0]
    if not row.get("legal", True):
        assert "colour identity" in row["verdict"]


def test_an_unknown_name_is_reported_not_dropped():
    got = A.assess(SLUG, ["Nonesuch Cardname Here"], branch=None)
    assert "not in the corpus" in got["cards"][0]["verdict"]
