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
    """THIS TEST HAD NEVER ASSERTED ANYTHING.

    It was guarded by `if not row.get("legal", True)` — so if `assess` ever
    stopped refusing an off-identity card the guard went false and the test
    passed silently. Worse, `SLUG` is FIVE-COLOUR ur-dragon, so Counterspell is
    on-identity and the branch never ran even once.

    Refusal needs a deck that can refuse. radagast is mono-green.
    """
    mono = "radagast"
    if not (DECKS_DIR / mono / "cards.json").exists():
        pytest.skip(f"no {mono} fixture")
    row = A.assess(mono, ["Counterspell"])["cards"][0]
    assert row["legal"] is False, (
        "a mono-green deck accepted Counterspell — the identity check is not "
        "running, and on a five-colour deck nothing would show it")
    assert "colour identity" in row["verdict"]
    # And the control: an on-identity card must NOT be refused, or the check
    # could be a constant `False` and still pass the line above.
    assert A.assess(mono, ["Llanowar Elves"])["cards"][0]["legal"] is True


def test_an_unknown_name_is_reported_not_dropped():
    got = A.assess(SLUG, ["Nonesuch Cardname Here"], branch=None)
    assert "not in the corpus" in got["cards"][0]["verdict"]


@needs
def test_opponent_gated_means_opponent_agency_not_opponent_as_a_target():
    """MEASURED BEFORE IT SHIPPED, AND THE FIRST CUT FAILED IT.

    Matching "each opponent" anywhere is the wording every drain payoff in the
    game uses: it classed 8 of this branch's 95 cards as gated on the pod and 5
    were wrong — Reckless Fireweaver is the branch's OWN drain, gated on your
    artifacts entering, and Exotic Orchard is a land. A card is gated on the
    opponents only when THEY have to act first.
    """
    assert A.gate_of("Whenever an artifact you control enters, Reckless "
                     "Fireweaver deals 1 damage to each opponent.") != A.OPPONENT
    assert A.gate_of("{T}: Add one mana of any color that a land an opponent "
                     "controls could produce.") != A.OPPONENT
    assert A.gate_of("Whenever an opponent draws a card, that player may pay "
                     "{2}. If they don't, you create a Treasure token.") == A.OPPONENT
    assert A.gate_of("At the beginning of your upkeep, each opponent may create "
                     "a Treasure token.") == A.OPPONENT


def test_the_tribe_is_named_in_english():
    """"needs Fishs" makes a correct verdict read like broken software, which is
    the whole cost of a sentence a tool says out loud. Invariant plurals and the
    -es rule, and nothing over-claimed: a Rhino pluralises normally."""
    assert A._plural("Fish") == "Fish"
    assert A._plural("Fox") == "Foxes"
    assert A._plural("Dwarf") == "Dwarves"
    assert A._plural("Rhino") == "Rhinos"
    assert A._plural("Dragon") == "Dragons"


@needs
def test_it_names_the_cards_no_channel_of_the_model_can_see():
    """STEP 6 OF THE ORDER, WHICH ONLY COVERED OPPONENT-GATED CARDS BEFORE.

    A sweep prices what the goldfish simulates and prices everything else only
    by displacement — which reads as noise and costs a full run per card to say
    so. Measured on the 29-card pool `close` proposed for the treasure branch,
    14 were invisible: Mana Reflection doubles MANA rather than tokens, and
    Oath of Lieges and Greener Pastures are land-matters cards the centroid
    pulled in on similar phrasing. Naming them is 16 runs not spent.
    """
    got = A.assess(SLUG, ["Mana Reflection", "Primal Vigor"], branch=BRANCH)
    by = {r["card"]: r for r in got["cards"]}
    assert by["Mana Reflection"]["model_sees"] == []
    assert "NO CHANNEL" in by["Mana Reflection"]["verdict"]
    # Primal Vigor is the control: same shelf, and the model DOES see it.
    assert "treasure doubler" in by["Primal Vigor"]["model_sees"]
    assert "NO CHANNEL" not in by["Primal Vigor"]["verdict"]
