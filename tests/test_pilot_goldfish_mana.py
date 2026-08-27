"""`produced_mana` — a tap-for-mana ability, written three ways.

The pattern matched an explicit symbol list and nothing else, so `{T}: Add
{C}{C}` parsed and `{T}: Add one mana of any color` did not. Measured across the
fleet: **71 of 110 tap-for-mana cards (65%) read zero** — Arcane Signet x7, Birds
of Paradise x5, Relic of Legends, Sanctum Weaver. ur-dragon's model could see 2
of its 11 non-land mana, and turn-seven mana came out ~19% low on every deck.

Not a stated assumption: the module's assumption list names rituals and cost
reducers and says nothing about rocks, because nobody knew.
"""

import pytest

from manamap.pilot import goldfish


@pytest.mark.parametrize("text,want", [
    # the shape that always worked
    ("{T}: Add {C}{C}.", 2),
    ("{T}: Add {G}{G}{G}.", 3),
    # the shape that never did — 65% of the fleet's rocks
    ("{T}: Add one mana of any color in your commander's color identity.", 1),
    ("Flying\n{T}: Add one mana of any color.", 1),
    ("{1}, {T}: Add two mana of any one color.", 2),
    # X is board-dependent: the conservative 1, as `treasure_profile` does
    ("{T}: Add X mana of any one color, where X is the number of "
     "enchantments you control.", 1),
    ("Destroy target permanent.", 0),
])
def test_the_three_ways_a_rock_is_written(text, want):
    assert goldfish.produced_mana(text) == want


@pytest.mark.parametrize("text,want", [
    # ALTERNATIVES ARE A CHOICE, NOT A SUM. Widening the match to catch the
    # written-out form also let it span commas, and `Add {R}, {G}, or {W}`
    # became three mana a turn. The old narrow pattern got this right BY
    # ACCIDENT — `(?:\{..\})+` stopped at the first comma — so the fix had to
    # preserve behaviour it was never trying to change.
    ("{T}: Add {R}, {G}, or {W}.", 1),
    ("{T}: Add {U} or {C}{U}. Spend this mana only to pay cumulative upkeep.", 2),
])
def test_alternatives_are_a_choice_not_a_sum(text, want):
    assert goldfish.produced_mana(text) == want


@pytest.mark.parametrize("text", [
    # A COST THAT CONSUMES THE PERMANENT IS NOT A RATE. Jeweled Lotus would
    # have paid three mana every turn forever.
    "{T}, Sacrifice this artifact: Add three mana of any one color.",
    "{5}, {T}, Sacrifice this artifact: Add five mana of any one color.",
    "{B}, {T}, Sacrifice a creature: Add {C}{C}{C}.",
])
def test_an_ability_that_eats_its_own_source_is_not_a_rock(text):
    assert goldfish.produced_mana(text) == 0


def test_a_land_never_counts_as_a_rock():
    """Lands are the mana BASE and are counted by the land-drop model; counting
    them here would be the same mana twice."""
    got = goldfish.classify({"name": "Cavern of Souls", "cmc": 0,
                             "type_line": "Land",
                             "oracle_text": "{T}: Add one mana of any color. "
                                            "Spend this mana only to cast a "
                                            "creature spell of the chosen type."})
    assert got["produces"] == 0


def test_restricted_mana_is_counted_but_named():
    """`spend()` is a scalar and cannot represent "only to cast Dragon spells".
    Delighted Halfling's legendary-only mana is nearly free in Commander;
    Throne of Eldraine's four is not. Same contract as the Treasure blind
    spots — the assumption is NAMED rather than silently made or dropped."""
    got = goldfish.run("goblin-storm", iterations=120, quiet=True)
    named = got["meta"].get("restricted_mana_counted_as_free")
    assert named, "restricted mana was counted with nothing saying so"
    assert any("Throne of Eldraine" in n for n in named), named


@pytest.mark.parametrize("slug", ["sisay", "ur-dragon", "hapatra"])
def test_the_fleets_rocks_are_actually_visible_now(slug):
    """A regression floor, per deck, because the failure was silent: the model
    produced a plausible number that was a fifth low."""
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards(slug)
    seen = sum(goldfish.produced_mana(c.get("oracle_text")) * c.get("quantity", 1)
               for c in doc["cards"]
               if "Land" not in (c.get("type_line") or "")
               and not c.get("is_commander"))
    assert seen >= 7, f"{slug} shows only {seen} non-land mana; the rocks are invisible again"
