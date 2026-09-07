"""A Room is two cards under one type line, and the model used to charge for both.

CR 202.3d takes a split card's mana value from the combined costs of its halves,
so Scryfall reports `Bottomless Pool // Locker Room` as cmc 6.0 — while the door
you cast costs `{U}`. The goldfish spent straight from that field.
"""

import pytest

from manamap.pilot import goldfish as g


def room(name, type_line, mana_cost, text=""):
    return {"name": name, "type_line": type_line, "mana_cost": mana_cost,
            "oracle_text": text, "cmc": 99.0}


ROOM_TYPE = "Enchantment — Room // Enchantment — Room"


def test_a_room_costs_one_door_and_not_scryfalls_combined_mana_value():
    """THE BUG: a {U} enchantment was charged six mana and sat in hand.

    `cmc` 99.0 below stands in for Scryfall's combined value. If `classify` ever
    reads it again for a Room this fails, which is the point — the assertion is
    on the door, not merely on "less than 99".
    """
    card = g.classify(room("Bottomless Pool // Locker Room", ROOM_TYPE, "{U} // {4}{U}"))
    assert card["cmc"] == 1, card["cmc"]
    assert card["room"]["unlock_cost"] == 5
    assert card["room"]["full_mv"] == 6


def test_the_cheaper_door_is_cast_even_when_it_is_the_BACK_one():
    """The corpus sweep is what forces this, and it is not a corner case.

    The front door is cheaper-or-equal on 24 of the 30 Rooms in the corpus, so a
    front-door rule looks right and passes casual inspection. It is a FOURFOLD
    error on `Defiled Crypt // Cadaver Lab`, whose doors are {3}{B} and {B}.
    """
    card = g.classify(room("Defiled Crypt // Cadaver Lab", ROOM_TYPE, "{3}{B} // {B}"))
    assert card["cmc"] == 1, card["cmc"]
    assert card["room"]["unlock_cost"] == 4


def test_the_pips_come_from_the_door_the_cost_came_from():
    """`front_field` always answers with the left half — the wrong half exactly
    when the cost is. A back-door Room would otherwise ask for the front door's
    colours at the back door's price."""
    card = g.classify(room("Roaring Furnace // Steaming Sauna", ROOM_TYPE,
                           "{1}{R} // {3}{U}{U}"))
    assert card["cmc"] == 2
    assert card["pips"] == [frozenset({"R"})], card["pips"]
    back = g.classify(room("Experimental Lab // Staff Room", ROOM_TYPE,
                           "{3}{G} // {2}{G}"))
    assert back["cmc"] == 3
    assert back["pips"] == [frozenset({"G"})], back["pips"]


def test_every_other_card_in_the_game_is_untouched():
    """The widening must be pure: no Room, no change to cmc or pips."""
    for card in ({"name": "Sol Ring", "type_line": "Artifact", "mana_cost": "{1}",
                  "cmc": 1.0, "oracle_text": ""},
                 {"name": "Fire // Ice", "type_line": "Instant // Instant",
                  "mana_cost": "{1}{R} // {1}{U}", "cmc": 4.0, "oracle_text": ""}):
        out = g.classify(card)
        assert out["room"] is None, card["name"]
        assert out["cmc"] == int(card["cmc"]), card["name"]


def test_eerie_reads_an_unlock_and_constellation_does_not():
    """The distinction that makes a Room worth more than one enchantment.

    Both idioms open "whenever an enchantment you control enters". Only Eerie
    adds "and whenever you fully unlock a Room", so only Eerie fires a second
    time for no card from hand. Crediting an unlock to every per-enchantment
    payoff would over-pay the five plain constellation cards in the deck this
    was written for.
    """
    eerie = g.classify({
        "name": "Balemurk Leech", "type_line": "Enchantment Creature — Leech",
        "mana_cost": "{1}{B}", "cmc": 2.0,
        "oracle_text": ("Eerie — Whenever an enchantment you control enters and "
                        "whenever you fully unlock a Room, each opponent loses 1 life.")})
    constellation = g.classify({
        "name": "Grim Guardian", "type_line": "Enchantment Creature — Zombie",
        "mana_cost": "{2}{B}", "cmc": 3.0,
        "oracle_text": ("Constellation — Whenever this creature or another "
                        "enchantment you control enters, each opponent loses 1 life.")})
    assert eerie["drain"]["eerie"] is True
    assert constellation["drain"]["eerie"] is False
    # Both must still read the ENTERS half — the flag adds a trigger, never
    # replaces one.
    assert eerie["drain"]["drain_per_enchantment"] == 1
    assert constellation["drain"]["drain_per_enchantment"] == 1


def test_a_half_open_room_animates_at_one_door_and_a_full_one_at_both():
    """CR 709.5: a permanent does not have the mana cost of a LOCKED half, so
    CR 202.3d gives a half-open Room the mana value of the open door alone.
    `model_commander_animate` sets base power from that, so the 8/8 is only real
    once both doors are open."""
    card = g.classify(room("Unholy Annex // Ritual Chamber", ROOM_TYPE,
                           "{2}{B} // {3}{B}{B}"))
    assert card["cmc"] == 3, "half open — the body is the open door"
    assert card["room"]["full_mv"] == 8, "fully unlocked — both doors"


def test_a_locked_door_gives_you_nothing_until_you_open_it():
    """THE BUG THIS EXISTS FOR, and it is the one that flattered Rooms.

    Scryfall concatenates both halves into `oracle_text`, so every text-derived
    profile read the LOCKED door too. `Unholy Annex // Ritual Chamber` was
    charged 3 for its {2}{B} front half and handed the 6/6 flying Demon printed
    on the {3}{B}{B} back half at the same instant. Correcting it moved rooms-v1
    from 0.3842 to 0.3192 — the over-credit was larger than the under-credit the
    same commit fixed.

    `token_power` is the TOTAL across bodies, so a lone 6/6 is (6, 1). Reaching
    for `combat_profile(...)["power"]` instead gives 0 — a Room has no power of
    its own — and the Demon enters as a 1/1.
    """
    card = g.classify(room(
        "Unholy Annex // Ritual Chamber", ROOM_TYPE, "{2}{B} // {3}{B}{B}",
        "At the beginning of your end step, draw a card. If you control a Demon, "
        "each opponent loses 2 life and you gain 2 life. Otherwise, you lose 2 life."
        " // When you unlock this door, create a 6/6 black Demon creature token "
        "with flying."))
    assert card["cmc"] == 3
    assert card["combat"]["token_bodies"] == 0, (
        "the Demon is printed on the LOCKED door and must not arrive at cast time")
    on_unlock = card["room"]["on_unlock"]
    assert on_unlock["combat"]["token_bodies"] == 1
    assert on_unlock["combat"]["token_power"] == 6, (
        "total power across bodies, so a lone 6/6 is (6, 1) — reading the card's "
        "own power would enter it as a 1/1")


def test_casting_a_door_fires_that_doors_own_unlock_clause():
    """CR 709.5d: a permanent is given the 'left half unlocked' designation as it
    ENTERS if its left half was cast. So a "when you unlock this door" ability on
    the door you cast triggers immediately, and belongs to the CAST profile — not
    to `on_unlock`. Grand Entryway really does make its Glimmer on turn two."""
    card = g.classify(room(
        "Grand Entryway // Elegant Rotunda", ROOM_TYPE, "{1}{W} // {2}{W}",
        "When you unlock this door, create a 1/1 white Glimmer enchantment "
        "creature token."
        " // When you unlock this door, put a +1/+1 counter on each of up to two "
        "target creatures."))
    assert card["cmc"] == 2
    assert card["combat"]["token_bodies"] == 1, (
        "the door you cast is unlocked as it enters, so its own clause fires now")
