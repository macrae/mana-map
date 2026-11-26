"""The wheel deck's own cards (2026-11-25, sharknado).

The pilot asked whether the goldfish measures a deck "built around wheels". Two cards on
jaws-v1 did something the model never read:

* LIBRARY OF LENG: "If an effect causes you to discard a card, discard it, but you may put it
  on top of your library instead of into your graveyard." The discard still happens -- Brallin
  is paid -- and a wheel's draw takes the hand back first, so a wheel no longer costs the hand.
* INTI, SENESCHAL OF THE SUN: a discard on every swing for a +1/+1 counter, and the top card
  exiled and playable until the next end step whenever you discard -- not a draw.

Every test compares two decks on the same seeds, one with the reading's text and one without,
and each was proved red by putting its bug back (named in the test).
"""
import random

from manamap.pilot import goldfish

LENG = ("You have no maximum hand size.\nIf an effect causes you to discard a card, discard "
        "it, but you may put it on top of your library instead of into your graveyard.")
INTI = ("Whenever you attack, you may discard a card. When you do, put a +1/+1 counter on "
        "target attacking creature. It gains trample until end of turn.\nWhenever you discard "
        "one or more cards, exile the top card of your library. You may play that card until "
        "your next end step.")
INTI_ATTACK_ONLY = INTI.split("\n")[0]
INTI_IMPULSE_ONLY = INTI.split("\n")[1]
WHEEL = "Each player discards their hand, then draws seven cards."
PING = "Whenever you discard a card, this deals 1 damage to each opponent."


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def _doc(*cards):
    cards = list(cards)
    cards.append(_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land",
                    quantity=99 - sum(c["quantity"] for c in cards)))
    return {"cards": cards}


def _games(doc, iterations=400, seed=7, **flags):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(seed)
    kw = dict(model_draw=True, model_discard=True, model_combat=True)
    kw.update(flags)
    return [goldfish.simulate_once(rng, library, 3, [], 10, **kw) for _ in range(iterations)]


def _mean(games, key, turn=9):
    return sum(g[key][turn] for g in games) / len(games)


# ── Library of Leng ─────────────────────────────────────────────────────────

def _leng_deck(leng_text):
    # Eight payoffs, so the wheel's hold is open in both arms almost from the start; FREE
    # wheels, so the extra wheels Leng allows cost no mana; and scarce five-drop bodies, so a
    # body a wheel throws away is one the deck does not get back. The gap is the redirect.
    return _doc(_c("Leng", leng_text, 1, "Artifact", quantity=4),
                _c("Pinger", PING, 1, "Artifact", quantity=8),
                _c("Wheel", WHEEL, 0, "Sorcery", quantity=10),
                _c("Big Body", "", 5, "Creature — Shark", quantity=8, power="5", toughness="5"))


def test_a_wheel_under_leng_keeps_the_spells_it_discards():
    """Under Leng the spells a wheel discards go on top and the seven drawn bring them back;
    without it a body held for later is thrown away. Measured on three seeds: +0.54 to +0.66
    bodies by turn ten, and 0.00 +/- 0.03 with the bug put back (`_to_yard(hand)` whatever
    `discard_to_top` says). The POLICY is tested too: putting the whole hand back, spare lands
    included, measured -0.11 -- fewer bodies than no Leng -- which is why lands go to the
    graveyard."""
    leng = _games(_leng_deck(LENG))
    blank = _games(_leng_deck("You have no maximum hand size."))
    assert _mean(leng, "bodies_by_turn") > _mean(blank, "bodies_by_turn") + 0.3


def test_a_discard_under_leng_still_pays_the_discard_payoffs():
    """The redirect is a REPLACEMENT on where the card goes, not on whether it was discarded:
    Brallin and the Pinger still see every card. Put the bug back by not counting a redirected
    discard and this reads fewer discards under Leng, not at least as many."""
    leng = _games(_leng_deck(LENG))
    blank = _games(_leng_deck("You have no maximum hand size."))
    assert _mean(leng, "discarded_by_turn") >= _mean(blank, "discarded_by_turn")


# ── Inti, Seneschal of the Sun ──────────────────────────────────────────────

def _inti_deck(inti_text, brick=False):
    filler = (_c("Brick", "", 20, "Sorcery", quantity=30) if brick else
              _c("Small Body", "", 2, "Creature — Human", quantity=30, power="2", toughness="2"))
    return _doc(_c("Inti", inti_text, 2, "Legendary Creature — Human Knight", quantity=6,
                   power="2", toughness="2"),
                _c("Pinger", PING, 1, "Artifact", quantity=4),
                _c("Wheel", WHEEL, 2, "Sorcery", quantity=8),
                filler)


def test_inti_impulse_gives_cards_to_play_and_is_not_a_draw():
    """Every discard batch exiles the top card, playable until the next end step: more bodies
    reach the board, and the draw count does not move -- Shabraz is not paid for an impulse.
    Put the bug back by sending the impulse through `draw_n` and the draw count rises."""
    with_impulse = _games(_inti_deck(INTI_IMPULSE_ONLY))
    without = _games(_inti_deck("Vigilance"))
    assert _mean(with_impulse, "bodies_by_turn") > _mean(without, "bodies_by_turn") + 0.3
    assert abs(_mean(with_impulse, "drawn_extra_by_turn")
               - _mean(without, "drawn_extra_by_turn")) < 0.5


def test_an_unplayed_impulse_card_leaves_by_the_end_step_after_next():
    """Bricks nothing can cast: an impulse brick must leave by the end step after next, so the
    hand under Inti holds only the impulses still live. Measured: +0.27 cards over the control
    by turn ten; with the bug put back -- never prune `impulse_cards` or the hand -- +1.10, every
    brick exiled all game piling up."""
    with_impulse = _games(_inti_deck(INTI_IMPULSE_ONLY, brick=True), model_hand_limit=False)
    without = _games(_inti_deck("Vigilance", brick=True), model_hand_limit=False)
    assert _mean(with_impulse, "hand_size_by_turn") - _mean(without, "hand_size_by_turn") < 0.7


def test_inti_attack_pitches_a_spare_land_for_a_counter():
    """On every swing with two or more lands in hand, a land is discarded (Brallin and the
    Pinger are paid) and the biggest attacker gets a counter. Put the bug back -- drop
    `attack_discard_counter` from the combat-engine registration -- and neither moves."""
    attack = _games(_inti_deck(INTI_ATTACK_ONLY))
    plain = _games(_inti_deck("Vigilance"))
    assert _mean(attack, "discarded_by_turn") > _mean(plain, "discarded_by_turn") + 0.5
    assert _mean(attack, "total_damage_by_turn") > _mean(plain, "total_damage_by_turn")


# ── what the wheels cost (item 4) ───────────────────────────────────────────

ONE_SIDED = "Discard your hand, then draw seven cards."


def _cost_deck(wheel_text):
    return _doc(_c("Pinger", PING, 1, "Artifact", quantity=6),
                _c("Wheel", wheel_text, 2, "Sorcery", quantity=10),
                _c("Body", "", 3, "Creature — Shark", quantity=20, power="3", toughness="3"))


def test_a_symmetric_wheel_hands_each_opponent_its_seven():
    """The cost row (2026-11-25): cards each opponent drew off our effects, beyond their own
    draw step. In a deck whose only such effect is a fixed draw-seven symmetric wheel, every
    game reads exactly seven per wheel cast. Put the bug back -- never add to
    `opponent_cards_given` -- and every game reads zero."""
    games = _games(_cost_deck(WHEEL), iterations=200)
    assert any(g["wheels_cast_by_turn"][7] for g in games)
    for g in games:
        assert g["opponent_cards_given_by_turn"][7] == 7 * g["wheels_cast_by_turn"][7], g


def test_a_wheel_of_our_hand_alone_hands_nobody_anything():
    """"Discard your hand, then draw seven" refills only us: the row stays at zero while
    the wheels still fire. Count every wheel as symmetric and this fails."""
    games = _games(_cost_deck(ONE_SIDED), iterations=200)
    assert any(g["wheels_cast_by_turn"][7] for g in games)
    assert all(g["opponent_cards_given_by_turn"][7] == 0 for g in games)


def test_the_cost_row_is_absent_where_no_wheel_is_modelled():
    """A deck without `model_discard` gets no series, so the net-change row is absent on both
    arms rather than a column of zeros that reads as "hands nothing"."""
    games = _games(_cost_deck(WHEEL), iterations=20, model_discard=False)
    assert all(g["opponent_cards_given_by_turn"] is None for g in games)
    from manamap.pilot import diagnostic
    assert diagnostic.PER_GAME[("wheels", "opponent_cards_given_by_turn", "8")](games[0]) is None


def test_an_impulse_card_is_never_cycled_or_pitched():
    """An exiled impulse card may be played, never cycled or pitched from hand. Cycling one
    crashed net-change on jaws-v1 (`list.remove`: the card was set aside, not in hand). Put the
    bug back -- cycle from `hand` rather than `_held()` -- and this raises."""
    deck = _doc(_c("Inti", INTI_IMPULSE_ONLY, 2, "Legendary Creature — Human Knight", quantity=6,
                   power="2", toughness="2"),
                _c("Pinger", PING, 1, "Artifact", quantity=4),
                _c("Wheel", WHEEL, 2, "Sorcery", quantity=8),
                _c("Cycler", "Cycling {2} ({2}, Discard this card: Draw a card.)", 9,
                   "Creature — Shark", quantity=30, power="5", toughness="5"))
    games = _games(deck, iterations=300)
    assert sum(g["discarded_by_turn"][9] for g in games) > 0


# ── the noncreature discount, Valeria's draw, Will of the Jeskai (2026-11-25) ───────────────

VALERIA = ("Noncreature spells you cast cost {1} less to cast. Whenever you cast your first "
           "noncreature spell each turn, draw a card.")
VALERIA_DISCOUNT = VALERIA.split(". ")[0] + "."
VALERIA_DRAW = VALERIA.split(". ")[1]
WILL = ("Choose one. If you control a commander as you cast this spell, you may choose both "
        "instead. • Each player may discard their hand and draw five cards. • Each "
        "instant and sorcery card in your graveyard gains flashback until end of turn. The "
        "flashback cost is equal to its mana cost.")


def test_a_noncreature_discount_is_read_and_spares_creatures():
    """Valeria Richards and Longshot, Rebel Bowman read as nothing: the reduction reader refused
    "Noncreature" as not a creature type. Read directly off the production functions."""
    from manamap.pilot.goldfish_profiles import cost_reduction, reduced_cost
    red = cost_reduction({"oracle_text": VALERIA})
    assert red == (1, ("noncreature",), False)
    spell = {"cmc": 6.0, "pips": [frozenset({"U"})], "is_creature": False,
             "type_line": "Enchantment", "subtypes": []}
    body = dict(spell, is_creature=True, type_line="Creature — Shark")
    assert reduced_cost(spell, [red]) == 5 and reduced_cost(body, [red]) == 6


def test_the_noncreature_discount_casts_expensive_spells_sooner():
    """With the discount out, six-mana wheels come down sooner and more often: +0.125 wheels by
    turn ten measured. Put the bug back -- drop the "noncreature" branch from `reduced_cost` --
    and the two arms read the same (0.00)."""
    def deck(text):
        return _doc(_c("Valeria", text, 3, "Creature — Human", quantity=6, power="2",
                       toughness="2"),
                    _c("Pinger", PING, 1, "Artifact", quantity=4),
                    _c("Big Wheel", WHEEL, 6, "Sorcery", quantity=10))
    assert _mean(_games(deck(VALERIA_DISCOUNT)), "wheels_cast_by_turn") \
        > _mean(_games(deck("Vigilance")), "wheels_cast_by_turn") + 0.06


def test_valeria_draws_on_the_first_noncreature_spell_each_turn_only():
    """Once a turn, not once a spell: +3.05 cards by turn eight on a deck of cantrips; firing on
    every noncreature cast (the bug, put back) read +12.9."""
    def deck(text):
        return _doc(_c("Valeria", text, 2, "Creature — Human", quantity=8, power="2",
                       toughness="2"),
                    _c("Cantrip", "Draw a card.", 1, "Sorcery", quantity=30))
    diff = _mean(_games(deck(VALERIA_DRAW)), "drawn_extra_by_turn", 7) \
        - _mean(_games(deck("Vigilance")), "drawn_extra_by_turn", 7)
    assert 1.0 < diff < 6.0, diff


def test_will_of_the_jeskai_is_a_wheel_and_grants_flashback_with_a_commander_out():
    """Will read as nothing: the wheel pattern wanted "discards ..., then draws" and Will says
    "may discard ... and draw". It is a symmetric five-card wheel, and with a commander out the
    wheels it discards can be cast again from the graveyard this turn: +0.61 wheels cast against
    the same card without the grant. Put the bug back -- never grant -- and the arms match."""
    from manamap.pilot.goldfish_profiles import draw_profile
    d = draw_profile({"name": "Will of the Jeskai", "oracle_text": WILL, "type_line": "Sorcery",
                      "mana_cost": "{3}{R}", "cmc": 4.0})
    assert d["wheel_draws"] == 5 and d["wheel_symmetric"] and d["grant_flashback"] == "commander"

    def deck(text):
        return _doc(_c("Will", text, 4, "Sorcery", quantity=6),
                    _c("Pinger", PING, 1, "Artifact", quantity=4),
                    _c("Wheel", WHEEL, 3, "Sorcery", quantity=10))
    plain = "Each player may discard their hand and draw five cards."
    assert _mean(_games(deck(WILL)), "wheels_cast_by_turn") \
        > _mean(_games(deck(plain)), "wheels_cast_by_turn") + 0.3


def test_the_widened_wheel_reader_refuses_the_wheels_it_must_not_read():
    """A die roll, an energy-sized draw and a vote the opponents carry are not plain wheels; the
    corpus sweep that widened the pattern found them (2026-11-25)."""
    from manamap.pilot.goldfish_profiles import _WHEEL_EXCLUDED
    assert {"Strategy, Schmategy", "Wheel of Potential", "Sail into the West"} <= _WHEEL_EXCLUDED


def test_a_class_level_reduction_is_not_the_card_s_until_paid():
    """Artist's Talent's "Noncreature spells you cast cost {1} less" sits behind "{2}{R}: Level 2":
    the widened reader applied it from the cast. Only text before the first level counts. Put the
    bug back -- drop the level cut in `cost_reduction` -- and this reads a reduction."""
    from manamap.pilot.goldfish_profiles import cost_reduction
    talent = ("(Gain the next level as a sorcery to add its ability.) Whenever you cast a "
              "noncreature spell, you may discard a card. If you do, draw a card. {2}{R}: Level 2 "
              "Noncreature spells you cast cost {1} less to cast.")
    assert cost_reduction({"oracle_text": talent}) is None
