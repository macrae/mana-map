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
