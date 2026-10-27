"""A ritual is spent only on a card the turn will actually cast.

The ritual loop casts a ritual when its net mana bridges the hand to a card it
could not otherwise afford (`_ritual_yield` minus its cost, against `_wants`).
It read a want as a mana TOTAL, so it spent rituals on two kinds of card the
turn then refused (2026-10-26, found by sharknado's ritual test reading
backwards):

  a wheel the pilot's policy holds (`hold_until.wheel`), which the draw loop's
  own gate refuses until a commander is out;
  a card whose colours the turn cannot pay, which `spend` refuses under
  `model_colors`.

Either way the ritual left the hand and nothing was bought. Each test builds a
deck where the ONLY thing a ritual could bridge to is the refused card, so a
correct loop never casts one: the hand reads the same as with the ritual blinded
(a dead card that stays in hand). Re-introduce either bug and the ritual is
cast, the hand is a card smaller, and the arms differ.
"""
import random

import pytest

from manamap.pilot import goldfish, goldfish_library, try_swap

from conftest import requires_data


def _deck(*spells):
    commander = {"name": "Unaffordable", "type_line": "Legendary Creature — Human",
                 "mana_cost": "{8}{R}{R}", "cmc": 10, "oracle_text": "",
                 "is_commander": True, "quantity": 1, "power": "1", "toughness": "1",
                 "colors": ["R"], "color_identity": ["R", "U"]}
    mountains = dict(try_swap.corpus_card("Mountain"), quantity=40)
    cards = [commander, mountains]
    for name, n in spells:
        cards.append(dict(try_swap.corpus_card(name), quantity=n))
    return {"cards": cards}


def _hands(deck, monkeypatch, blind, **kw):
    if blind:
        monkeypatch.setattr(goldfish_library, "ritual_profile", lambda card: None)
    library, _ = goldfish.build_library(deck)
    monkeypatch.undo()
    rng = random.Random(4)
    return [goldfish.simulate_once(rng, library, 10, [], 6, model_draw=True, **kw)
            ["hand_size_by_turn"] for _ in range(200)]


@requires_data
def test_a_ritual_is_not_spent_on_a_wheel_the_policy_holds(monkeypatch):
    deck = _deck(("Seething Song", 20), ("Reforge the Soul", 20))
    policy = {"wheel": {"commanders_out": 1}}
    read = _hands(deck, monkeypatch, blind=False, hold_until=policy)
    blind = _hands(deck, monkeypatch, blind=True, hold_until=policy)
    assert read == blind, "a ritual was spent to afford a wheel the policy holds"


@requires_data
def test_a_ritual_is_not_spent_on_a_card_whose_colours_cannot_be_paid(monkeypatch):
    deck = _deck(("Seething Song", 20), ("Air Elemental", 20))
    read = _hands(deck, monkeypatch, blind=False, model_colors=True)
    blind = _hands(deck, monkeypatch, blind=True, model_colors=True)
    assert read == blind, "a ritual was spent on a blue card with no blue source"


@requires_data
def test_the_control_a_ritual_still_bridges_to_a_card_it_can_buy(monkeypatch):
    """The other half, or the two tests above pass on a loop that never casts a
    ritual at all: with a red five-drop it CAN pay for, the ritual is cast."""
    deck = _deck(("Seething Song", 20), ("Hellkite Charger", 20))
    read = _hands(deck, monkeypatch, blind=False, model_colors=True)
    blind = _hands(deck, monkeypatch, blind=True, model_colors=True)
    assert read != blind, "the ritual never bridged — the two tests above prove nothing"
