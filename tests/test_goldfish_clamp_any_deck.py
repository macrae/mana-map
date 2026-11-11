"""Skullclamp clamps under `model_sacrifice`, not under Meren's declaration alone (2026-11-11).

The clamp ("Equipped creature gets +1/-1. Whenever equipped creature dies, draw two cards.
Equip {1}") was read on 2026-11-07 for Meren and acted on only inside her commander block,
so on Edgar's threat-v1 branch, the deck that runs it on 1/1 eminence tokens, it was a dark
card that drew nothing. The draw is a death trigger, so a Teysa doubles it.

Proved by re-introducing the bug: put the clamp loop back under `commander_recursion` and
`test_a_clamp_draws_on_a_sacrifice_deck` reads the same draws for both arms.
"""
import random

from manamap.pilot import goldfish, model_coverage

CLAMP = "Equipped creature gets +1/-1.\nWhenever equipped creature dies, draw two cards.\nEquip {1}"


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def _extra_draws(with_clamp, iterations=300):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Skullclamp" if with_clamp else "Blank Rock", CLAMP if with_clamp else "", 1,
                "Artifact — Equipment" if with_clamp else "Artifact", quantity=8),
             _c("Body", "", 1, "Creature — Vampire", quantity=30, power="1", toughness="1")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    library, _ = goldfish.build_library({"cards": cards})
    rng = random.Random(9)
    runs = [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_draw=True,
                                   model_sacrifice=True)
            for _ in range(iterations)]
    return sum(r["drawn_extra_by_turn"][9] for r in runs) / iterations


def test_the_clamp_is_cast_and_read_under_the_sacrifice_channel():
    p = goldfish.classify(_c("Skullclamp", CLAMP, 1, "Artifact — Equipment"))
    assert "sacrifice" in model_coverage.channels_for(p)
    assert "recursion" not in model_coverage.channels_for(p)
    assert not model_coverage.never_cast(p, {"model_sacrifice": True})


def test_a_clamp_draws_on_a_sacrifice_deck():
    with_clamp, without = _extra_draws(True), _extra_draws(False)
    assert with_clamp > without + 2, (with_clamp, without)
