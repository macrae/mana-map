"""Meren of Clan Nel Toth's engine, declared as `model_commander_recursion` (2026-11-06).

"Whenever another creature you control dies, you get an experience counter. At the beginning
of your end step, choose target creature card in your graveyard. If that card's mana value is
less than or equal to the number of experience counters you have, return it to the
battlefield. Otherwise, put it into your hand."

Before this, meren-recursion's goldfish never counted experience, never returned a creature,
and sacrificed tokens only, so its drain read 0 at every turn: every figure on the deck
described a pile of creatures attacking. The pilot half is a stated policy: with a free
outlet and Meren out, fodder (a creature that is not itself a payoff, engine or outlet) is
sacrificed after combat.
"""

import json

import pytest

from manamap.pilot import goldfish
from manamap.pilot.common import deck_file

from conftest import requires_deck

SLUG = "meren-recursion"


def _targets(**over):
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    t.update(over)
    return t


def _run(targets, n=1500):
    return goldfish.run(SLUG, iterations=n, seed=3, quiet=True, _band=False,
                        _targets_doc=targets)["metrics"]


@requires_deck
def test_the_declaration_is_checked_against_the_commanders_text():
    t = json.loads(deck_file("sharknado", "goldfish_targets.json").read_text())
    t["model_commander_recursion"] = {"sacrifice": "one_fodder_per_turn"}
    with pytest.raises(goldfish.DeclarationError, match="Meren"):
        goldfish.run("sharknado", iterations=10, seed=1, quiet=True, _band=False,
                     _targets_doc=t)


@requires_deck
def test_absent_when_not_declared():
    t = _targets()
    t.pop("model_commander_recursion", None)
    assert "commander_recursion" not in _run(t, 200)


@requires_deck
def test_meren_returns_creatures_and_her_sacrifices_pay_the_drain():
    """PROVED BY REMOVING THE DECLARATION: no experience, no returns, and the sacrifices
    and the drain fall back to the token sweep alone."""
    on = _run(_targets(model_commander_recursion={"sacrifice": "one_fodder_per_turn"}))
    off_t = _targets()
    off_t.pop("model_commander_recursion", None)
    off = _run(off_t)
    rec = on["commander_recursion"]
    assert rec["mean_experience_by_turn"]["10"] > 1
    assert rec["mean_returned_to_battlefield"] > 0.5
    assert on["mean_sacrifices_by_turn"]["10"] > off["mean_sacrifices_by_turn"]["10"]
    assert (on["drain"]["mean_cumulative_drain_by_turn"]["10"]
            > off["drain"]["mean_cumulative_drain_by_turn"]["10"])


@requires_deck
def test_the_aggressive_policy_sacrifices_more():
    one = _run(_targets(model_commander_recursion={"sacrifice": "one_fodder_per_turn"}))
    every = _run(_targets(model_commander_recursion={"sacrifice": "all_fodder_with_payoff"}))
    assert every["mean_sacrifices_by_turn"]["10"] > one["mean_sacrifices_by_turn"]["10"]
