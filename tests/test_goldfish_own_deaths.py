"""Every death of ours pays Meren's experience (2026-11-07).

"Whenever another creature you control dies, you get an experience counter." Until this
date the model counted a death only where its own outlets sacrificed one, so Sakura-Tribe
Elder, Spore Frog, Cankerbloom and Haywire Mite sacrificing themselves, Plaguecrafter's
edict, Skullclamp and the body a sacrifice-tutor costs all left her counter where it was:
45% of games reached turn ten with no experience at all.

`own_death_profile` reads the three shapes from the card's own text; the turn loop kills
the body through `_deaths_after_combat`, the one helper that now pays the counter.
"""

import json

import pytest

from manamap.pilot import goldfish, goldfish_library, model_coverage
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish_profiles import own_death_profile

from conftest import requires_deck

SLUG = "meren-recursion"


def _card(name, text, type_line):
    return {"name": name, "oracle_text": text, "type_line": type_line}


def test_the_three_shapes_read_from_the_text():
    elder = own_death_profile(_card("Sakura-Tribe Elder", "Sacrifice this creature: Search your "
                                    "library for a basic land card, put that card onto the "
                                    "battlefield tapped, then shuffle.", "Creature — Snake Shaman"))
    assert elder["self_sac"] and elder["self_sac_cost"] == 0
    mite = own_death_profile(_card("Haywire Mite", "{G}, Sacrifice this creature: Exile target "
                                   "noncreature artifact or noncreature enchantment.",
                                   "Artifact Creature — Insect"))
    assert mite["self_sac"] and mite["self_sac_cost"] == 1
    crafter = own_death_profile(_card("Plaguecrafter", "When this creature enters, each player "
                                      "sacrifices a creature or planeswalker of their choice. Each "
                                      "player who can't discards a card.", "Creature — Human Shaman"))
    assert crafter["self_edict"] and not crafter["self_sac"]
    clamp = own_death_profile(_card("Skullclamp", "Equipped creature gets +1/-1.\nWhenever "
                                    "equipped creature dies, draw two cards.\nEquip {1}",
                                    "Artifact — Equipment"))
    assert clamp["clamp"] == {"toughness_minus": 1, "draw": 2, "equip": 1}


def test_an_equipment_that_pays_nothing_on_the_death_is_not_a_clamp():
    """The corpus sweep matched Barbed Battlegear (+4/-1, no death clause) beside Skullclamp."""
    gear = own_death_profile(_card("Barbed Battlegear", "Equipped creature gets +4/-1.\nEquip {2}",
                                   "Artifact — Equipment"))
    assert gear["clamp"] is None


def test_a_trigger_that_sacrifices_is_not_an_activated_cost():
    """"Sacrifice this creature" inside an effect (no colon after it) is not a choice the pilot
    makes; only a COST is read."""
    p = own_death_profile(_card("Ball Lightning", "Trample, haste\nAt the beginning of the end "
                                "step, sacrifice this creature.", "Creature — Elemental"))
    assert not p["self_sac"]


def test_the_clamp_is_cast_and_its_mirror_agrees():
    prof = {"is_land": False, "bodies": 0, "produces": 0, "tutor": False, "reduces": None,
            "sac_outlet": None, "death": {}, "own_death": {"clamp": {"toughness_minus": 1,
                                                                     "draw": 2, "equip": 1}}}
    assert model_coverage.never_cast(dict(prof), {"model_commander_recursion": True}) is False
    assert model_coverage.never_cast(dict(prof), {}) is True


def _xp(monkeypatch=None, blank=None, n=1500):
    if blank is not None:
        real = goldfish_library.own_death_profile

        def _without(card):
            p = real(card)
            return {**p, **blank}
        monkeypatch.setattr(goldfish_library, "own_death_profile", _without)
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    m = goldfish.run(SLUG, iterations=n, seed=7, quiet=True, _band=False, _targets_doc=t)["metrics"]
    return m["commander_recursion"]["mean_experience_by_turn"]["10"], m


@requires_deck
@pytest.mark.parametrize("shape,blank", [
    ("self-sacrifice", {"self_sac": False}),
    ("edict", {"self_edict": False}),
    ("clamp", {"clamp": None}),
])
def test_each_death_source_pays_experience(monkeypatch, shape, blank):
    """PROVED BY REMOVING EACH SHAPE: her experience at turn ten falls without it."""
    on, _ = _xp()
    off, _ = _xp(monkeypatch, blank)
    assert on > off, (shape, on, off)


@requires_deck
def test_the_clamp_draws_on_the_death(monkeypatch):
    _, on = _xp()
    _, off = _xp(monkeypatch, {"clamp": None})
    assert on["mean_extra_cards_drawn_by_turn"]["10"] > off["mean_extra_cards_drawn_by_turn"]["10"]
