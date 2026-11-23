"""Bodies that feed Meren on their own (2026-11-23, study 010).

The pilot: "bodies with built in sac outlets solve for the early engine assembly problem."
A card that needs no second piece starts paying experience the turn it lands. Read from the
card's own text by `own_death_profile`, applied under Meren's declaration:
  dies_on_entry    "sacrifice it unless" left unpaid (Phyrexian Dreadnought), or a printed 0/0
                   that enters with X +1/+1 counters and costs only X, cast at X=0 (Hangarback)
  exploit          the exploiter sacrifices a spare body or itself; its payoff's draw is read
  self_sac_tokens  Eldrazi Spawn and Scion sacrifice themselves, so they need no altar
"""

import json

import pytest

from manamap.pilot import goldfish, goldfish_library, model_coverage
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish_profiles import own_death_profile

from conftest import requires_deck, with_card

SLUG = "meren-recursion"


def _card(name, text, type_line, toughness=None, mana_cost=""):
    return {"name": name, "oracle_text": text, "type_line": type_line,
            "toughness": toughness, "mana_cost": mana_cost}


HANGARBACK = _card("Hangarback Walker", "This creature enters with X +1/+1 counters on it.\n"
                   "When this creature dies, create a 1/1 colorless Thopter artifact creature "
                   "token with flying for each +1/+1 counter on this creature.\n{1}, {T}: Put a "
                   "+1/+1 counter on this creature.", "Artifact Creature — Construct", "0",
                   "{X}{X}")
DREADNOUGHT = _card("Phyrexian Dreadnought", "Trample\nWhen this creature enters, sacrifice it "
                    "unless you sacrifice any number of creatures with total power 12 or "
                    "greater.", "Artifact Creature — Phyrexian Dreadnought", "12", "{1}")
STINGER = _card("Fell Stinger", "Deathtouch\nExploit (When this creature enters, you may "
                "sacrifice a creature.)\nWhen this creature exploits a creature, target player "
                "draws two cards and loses 2 life.", "Creature — Zombie Scorpion", "2", "{2}{B}")
WATCHER = _card("Eyeless Watcher", "Emerge {2}{G}\nWhen this creature enters, create two 1/1 "
                "colorless Eldrazi Scion creature tokens. They have \"Sacrifice this token: Add "
                "{C}.\"", "Creature — Eldrazi Drone", "6", "{3}{G}")


def test_the_self_feeding_shapes_read_from_the_text():
    assert own_death_profile(HANGARBACK)["dies_on_entry"]
    assert own_death_profile(DREADNOUGHT)["dies_on_entry"]
    assert own_death_profile(STINGER)["exploit"] == {"draw": 2}
    assert own_death_profile(WATCHER)["self_sac_tokens"]


def test_an_exploit_is_a_choice_not_an_edict():
    """Exploit's reminder text says "you may sacrifice": it must not read as a mandatory edict,
    which would kill a body every time it entered."""
    p = own_death_profile(STINGER)
    assert p["exploit"] is not None and not p["self_edict"]


def test_an_x_creature_that_costs_coloured_mana_too_is_not_cast_at_zero():
    """A hydra ({X}{G}, a 0/0 with X counters) is cast for X, not thrown away at X=0."""
    hydra = _card("Some Hydra", "This creature enters with X +1/+1 counters on it.",
                  "Creature — Hydra", "0", "{X}{G}")
    assert not own_death_profile(hydra)["dies_on_entry"]


def test_a_body_with_toughness_that_is_not_x_does_not_die_on_entry():
    bear = _card("Grizzly Bears", "", "Creature — Bear", "2", "{1}{G}")
    p = own_death_profile(bear)
    assert not (p["dies_on_entry"] or p["exploit"] or p["self_sac_tokens"] or p["self_edict"])


def test_the_plain_edict_on_our_own_board_reads_as_an_edict():
    centaur = _card("Accursed Centaur", "When this creature enters, sacrifice a creature.",
                    "Creature — Zombie Centaur", "2", "{B}")
    assert own_death_profile(centaur)["self_edict"]


def test_each_shape_counts_as_read_by_meren_s_channel_and_no_other():
    """`try` must not call these cards invisible once the model reads them -- and the shapes
    are applied under Meren's declaration alone, so they are credited to `recursion`, never to
    `sacrifice`, which a deck without her switches on."""
    from manamap.pilot import goldfish_library as gl
    for card in (HANGARBACK, DREADNOUGHT, STINGER, WATCHER):
        prof = gl.classify(dict(card, cmc=1, colors=[], color_identity=[], keywords=[]))
        got = model_coverage.channels_for(prof)
        assert "recursion" in got, card["name"]
        assert "sacrifice" not in got, card["name"]


def _xp(doc, monkeypatch=None, blank=None, n=1500):
    if blank is not None:
        real = goldfish_library.own_death_profile
        monkeypatch.setattr(goldfish_library, "own_death_profile",
                            lambda card: {**real(card), **blank})
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    m = goldfish.run(SLUG, iterations=n, seed=7, quiet=True, _band=False, _targets_doc=t,
                     doc=doc)["metrics"]
    return m["commander_recursion"]["mean_experience_by_turn"]["8"]


@requires_deck
@pytest.mark.parametrize("add,cut,blank", [
    ("Hangarback Walker", "Tergrid, God of Fright // Tergrid's Lantern", {"dies_on_entry": False}),
    ("Fell Stinger", "Gray Merchant of Asphodel", {"exploit": None}),
    ("Eyeless Watcher", "Nyx Weaver", {"self_sac_tokens": False}),
])
def test_each_self_feeding_shape_pays_experience(monkeypatch, add, cut, blank):
    """PROVED BY REMOVING EACH SHAPE: with the card in the list, her experience by turn eight
    falls when the model stops reading what the card does to itself."""
    doc = with_card(SLUG, add, cut)
    on = _xp(doc)
    off = _xp(doc, monkeypatch, blank)
    assert on > off, (add, on, off)
