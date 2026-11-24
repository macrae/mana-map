"""Reducers that are not about a creature, and the Tithe infinites' finishers (2026-11-23).

Sapphire Medallion read as nothing: "Blue spells you cast cost {1} less" matched the reducer
pattern and was refused because Blue is not a creature type. And the two Smothering Tithe
infinites on Heliod b3-v1 produce near-infinite mana and infinite card draw -- a kill only with
something to spend it on: Walking Ballista, or Approach of the Second Sun cast twice.
"""

import json

from manamap.pilot import goldfish
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish_profiles import cost_reduction, reduced_cost, window_profile

from conftest import requires_deck


def _c(name, text, type_line="Artifact"):
    return {"name": name, "oracle_text": text, "type_line": type_line}


def _spell(cmc, pips, type_line, creature=False):
    return {"cmc": cmc, "pips": [frozenset(p) for p in pips], "type_line": type_line,
            "is_creature": creature, "subtypes": []}


def test_a_colour_reducer_cuts_only_that_colour_s_spells_and_never_a_pip():
    red = cost_reduction(_c("Sapphire Medallion", "Blue spells you cast cost {1} less to cast."))
    assert red == (1, ("color", "U"), False)
    blue = _spell(4, ["U"], "Sorcery")
    white = _spell(7, ["W"], "Sorcery")
    assert reduced_cost(blue, [red]) == 3
    assert reduced_cost(white, [red]) == 7
    assert reduced_cost(_spell(1, ["U"], "Instant"), [red]) == 1       # floored at the pip


def test_the_instant_sorcery_and_every_spell_reducers_read():
    inst = cost_reduction(_c("Goblin Electromancer", "Instant and sorcery spells you cast cost "
                             "{1} less to cast.", "Creature — Goblin Wizard"))
    assert inst == (1, ("types", ("Instant", "Sorcery")), False)
    assert reduced_cost(_spell(3, ["R"], "Instant"), [inst]) == 2
    assert reduced_cost(_spell(3, ["R"], "Creature — Goblin", True), [inst]) == 3
    helm = cost_reduction(_c("Helm of Awakening", "Spells cost {1} less to cast."))
    assert helm == (1, ("any",), False)


def test_a_creature_reducer_still_reads_as_before():
    assert cost_reduction(_c("Goblin Warchief", "Goblin spells you cast cost {1} less to cast.\n"
                             "Goblins you control have haste.", "Creature — Goblin")) == \
        (1, "Goblin", False)


def test_walking_ballista_reads_as_a_mana_sink_that_kills():
    assert window_profile(_c("Walking Ballista", "This creature enters with X +1/+1 counters on "
                             "it.\n{4}: Put a +1/+1 counter on this creature.\nRemove a +1/+1 "
                             "counter from this creature: It deals 1 damage to any target.",
                             "Artifact Creature — Construct"))["x_pinger"]


@requires_deck
def test_the_tithe_infinites_end_games_only_with_the_combo_channel_on():
    """PROVED BY SWITCHING IT OFF: b3-v1 assembles a Tithe line and kills by combo with the
    channel declared, and never without it."""
    t = json.loads(deck_file("heliod", "goldfish_targets.json").read_text())
    on = goldfish.run("heliod", branch="b3-v1", iterations=2000, seed=7, quiet=True, _band=False,
                      with_results=True, _targets_doc=dict(t, model_combos=True))["_results"]
    off = goldfish.run("heliod", branch="b3-v1", iterations=2000, seed=7, quiet=True, _band=False,
                       with_results=True, _targets_doc=dict(t, model_combos=False))["_results"]
    assert any(r["kill_by"] == "combo" for r in on)
    assert not any(r["kill_by"] == "combo" for r in off)
