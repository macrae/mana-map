"""The answers block: answers, counterspells and protection our seat cast, per game
(2026-10-21). Unit on synthetic events and a hand-written roles map; regression on
sharknado's committed logs."""

import pathlib

import pytest

from manamap.sim import answers as ix
from manamap.sim import parse as sim_parse

from conftest import requires_deck

ROLES = {"Swords to Plowshares": ["removal:spot"], "Negate": ["counterspell"],
         "Brallin, Skyshark Rider": ["removal:damage", "threat:body"],
         "Collective Defiance": ["removal:damage"],
         "Lightning Greaves": ["buff:attached", "protection:self"]}
DOC = {"cards": [
    {"name": "Swords to Plowshares", "type_line": "Instant"},
    {"name": "Negate", "type_line": "Instant"},
    {"name": "Brallin, Skyshark Rider", "type_line": "Legendary Creature — Human Shaman"},
    {"name": "Collective Defiance", "type_line": "Sorcery"},
    {"name": "Lightning Greaves", "type_line": "Artifact — Equipment"},
    {"name": "Grunt", "type_line": "Creature — Goblin"}]}


def test_classes_come_from_the_roles_and_a_creatures_ping_is_not_an_answer():
    """Proven by dropping the noncreature condition: Brallin then reads as an answer."""
    cls = ix.classes(DOC, ROLES)
    assert cls["Swords to Plowshares"] == {"answer"}
    assert cls["Negate"] == {"answer", "counter"}
    assert cls["Collective Defiance"] == {"answer"}
    assert cls["Lightning Greaves"] == {"protection"}
    assert "Brallin, Skyshark Rider" not in cls and "Grunt" not in cls


def test_per_game_counts_our_casts_and_activations_only():
    L, O = "Ai(1)-mm-x", "Ai(2)-mm-y"
    game = {"seats": [L, O], "events": [
        {"kind": "cast", "seat": L, "what": "Swords to Plowshares (12)"},
        {"kind": "cast", "seat": L, "what": "Negate"},
        {"kind": "activated", "seat": L, "what": "Lightning Greaves"},
        {"kind": "cast", "seat": O, "what": "Swords to Plowshares"},
        {"kind": "cast", "seat": L, "what": "Grunt"},
    ]}
    row = ix.per_game(game, L, ix.classes(DOC, ROLES))
    assert row == {"answers_cast": 2, "counters_cast": 1, "protection_cast": 1}


def test_the_block_validates_and_absent_means_absent():
    block = ix.aggregate([{"answers_cast": 2, "counters_cast": 1, "protection_cast": 0},
                          {"answers_cast": 0, "counters_cast": 0, "protection_cast": 3}],
                         "x", ix.classes(DOC, ROLES))
    assert ix.validate(block, 2) == []
    assert ix.validate(block, 3)
    block["per_game"][0]["counters_cast"] = 5
    assert any("more counterspells" in e for e in ix.validate(block, 2))
    assert ix.from_logs([], "x", {"cards": [DOC["cards"][-1]]}, ROLES) is None, "no class, no block"
    assert ix.from_logs([], "x", DOC, roles=None) is not None or ix._roles() is None


@requires_deck
def test_sharknados_logs_yield_an_interaction_block():
    from manamap.config import DECKS_DIR
    from manamap.pilot.common import load_deck_cards
    logs = sorted((DECKS_DIR / "sharknado" / "sim" / "logs").glob("*n200*tl9473cf35*/part-00.log"))
    if not logs or ix._roles() is None:
        pytest.skip("no telemetry log or no card_roles.json on disk")
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    games = sim_parse.parse_games(text)
    block = ix.from_logs([text], "sharknado", load_deck_cards("sharknado"))
    assert block and block["games"] == len(games) >= 5
    assert "Swords to Plowshares" in block["cards"]
    assert block["answers_cast"]["total"] > 0 and block["protection_cast"]["total"] > 0
    assert ix.validate(block, len(games)) == []
