"""The wheel timing block: when our seat's wheels fired and onto what, from a run's
logs (2026-10-21). Unit on the two-seat fixture with a card the seat really cast
standing in for a wheel; regression on sharknado's committed telemetry logs."""

import json
import pathlib

import pytest

from manamap.sim import board_series as bs
from manamap.sim import parse as sim_parse
from manamap.sim import wheel_timing as wt

from conftest import requires_deck

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game.log"


def _fixture_game():
    text = FIXTURE.read_text(encoding="utf-8", errors="replace")
    game = sim_parse.parse_games(text)[0]
    label = game["seats"][0]
    return text, game, label, label.split("-", 1)[1]


def test_wheel_names_come_from_the_goldfishs_own_readers():
    doc = {"cards": [
        {"name": "Windfall", "type_line": "Sorcery", "cmc": 3, "mana_cost": "{2}{U}",
         "oracle_text": "Each player discards their hand, then draws cards equal to the greatest number of cards a player discarded this way."},
        {"name": "Grunt", "type_line": "Creature — Human", "cmc": 2, "mana_cost": "{1}{R}"},
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0},
    ]}
    assert wt.wheel_names(doc) == {"Windfall"}
    assert wt.from_logs([], {}, "x", {"cards": doc["cards"][1:]}) is None, "no wheel, no block"


def test_a_cast_on_a_turn_with_the_commander_out_counts_and_one_without_does_not(monkeypatch):
    """The fixture seat cast two distinct spells; call the second one a wheel and
    the first the commander, and the share follows the cleanup board. Proven by
    reading the START-of-turn board instead: a commander cast that turn then reads
    as absent and the count drops."""
    monkeypatch.setattr(wt, "_wheeled", lambda *a: True)   # the stand-in never discards
    text, game, label, ours = _fixture_game()
    casts = []
    for ev in game["events"]:
        if ev.get("kind") == "cast" and ev.get("seat") == label and ev["what"] not in casts:
            casts.append(ev["what"])
    assert len(casts) >= 2
    cmdr, wheel = casts[0], casts[1]
    row = wt.per_game(game, {label: cmdr}, label, {wt._clean(wheel)})
    assert row["wheels_cast"] >= 1 and row["first_wheel_turn"] is not None
    assert 0 <= row["wheels_with_commander"] <= row["wheels_cast"]
    # the same count by hand, off the series this module reads
    rows = {r["global_turn"]: r for r in bs.series(game, {label: cmdr})[label]}
    by_hand = sum(1 for ev in game["events"]
                  if ev.get("kind") == "cast" and ev.get("seat") == label
                  and wt._clean(ev["what"]) == wt._clean(wheel)
                  and rows.get(ev["turn"], {}).get("commander_on_battlefield"))
    assert row["wheels_with_commander"] == by_hand
    # no commander named: nothing can be "with a commander"
    assert wt.per_game(game, {}, label, {wt._clean(wheel)})["wheels_with_commander"] == 0


def test_the_block_aggregates_and_validates_and_absent_means_absent(monkeypatch):
    monkeypatch.setattr(wt, "_wheeled", lambda *a: True)   # the stand-in never discards
    text, game, label, ours = _fixture_game()
    casts = [ev["what"] for ev in game["events"] if ev.get("kind") == "cast" and ev.get("seat") == label]
    wheel = wt._clean(casts[-1])
    doc = {"cards": [{"name": wheel, "type_line": "Sorcery", "cmc": 3, "mana_cost": "{2}{U}",
                      "oracle_text": "Each player discards their hand, then draws seven cards."}]}
    block = wt.from_logs([text, text], {label: {casts[0]}}, ours, doc)
    assert block and block["games"] == 2 and block["wheel_names"] == [wheel]
    assert block["wheels_cast"]["total"] == 2 * len([c for c in casts if wt._clean(c) == wheel])
    assert block["wheels_with_commander"]["n"] == block["wheels_cast"]["total"]
    assert wt.validate(block, 2) == []
    assert wt.validate(block, 3)
    bad = json.loads(json.dumps(block))
    bad["per_game"][0]["wheels_with_commander"] = 99
    assert any("more wheels with a commander" in e for e in wt.validate(bad, 2))
    assert any("kept seven is NOT in the log" in l for l in block["limits"])


@requires_deck
def test_sharknados_telemetry_logs_yield_wheel_timing_for_both_partners():
    """The committed V6 run's logs: the deck's wheels are read by name and both
    partners count toward "a commander was out"."""
    from manamap.config import DECKS_DIR
    from manamap.pilot.common import load_deck_cards
    logs = sorted((DECKS_DIR / "sharknado" / "sim" / "logs").glob("*n200*tl9473cf35*/part-00.log"))
    if not logs:
        pytest.skip("no telemetry log on disk for sharknado")
    text = logs[0].read_text(encoding="utf-8", errors="replace")
    games = sim_parse.parse_games(text)
    label = next(s for s in games[0]["seats"] if s.endswith("-sharknado"))
    both = {label: {"Shabraz, the Skyshark", "Brallin, Skyshark Rider"}}
    one = {label: {"Brallin, Skyshark Rider"}}
    doc = load_deck_cards("sharknado")
    block = wt.from_logs([text], both, "sharknado", doc)
    assert block and block["games"] == len(games) >= 5
    assert "Windfall" in block["wheel_names"]
    assert block["wheels_cast"]["total"] > 0
    assert wt.validate(block, len(games)) == []
    brallin_only = wt.from_logs([text], one, "sharknado", doc)
    assert block["wheels_with_commander"]["k"] >= brallin_only["wheels_with_commander"]["k"], (
        "a commander out means EITHER partner; reading Brallin alone can only lose counts")


def test_a_split_card_counts_only_the_face_that_wheels():
    """Forge logs `cast Commit` and `cast Memory` as two names; Commit (a bounce)
    was read as a wheel on the eight-game test of 2026-10-21. Only the face whose
    own text wheels is a wheel name. Proven by adding the front face unconditionally."""
    doc = {"cards": [{
        "name": "Commit // Memory", "type_line": "Instant // Sorcery", "cmc": 6,
        "mana_cost": "{3}{U} // {4}{U}{U}",
        "oracle_text": "Put target spell or nonland permanent into its owner's library second from the top.\n//\nEach player shuffles their hand and graveyard into their library, then draws seven cards.",
        "card_faces": [
            {"name": "Commit", "oracle_text": "Put target spell or nonland permanent into its owner's library second from the top."},
            {"name": "Memory", "oracle_text": "Each player shuffles their hand and graveyard into their library, then draws seven cards."}]}]}
    names = wt.wheel_names(doc)
    assert "Memory" in names and "Commit" not in names, names


def test_a_modal_wheel_cast_for_its_burn_is_not_a_wheel():
    """Collective Defiance cast seven times at olivers-table (2026-10-21) and never
    chose the wheel mode. Proven by deleting the discard requirement: the burn cast
    then counts and `wheels_cast` reads 2."""
    L = "Ai(1)-mm-x"
    game = {"seats": [L], "events": [
        {"kind": "cast", "seat": L, "what": "Collective Defiance", "turn": 3},
        {"kind": "damage", "turn": 3},
        {"kind": "resolve", "seat": L, "text": "Collective Defiance (8) - Choose one or more", "turn": 3},
        {"kind": "discard", "seat": L, "card": "Grunt", "turn": 3},
        {"kind": "cast", "seat": L, "what": "Collective Defiance", "turn": 5},
        {"kind": "discard", "seat": L, "card": "Grunt", "turn": 5},
        {"kind": "resolve", "seat": L, "text": "Collective Defiance (8) - Choose one or more", "turn": 5},
    ], "turns_by_seat": {}, "outcome": {}}
    from unittest import mock
    with mock.patch.object(wt.board_series, "series", return_value={L: [
            {"global_turn": 3, "own_turn": 2, "commander_on_battlefield": False},
            {"global_turn": 5, "own_turn": 3, "commander_on_battlefield": True}]}):
        row = wt.per_game(game, {}, L, {"Collective Defiance"})
    assert row["wheels_cast"] == 1 and row["first_wheel_turn"] == 3
    assert row["wheels_with_commander"] == 1
    # a shuffle wheel logs zone moves, no discard line (Echo of Eons, 2026-10-21)
    shuffle = {"seats": [L], "events": [
        {"kind": "cast", "seat": L, "what": "Echo of Eons", "turn": 5},
        {"kind": "zone", "card": "Grunt", "from": "Hand", "to": "Library", "owner": L, "turn": 5},
        {"kind": "resolve", "seat": L, "text": "Echo of Eons (46) - Each player", "turn": 5}]}
    with mock.patch.object(wt.board_series, "series", return_value={L: [
            {"global_turn": 5, "own_turn": 3, "commander_on_battlefield": True}]}):
        assert wt.per_game(shuffle, {}, L, {"Echo of Eons"})["wheels_cast"] == 1


def test_an_instant_wheel_on_an_opponents_turn_reads_our_last_board(monkeypatch):
    """Until 2026-10-25 only our own turns were looked up, so Tolarian Winds cast on an
    opponent's turn read as "no commander" with Brallin out on both sides of it (run
    caeeec41, one of the three misses behind the 86% seat-audit fail). Proven by
    reverting `_board_at` to `own.get(t)`: the count drops to 0."""
    monkeypatch.setattr(wt, "_wheeled", lambda *a: True)
    monkeypatch.setattr(wt.board_series, "series", lambda game, cmd: {"1-us": [
        {"global_turn": 1, "own_turn": 1, "commander_on_battlefield": True},
        {"global_turn": 3, "own_turn": 2, "commander_on_battlefield": True}]})
    game = {"events": [{"kind": "cast", "seat": "1-us", "what": "Tolarian Winds", "turn": 2}]}
    row = wt.per_game(game, {}, "1-us", {"Tolarian Winds"})
    assert row == {"first_wheel_turn": 1, "wheels_cast": 1, "wheels_with_commander": 1}
    assert wt._board_at({}, 2) is None, "a wheel before our first turn has no board"
