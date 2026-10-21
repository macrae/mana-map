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


def test_a_cast_on_a_turn_with_the_commander_out_counts_and_one_without_does_not():
    """The fixture seat cast two distinct spells; call the second one a wheel and
    the first the commander, and the share follows the cleanup board. Proven by
    reading the START-of-turn board instead: a commander cast that turn then reads
    as absent and the count drops."""
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


def test_the_block_aggregates_and_validates_and_absent_means_absent():
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
