"""What the AI actually CAST from our seat, per card, in the record.

Every cast-count table in docs/gotchas-bench.md was made by an ad-hoc parser
after the run; the parser had the card name in hand at every `Add To Stack`
line and threw it away one function later. On 2026-09-10 sharknado's seat cast
Wheel of Fortune once and Windfall never in 60 games while DISCARDING Windfall
three times and Faithless Looting six — the log's own statement that the card
was held and not played — and that had to be counted by hand to be seen.

The block is MEASURED ONLY: names and counts, our seat, plus the seat's own
turns (Forge's `Game Outcome: Turn N` is the global turn halved, not a seat's
count). No engine set, no verdict — those are read at print time from an
authored file and must never be written into a record that has to re-derive
from itself and its logs.
"""

import json

import pytest

from manamap.sim import parse, validate_sim

LOG = """\
Mulligan: Ai(1)-mm-sharky has kept a hand of 7 cards.
Mulligan: Ai(2)-mm-rival has kept a hand of 6 cards.
Turn: Turn 1 (Ai(1)-mm-sharky)
Land: Ai(1)-mm-sharky played Island (5)
Turn: Turn 2 (Ai(2)-mm-rival)
Land: Ai(2)-mm-rival played Swamp (9)
Add To Stack: Ai(2)-mm-rival cast Sign in Blood targeting [Ai(2)-mm-rival]
Turn: Turn 3 (Ai(1)-mm-sharky)
Add To Stack: Ai(1)-mm-sharky cast Commit targeting [Swamp - Land (9)]
Add To Stack: Ai(1)-mm-sharky activated Lonely Sandbar
Discard: Ai(1)-mm-sharky discards Windfall (12).
Discard: Ai(1)-mm-sharky discards Windfall (12).
Discard: Ai(1)-mm-sharky discards Faithless Looting (14).
Turn: Turn 4 (Ai(2)-mm-rival)
Add To Stack: Ai(2)-mm-rival cast Sign in Blood targeting [Ai(2)-mm-rival]
Turn: Turn 5 (Ai(1)-mm-sharky)
Add To Stack: Ai(1)-mm-sharky cast Commit targeting [Swamp - Land (9)]
Game Outcome: Turn 3
Game Outcome: Ai(2)-mm-rival has won because all opponents have lost
Game Result: Game 1 ended in 1000 ms
"""
LABEL = {"Ai(1)-mm-sharky": "sharky", "Ai(2)-mm-rival": "rival"}


def _facts():
    games = parse.parse_games(LOG)
    assert len(games) == 1
    return [parse.game_facts(games[0])]


def test_the_parser_counts_by_card_and_own_turns():
    f = _facts()[0]
    ours = f["by_card"]["Ai(1)-mm-sharky"]
    assert ours["cast"] == {"Commit": 2}, "the front face, targets stripped"
    assert ours["activated"] == {"Lonely Sandbar": 1}, "cycling from hand is an activation"
    assert ours["discarded"] == {"Windfall": 2, "Faithless Looting": 1}, "the id tail is stripped"
    assert f["by_card"]["Ai(2)-mm-rival"]["cast"] == {"Sign in Blood": 2}
    # Own turns: 3 for us, 2 for the rival, against a global count of 5.
    assert f["per_seat"]["Ai(1)-mm-sharky"]["turns"] == 3
    assert f["per_seat"]["Ai(2)-mm-rival"]["turns"] == 2
    assert f["global_turn"] == 5


def test_the_roll_up_is_our_seat_only_and_compact_never_carries_it():
    facts = _facts()
    ec = parse.engine_casts(facts, LABEL, "sharky")
    assert ec == {"seat": "sharky", "games": 1, "turns": 3, "kept_hand_mean": 7.0,
                  "by_card": {"Commit": {"cast": 2, "activated": 0, "discarded": 0},
                              "Faithless Looting": {"cast": 0, "activated": 0, "discarded": 1},
                              "Lonely Sandbar": {"cast": 0, "activated": 1, "discarded": 0},
                              "Windfall": {"cast": 0, "activated": 0, "discarded": 2}}}
    assert "Sign in Blood" not in ec["by_card"], "the rival's casts are not ours"
    row = parse.compact(facts[0], LABEL)
    assert "by_card" not in row and "by_card" not in json.dumps(row), \
        "ninety-nine names times four seats must not ride into the tracked record"
    assert row["per_seat"]["sharky"]["turns"] == 3, "own turns DO ride: one scalar"


def test_the_validator_accepts_an_absent_block_and_rejects_a_malformed_one():
    """Absent means not measured — a record made before the block existed is
    not wrong. Present means it must be well-formed."""
    rec = {"run_id": "r", "slug": "sharky", "at": "2026-09-10", "engine": {},
           "seats": [{"slug": "sharky", "forge_name": "mm-sharky",
                      "decklist_sha256": "0" * 64, "commander": ["A"]}],
           "games_requested": 1, "games_completed": 1,
           "summary": {"wins": {"sharky": 1}, "draws": 0, "truncated": 0, "decided": 1, "win_rate": 1.0},
           "outcomes": [{"winner": "sharky", "round": 3, "global_turn": 5}],
           "analysis": {"games": 1, "seats": {"sharky": {"wins": 1, "win_rate": 1.0,
                                                          "win_rate_ci95": [0.2, 1.0]}}},
           "assumptions": ["SEEDED"], "seeds": [1], "jobs": 1}
    assert validate_sim.validate(rec, "sharky") == []
    rec["engine_casts"] = parse.engine_casts(_facts(), LABEL, "sharky")
    assert validate_sim.validate(rec, "sharky") == []
    rec["engine_casts"]["games"] = 7
    assert any("engine_casts.games" in e for e in validate_sim.validate(rec, "sharky"))
    rec["engine_casts"]["games"] = 1
    rec["engine_casts"]["by_card"]["Windfall"]["discarded"] = -1
    assert any("malformed" in e for e in validate_sim.validate(rec, "sharky"))
