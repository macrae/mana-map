"""The hand, read exactly from the telemetry patch's zone lines — and absent, never
zero, from a shipped-formatter log.

Pinned on a real two-seat game played under the patched jar
(tests/fixtures/forge/two-seat-one-game-telemetry.log, the spike's control game) whose
hand-by-hand read was done by hand first: goblin-storm kept five after two mulligans,
held Goblin Bombardment and Great Train Heist for all fourteen turns, and missed four
land drops with no land in hand.
"""
from pathlib import Path

import pytest

from manamap.sim import parse

FIX_DIR = Path(__file__).parent / "fixtures" / "forge"
LABEL = {"Ai(1)-mm-goblin-storm": "goblin-storm", "Ai(2)-mm-giada-angels": "giada-angels"}
GS, GA = "Ai(1)-mm-goblin-storm", "Ai(2)-mm-giada-angels"


@pytest.fixture(scope="module")
def patched():
    text = (FIX_DIR / "two-seat-one-game-telemetry.log").read_text()
    facts, agg = parse.analyze_logs([text], LABEL)
    return facts, agg


def test_a_shipped_log_has_no_hand_key_anywhere():
    """Absent means absent: the old fixture carries no Hand line, so no seat gets a
    `hand` and the aggregate has no block — every record on disk re-analyses to the
    same bytes."""
    text = (FIX_DIR / "two-seat-one-game.log").read_text()
    facts, agg = parse.analyze_logs([text], {"Ai(1)-radagast": "radagast", "Ai(2)-edgar-vampires": "edgar-vampires"})
    assert facts and all("hand" not in p for f in facts for p in f["per_seat"].values())
    assert all("hand" not in seat for seat in agg["seats"].values())
    assert any(l.startswith("Card advantage, tutoring and recursion are ABSENT") for l in agg["limits"])


def test_library_to_hand_counts_from_turn_one_not_the_deal(patched):
    """The bug this guards: counting the opening deal and two mulligans (7 + 7 + 6
    cards dealt to goblin-storm before turn 1) as draws."""
    facts, _ = patched
    h = facts[0]["per_seat"]
    assert h[GS]["hand"]["library_to_hand"] == 6      # seven own turns, no draw on the play
    assert h[GA]["hand"]["library_to_hand"] == 7


def test_the_hand_is_exact_at_the_end_of_every_own_turn(patched):
    facts, _ = patched
    gs = facts[0]["per_seat"][GS]["hand"]
    assert gs["end_of_turn_size"] == {1: 3, 3: 2, 5: 3, 7: 4, 9: 5, 11: 4, 13: 5}
    assert gs["empty_own_turns"] == 0
    assert gs["own_turns_without_land_drop"] == 4          # T5, T7, T9, T13
    assert gs["missed_land_drops_with_land_in_hand"] == 0  # no land in hand on any of them
    held = {c["card"] for c in gs["hand_at_end"]}
    assert {"Goblin Bombardment", "Great Train Heist"} <= held, "held all game, exactly — not inferred"
    assert all(c["since"] == 0 for c in gs["hand_at_end"] if c["card"] in ("Goblin Bombardment", "Great Train Heist")), \
        "kept in the opening hand"


def test_a_missed_drop_with_a_land_in_hand_is_counted_and_the_predicate_is_a_floor():
    """Built events: seat A holds a card some seat played as a land and passes without
    a drop -> counted; seat B holds a land NOBODY ever played -> invisible (the floor)."""
    A, B = "Ai(1)-a", "Ai(2)-b"
    ev = [
        {"kind": "zone", "card": "Swamp", "id": "1", "to": "Hand", "from": "Library", "owner": A, "turn": 0, "active": None},
        {"kind": "zone", "card": "Lonely Sandbar", "id": "2", "to": "Hand", "from": "Library", "owner": B, "turn": 0, "active": None},
        {"kind": "land", "seat": B, "card": "Swamp", "id": "9", "turn": 1, "active": A},   # somebody played a Swamp
        {"kind": "zone", "card": "Bolt", "id": "3", "to": "Hand", "from": "Library", "owner": A, "turn": 1, "active": A},
        {"kind": "zone", "card": "Bolt", "id": "4", "to": "Hand", "from": "Library", "owner": B, "turn": 2, "active": B},
    ]
    out = parse.hand_facts(ev, [A, B])
    assert out[A]["missed_land_drops_with_land_in_hand"] == 1 and out[A]["own_turns_without_land_drop"] == 1
    assert out[B]["missed_land_drops_with_land_in_hand"] == 0 and out[B]["own_turns_without_land_drop"] == 1
    assert out[A]["library_to_hand"] == 1 and out[B]["library_to_hand"] == 1


def test_the_aggregate_carries_the_block_and_the_limit_only_when_a_game_did(patched):
    _, agg = patched
    gs = agg["seats"]["goblin-storm"]["hand"]
    assert gs["games"] == 1 and gs["library_to_hand"]["mean"] == 6
    assert gs["end_of_turn_size_by_turn"][1] == 3 and gs["end_of_turn_size_by_turn"][13] == 5
    assert {"card": "Goblin Bombardment", "games": 1} in gs["held_at_end"]
    assert any(l.startswith("HAND FACTS") for l in agg["limits"])
    assert not any(l.startswith("Card advantage, tutoring and recursion are ABSENT") for l in agg["limits"])
