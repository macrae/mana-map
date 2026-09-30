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
    assert gs["end_of_turn_size_by_turn"]["1"] == 3 and gs["end_of_turn_size_by_turn"]["13"] == 5
    assert {"card": "Goblin Bombardment", "games": 1} in gs["held_at_end"]
    assert any(l.startswith("HAND FACTS") for l in agg["limits"])
    assert not any(l.startswith("Card advantage, tutoring and recursion are ABSENT") for l in agg["limits"])


CMC = {"Goblin Bombardment": 2, "Great Train Heist": 1, "Vandalblast": 1, "Grapeshot": 2,
       "Broadside Bombardiers": 3, "Past in Flames": 4, "Mountain": 0}


def test_castable_and_uncast_turns_are_counted_on_lands_alone():
    """Hand-checked on the fixture: goblin-storm's lands stood at 1 (T1), 2 (T3-T9), 3
    (T11-T13). Goblin Bombardment (2) sat in hand all game: castable at the end of six
    own turns, T3 through T13. Great Train Heist (1): all seven. The bug this guards is
    counting turns the seat could NOT have cast it — the land count is the gate."""
    text = (FIX_DIR / "two-seat-one-game-telemetry.log").read_text()
    facts, _ = parse.analyze_logs([text], LABEL, cmc=CMC)
    cards = facts[0]["per_seat"][GS]["hand"]["cards"]
    assert cards["Goblin Bombardment"] == {"turns_in_hand": 7, "castable_uncast": 6}
    assert cards["Great Train Heist"] == {"turns_in_hand": 7, "castable_uncast": 7}
    # a card whose mana value is unknown gets turns_in_hand and no castability claim
    assert "castable_uncast" not in cards["Dragon Fodder"]
    # the opponent's cards carry no castability either: no cmc, no claim
    assert all("castable_uncast" not in v for v in facts[0]["per_seat"][GA]["hand"]["cards"].values())


def test_engine_casts_carries_the_measured_columns_only_where_the_hand_was_logged():
    text = (FIX_DIR / "two-seat-one-game-telemetry.log").read_text()
    facts, _ = parse.analyze_logs([text], LABEL, cmc=CMC)
    ec = parse.engine_casts(facts, LABEL, "goblin-storm")
    gb = ec["by_card"]["Goblin Bombardment"]
    assert gb["cast"] == 0 and gb["in_hand_games"] == 1 and gb["castable_uncast"] == 6
    shipped = (FIX_DIR / "two-seat-one-game.log").read_text()
    lbl = {"Ai(1)-radagast": "radagast", "Ai(2)-edgar-vampires": "edgar-vampires"}
    facts2, _ = parse.analyze_logs([shipped], lbl, cmc={"Sol Ring": 1})
    ec2 = parse.engine_casts(facts2, lbl, "radagast")
    assert all(set(r) == {"cast", "activated", "discarded"} for r in ec2["by_card"].values()), \
        "a plain-formatter record must re-derive byte for byte"


def test_held_while_castable_is_measured_never_inferred():
    from manamap.sim import engine_casts as ec_mod
    text = (FIX_DIR / "two-seat-one-game-telemetry.log").read_text()
    facts, _ = parse.analyze_logs([text], LABEL, cmc=CMC)
    rec = {"engine_casts": parse.engine_casts(facts, LABEL, "goblin-storm")}
    held = ec_mod.held_while_castable(rec)
    names = [r["card"] for r in held]
    assert names[:2] == ["Great Train Heist", "Goblin Bombardment"]
    assert ec_mod.held_while_castable({"engine_casts": {"by_card": {"X": {"cast": 0, "activated": 0, "discarded": 3}}}}) == [], \
        "a discarded-but-never-seen-castable card is the OLD inference, not this list"
    assert ec_mod.held_while_castable({}) is None


def test_turns_on_the_battlefield_are_counted_from_the_arrivals_and_lands_are_not():
    """goblin-storm's Impact Tremors resolved on turn 3 and stayed; lands never count."""
    text = (FIX_DIR / "two-seat-one-game-telemetry.log").read_text()
    facts, _ = parse.analyze_logs([text], LABEL, cmc=CMC)
    cards = facts[0]["per_seat"][GS]["hand"]["cards"]
    assert "turns_on_battlefield" not in cards.get("Mountain", {})
    on_board = {n: v["turns_on_battlefield"] for n, v in cards.items() if "turns_on_battlefield" in v}
    assert on_board, "nothing resolved?"
    assert all(v >= 1 for v in on_board.values())
    ec = parse.engine_casts(facts, LABEL, "goblin-storm")
    assert any("turns_on_battlefield" in r for r in ec["by_card"].values())
