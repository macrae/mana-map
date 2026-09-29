"""The board series: an estimate of the board at the end of every turn, from
the resolve lines the bridge already reads — with every floor named.

Pinned on hand-built states (the rules) and on the two-seat fixture log (the
integration: the series at a cut equals the bridge's own reconstruction at
that cut, field for field, so two consumers of one function cannot drift).
"""

import json
import pathlib

import pytest

from manamap.sim import board_series as bs
from manamap.sim import bridge
from manamap.sim import parse as sim_parse

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game.log"


def _state(**over):
    st = bridge._seat_state()
    st.update(over)
    return st


def test_a_star_creature_is_a_body_and_not_power_and_a_treasure_is_not_a_body():
    """THE RULES, on a hand-built state. Re-introduce by mapping `*` to 0 and
    `power_unknown` reads 0; by counting a pt-less token as a body and
    `bodies` reads 3."""
    st = _state(
        perms={"a": {"name": "Lord of Extinction", "pt": "*/*", "tapped": False},
               "b": {"name": "Grizzly Bears", "pt": "2/2", "tapped": True},
               "c": {"name": "Sol Ring", "pt": None, "tapped": True},
               "d": {"name": "Rancor", "pt": None, "tapped": False, "attached_to": "Grizzly Bears"}},
        tokens={"t1": {"name": "Treasure Token", "pt": None, "tapped": False, "token": True},
                "t2": {"name": "Goblin Token", "pt": "1/1", "tapped": False, "token": True}},
        lands={"l1": {"name": "Mountain", "tapped": True}, "l2": {"name": "Forest", "tapped": False}},
        commander_zone="battlefield")
    got = bs.snapshot(st)
    assert got["bodies"] == 3, got                  # Lord, Bears, Goblin token
    assert got["printed_power"] == 3 and got["power_unknown"] == 1
    assert got["permanents"] == 8 and got["tokens"] == 2
    assert got["lands"] == 2 and got["open_lands"] == 1
    assert got["rocks_tapped"] == 1, "Sol Ring tapped for mana; the aura is not a rock"
    assert got["commander_on_battlefield"] is True


def test_commander_uptime_counts_own_turns_after_the_first_resolve():
    rows = [{"own_turn": 1, "commander_on_battlefield": False},
            {"own_turn": 2, "commander_on_battlefield": True},
            {"own_turn": 3, "commander_on_battlefield": True},
            {"own_turn": 4, "commander_on_battlefield": False},
            {"own_turn": 5, "commander_on_battlefield": True}]
    assert bs.commander_uptime(rows) == 0.75
    assert bs.commander_uptime([{"own_turn": 1, "commander_on_battlefield": False}]) is None


def test_the_series_at_a_cut_equals_the_bridges_own_reconstruction():
    """Two consumers of one function. For every own turn of every seat in the
    fixture game, the series row equals `snapshot(reconstruct(...))` at the
    start of that turn's cleanup — re-introduce by snapshotting before the
    untap reset and `open_lands` disagrees."""
    game = sim_parse.parse_games(FIXTURE.read_text(encoding="utf-8", errors="replace"))[0]
    cmd = {}
    ser = bs.series(game, cmd)
    checked = 0
    for seat, rows in ser.items():
        assert rows, seat
        for r in rows:
            states, *_ = bridge.reconstruct(game, r["global_turn"], "ending", "cleanup", cmd)
            want = bs.snapshot(states[seat])
            got = {k: r[k] for k in want}
            assert got == want, (seat, r["global_turn"])
            assert r["bodies"] <= r["permanents"]
            checked += 1
    assert checked >= 10
    # own turns alternate between the two seats and never share a global turn
    turns = bs.own_turns(game)
    a, b = (turns[s] for s in game["seats"])
    assert not set(a) & set(b) and len(a) + len(b) >= 10


def test_the_record_block_is_our_seat_only_with_means_and_floors_named():
    text = FIXTURE.read_text(encoding="utf-8", errors="replace")
    game = sim_parse.parse_games(text)[0]
    ours = game["seats"][0].split("-", 1)[1]           # "radagast" from "Ai(1)-radagast"
    block = bs.from_logs([text], {}, ours)
    assert block["seat"] == ours and block["games"] == 1 and block["since"] == bs.SINCE
    assert len(block["per_game"]) == 1 and block["per_game"][0]["turns"] >= 5
    # one game: no interval can be formed, so the aggregate is honestly empty
    assert all(v == {} for v in block["by_turn"].values())
    assert block["commander_uptime"]["mean_ci"] is None
    assert any("ESTIMATE" in l for l in block["limits"])
    assert bs.validate(block, 1) == []
    assert bs.validate(block, 2) and "games" in bs.validate(block, 2)[0]
    twice = bs.from_logs([text, text], {}, ours)
    assert twice["games"] == 2 and twice["by_turn"]["bodies"], "two games form an interval"
    assert all(v["n"] == 2 for v in twice["by_turn"]["bodies"].values())


def test_a_double_faced_commander_is_matched_by_its_front_face():
    """FOUND ON THE FLEET SWEEP: heliod's uptime read None on every run.
    `record_commanders` carries `Front // Back`; the resolve line prints the
    front face. Re-introduce by taking the whole name and this reads None."""
    text = FIXTURE.read_text(encoding="utf-8", errors="replace")
    game = sim_parse.parse_games(text)[0]
    label = game["seats"][0]
    ours = label.split("-", 1)[1]
    # find a creature our seat actually resolved, and call it the commander
    cast = next((ev["what"] for ev in game["events"]
                 if ev.get("kind") == "cast" and ev.get("seat") == label), None)
    assert cast, "the fixture seat cast nothing"
    plain = bs.from_logs([text], {label: {cast}}, ours)
    dfc = bs.from_logs([text], {label: {f"{cast} // Some Back Face"}}, ours)
    assert plain["per_game"][0]["commander_uptime"] is not None
    assert dfc["per_game"][0]["commander_uptime"] == plain["per_game"][0]["commander_uptime"]
