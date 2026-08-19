"""The Forge log parser (S2) — deterministic over non-deterministic games.

Pinned on a real complete 2-seat game (tests/fixtures/forge/two-seat-one-game.log):
events split by kind; seats are learned from assignment/land lines; tokens are counted
two honest ways (resolutions that create, ids observed acting) and never by guessing;
combat damage is attributed to the controller who assigned the source; the final blow
is attributed through that map; and the aggregate's intervals behave.
"""

import json
from pathlib import Path

import pytest

from manamap.sim import parse, validate_sim

FIX = (Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game.log").read_text()
LABEL = {"Ai(1)-radagast": "radagast", "Ai(2)-edgar-vampires": "edgar-vampires"}


@pytest.fixture(scope="module")
def game():
    games = parse.parse_games(FIX)
    assert len(games) == 1
    return games[0]


def test_events_split_by_kind_and_seats_are_learned(game):
    kinds = {e["kind"] for e in game["events"]}
    assert {"land", "cast", "triggered", "resolve", "attack", "block", "noblock", "damage", "life", "zone"} <= kinds
    assert set(game["seats"]) == set(LABEL)
    # the owner map is learned from who assigned a permanent, never assumed
    assert game["owner"]["17"] == "Ai(1)-radagast"          # Fauna Shaman attacked for radagast
    assert game["owner"]["205"] == "Ai(2)-edgar-vampires"   # Vampire Token blocked for edgar


def test_per_game_facts_on_a_real_game(game):
    f = parse.game_facts(game)
    r, e = f["per_seat"]["Ai(1)-radagast"], f["per_seat"]["Ai(2)-edgar-vampires"]
    assert f["winner"] == "Ai(1)-radagast" and f["round"] == 9 and f["global_turn"] == 18
    assert r["combat_damage_dealt_to_players"] == e["combat_damage_taken"] == 42
    assert e["combat_damage_dealt_to_players"] == r["combat_damage_taken"] == 11
    assert e["eliminated_turn"] == 18 and e["eliminated_by"] == "Ai(1)-radagast"
    assert r["eliminated_turn"] is None
    assert r["lands"] == 7 and e["lands"] == 6 and r["casts"] == 10 and e["casts"] == 9
    assert r["first_attack_turn"] == 8 and e["first_attack_turn"] == 15


def test_tokens_are_counted_two_honest_ways(game):
    f = parse.game_facts(game)
    e = f["per_seat"]["Ai(2)-edgar-vampires"]
    # Edgar's eminence made tokens (two creation resolutions attributed via Activator:),
    # and exactly one Vampire Token was ever SEEN acting — it blocked.
    assert e["token_resolutions"] == 2
    assert e["tokens_observed"] == 1 and e["token_blockers"] == 1 and e["token_attackers"] == 0
    assert e["token_combat_damage_to_players"] == 0 and e["token_damage_share"] == 0.0
    r = f["per_seat"]["Ai(1)-radagast"]
    assert r["token_resolutions"] == 5 and r["tokens_observed"] == 0, (
        "Scute Swarm and friends resolved token makers; no radagast token ever attacked or blocked")


def test_the_activator_tag_with_a_trailing_comma_attributes_correctly():
    """The first version's seat pattern was \\S+ and swallowed the comma after an
    `Activator:` tag, so every token resolution fell to the active seat."""
    line = ("Resolve Stack: Eminence — create a 1/1 black Vampire creature token. "
            "[Card: Charismatic Conqueror (115), Activator: Ai(2)-edgar-vampires, SpellAbility: x]")
    g = parse._new_game(); g["active"] = "Ai(1)-radagast"
    ev = parse._event(line, g)
    assert ev["kind"] == "resolve" and ev["creates_token"] and ev["seat"] == "Ai(2)-edgar-vampires"


def test_damage_lines_distinguish_combat_noncombat_and_targets():
    g = parse._new_game()
    c = parse._event("Damage: Fauna Shaman (17) deals 2 combat damage to Ai(2)-edgar-vampires.", g)
    n = parse._event("Damage: Warleader's Call (104) deals 1 non-combat damage to Ai(3)-mm-yawgmoth-swarm.", g)
    p = parse._event("Damage: Vampire Token (205) deals 1 damage to Fauna Shaman (17).", g)
    d = parse._event("Damage: Ayara (230) deals 3 damage (Deathtouch) to Scute Swarm (9).", g)
    assert c["combat"] and c["to_player"] == "Ai(2)-edgar-vampires" and c["amount"] == 2
    assert n["noncombat"] and n["to_player"] == "Ai(3)-mm-yawgmoth-swarm"
    assert not p["combat"] and p["to_player"] is None and p["to_perm"] == "Fauna Shaman (17)"
    assert d["tag"] == "Deathtouch"


def test_assignment_lists_lose_their_separators():
    """Measured on a lifted board: attackers read ', Insect Token' and 'and Insect Token'."""
    got = parse._perms("Hornet Queen (53), Insect Token (473), Insect Token (474) and Insect Token (476)")
    assert [n for n, _ in got] == ["Hornet Queen", "Insect Token", "Insect Token", "Insect Token"]


def test_a_drain_kill_is_attributed_to_the_ability_controller_not_the_last_damage():
    """Measured on the pod run: Vito wins 9 of 20 on 7 combat damage a game — every one a
    drain — and damage-only attribution got all of them wrong."""
    text = ("Mulligan: Ai(1)-mm-a has kept a hand of 7 cards\nMulligan: Ai(2)-mm-vito has kept a hand of 7 cards\n"
            "Turn: Turn 5 (Ai(2)-mm-vito)\n"
            "Combat: Ai(1)-mm-a assigned Bear (9) to attack Ai(2)-mm-vito.\n"
            "Damage: Bear (9) deals 2 combat damage to Ai(2)-mm-vito.\nLife: Life: Ai(2)-mm-vito 40 > 38\n"
            "Resolve Stack: Whenever you gain life, each opponent loses that much life. [Player: Ai(2)-mm-vito]\n"
            "Life: Life: Ai(1)-mm-a 3 > -7\n"
            "Game Outcome: Turn 3\nGame Outcome: Ai(2)-mm-vito has won because all opponents have lost\n"
            "Game Outcome: Ai(1)-mm-a has lost because life total reached 0\n"
            "Game Result: Game 1 ended in 10 ms. Ai(2)-mm-vito has won!\n")
    f = parse.game_facts(parse.parse_games(text)[0])
    a = f["per_seat"]["Ai(1)-mm-a"]
    assert a["eliminated_by"] == "Ai(2)-mm-vito" and a["eliminated_how"] == "life loss"
    assert a["combat_damage_taken"] == 0, "a drain is not damage and never enters a damage total"


def test_aggregate_intervals_behave():
    facts = [parse.game_facts(parse.parse_games(FIX)[0])]
    agg = parse.aggregate(facts, "Ai(1)-radagast", LABEL)
    r = agg["seats"]["radagast"]
    assert r["wins"] == 1 and r["win_rate"] == 1.0 and r["win_rate_ci95"] == [0.207, 1.0]
    assert agg["round"] == {"mean": 9.0, "n": 1}, "no interval below two games"
    assert parse.wilson(0, 0) == (None, None)
    assert parse.wilson(0, 8) == (0.0, 0.324)
    m = parse.mean_ci([1, 2, 3, 4])
    assert m["mean"] == 2.5 and m["ci95"][0] < 2.5 < m["ci95"][1]
    assert any("tokens_observed" in l for l in agg["limits"])
    curve = agg["our_cumulative_combat_damage_by_round"]
    assert curve[-1]["mean"] == 42.0 and curve[0]["round"] == 1


def test_validate_sim_reproves_the_analysis_from_logs_and_catches_drift():
    facts, analysis = parse.analyze_logs([FIX], LABEL)
    rec = {"run_id": "x", "slug": "radagast", "at": "2026-08-19", "engine": {},
           "seats": [{"slug": "radagast", "forge_name": "radagast", "decklist_sha256": "a" * 64},
                     {"slug": "edgar-vampires", "forge_name": "edgar-vampires", "decklist_sha256": "b" * 64}],
           "games_requested": 1, "games_completed": 1,
           "summary": {"wins": {"radagast": 1, "edgar-vampires": 0}, "draws": 0},
           "outcomes": [{"winner": "radagast"}], "analysis": analysis,
           "assumptions": ["SAMPLED, NOT SEEDED"]}
    assert validate_sim.validate(rec, "radagast", [FIX]) == []
    bad = json.loads(json.dumps(rec)); bad["analysis"]["seats"]["radagast"]["wins"] = 0
    errs = validate_sim.validate(bad, "radagast", [FIX])
    assert any("disagree" in e for e in errs) and any("does not match what the logs derive" in e for e in errs)
    bad2 = json.loads(json.dumps(rec)); bad2["assumptions"] = ["seeded, honest"]
    assert any("SAMPLED" in e for e in validate_sim.validate(bad2, "radagast"))
    bad3 = json.loads(json.dumps(rec)); bad3["seats"][1]["decklist_sha256"] = "short"
    assert any("sha256" in e for e in validate_sim.validate(bad3, "radagast"))
