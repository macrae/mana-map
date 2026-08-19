"""The v2 vocabulary (`pilot/game_state.py`) and the bridge (`sim/bridge.py`): a Forge
game lifted at a cut into a `game_state` v2 scenario the resolve loop can take.

Pinned on the real complete 2-seat game: lands are exact (name, id, tapped since the
controller's last untap); cast permanents enter when their resolve line prints and leave
by id; a face-down Morph becomes the card it was; a token exists from its first use;
a commander's logged exit reads as `command`; hand is an estimate and says so; cutting
past the game's end says so; and a lifted scenario fails the preflight exactly until a
question and a stack/action are posed.
"""

import json
from pathlib import Path

import pytest

from manamap.pilot import game_state, scenario_facts, validate_stack
from manamap.sim import bridge, parse

FIX = (Path(__file__).parent / "fixtures" / "forge" / "two-seat-one-game.log").read_text()
CMD = {"Ai(1)-radagast": "Radagast of Rhosgobel", "Ai(2)-edgar-vampires": "Edgar Markov"}


@pytest.fixture(scope="module")
def game():
    return parse.parse_games(FIX)[0]


def _state(game, turn, step):
    ph, st = bridge.resolve_cut(step)
    seats, notes, active, _, _ = bridge.reconstruct(game, turn, ph, st, CMD)
    return seats, notes, active


# ── game_state vocabulary ───────────────────────────────────────────────────

def test_v2_form_check_catches_what_the_spec_names():
    good = {"version": 2, "turn": 4, "active_seat": "you", "phase": "combat", "step": "declare blockers",
            "priority": "seat-2",
            "seats": [{"seat": "you", "life": 40, "board": ["Forest (untapped)"], "hand": ["Craterhoof Behemoth"]},
                      {"seat": "seat-2", "life": 31, "board": [], "hand": {"unknown": 4}}],
            "stack": [], "actions": [{"seat": "you", "kind": "attack", "attackers": []}], "question": "?"}
    assert game_state.validate_v2(good) == []
    bad = json.loads(json.dumps(good))
    bad["phase"] = "main 1"; bad["step"] = "blocks"; bad["actions"][0]["kind"] = "trigger"
    bad["seats"][1]["hand"] = "four cards"; bad["seats"][0]["seat"] = "me"
    errs = game_state.validate_v2(bad)
    assert any("phase" in e for e in errs) and any("step" in e for e in errs)
    assert any("actions[0].kind" in e for e in errs), "triggers are never actions"
    assert any("hand" in e for e in errs) and any('"you"' in e for e in errs)
    empty = dict(good, stack=[], actions=[])
    assert any("nothing to resolve" in e for e in game_state.validate_v2(empty))


def test_entry_helpers_read_both_forms():
    assert game_state.entry_name("Fume Spitter (1/1) — already sacrificed to pay the cost") == "Fume Spitter"
    assert game_state.entry_name({"name": "Scute Swarm", "pt": "1/1"}) == "Scute Swarm"
    assert game_state.entry_is_token("Vampire Token (1/1)") and game_state.entry_is_token({"name": "X", "token": True})
    assert game_state.entry_annotations("A — already sacrificed to pay the cost of the ability now on the stack") == [
        "already sacrificed to pay the cost of the ability now on the stack"]


def test_validate_stack_and_scenario_facts_take_v2():
    doc = {"id": "099", "slug": "x", "deck": "x", "title": "t",
           "scenario": {"version": 2, "turn": 6, "active_seat": "you", "phase": "precombat main", "step": None,
                        "seats": [{"seat": "you", "life": 40, "board": [
                                       {"name": "Scute Swarm", "pt": "1/1", "token": False},
                                       {"name": "Insect Token", "pt": "1/1", "token": True},
                                       {"name": "Forest", "type": "Land"},
                                       {"name": "Fume Spitter", "pt": "1/1",
                                        "annotations": ["already sacrificed to pay the cost of the ability now on the stack"]}]},
                                  {"seat": "seat-2", "life": 12, "board": ["two 2/2s"], "archetype": "aggro"}],
                        "stack": [{"pos": 0, "item": "Craterhoof"}], "actions": [], "question": "lethal?"}}
    errs, _ = validate_stack.validate_preflight(doc)
    assert errs == []
    b = scenario_facts.board_bodies(scenario_facts.your_board(doc["scenario"]))
    assert b["creature_bodies"] == ["Scute Swarm", "Insect Token"]
    assert b["lands"] == ["Forest"] and b["spent_paying_a_cost"] == ["Fume Spitter"]
    opps = scenario_facts.opponents_of(doc["scenario"])
    assert opps == [{"seat": "seat-2", "life": 12, "board": ["two 2/2s"], "archetype": "aggro"}]
    assert game_state.our_named_cards(doc["scenario"]) == ["Scute Swarm", "Forest", "Fume Spitter"], "tokens excluded"


# ── the bridge on a real game ───────────────────────────────────────────────

def test_lands_are_exact_and_tapped_since_the_controllers_last_untap(game):
    seats, _, active = _state(game, 14, "declare blockers")
    r, e = seats["Ai(1)-radagast"], seats["Ai(2)-edgar-vampires"]
    assert active == "Ai(1)-radagast"
    assert len(r["lands"]) == 6 and not any(l["tapped"] for l in r["lands"].values()), \
        "radagast untapped at its own untap step and has not tapped a land before blockers"
    assert len(e["lands"]) == 6 and sum(l["tapped"] for l in e["lands"].values()) == 5, \
        "edgar's lands stay tapped from his own turn until HIS next untap"


def test_cast_permanents_enter_on_resolve_leave_by_id_and_morph_unmorphs(game):
    seats, _, _ = _state(game, 8, "upkeep")
    names = {p["name"] for p in seats["Ai(1)-radagast"]["perms"].values()}
    assert names == {"Fauna Shaman", "Morph"}, "face-down creature is logged as Morph"
    seats, _, _ = _state(game, 8, "precombat main")
    names = {p["name"] for p in seats["Ai(1)-radagast"]["perms"].values()}
    assert "Nantuko Vigilante" in names and "Morph" not in names, \
        "the AI turned it face up during its turn-8 upkeep — before precombat main"
    seats, _, _ = _state(game, 15, "precombat main")
    assert "Fauna Shaman" not in {p["name"] for p in seats["Ai(1)-radagast"]["perms"].values()}, \
        "died in combat on turn 14 and left by id"
    assert "Cruel Celebrant" not in {p["name"] for p in seats["Ai(2)-edgar-vampires"]["perms"].values()}


def test_a_token_exists_from_its_first_use_and_the_commander_exit_reads_as_command(game):
    seats, _, _ = _state(game, 14, "declare blockers")
    assert seats["Ai(2)-edgar-vampires"]["tokens"] == {}, "the Vampire Token has not acted yet"
    seats, notes, _ = _state(game, 15, "precombat main")
    assert "205" in seats["Ai(2)-edgar-vampires"]["tokens"], "it blocked on turn 14"
    seats, _, _ = _state(game, 14, "declare blockers")
    assert seats["Ai(1)-radagast"]["commander_zone"] == "battlefield" and \
        seats["Ai(1)-radagast"]["commander_casts"] == 1


def test_hand_is_an_estimate_and_a_cut_past_the_end_says_so(game):
    seats, notes, _ = _state(game, 8, "precombat main")
    r = seats["Ai(1)-radagast"]
    assert r["kept"] == 7 and r["draw_steps"] == 4 and r["lands_n"] == 3 and r["cast_n"] == 2
    seats, notes, _ = _state(game, 99, "precombat main")
    assert any("did not reach turn 99" in n for n in notes)


def test_lift_writes_a_v2_scenario_that_needs_only_a_question(tmp_path, monkeypatch):
    decks = tmp_path / "decks"; base = decks / "radagast"
    (base / "sim" / "logs" / "run-x").mkdir(parents=True)
    (base / "sim" / "logs" / "run-x" / "part-00.log").write_text(FIX)
    (base / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    (decks / "edgar-vampires").mkdir(); (decks / "edgar-vampires" / "decklist.txt").write_text("1 Edgar Markov *CMDR*\n1 Swamp\n")
    (base / "sim" / "run-x.json").write_text(json.dumps({
        "run_id": "run-x", "slug": "radagast",
        "seats": [{"slug": "radagast", "forge_name": "radagast", "decklist_sha256": "a" * 64},
                  {"slug": "edgar-vampires", "forge_name": "edgar-vampires", "decklist_sha256": "b" * 64}],
        "outcomes": [{"winner": "radagast", "round": 9, "global_turn": 18, "log": "part-00.log",
                      "seed": 42, "game_in_job": 1}]}))
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    monkeypatch.setattr("manamap.sim.forge.DECKS_DIR", decks)
    out, doc = bridge.lift("radagast", "run-x", 1, 14, "declare blockers")
    assert out.parent.name == "scenarios" and out.exists()
    sc = doc["scenario"]
    assert sc["version"] == 2 and sc["source"]["replay"] == "-n 1 -s 42"
    assert [s["seat"] for s in sc["seats"]] == ["you", "seat-2"]
    you = sc["seats"][0]
    assert you["commander"] == {"name": "Radagast of Rhosgobel", "zone": "battlefield", "casts": 1}
    assert you["hand"]["estimate"] is True and you["mana"]["open"] == 6
    assert sc["question"] == "" and sc["stack"] == [] and sc["actions"] == []
    errs, _ = validate_stack.validate_preflight(doc)
    assert any("question is empty" in e for e in errs) and any("nothing to resolve" in e for e in errs)
    doc["scenario"]["question"] = "Does the block kill Fauna Shaman?"
    doc["scenario"]["actions"] = [{"seat": "seat-2", "kind": "block", "blocks": []}]
    assert validate_stack.validate_preflight(doc)[0] == []
    out2, doc2 = bridge.lift("radagast", "run-x", 1, 14, "declare blockers", to_stack=True)
    assert out2.parent.name == "stacks" and out2.name.startswith("001-sim-g1-t14-") and doc2["id"] == "001"
