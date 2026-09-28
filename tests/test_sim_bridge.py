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

from manamap import config
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


def test_a_token_exists_from_its_CREATION_and_the_commander_exit_reads_as_command(game):
    """THIS TEST ASSERTED THE BUG. It read "a token exists from its first USE" and
    required `tokens == {}` on turn 14 because the Vampire Token "has not acted
    yet" — while two of them had been on the battlefield since turn 5.

    MEASURED across one 100-game run, 804 of our precombat mains: the lift listed
    187 tokens where 1843 were alive. It saw 10%. On a token deck the board it
    described was not a simplification, it was a different board — ">= 6 other
    bodies beside the commander" read 2.0% of turns where the true figure is
    36.5%, and a conclusion was drawn from the wrong one.

    The contract now: a token exists when it is CREATED, unbound until it acts,
    and binding an id must CONSUME the placeholder rather than add a second copy.
    """
    seats, _, _ = _state(game, 14, "declare blockers")
    toks = seats["Ai(2)-edgar-vampires"]["tokens"]
    assert len(toks) == 2 and {t["name"] for t in toks.values()} == {"Vampire Token"}, \
        f"two Vampire Tokens were made by turn 14 and must be on the board: {toks}"
    assert all(t["id"] is None for t in toks.values()), \
        "neither has acted yet, so neither has an id"
    # Turns 5 AND 7 — two separate eminence triggers, not one resolution making
    # two. I asserted both on turn 5 and the fixture corrected me.
    assert {t["first_seen_turn"] for t in toks.values()} == {5, 7}, \
        "the turn recorded must be when each was created, not when it acted"

    # It blocks on turn 14, which gives ONE of them its id — and the count must
    # not move. A second entry here is the double-count this consumes.
    seats, notes, _ = _state(game, 15, "precombat main")
    after = seats["Ai(2)-edgar-vampires"]["tokens"]
    assert "205" in after, "it blocked on turn 14 and should now be bound by id"
    assert len(after) == 2, (
        f"binding an id added a token instead of consuming the placeholder: {after}")
    assert sum(1 for t in after.values() if t["id"] is None) == 1, \
        "exactly one of the two should still be unbound"

    seats, _, _ = _state(game, 14, "declare blockers")
    assert seats["Ai(1)-radagast"]["commander_zone"] == "battlefield" and \
        seats["Ai(1)-radagast"]["commander_casts"] == 1


def test_a_token_that_never_acted_still_dies(game):
    """Registering tokens at creation leaks every chump blocker unless an unbound
    token can also LEAVE. Its zone change carries an id that was never bound, so
    the by-id removal misses it and one placeholder of the same name must go."""
    log = HEADER + (
        "Turn: Turn 1 (Ai(1)-us)\n"
        "Phase: Ai(1)-us' Main phase, precombat\n"
        "Resolve Stack: Ai(1)-us creates two 1/1 red Goblin creature tokens\n"
        "Phase: Ai(1)-us' End step\n"
        "Turn: Turn 2 (Ai(2)-them)\n"
        "Phase: Ai(2)-them' Main phase, precombat\n"
        "Zone Change: Goblin Token (901) was put into Graveyard from Battlefield.\n"
        "Phase: Ai(2)-them' End step\n")
    seats, _, _, _, _ = bridge.reconstruct(_one_game(log), 2, "ending", "end", {})
    toks = seats["Ai(1)-us"]["tokens"]
    assert len(toks) == 1, (
        f"a token that died without ever acting is still on the board: {toks}")


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
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
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


# ── two reconstruction gaps a rules-checker found on a real board (2026-09-28) ──
#
# Both were found by the adversarial checker in the `/resolve-stack` loop reading
# a board this bridge had lifted, and both make a lifted battlefield INCOMPLETE —
# which on the defender's side silently flatters any lethal claim.

def _one_game(text):
    return parse.parse_games(text)[0]


# THE HEADER IS COPIED FROM THE REAL FIXTURE'S OWN SHAPE, not invented. A
# hand-written preamble parsed to zero games and the tests died on
# `parse_games(...)[0]` with an IndexError — which says nothing about the bug
# under test.
HEADER = ("Simulation mode\n"
          "Ai(1)-us vs Ai(2)-them - one game of Commander\n"
          "Mulligan: Ai(1)-us has kept a hand of 7 cards\n"
          "Mulligan: Ai(2)-them has kept a hand of 7 cards\n")


def test_a_characteristic_defining_power_is_a_creature_not_a_dropped_cast():
    """`Lord of Extinction - Creature * / *`.

    `_CREATURE` demanded `\\d+`, so the `*` form matched nothing, `text == name`
    matched nothing either, and the cast stayed in `pending_casts` FOREVER — the
    creature never reached the board and the lift described a battlefield it was
    not on. Stack 008 claimed seat-2 held "Ripples of Undeath and three tapped
    lands"; the log had Splinterfright there too, and the kill the artifact
    proved survived only because it happened to be tapped. ONE untapped blocker
    turns that kill into a survival.

    SWEEP, one 100-game run: ~107 such resolutions over six distinct creatures,
    every one an opponent's. The P/T is kept as the literal `*/*` because the log
    does not carry the value — "a real creature whose size I cannot give you" is
    what a resolver must reason about, and a fabricated number would be worse
    than the absence it replaces.
    """
    log = HEADER + (
        "Turn: Turn 1 (Ai(1)-us)\n"
        "Phase: Ai(1)-us' Untap step\n"
        "Phase: Ai(1)-us' Main phase, precombat\n"
        "Add To Stack: Ai(2)-them cast Lord of Extinction\n"
        "Resolve Stack: Lord of Extinction - Creature * / *\n"
        "Add To Stack: Ai(2)-them cast Grizzly Bears\n"
        "Resolve Stack: Grizzly Bears - Creature 2 / 2\n"
        "Phase: Ai(1)-us' End step\n")
    seats, _, _, _, _ = bridge.reconstruct(_one_game(log), 1, "ending", "end", {})
    them = seats["Ai(2)-them"]["perms"]
    got = {p["name"]: p["pt"] for p in them.values()}
    assert got == {"Lord of Extinction": "*/*", "Grizzly Bears": "2/2"}, (
        f"a characteristic-defining P/T was dropped from the board: {got}")
    assert not seats["Ai(2)-them"]["pending_casts"], (
        "the cast is still pending, which is the mechanism: it never resolves "
        "into a permanent and is invisible to every consumer")


def test_an_aura_enters_the_battlefield_instead_of_reading_as_a_spell():
    """`Rancor (203) -  Attach to Sythis (12)` matches `_SPELL`, so the branch
    meaning "an instant or sorcery resolved" DISCARDED it, and every Aura was
    absent from every lift.

    Found on stack 009 the same day: seat-4's Sphere of Safety taxes attackers
    {X} where X counts its controller's enchantments, and an Aura the lift had
    dropped made the real tax {3} against the {2} the artifact reasoned from.

    Equipment is deliberately unaffected — it enters as a bare name when cast and
    its later equip ability is not a cast, so it is no longer pending when it
    attaches. The test pins both halves.
    """
    log = HEADER + (
        "Turn: Turn 1 (Ai(1)-us)\n"
        "Phase: Ai(1)-us' Untap step\n"
        "Phase: Ai(1)-us' Main phase, precombat\n"
        "Add To Stack: Ai(2)-them cast Rancor targeting [Bear (12)]\n"
        "Resolve Stack: Rancor (203) -  Attach to Bear (12)\n"
        "Add To Stack: Ai(2)-them cast Lightning Bolt targeting [Ai(1)-us]\n"
        "Resolve Stack: Lightning Bolt (204) - Deals 3 damage to any target\n"
        "Phase: Ai(1)-us' End step\n")
    seats, _, _, _, _ = bridge.reconstruct(_one_game(log), 1, "ending", "end", {})
    them = seats["Ai(2)-them"]["perms"]
    names = {p["name"] for p in them.values()}
    assert "Rancor" in names, "the Aura is missing from the battlefield"
    assert "Lightning Bolt" not in names, (
        "an instant became a permanent — the Attach check is matching too widely")
    rancor = next(p for p in them.values() if p["name"] == "Rancor")
    assert rancor["attached_to"] == "Bear", (
        "the Aura is on the board but does not say what it is attached to, which "
        "is the part a rules question needs")


def test_an_unattributable_token_death_is_removed_from_NOBODY(game):
    """WHO OWNED THE TOKEN THAT DIED. `owner` is learned from lines that name a
    controller outright, so a token that never attacked, blocked or dealt damage
    never entered it — and its death line carries an id but no seat.

    MEASURED over 86 games: 705 of 857 token deaths (82%) had no `owner` entry.
    They were being applied to whichever seat happened to hold a matching
    unlisted token, which can delete an OPPONENT'S blocker — the direction that
    flatters a kill. The name layer ("only one seat ever makes a Goblin Token")
    resolves nearly all of them; a genuine tie remains in 24% of games.

    A TIE REMOVES FROM NOBODY and says so, which overstates a token rather than
    deleting a blocker, and the note tells the resolver which way to read it.
    This is `parse.py`'s own rule for its name fallback: consulted only where
    `owner` is silent, and ambiguity left unattributed rather than guessed.
    """
    log = HEADER + (
        "Turn: Turn 1 (Ai(1)-us)\n"
        "Phase: Ai(1)-us' Main phase, precombat\n"
        "Resolve Stack: Ai(1)-us creates a 1/1 red Goblin creature token\n"
        "Phase: Ai(1)-us' End step\n"
        "Turn: Turn 2 (Ai(2)-them)\n"
        "Phase: Ai(2)-them' Main phase, precombat\n"
        "Resolve Stack: Ai(2)-them creates a 1/1 red Goblin creature token\n"
        # Neither token ever acts, so `owner` never learns either id.
        "Zone Change: Goblin Token (901) was put into Graveyard from Battlefield.\n"
        "Phase: Ai(2)-them' End step\n")
    seats, notes, _, _, _ = bridge.reconstruct(_one_game(log), 2, "ending", "end", {})
    both = {s: len(seats[s]["tokens"]) for s in seats}
    assert both == {"Ai(1)-us": 1, "Ai(2)-them": 1}, (
        f"an unattributable death was charged to a seat on a guess: {both}")
    assert any("removed from NOBODY" in n for n in notes), (
        "the board overstates a token and the artifact does not say so")

    # And when only ONE seat makes that token name, the inference IS available
    # and must be used — otherwise every chump blocker leaks.
    log_one = HEADER + (
        "Turn: Turn 1 (Ai(1)-us)\n"
        "Phase: Ai(1)-us' Main phase, precombat\n"
        "Resolve Stack: Ai(1)-us creates two 1/1 red Goblin creature tokens\n"
        "Phase: Ai(1)-us' End step\n"
        "Turn: Turn 2 (Ai(2)-them)\n"
        "Phase: Ai(2)-them' Main phase, precombat\n"
        "Zone Change: Goblin Token (901) was put into Graveyard from Battlefield.\n"
        "Phase: Ai(2)-them' End step\n")
    seats, _, _, _, _ = bridge.reconstruct(_one_game(log_one), 2, "ending", "end", {})
    assert len(seats["Ai(1)-us"]["tokens"]) == 1, (
        "only one seat makes a Goblin Token, so the death is attributable and "
        "must be applied")
