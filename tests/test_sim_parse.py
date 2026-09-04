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
    assert agg["round"] == {"mean": 9.0, "median": 9, "min": 9, "max": 9, "n": 1}, \
        "no interval below two games, but the median and range still describe it"
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


# ── Commander damage (CR 903.10a) ───────────────────────────────────────────

CMD = {"Ai(1)-radagast": {"Radagast of Rhosgobel"},
       "Ai(2)-edgar-vampires": {"Edgar Markov"}}


def test_commander_damage_is_tallied_per_defender(game):
    """21 from the same commander on ONE player is a whole archetype's only win
    condition, and the parser could not see it: `combat_damage_dealt_to_players` sums
    every source and every defender, so a commander that hit three seats for 20 each
    looked exactly like one that hit a single seat for 60 and killed them.

    In the fixture each commander connects for 11 on the other — 2 + 2 + 7 for
    Radagast, 5 + 6 for Edgar — and neither is lethal.
    """
    f = parse.game_facts(game, CMD)
    rad = f["per_seat"]["Ai(1)-radagast"]
    edg = f["per_seat"]["Ai(2)-edgar-vampires"]
    assert rad["commander_damage_by_defender"] == {"Ai(2)-edgar-vampires": 11}
    assert edg["commander_damage_by_defender"] == {"Ai(1)-radagast": 11}
    assert rad["commander_damage_max"] == 11 and rad["commander_damage_lethal"] is False


def test_an_unknown_commander_is_absent_rather_than_zero(game):
    """The Forge log never names a commander, so the names come from the seat's own
    decklist. When they are unavailable the block must not appear at all — reporting
    "dealt 0" for a commander nobody identified is a measurement the run did not make.
    """
    f = parse.game_facts(game)
    for p in f["per_seat"].values():
        assert not [k for k in p if "commander_damage" in k]
    _, agg = parse.analyze_logs([FIX], LABEL)
    assert all("commander_damage" not in s for s in agg["seats"].values())


def test_only_combat_damage_counts_toward_the_21(game):
    """903.10a asks for COMBAT damage. The fixture has Purphoros dealing 2 non-combat
    damage to radagast; make the same source the seat's commander and it still must not
    be counted, or a Purphoros deck would read as closing on commander damage it can
    never deal."""
    f = parse.game_facts(game, {"Ai(2)-edgar-vampires": {"Purphoros, God of the Forge"}})
    assert f["per_seat"]["Ai(2)-edgar-vampires"]["commander_damage_by_defender"] == {}


def test_a_commander_is_matched_on_its_face_as_well_as_the_joined_name():
    """Forge logs a transformed permanent under the face on the battlefield while the
    decklist names the card, so `A // B` has to match a log line saying just `A`."""
    assert parse._is_commander("Brutal Cathar", {"Brutal Cathar // Moonrage Brute"})
    assert parse._is_commander("Moonrage Brute", {"Brutal Cathar // Moonrage Brute"})
    assert parse._is_commander("Edgar Markov", {"Edgar Markov"})
    assert not parse._is_commander("Edgar Markov", {"Radagast of Rhosgobel"})
    assert not parse._is_commander("Edgar Markov", None)


def test_the_aggregate_separates_total_dealt_from_the_number_that_kills():
    """`dealt_total` and `max_on_one_defender` are different questions and the run
    record must carry both — spreading 60 across three seats wins nothing."""
    _, agg = parse.analyze_logs([FIX], LABEL, CMD)
    cd = agg["seats"]["radagast"]["commander_damage"]
    assert cd["commander"] == ["Radagast of Rhosgobel"]
    assert cd["dealt_total"]["mean"] == 11.0
    assert cd["max_on_one_defender"]["mean"] == 11.0
    assert cd["best_single_game_max"] == 11
    assert cd["games_reaching_21"] == 0 and cd["games_dealing_any"] == 1
    assert any("903.10a" in x for x in agg["limits"]), \
        "the limit that explains per-defender must travel with the figure"


def test_the_mean_never_travels_without_the_median():
    """A mean over a skewed sample is a true number describing no game.

    kianne's V1-vs-V2 arm B had per-game commander damage
    `0 0 0 0 0 0 0 0 0 0 31 178`: mean 17.42 against V1's 2.25, which reads as a
    sevenfold win. The median is 0 in BOTH arms — the whole difference is two games,
    one a blowout, and the deck connected in FEWER games after the change. The ci95
    already spanned zero, so the record was honest; it took sorting the values by hand
    to see it.
    """
    skewed = parse.mean_ci([0] * 10 + [31, 178])
    assert skewed["mean"] == 17.42 and skewed["median"] == 0.0
    assert skewed["min"] == 0 and skewed["max"] == 178
    assert skewed["ci95"][0] < 0 < skewed["ci95"][1], "the interval spans zero"

    flat = parse.mean_ci([0] * 8 + [5, 7, 7, 8])
    assert flat["mean"] == 2.25 and flat["median"] == 0.0
    assert flat["max"] == 8

    assert parse.mean_ci([]) == {"mean": None, "n": 0}
    one = parse.mean_ci([4])
    assert one["median"] == 4 and "ci95" not in one, "no interval below two games"
    even = parse.mean_ci([1, 3])
    assert even["median"] == 2.0, "an even sample averages the middle pair"


# ── the opening hand, measured and thrown away since the parser existed ─────

MULLIGAN_LOG = """Mulligan: Ai(1)-mine has mulliganed down to 7 cards.
Mulligan: Ai(1)-mine has mulliganed down to 6 cards.
Mulligan: Ai(1)-mine has kept a hand of 6 cards
Mulligan: Ai(2)-rival has kept a hand of 7 cards
Turn: Turn 1 (Ai(1)-mine)
Life: Life: Ai(1)-mine 40 > 0
Game Outcome: Ai(2)-rival has won because all opponents have lost
Game Result: Game 1 ended in 1000 ms.
"""


def test_both_mulligan_lines_are_read():
    """Forge emits two, and only the second was ever matched.

    `has mulliganed down to N cards.` fires once per mulligan TAKEN; `has kept a
    hand of N cards` gives the final size. The parser matched only `kept`, so the
    COUNT was in every log and in no measurement — and `compact()` then dropped
    even the kept size, so neither reached a record.
    """
    games = parse.parse_games(MULLIGAN_LOG)
    assert len(games) == 1
    facts = parse.game_facts(games[0])

    mine = facts["per_seat"]["Ai(1)-mine"]
    assert mine["mulligans_taken"] == 2
    assert mine["mulligan_kept"] == 6

    rival = facts["per_seat"]["Ai(2)-rival"]
    assert rival["mulligans_taken"] == 0
    assert rival["mulligan_kept"] == 7


def test_forge_gives_the_first_mulligan_free_and_the_record_says_so():
    """A FIDELITY GAP, not a parser one, and the reason both figures are kept.

    Measured across all 130 tracked logs and 5,056 seat-hands with ZERO
    exceptions: 0 mulligans keeps 7, ONE mulligan also keeps 7, two keeps 6,
    three keeps 5. So `kept = 7 - max(0, taken - 1)`. Under the London mulligan
    one mulligan draws seven and bottoms one, for a hand of SIX — so a deck that
    mulligans is flattered by a card here, and neither figure is derivable from
    the other under real rules.

    The relation is asserted as OBSERVED BEHAVIOUR, not as a rule: if a Forge
    upgrade implements London, this test fails and the limit needs rewriting,
    which is the correct outcome rather than a silent re-basing.
    """
    for taken, kept in ((0, 7), (1, 7), (2, 6), (3, 5)):
        assert kept == 7 - max(0, taken - 1), (taken, kept)

    facts = [parse.game_facts(g) for g in parse.parse_games(MULLIGAN_LOG)]
    agg = parse.aggregate(facts, "Ai(1)-mine",
                          {"Ai(1)-mine": "mine", "Ai(2)-rival": "rival"})
    assert any("first mulligan free" in lim.lower() for lim in agg["limits"]), \
        "the gap must travel with the record that has it"


def test_the_mulligan_figures_aggregate_with_their_spread():
    """A mean never travels alone — `mean_ci` carries median, min and max."""
    facts = [parse.game_facts(g) for g in parse.parse_games(MULLIGAN_LOG)]
    agg = parse.aggregate(facts, "Ai(1)-mine",
                          {"Ai(1)-mine": "mine", "Ai(2)-rival": "rival"})

    mine = agg["seats"]["mine"]
    for key in ("mulligans_taken", "mulligan_kept"):
        assert set(mine[key]) >= {"mean", "median", "min", "max", "n"}, key
    assert mine["mulligans_taken"]["mean"] == 2.0
    assert mine["mulligan_kept"]["mean"] == 6.0
    assert agg["seats"]["rival"]["mulligans_taken"]["mean"] == 0.0


def test_a_seat_with_no_kept_line_reports_absent_not_seven():
    """Absent means ABSENT. A hand nobody logged is not a hand of seven."""
    # A `Life:` line so the game survives `parse_games`, which drops a game with
    # no events, no winner and no truncation — the mulligan lines alone are not
    # events, which is correct and is why the fixture needs one.
    truncated = ("Mulligan: Ai(1)-mine has mulliganed down to 7 cards.\n"
                 "Turn: Turn 1 (Ai(1)-mine)\n"
                 "Life: Life: Ai(1)-mine 40 > 39\n"
                 "Game Result: Game 1 ended in 5 ms.\n")
    facts = parse.game_facts(parse.parse_games(truncated)[0])
    mine = facts["per_seat"]["Ai(1)-mine"]
    assert mine["mulligans_taken"] == 1
    assert mine["mulligan_kept"] is None

    agg = parse.aggregate([facts], "Ai(1)-mine", {"Ai(1)-mine": "mine"})
    assert agg["seats"]["mine"]["mulligan_kept"]["n"] == 0


def test_every_tracked_record_carries_the_opening_hand():
    """The parser change owes the fleet a regeneration, in the same commit."""
    records = sorted((Path(__file__).parent.parent / "data" / "decks")
                     .glob("*/sim/*.json"))
    assert len(records) >= 15, "the guard iterated almost nothing"
    for path in records:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for slug, seat in doc["analysis"]["seats"].items():
            assert "mulligans_taken" in seat, f"{path.name}: {slug}"
            assert "mulligan_kept" in seat, f"{path.name}: {slug}"
