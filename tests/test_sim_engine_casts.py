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


# ── the fleet ───────────────────────────────────────────────────────────────

#: Runs KEPT although the AI never cast part of the deck's engine, each with
#: the reason. A named set rather than a widened threshold, the same shape as
#: `KNOWN_FLAGGED` in test_sim_pilot_quality: the record is evidence about the
#: harness and a floor on the deck, and its win rate is not read as a result.
KNOWN_UNCAST = {
    # ── goblin-storm, three runs made 2026-09-26/28 on the CURRENT lists ──
    #
    # NOT the stale-join shape below: every card named was in the 99 when the
    # games were played, so these are TRUE POSITIVES about the AI. Two are
    # proven from the record itself — `by_card` shows **Faithless Looting
    # cast 0 / discarded 1** and **Goblin Bombardment cast 0 / discarded 1**, so
    # the AI held each and threw it away rather than casting it. Both are
    # documented classes: Forge's AI will not discard its own hand (the sharknado
    # entry below records the same thing), and it will not sacrifice for a benefit
    # its evaluator cannot price, which is what a free sacrifice outlet is.
    #
    # Past in Flames and Witch's Mark carry NO `by_card` record at all, so the
    # cause is unknown rather than proven — stated rather than assumed, because a
    # card that was never drawn and a card that was held and passed over are
    # different facts and only one of them is about the AI.
    #
    # The champion at standard-v3, 100 games:
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n100-ef47e836-s14467126-podExperimental-c600.json",
    # zada-v1 at standard-v3, 100 games, same four cards:
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n100-cb6398af-s1412301999-podExperimental-c600.json",
    # copy-burst-v1 at standard-v3, 100 games. Only Haze of Rage flags here, and
    # the run is doubly caveated in its own record: it also trips the piloting
    # gate. Read with the Zada ceiling — the AI cast the four copy effects 84
    # times across these games while Zada's ability triggered 39 times in total,
    # so most casts did not target her and produced one token instead of N.
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n100-af01a068-s936119400-podExperimental-c600.json",
    # The OVERRIDDEN companion to the run above — same list, same pod, same clock, eleven
    # `AITgts$` hints installed. Flags for the same four cards and two of them are
    # documented at the engine level rather than the deck's: Faithless Looting and
    # Goblin Bombardment both carry Forge's own `AI:RemoveDeck:All`, its marker for a card
    # its AI has no logic for (`sim/forge_cards.py` reads it). So "never cast" here is not
    # a fact about the 99; it is Forge declaring in its own card scripts that it cannot
    # fly those two, which no `AITgts$` hint addresses — the hint narrows a TARGET and
    # these were never cast at all. The run remains a FLOOR and its record says so.
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n100-af01a068-s909090-podExperimental-c600.json",
    # A RECORD IS JUDGED AGAINST TODAY'S DECLARATION, NOT THE ONE IT RAN UNDER.
    # These two flag for a reason that is not the AI's: the deck changed after
    # the run, and the cards now named in `goldfish_targets.json` were not in
    # the 99 when the games were played. The record is honest, the declaration
    # is honest, and the join between them is what is stale.
    #
    # 2026-09-02, ur-dragon at ~standard, 100 games: Steel Hellkite,
    # Transcendent Dragon and Whirlwing Stormbrood read as never cast. All
    # three ARRIVED in v1.3.0 on 2026-09-20 and were added to the castable-
    # dragon target legs the same day. The seat could not cast cards the deck
    # did not contain.
    "giada-angels-vs-baylen-tokens-vs-abaddon-n100-45f502ef-s1173684975-podExperimental-c600.json",
    # Same shape, gishath at standard-v3, 40 games: World War Hulk arrived in
    # v1.1.0 on 2026-09-20, after this run.
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n40-0ed9ed6f-s249163119-podExperimental-c600.json",
    # Same shape again, ur-dragon at standard-v2, 60 games (2026-09-06): the
    # same three v1.3.0 arrivals. Every ur-dragon record predating 2026-09-20
    # flags this way, which is the tell that it is the JOIN and not the AI.
    "abaddon-vs-nekusar-discard-vs-muldrotha-value-n60-acc77874-s898753652-podExperimental-c600.json",
    # 2026-09-10, sharknado@recon-v1 at standard-v3: Wheel of Fortune cast once,
    # Windfall / Magus / Jace's Archivist / Faithless Looting never, in 60
    # games. Forge's AI will not discard its own hand. Kept as the record that
    # made this block exist. (Its 20-game Experimental sibling shows the same
    # casts but is under the per-card lines at that N and does not flag.)
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n60-c1d1131f-s1251704607-podExperimental-c600.json",
    # 2026-09-13, sharknado (the DECK, not the branch) at standard-v3, 120
    # games: Jace's Archivist, Winds of Change, Faithless Looting and
    # Improbable Alliance never cast. The seat cast 95 wheels and DISCARDED
    # 105, and nine of seventeen were never cast once -- Wheel of Fortune got
    # one cast in 120 games. The escape hatch was checked and closed the next
    # day: `--profile Experimental`, same pod and seeds, was abandoned at 37
    # games with Jace's Archivist still at zero activations and the rate no
    # better, so the refusal is a limit of the evaluator rather than a setting.
    # docs/gotchas-bench.md "sharknado is a deck Forge cannot pilot". The
    # record is kept because `engine_casts` is the evidence FOR that claim.
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n120-ebff4d05-s1959377157-podExperimental-c600.json",
    # THE SACRIFICE CLASS, edgar-vampires, every run since 2026-08-24: Viscera
    # Seer, Altar of Dementia, Ashnod's Altar, Bloodflow Connoisseur, Vish Kal
    # never cast. docs/gotchas-bench.md "The Forge AI will not press a sacrifice
    # button" measured it on 08-30; these are the same fact on every table the
    # deck sat at, and the block now says so per record instead of per essay.
    "abaddon-vs-nekusar-discard-vs-muldrotha-value-n60-3e064845-s1040599109-podExperimental-c600.json",
    "giada-angels-vs-baylen-tokens-vs-abaddon-n100-42dc6d00-s1121742080-podExperimental-c600.json",
    "giada-angels-vs-vito-vs-baylen-tokens-n100-717196e1-s1903269601.json",
    "giada-angels-vs-vito-vs-baylen-tokens-n400-717196e1-s1903269601.json",
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n60-cdaddf26-s1450724134-podExperimental-c600.json",
    # goblin-storm, 2026-09-02, both tables: Goblin Bombardment and Past in
    # Flames never cast -- an activation outlet and a graveyard storm turn,
    # the two things Forge's AI is documented not to do ("pretty bad for most
    # combo decks", its own words in every record).
    "giada-angels-vs-baylen-tokens-vs-abaddon-n100-62a56c7d-s1655008381-podExperimental-c600.json",
    "giada-angels-vs-vito-vs-baylen-tokens-n100-a209c8f5-s718550261-podExperimental.json",
    # heliod, 2026-09-07 and 09-09: Psychosis Crawler (0 casts, the gotcha's own
    # example) with Long-Term Plans and a handful of defensive bodies. The rest
    # of a 45-card engine played; the deck's rates are read with this beside them.
    "giada-angels-vs-baylen-tokens-vs-abaddon-n120-996adb84-s573917060-podExperimental-c600.json",
    "giada-angels-vs-baylen-tokens-vs-abaddon-n120-b0f44e4a-s968800842-podExperimental-c600.json",
    # ur-dragon, 2026-08-24, the first tracked record, vito era, before seats
    # rotated: six Dragons never cast in 100 games. Every later ur-dragon run
    # reads ENGINE PLAYED, so this is the harness of that day, kept as history.
    "giada-angels-vs-vito-vs-baylen-tokens-n100-c040c7ac-s1225470892.json",
    # zur-enchantress (archived 09-10), 2026-09-06: one card of 26 in each --
    # The Meathook Massacre at one table, Eidolon of Astral Winds at the other.
    "abaddon-vs-nekusar-discard-vs-muldrotha-value-n60-91ac828c-s444001932-podExperimental-c600.json",
    "giada-angels-vs-baylen-tokens-vs-abaddon-n60-07b69243-s129405507-podExperimental-c600.json",
    # 2026-09-10, heliod at standard-v3, 40 games: Psychosis Crawler never cast
    # (discarded once), one card of a 53-card engine set; the rest of the
    # engine played at its expected rate. docs/gotchas-bench.md already records
    # the Crawler at 0 casts in 20 games — "read correctly, never played" — so
    # this is the same true positive measured again. The run's 0.278 is read
    # as heliod's because the declared engine was otherwise played.
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n40-4b0e1964-s1259215204-podExperimental-c600.json",
    # 2026-09-25, goblin-storm@zada-v1 on standard-v3, BOTH profiles. Faithless
    # Looting, Past in Flames, Witch's Mark (and Goblin Bombardment on the Default
    # run) read as never cast. A TRUE POSITIVE and a DIFFERENT reason from the two
    # above: the declaration is current and the cards are in the 99 — the AI drew
    # them and passed. Faithless Looting was DISCARDED 3 times, which is the log's
    # own statement that it was held and thrown away.
    #
    # Both records are kept because the run's value is not its rate. Zada's
    # trigger fired 21 times in 60 games against 100 casts of her, so the win
    # rate measures a Zada deck played without Zada's ability. The four uncast
    # cards are a footnote next to that. → docs/gotchas-bench.md
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n60-c9abd341"
    "-s1383481153-podExperimental-c600.json",
    "sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n60-c9abd341"
    "-s1383481153-meExperimental-podExperimental-c600.json",
}


def _tracked_records():
    import glob, pathlib as _pl
    root = _pl.Path(__file__).resolve().parent.parent
    return sorted(p for p in glob.glob(str(root / "data/decks/*/sim/*.json"))
                  + glob.glob(str(root / "data/decks/*/branches/*/sim/*.json")) if "/logs/" not in p)


def test_every_record_made_after_the_block_existed_carries_it():
    """Absent means not measured — and from 2026-09-10 on, nothing is unmeasured."""
    paths = _tracked_records()
    if len(paths) < 5:
        pytest.skip("deck data not present")
    checked, missing = 0, []
    for p in paths:
        rec = json.load(open(p))
        if (rec.get("at") or "") < "2026-09-10":
            continue
        checked += 1
        if "engine_casts" not in rec:
            missing.append(p.split("/")[-1])
    assert checked >= 1, "the loop iterated nothing — no record dated after the block"
    assert not missing, f"records dated after 2026-09-10 with no engine_casts block: {missing}"


def test_no_kept_record_has_an_uncast_engine_by_accident():
    """A run whose declared engine went uncast is a floor, not a result. Every
    such record is listed by name with its reason, or this fails."""
    from manamap.sim import engine_casts as ec
    from manamap.pilot.common import load_deck_cards
    from manamap.sim.forge import split_seat
    paths = _tracked_records()
    if len(paths) < 5:
        pytest.skip("deck data not present")
    checked, bad = 0, []
    for p in paths:
        rec = json.load(open(p))
        if "engine_casts" not in rec:
            continue
        slug = rec["slug"]
        base, branch = split_seat(slug)
        try:
            names = ec.nonland_names(load_deck_cards(base, branch))
        except FileNotFoundError:
            continue
        q = ec.from_record(rec, names, ec.engine_set(base, branch))
        checked += 1
        if q and q["covered"] is False and p.split("/")[-1] not in KNOWN_UNCAST:
            bad.append((p.split("/")[-1], q["engine"]["never_cast"][:4]))
    assert checked >= 1, "no record carried the block — the loop checked nothing"
    assert not bad, (f"the AI never cast part of the engine in {bad}. Check it is a true "
                     f"positive; if it is, add the record to KNOWN_UNCAST with its reason.")
    for name in KNOWN_UNCAST:
        assert any(p.endswith(name) for p in paths), f"{name} is in KNOWN_UNCAST but no longer tracked"


def test_compact_carries_the_per_turn_series():
    """`life_by_turn` and `damage_to_players_by_turn` used to be dropped here;
    a table model is calibrated from them, so they ride, for every seat."""
    facts = _facts()
    row = parse.compact(facts[0], LABEL)
    for seat in ("sharky", "rival"):
        assert "life_by_turn" in row["per_seat"][seat]
        assert isinstance(row["per_seat"][seat]["damage_to_players_by_turn"], dict)


def test_every_tracked_record_row_carries_the_series():
    """Re-derived on 2026-09-10 for every record whose logs exist; a record made
    by `run` carries them from the start."""
    paths = _tracked_records()
    if len(paths) < 5:
        pytest.skip("deck data not present")
    checked, missing = 0, []
    for p in paths:
        rec = json.load(open(p))
        for g in rec.get("games") or []:
            checked += 1
            if any("life_by_turn" not in v for v in g["per_seat"].values()):
                missing.append(p.split("/")[-1]); break
    assert checked >= 100, "the loop iterated almost nothing"
    assert not missing, f"records whose rows lack the per-turn series: {missing[:6]}"


# ── gate (b): the table's behaviour, per seat ───────────────────────────────

COMBAT_LOG = """\
Mulligan: Ai(1)-mm-sharky has kept a hand of 7 cards.
Mulligan: Ai(2)-mm-rival has kept a hand of 7 cards.
Turn: Turn 1 (Ai(1)-mm-sharky)
Add To Stack: Ai(1)-mm-sharky cast Grizzly Bears
Turn: Turn 2 (Ai(2)-mm-rival)
Add To Stack: Ai(2)-mm-rival cast Wall of Omens
Discard: Ai(2)-mm-rival discards Island (3).
Turn: Turn 3 (Ai(1)-mm-sharky)
Combat: Ai(1)-mm-sharky assigned Grizzly Bears (11), Runeclaw Bear (12) to attack Ai(2)-mm-rival.
Combat: Ai(2)-mm-rival assigned Wall of Omens (21) to block Grizzly Bears (11).
Ai(2)-mm-rival didn't block Runeclaw Bear (12)
Damage: Runeclaw Bear (12) deals 2 combat damage to Ai(2)-mm-rival.
Life: Life: Ai(2)-mm-rival 40 > 38
Game Outcome: Turn 2
Game Outcome: Ai(1)-mm-sharky has won because all opponents have lost
Game Result: Game 1 ended in 1000 ms
"""


def test_the_table_s_behaviour_is_counted_per_seat():
    games = parse.parse_games(COMBAT_LOG)
    f = parse.game_facts(games[0])
    me, them = f["per_seat"]["Ai(1)-mm-sharky"], f["per_seat"]["Ai(2)-mm-rival"]
    assert me["attackers_declared"] == 2 and me["attackers_blocked"] == 1
    assert them["blocks_declared"] == 1 and them["discards"] == 1
    assert them["life_by_turn"] == {3: 38}
    assert me["damage_to_players_by_turn"] == {3: 2}
    _, agg = parse.analyze_logs([COMBAT_LOG], LABEL)
    for name in ("sharky", "rival"):
        seat = agg["seats"][name]
        for key in ("turns", "discards", "blocks_declared", "attackers_declared", "attackers_blocked"):
            assert key in seat, key
        assert seat["cumulative_combat_damage_by_round"][0]["round"] == 1
        assert seat["wipe_recovery"]["available"] is False, "no wipe in one small game"
    assert agg["seats"]["rival"]["cumulative_combat_damage_by_round"][-1]["mean"] == 0.0
    assert agg["seats"]["sharky"]["cumulative_combat_damage_by_round"][-1]["mean"] == 2.0
    assert agg["our_cumulative_combat_damage_by_round"] == agg["seats"]["sharky"]["cumulative_combat_damage_by_round"]


def test_never_cast_names_whether_each_card_was_SEEN_or_only_INFERRED():
    """THE TWO FLOORS THIS BLOCK DEFINES AND NEVER DISTINGUISHED.

    `expected_is` is documented as "a MODEL floor" and `held_is` as "a MEASURED floor",
    and the predicate `plays == 0 and (discarded >= 3 or expected >= 8.0)` collapsed them:
    `expected` is a RUN-LEVEL scalar, games x (kept + own turns) / library, so
    `expected >= 8.0` asks "is this run longer than about 25 games" and is TRUE on 23 of
    the 25 tracked runs. Past that length every zero-play nonland lands in the list on the
    modelled floor alone.

    Which is not wrong — never played across ~27 expected draws is a real finding — but it
    is an INFERENCE, and this is the gate a reader is told to check before trusting a rate.
    Measured on the tracked runs: ur-dragon at 100 games lists 9 cards and the log shows 0
    of them discarded, while sharknado lists 17 of which 11 were genuinely seen.
    """
    from manamap.sim import engine_casts as ec

    rec = {"engine_casts": {"seat": "x", "games": 100, "turns": 2000,
                            "kept_hand_mean": 7.0,
                            "by_card": {"Seen": {"cast": 0, "activated": 0, "discarded": 5},
                                        "Unseen": {"cast": 0, "activated": 0, "discarded": 0},
                                        "Played": {"cast": 4, "activated": 0, "discarded": 0}}},
           "seats": [{"commander": ["C"]}]}
    got = ec.from_record(rec, deck_names={"Seen", "Unseen", "Played"})
    assert got["expected_natural_draws_per_card"] >= ec.NEVER_CAST_EXPECTED, (
        "this fixture exists to exercise the run length where the modelled floor covers "
        "everything; if it no longer does, the premise moved")

    by = {r["card"]: r["basis"] for r in got["never_cast"]}
    assert by == {"Seen": "measured", "Unseen": "modelled"}, by
    assert "Played" not in by, "a card that was cast is not in the list at all"

    basis = got["never_cast_basis"]
    assert basis["measured"] == 1 and basis["modelled"] == 1
    assert "inference" in basis["what"]

    out = ec.render(got)
    text = "\n".join(out) if isinstance(out, list) else out
    assert "1 seen discarded (MEASURED)" in text
    assert "1 inferred from" in text and "MODELLED" in text


def test_the_headline_only_says_held_when_something_was_seen():
    """"HELD" ASSERTS AN OBSERVATION. ur-dragon's run printed "held and never cast" over
    nine cards the log never shows being discarded."""
    from manamap.sim import engine_casts as ec

    def rendered(discarded):
        rec = {"engine_casts": {"seat": "x", "games": 100, "turns": 2000,
                                "kept_hand_mean": 7.0,
                                "by_card": {"A": {"cast": 0, "activated": 0,
                                                  "discarded": discarded}}},
               "seats": [{"commander": ["C"]}]}
        out = ec.render(ec.from_record(rec, deck_names={"A"}))
        return "\n".join(out) if isinstance(out, list) else out

    assert "held and never cast" in rendered(5), "seen in hand — 'held' is earned"
    seen_none = rendered(0)
    assert "never cast" in seen_none
    assert "held and never cast" not in seen_none, (
        "nothing was observed in hand, so 'held' is a claim the log cannot support")


# --------------------------------------------- the live view's never-cast line

def test_progress_never_cast_excludes_lands_and_names_an_unfiltered_fallback():
    """`sim-progress` printed "held and never cast: Smoldering Crater, Castle Embereth,
    Forgotten Cave" on a live run — LANDS — two commits after I wrote that this site
    "earns the word" because it filters on `discarded`. Lands get discarded too.

    And the same fix in `forge.py` used to fall back to the unfiltered table SILENTLY when
    the deck's list could not be loaded, so a seat with no cards.json read every land as
    held-and-never-cast with nothing distinguishing that mode. `nonland=None` is allowed,
    but the caller must say so; this asserts the function makes that possible by not
    filtering, rather than guessing.
    """
    from manamap.sim.progress import held_never_cast

    by_card = {"Mountain":           {"cast": 0, "activated": 0, "discarded": 8},
               "Smoldering Crater":  {"cast": 0, "activated": 0, "discarded": 2},
               "Faithless Looting":  {"cast": 0, "activated": 0, "discarded": 3},
               "Zada, Hedron Grinder": {"cast": 16, "activated": 0, "discarded": 0},
               "Never Seen":         {"cast": 0, "activated": 0, "discarded": 0}}
    nonland = {"Faithless Looting", "Zada, Hedron Grinder", "Never Seen"}

    got = held_never_cast(by_card, nonland)
    assert got == [("Faithless Looting", 3)], got
    # A land can only ever read as "never cast", so it must not be there —
    # and "Never Seen" was never discarded, so the log gives no sign it was held.

    # With no deck list, NOTHING is filtered — and that is the caller's cue to say so.
    unfiltered = held_never_cast(by_card, None)
    assert ("Mountain", 8) in unfiltered and ("Smoldering Crater", 2) in unfiltered
    assert unfiltered[0] == ("Mountain", 8), "most-discarded first"
