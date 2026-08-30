"""Post-wipe value: what a deck does on the turn its board dies.

THE METRIC THE GOLDFISH CANNOT HAVE. Nothing dies in `pilot/goldfish.py` — no
blockers, no removal, no sacrifice — so Blood Artist, Zulaport Cutthroat and
Bastion of Remembrance contribute exactly zero to every figure it publishes.
edgar-vampires' captain's log 002 is about that gap: "both games were rebuild,
wipe, rebuild, and neither of my decks generates value on the way down."

Forge kills things, so this is where the question can be asked — but only in the
shape the log supports. The obvious metric, board before against board two turns
later, is NOT AVAILABLE and these tests pin that: Forge emits a `Zone Change`
when a permanent leaves the battlefield and nothing when one arrives, measured
at 0 `to Battlefield` lines across a 100-game run. A board series reconstructed
from that would be zeros wearing the name of a measurement.
"""

import pytest

from manamap.sim import parse

from conftest import requires_deck


def _fact(seats, lost, damage, global_turn=30):
    """A minimal game_facts shape: per-turn losses and per-turn damage."""
    return {
        "seats": list(seats), "winner": None, "global_turn": global_turn,
        "per_seat": {
            s: {"permanents_lost_by_turn": dict(lost.get(s, {})),
                "damage_to_players_by_turn": dict(damage.get(s, {}))}
            for s in seats},
    }


ME, A, B = "me", "opp-a", "opp-b"


def test_attrition_is_not_a_wipe():
    """Creatures die in combat every turn of every game. The signature of mass
    removal is BREADTH — one attack cannot hit two defenders' boards at once."""
    f = _fact([ME, A, B], {ME: {5: 4}}, {})
    assert parse.wipes(f, ME) == [], "four of one seat's permanents is a combat"


def test_breadth_across_seats_is_the_signature():
    f = _fact([ME, A, B], {ME: {5: 3}, A: {5: 2}, B: {5: 1}}, {})
    got = parse.wipes(f, ME)
    assert len(got) == 1
    assert got[0]["turn"] == 5
    assert got[0]["permanents_lost_table"] == 6
    assert got[0]["seats_hit"] == 3
    assert got[0]["permanents_lost_mine"] == 3


def test_the_threshold_is_a_threshold_in_both_directions():
    """RE-INTRODUCING THE CONDITION. One under on either axis and it is not a
    wipe; one over on both and it is."""
    n, k = parse.WIPE_MIN_PERMANENTS, parse.WIPE_MIN_SEATS
    assert k == 2, "the fixtures below are written against two"
    too_few = _fact([ME, A], {ME: {4: n - 2}, A: {4: 1}}, {})
    assert parse.wipes(too_few, ME) == []
    one_seat = _fact([ME, A], {ME: {4: n + 3}}, {})
    assert parse.wipes(one_seat, ME) == []
    both = _fact([ME, A], {ME: {4: n - 1}, A: {4: 1}}, {})
    assert len(parse.wipes(both, ME)) == 1


def test_value_on_the_way_down_is_separated_from_value_after():
    """The captain's log describes the first one: "a board wipe going off with
    the drain package online — eleven creatures died, every opponent lost 11".
    A deck that only rebuilds scores on `damage_after` and zero on the turn."""
    f = _fact([ME, A], {ME: {6: 4}, A: {6: 2}},
              {ME: {5: 99, 6: 11, 7: 3, 8: 4, 9: 99}})
    w = parse.wipes(f, ME)[0]
    assert w["damage_on_wipe"] == 11, "the turn the board died"
    assert w["damage_after"] == 7, "turns 7 and 8 only — not 5, not 9"
    assert not w["truncated"]


def test_a_wipe_the_game_did_not_outlive_is_flagged_not_averaged():
    """A short window quietly averaged in beside full ones understates recovery
    for exactly the games where the wipe ended it."""
    f = _fact([ME, A], {ME: {9: 4}, A: {9: 2}}, {ME: {10: 5}}, global_turn=10)
    assert parse.wipes(f, ME)[0]["truncated"] is True


def test_no_wipe_is_ABSENT_and_says_why():
    """Absent means ABSENT. `damage_on_wipe: 0.0` against no wipes at all is a
    measurement nobody made wearing the shape of one that was."""
    facts = [_fact([ME, A], {ME: {3: 1}}, {}) for _ in range(5)]
    r = parse.wipe_recovery(facts, ME)
    assert r["available"] is False
    assert "absent measurement" in r["why"]
    for k in ("damage_on_wipe", "damage_after", "wipes_seen"):
        assert k not in r, f"{k} must not be reported when nothing was seen"


def test_an_unknown_seat_is_not_a_deck_with_no_wipes():
    assert parse.wipes(_fact([A, B], {A: {4: 9}, B: {4: 2}}, {}), ME) == []


def test_the_aggregate_carries_medians_and_its_own_limits():
    """A MEAN IS NOT A RESULT. One 15-damage game carried edgar-vampires' real
    `damage_after` mean to 4.0 against a median of 0."""
    facts = [_fact([ME, A], {ME: {5: 4}, A: {5: 2}}, {ME: {6: d}})
             for d in (0, 0, 0, 0, 30)]
    r = parse.wipe_recovery(facts, ME)
    assert r["available"] is True
    assert r["wipes_seen"] == 5
    assert r["damage_after"]["mean"] == 6.0
    assert r["damage_after"]["median"] == 0, "the mean is one game"
    assert r["damage_after"]["max"] == 30
    blob = " ".join(r["limits"])
    assert "Board size before and after is NOT here" in blob, (
        "the unmeasurable half must be named where the figures are printed")
    assert "HEURISTIC" in blob
    assert str(parse.WIPE_MIN_PERMANENTS) in r["definition"]


@requires_deck
def test_against_the_real_pod_run_the_instrument_is_shown_to_work():
    """THE CONTROL THAT MAKES THE FINDING A FINDING. edgar-vampires scores 0
    damage on all nine wipes in its 100-game pod run. That reads identically to
    a broken instrument, so the same run must show the seat scoring damage on
    OTHER turns — it deals 2,909 across 328 scoring turns."""
    import pathlib
    from manamap.config import DATA_DIR
    root = (pathlib.Path(DATA_DIR) / "decks/edgar-vampires/sim/logs"
            / "giada-angels-vs-vito-vs-baylen-tokens-n100-717196e1-s1903269601")
    if not root.is_dir():
        pytest.skip("the 100-game pod run is not present")
    facts = [parse.game_facts(g) for f in sorted(root.glob("*.log"))
             for g in parse.parse_games(f.read_text(errors="replace"))]
    seat = next(s for s in facts[0]["seats"] if "edgar" in s)
    scoring = sum(1 for f in facts
                  for v in f["per_seat"][seat]["damage_to_players_by_turn"].values() if v)
    assert scoring > 100, "the damage channel is dead — no finding can rest on it"
    r = parse.wipe_recovery(facts, seat)
    assert r["available"] and r["wipes_seen"] >= 5


# ── The seven fields aggregate() computed and threw away ──────────────────

def _agg(facts, seat):
    return parse.aggregate(facts, seat, {s: s for s in facts[0]["seats"]})["seats"][seat]


def test_the_record_publishes_the_DRAIN_half_of_a_drain_deck():
    """THE MEASUREMENT DEFECT THAT COST FOUR BRANCHES.

    `game_facts` has always accumulated `noncombat_damage_dealt_to_players` and
    `aggregate` published only the COMBAT figure — so a record for a deck whose
    stated engine is drain reported half of it. Audited card by card on
    edgar-vampires: of seven output channels the 99 produces, exactly TWO
    reached the page. Drain (9 cards), lifegain (17), +1/+1 counters (15), card
    draw (11) and removal (7) were invisible, and four branches were designed
    and killed against that view.
    """
    f = {"seats": ["me", "a"], "winner": "a", "global_turn": 10,
         "round": 10, "ms": 100, "lost": {}, "mulligan": {}, "won_by": None,
         "per_seat": {s: {"eliminated_turn": None, "eliminated_by": None,
                          "eliminated_how": None, "lands": 6, "casts": 8,
                          "combat_damage_dealt_to_players": 20,
                          "noncombat_damage_dealt_to_players": 7,
                          "combat_damage_taken": 5, "first_attack_turn": 4,
                          "creatures_lost": 2, "activations": 3, "triggers": 11,
                          "token_resolutions": 0, "tokens_observed": 0,
                          "token_attackers": 0, "token_blockers": 0,
                          "token_combat_damage_to_players": 0,
                          "token_damage_share": None, "tokens_chumped": 0,
                          "permanents_lost_by_turn": {},
                          "counter_events": 0, "mass_counter_events": 0,
                          "proliferate_events": 0,
                          "life_by_turn": {"1": 40, "2": 45, "3": 38},
                          "damage_to_players_by_turn": {}}
                      for s in ("me", "a")}}
    s = _agg([f], "me")
    assert s["noncombat_damage_dealt_to_players"]["mean"] == 7
    assert s["damage_dealt_total"]["mean"] == 27, "combat AND drain, together"
    for k in ("creatures_lost", "activations", "triggers"):
        assert k in s, f"{k} is computed and must be published"


def test_life_gained_and_lost_are_SEPARATE_facts():
    """A deck can gain forty and lose sixty and end where it started. Summing
    the movement into one number hides which deck it is. Measured on
    edgar-vampires: seventeen lifegain cards produce a MEDIAN of 2 life, which
    is what makes Vito, Sanguine Bond and Bloodthirsty Conqueror blanks."""
    per = {"life_by_turn": {"1": 40, "2": 45, "3": 30, "4": 33}}
    assert parse._life_delta(per, +1) == 8, "5 up then 3 up"
    assert parse._life_delta(per, -1) == 15, "15 down"


def test_a_seat_whose_life_never_moved_reports_zero_not_absent():
    assert parse._life_delta({"life_by_turn": {"1": 40, "2": 40}}, +1) == 0
    assert parse._life_delta({}, +1) == 0


# ── The counter channel ───────────────────────────────────────────────────

def _resolve_game(texts, seat="me"):
    """A game whose only content is resolved abilities, attributed to `seat`."""
    return {"seats": ["me", "a"], "turn": 5, "active": seat, "owner": {},
            "mulligan": {},
            "outcome": {"winner": "a", "won_by": None, "round": 5,
                        "global_turn": 5, "draw": False, "lost": {}, "ms": 1},
            "events": [
                {"kind": "resolve", "seat": seat, "turn": 1, "active": seat,
                 "text": t, "creates_token": False} for t in texts]}


def test_counter_placing_abilities_are_counted_and_attributed():
    """FIFTEEN CARDS IN edgar-vampires PUT +1/+1 COUNTERS AND NONE OF IT REACHED
    A RECORD. Forge emits no `Counter:` line — a counter only ever appears
    inside the ability text of a `Resolve Stack` line, 5,519 of them in one
    400-game run, and the parser read none. That made the whole
    counter-and-proliferate thesis unjudgeable."""
    g = _resolve_game([
        "Whenever Edgar Markov attacks, put a +1/+1 counter on each Vampire you control.",
        "Whenever this creature attacks, you may sacrifice another creature. "
        "If you do, put a +1/+1 counter on this creature.",
        "Blood Artist - Creature 0 / 1",
    ])
    p = parse.game_facts(g)["per_seat"]["me"]
    assert p["counter_events"] == 2, "two abilities placed counters, one did not"
    assert p["mass_counter_events"] == 1, "only the `on each` one scales with the board"


def test_proliferate_is_counted_separately_from_placing_a_counter():
    g = _resolve_game([
        "Whenever a commander you control enters or attacks, proliferate.",
        "Whenever Edgar Markov attacks, put a +1/+1 counter on each Vampire you control.",
    ])
    p = parse.game_facts(g)["per_seat"]["me"]
    assert p["proliferate_events"] == 1
    assert p["counter_events"] == 1, "proliferate places no counter of its own"


def test_an_opponents_counter_ability_is_not_credited_to_us():
    g = _resolve_game(["Whenever Youthful Valkyrie enters, put a +1/+1 counter on it."],
                      seat="a")
    per = parse.game_facts(g)["per_seat"]
    assert per["a"]["counter_events"] == 1
    assert per["me"]["counter_events"] == 0


def test_the_channel_lives_in_the_LIVE_resolve_branch():
    """RE-INTRODUCING THE BUG. The first cut of this put the counter logic in a
    SECOND `elif k == "resolve"` further down the dispatch chain, where the
    drain branch had already claimed every resolve event. It read a flat zero
    across 400 games while the regex matched seven times in the first log alone
    — a dead branch and a channel that never fires are indistinguishable from
    outside. This asserts there is exactly one such branch."""
    import inspect
    src = inspect.getsource(parse.game_facts)
    assert src.count('elif k == "resolve":') == 1, (
        "a second resolve branch is unreachable — merge it into the first")


def test_the_aggregate_publishes_the_counter_channel():
    g = _resolve_game([
        "Whenever Edgar Markov attacks, put a +1/+1 counter on each Vampire you control.",
        "Whenever a commander you control enters or attacks, proliferate.",
    ])
    f = parse.game_facts(g)
    for k in ("counter_events", "mass_counter_events", "proliferate_events"):
        assert k in f["per_seat"]["me"], k
    s = parse.aggregate([f], "me", {"me": "me", "a": "a"})["seats"]["me"]
    assert s["counter_events"]["mean"] == 1
    assert s["mass_counter_events"]["mean"] == 1
    assert s["proliferate_events"]["mean"] == 1


@requires_deck
def test_against_the_real_run_the_pilot_out_counters_the_pod():
    """THE CONTROL. A channel that reports the same number for every seat is
    measuring the log rather than the deck. bloodline is built on mass counter
    effects — Edgar's attack trigger, Cordial Vampire on every death — so it
    must lead its own pod on the `on each` figure."""
    import pathlib, glob
    from manamap.config import DATA_DIR
    ds = glob.glob(str(pathlib.Path(DATA_DIR)
                       / "decks/edgar-vampires/branches/bloodline/sim/logs/*/"))
    if not ds:
        pytest.skip("the bloodline pod run is not present")
    games = []
    for f in sorted(pathlib.Path(ds[0]).glob("*.log")):
        games += parse.parse_games(f.read_text(errors="replace"))
    facts = [parse.game_facts(g) for g in games if g.get("outcome", {}).get("winner")]
    assert len(facts) > 50
    seat = next(s for s in facts[0]["seats"] if "edgar" in s)
    mine = sum(f["per_seat"][seat]["mass_counter_events"] for f in facts) / len(facts)
    others = [sum(f["per_seat"][s]["mass_counter_events"] for f in facts) / len(facts)
              for s in facts[0]["seats"] if s != seat]
    assert mine > 1.0, f"the deck's own mass counter engine reads {mine}"
    assert mine > max(others), f"{mine} must lead the pod {others}"


# ── The AI is part of the instrument ──────────────────────────────────────

@requires_deck
def test_the_run_id_carries_a_non_default_pilot():
    """A SILENT OVERWRITE, and the same class as filing a branch measurement
    under the champion's name. The run id is built from the opponents, the game
    count, a digest over every seat's DECKLIST and the seed — none of which move
    when the AI does. So `--profile Experimental` wrote to exactly the path the
    Default run had already written and replaced it with no warning."""
    from manamap.sim import forge
    D, O = "edgar-vampires", ["vito"]          # config_digest reads real lists
    base = forge.run_id(D, O, 10, seed=7)
    assert forge.run_id(D, O, 10, seed=7, profile="Default") == base, (
        "the default pilot must leave every existing run id unchanged")
    assert forge.run_id(D, O, 10, seed=7, vs_profile="Default") == base
    mine = forge.run_id(D, O, 10, seed=7, profile="Experimental")
    pod = forge.run_id(D, O, 10, seed=7, vs_profile="Reckless")
    both = forge.run_id(D, O, 10, seed=7, profile="Experimental",
                        vs_profile="Reckless")
    assert len({base, mine, pod, both}) == 4, "four configurations, four paths"
    assert mine.endswith("-aiExperimental")
    assert pod.endswith("-vsaiReckless")


def test_the_pod_can_be_given_its_own_pilot():
    """THE POD IS PART OF THE INSTRUMENT. `--profile` set only our seat and left
    every opponent on Default, so the table could never be made to play
    differently — and a win rate is relative to the pod's competence as much as
    to the deck."""
    from manamap.sim import forge
    argv = forge.command(["me", "a", "b"], 1, 300, jar="j",
                         profiles=["Experimental", "Reckless", "Reckless"])
    i = argv.index("-a")
    assert argv[i + 1:i + 4] == ["Experimental", "Reckless", "Reckless"]
    assert forge.command(["me", "a"], 1, 300, jar="j") .count("-a") == 0, (
        "no profiles means no flag, so the default invocation is unchanged")


def test_profiles_are_positional_and_match_the_deck_order():
    """Forge reads `-a` positionally against `-d`. A profile list that is
    shorter or reordered silently hands your pilot to an opponent."""
    from manamap.sim import forge
    argv = forge.command(["me", "a", "b"], 1, 300, jar="j",
                         profiles=["Cautious", "Default", "Default"])
    def values_after(flag):
        """Everything after `flag` up to the next flag. `-d` and `-a` are not
        adjacent — `-f`, `-n` and `-c` sit between them — so a plain slice from
        one index to the other swallows those and reports nine decks."""
        i = argv.index(flag) + 1
        out = []
        while i < len(argv) and not argv[i].startswith("-"):
            out.append(argv[i]); i += 1
        return out

    decks, profs = values_after("-d"), values_after("-a")
    assert len(decks) == len(profs) == 3, f"{decks} vs {profs}"
    assert profs[0] == "Cautious", "our seat is first in both lists"
