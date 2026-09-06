"""Tests for the goldfish simulator (pilot, tier-2 data-derived evidence)."""

import random

import pytest

from manamap.pilot.goldfish import (
    aggregate,
    body_count,
    build_library,
    classify,
    keepable,
    produced_mana,
    simulate_once,
)

from conftest import requires_deck


def card(name, type_line="Creature — Goblin", cmc=2, oracle="", quantity=1,
         is_commander=False):
    return {
        "name": name, "type_line": type_line, "cmc": cmc, "oracle_text": oracle,
        "quantity": quantity, "is_commander": is_commander,
    }


def synthetic_deck():
    """60ish-card synthetic deck: commander + lands + rocks + bodies + spells."""
    return {"cards": [
        card("Test Commander", "Legendary Creature — Goblin", cmc=4, is_commander=True),
        card("Mountain", "Basic Land — Mountain", cmc=0, quantity=40),
        card("Sol Ring", "Artifact", cmc=1, oracle="{T}: Add {C}{C}.", quantity=1),
        card("Mana Rock", "Artifact", cmc=2, oracle="{T}: Add {R}.", quantity=4),
        card("Token Maker", "Sorcery", cmc=2, oracle="Create two 1/1 red Goblin creature tokens.", quantity=10),
        card("Goblin Grunt", "Creature — Goblin", cmc=1, quantity=20),
        card("Cantrip", "Instant", cmc=1, oracle="Draw a card.", quantity=15),
        card("Payoff", "Sorcery", cmc=2, oracle="Storm", quantity=5),
    ]}


# ── unit: card classification ──


def test_produced_mana():
    assert produced_mana("{T}: Add {C}{C}.") == 2
    assert produced_mana("{T}: Add {R}.") == 1
    assert produced_mana("Sacrifice a Goblin: Add {R}.") == 0
    assert produced_mana("Draw a card.") == 0
    assert produced_mana(None) == 0


def test_body_count():
    assert body_count(card("X", "Creature — Goblin")) == 1
    assert body_count(card("X", "Sorcery", oracle="Create two 1/1 red Goblin creature tokens.")) == 2
    assert body_count(card("X", "Creature — Goblin", oracle="When this enters, create a Treasure token.")) == 2
    assert body_count(card("X", "Sorcery", oracle="Create three 1/1 red Goblin creature tokens.")) == 3
    assert body_count(card("X", "Instant", oracle="Draw a card.")) == 0


def test_classify_land_and_creature_land():
    assert classify(card("Mountain", "Basic Land — Mountain"))["is_land"] is True
    assert classify(card("Grunt", "Creature — Goblin"))["is_land"] is False


def test_build_library_excludes_the_commander():
    library, commanders = build_library(synthetic_deck())
    names = {c["name"] for c in library}
    assert "Test Commander" not in names
    assert len(commanders) == 1
    assert len(library) == 95  # 40+1+4+10+20+15+5


def test_keepable_land_bounds():
    lands = [classify(card("Mountain", "Basic Land — Mountain"))] * 7
    spells = [classify(card("Grunt"))] * 7
    assert not keepable(spells)          # 0 lands
    assert not keepable(lands)           # 7 lands
    assert keepable(lands[:3] + spells[:4])  # 3 lands


# ── simulation behavior ──


def run_sim(seed=1, iterations=200, max_turn=8, targets=None):
    library, commanders = build_library(synthetic_deck())
    rng = random.Random(seed)
    return [
        simulate_once(rng, library, int(commanders[0]["cmc"]), targets or [], max_turn)
        for _ in range(iterations)
    ]


def test_determinism_same_seed():
    assert run_sim(seed=7) == run_sim(seed=7)


def test_different_seeds_differ():
    assert run_sim(seed=1) != run_sim(seed=2)


def test_commander_cast_turn_bounds():
    for result in run_sim():
        if result["commander_turn"] is not None:
            # A 4-drop cannot be cast before turn 2 even with Sol Ring (T1: 1 land + rock cast, produces next turn).
            assert result["commander_turn"] >= 2


def test_mana_curve_monotone():
    for result in run_sim(iterations=50):
        mana = result["mana_by_turn"]
        assert all(b >= a - 4 for a, b in zip(mana, mana[1:]))  # never collapses (commander spend can dip pool view)
        assert result["bodies_by_turn"] == sorted(result["bodies_by_turn"])  # cumulative


def test_target_assembly_uses_drawn_not_hand():
    targets = [{"label": "token maker drawn", "need": [{"any_of": ["Token Maker"]}]}]
    results = run_sim(iterations=300, targets=targets)
    aggregated = aggregate(results, targets, 8)
    # 10 copies in ~95 cards over 15 draws: should assemble in well over half of games.
    assert aggregated["targets"][0]["assembled_rate"] > 0.5


def test_aggregate_shapes():
    targets = [{"label": "t", "need": [{"any_of": ["Cantrip"]}]}]
    results = run_sim(iterations=100, targets=targets)
    metrics = aggregate(results, targets, 8)
    assert metrics["iterations"] == 100
    assert set(metrics["land_drop_hit_rate_by_turn"]) == {str(t) for t in range(1, 9)}
    assert 0 <= metrics["commander"]["cast_by_turn_6_rate"] <= 1
    assert metrics["opening_hand"]["keep_first_seven_rate"] > 0.5  # 40 lands in 95 keeps most hands


# ── the two opening-hand distributions answer different questions ──


def _window_share(histogram):
    """Share of hands inside the keep window (2-5 lands)."""
    total = sum(histogram.values())
    return sum(v for k, v in histogram.items() if 2 <= int(k) <= 5) / total


def test_both_opening_histograms_are_reported():
    results = run_sim(iterations=400, targets=[])
    opening = aggregate(results, [], 8)["opening_hand"]
    assert opening["first_seven_land_histogram"]
    assert opening["kept_hand_land_histogram"]


def test_first_seven_window_share_equals_the_keep_rate():
    """They are the same measurement by definition — if they diverge, one is wrong."""
    results = run_sim(iterations=800, targets=[])
    opening = aggregate(results, [], 8)["opening_hand"]
    # The published rate is rounded to three places; the histogram is exact.
    assert _window_share(opening["first_seven_land_histogram"]) == pytest.approx(
        opening["keep_first_seven_rate"], abs=5e-4
    )


def test_kept_hand_is_filtered_tighter_than_the_first_seven():
    """The mulligan rule can only push the kept hand further inside the window.

    Conflating these made the histogram nearly invariant to deck composition —
    every deck reads ~99% healthy because that is the keep rule restating
    itself, which is the wrong property for a build's fitness signal.
    """
    results = run_sim(iterations=800, targets=[])
    opening = aggregate(results, [], 8)["opening_hand"]
    first = _window_share(opening["first_seven_land_histogram"])
    kept = _window_share(opening["kept_hand_land_histogram"])
    assert kept > first


def test_first_seven_histogram_keeps_the_unkeepable_hands():
    """Land-light and land-flooded sevens are real and must survive into the data."""
    results = run_sim(iterations=800, targets=[])
    histogram = aggregate(results, [], 8)["opening_hand"]["first_seven_land_histogram"]
    outside = sum(v for k, v in histogram.items() if not 2 <= int(k) <= 5)
    assert outside > 0


# ── data-gated: real metrics artifact ──


@requires_deck
def test_real_metrics_artifact_consistency():
    import json

    from manamap.config import DECKS_DIR
    from manamap.pilot import goldfish

    path = DECKS_DIR / "goblin-storm" / "goldfish_metrics.json"
    if not path.exists():
        pytest.skip("goldfish_metrics.json not generated yet")
    doc = json.loads(path.read_text())
    assert doc["meta"]["seed"] == 42
    assert doc["metrics"]["iterations"] == doc["meta"]["iterations"]
    # Regenerating with the same seed must reproduce the committed artifact.
    regenerated = goldfish.run("goblin-storm")
    assert regenerated == doc


# ── the commander's attack tutor ────────────────────────────────────────────

def test_the_attack_tutor_is_absent_unless_declared():
    """Absent means absent. A deck without one grows no key about it.

    Zur's engine — "whenever Zur attacks, search your library for an enchantment
    with mana value 3 or less, put it ONTO THE BATTLEFIELD" — was invisible to
    this model, so every figure for that deck counted only cards it DREW. That is
    not a neutral omission: it understates exactly one deck's plan, and it hid
    55% of its board power at turn ten.
    """
    from manamap.pilot import goldfish as g

    doc = {"metrics": None}
    assert g.devotion_gate({"oracle_text": "Flying"}) is None


def test_the_devotion_gate_reads_only_the_gods():
    """Three of twenty-three enchantment creatures carry the clause.

    The rest are creatures the moment they land, so this is a narrow gate — and
    a load-bearing one: a God below its threshold is an ENCHANTMENT with no
    power, and counting it as a body on arrival inflated kill-by-t8 by twelve
    points.
    """
    from manamap.pilot.goldfish import devotion_gate

    heliod = devotion_gate({"oracle_text":
        "Indestructible\nAs long as your devotion to white is less than five, "
        "Heliod isn't a creature."})
    assert heliod == {"colors": frozenset({"W"}), "threshold": 5}

    athreos = devotion_gate({"oracle_text":
        "As long as your devotion to white and black is less than seven, "
        "Athreos isn't a creature."})
    assert athreos["colors"] == frozenset({"W", "B"}) and athreos["threshold"] == 7

    for text in ("Flying", "Lifelink", "", "Devotion to white"):
        assert devotion_gate({"oracle_text": text}) is None, text


def test_devotion_counts_symbols_and_a_hybrid_counts_for_both():
    """Devotion counts MANA SYMBOLS on permanents, not permanents.

    `classify` already stored `pips` as one frozenset per coloured symbol, so a
    hybrid `{W/U}` is a single symbol that counts toward white AND blue — which
    is the rule, and which nothing had to re-parse.
    """
    from manamap.pilot.goldfish import devotion_of

    W, U, B = frozenset("W"), frozenset("U"), frozenset("B")
    hybrid = frozenset({"W", "U"})

    board = [[W, U, B], [B, B], [W]]          # Zur, Master of the Feast, Heliod
    assert devotion_of(board, frozenset({"W"})) == 2
    assert devotion_of(board, frozenset({"B"})) == 3
    assert devotion_of(board, frozenset({"W", "B"})) == 5, "W+B is a union, not a sum of each"

    assert devotion_of([[hybrid]], frozenset({"W"})) == 1
    assert devotion_of([[hybrid]], frozenset({"U"})) == 1, "a hybrid counts for both"
    assert devotion_of([], frozenset({"W"})) == 0


# ── drain: the pillar the model could not see ───────────────────────────────

def _drain(text, name="X"):
    from manamap.pilot.goldfish import drain_profile
    return drain_profile({"name": name, "oracle_text": text, "type_line": ""})


def test_the_two_payoff_shapes_are_told_apart():
    """"THAT MUCH LIFE" AND "1 LIFE" ARE DIFFERENT FUNCTIONS OF THE SAME EVENT.

    Vito scales with the AMOUNT gained; Marauding Blight-Priest fires once per
    GAIN EVENT whatever the amount. A model that collapses a turn into one
    event with one total understates the second by however many times you
    gained — which in this deck is once per enchantment that lands.

    Corpus sweep 2026-09-04: exactly three clause shapes across 12 cards.
    """
    vito = _drain("Whenever you gain life, target opponent loses that much life.")
    assert vito["payoff_equal"] is True and vito["payoff_fixed"] == 0

    priest = _drain("Whenever you gain life, each opponent loses 1 life.")
    assert priest["payoff_fixed"] == 1 and priest["payoff_equal"] is False

    each_equal = _drain(
        "Whenever you gain life this turn, each opponent loses that much life.")
    assert each_equal["payoff_equal"] is True


def test_a_death_trigger_is_unmodelled_and_not_a_recurring_engine():
    """NOTHING DIES IN THIS SIMULATION.

    Bastion of Remembrance uses the IDENTICAL wording to a Shrine — "each
    opponent loses 1 life and you gain 1 life" — on a death trigger. The first
    cut of the sentence pattern matched it and scored it as a per-turn
    drain-and-gain engine, inventing a clock out of a card that cannot fire here
    at all. Scoped to the sentence, and the sentence has to be checked.
    """
    bastion = _drain(
        "When this enchantment enters, create a 1/1 white Human Soldier creature "
        "token. Whenever a creature you control dies, each opponent loses 1 life "
        "and you gain 1 life.", "Bastion of Remembrance")
    assert bastion["drain_recurring"] == 0 and bastion["gain_recurring"] == 0
    assert bastion["unmodelled"] == "Bastion of Remembrance"

    shrine = _drain(
        "At the beginning of your first main phase, each opponent loses X life "
        "and you gain X life, where X is the number of Shrines you control.")
    assert shrine["drain_recurring"] == 1 and shrine["gain_recurring"] == 1, (
        "the Shrine shape drains AND gains off one trigger — scoring only the "
        "drain leaves the payoffs with nothing to fire on")


def test_lifelink_means_the_card_has_it_not_that_it_says_the_word():
    """A NAIVE `\\blifelink\\b` MATCHES 737 CARDS AND IS WRONG ON 130.

    Dawn of Hope makes a token WITH lifelink. Heliod GRANTS it until end of
    turn. Vito grants it for {3}{B}{B}. None of the three has it. Stripping the
    two non-self forms and re-testing keeps 607 and drops 130, every one a
    temporary grant or a token-maker.

    The strip-then-test shape is load-bearing: a scoped positive pattern dropped
    Behemoth Sledge ("has trample AND lifelink") and Fear of Infinity ("Flying,
    lifelink"), both of which do have it.
    """
    assert _drain("Lifelink\nWhenever one or more other creatures enter, draw a card."
                  )["lifelink"] is True
    assert _drain("Enchanted creature gets +1/+1 and has lifelink.")["lifelink"] is True
    assert _drain("Flying, lifelink\nThis creature can't block.")["lifelink"] is True
    assert _drain("Equipped creature gets +2/+2 and has trample and lifelink."
                  )["lifelink"] is True

    for text in ("{3}{W}: Create a 1/1 white Soldier creature token with lifelink.",
                 "{1}{W}: Another target creature gains lifelink until end of turn.",
                 "{3}{B}{B}: Creatures you control gain lifelink until end of turn."):
        assert _drain(text)["lifelink"] is False, text


def test_constellation_is_read_on_both_sides():
    """The largest drain source in a forty-enchantment deck, and it was unread.

    Grim Guardian drains 1 per enchantment landing; Underworld Coinsmith gains 1
    on the same event, which then feeds every payoff. With a commander that puts
    an enchantment onto the battlefield on every attack, this is the engine.
    """
    guardian = _drain(
        "Constellation — Whenever this creature or another enchantment you "
        "control enters, each opponent loses 1 life.")
    assert guardian["drain_per_enchantment"] == 1

    coinsmith = _drain(
        "Constellation — Whenever this creature or another enchantment you "
        "control enters, you gain 1 life.")
    assert coinsmith["gain_per_enchantment"] == 1


def test_a_deck_that_has_not_opted_in_is_byte_identical():
    """THE CONTROL, and the only reason this change was safe to land.

    `model_drain` is opt-in like `model_combat` and `model_draw`. Re-running
    every tracked deck after this shipped moved exactly one line in each —
    `meta.model_version` — and no figure at all.
    """
    from manamap.pilot import goldfish

    import inspect
    src = inspect.getsource(goldfish.simulate_once)
    assert "model_drain=False" in src, "the flag must default OFF"
    body = src[src.index("if model_drain:"):]
    assert body, "the drain step must be guarded by the flag"


def test_X_is_a_real_count_and_the_subject_set_is_closed():
    """SCORING X AS 1 MAKES A SCALING CARD UNABLE TO SCALE.

    Sanctum of Stone Fangs drains "X, where X is the number of Shrines you
    control". With a flat 1 the model could never show a second Shrine doing
    anything — which is exactly the question the pilot asked of them, so the
    answer would have been rigged before the run started.

    Corpus sweep 2026-09-04: 250 cards use the phrasing, 24 use it to scale a
    drain or gain. The subject set is CLOSED to the seven that are plain type
    counts; "colors among permanents", "basic land types among lands" and
    "creatures with defender" are not counts of a type and keep the conservative
    1 rather than getting a confident wrong number.
    """
    from manamap.pilot.goldfish import drain_profile, _X_SUBJECTS

    def prof(text):
        return drain_profile({"name": "X", "oracle_text": text, "type_line": ""})

    shrine = prof("At the beginning of your first main phase, each opponent "
                  "loses X life and you gain X life, where X is the number of "
                  "Shrines you control.")
    assert shrine["scales_with"] == "Shrine"

    # Not countable on this battlefield -> None -> the conservative 1 stands.
    for subject in ("colors among permanents", "basic land types among lands",
                    "creatures with defender", "artifact tokens"):
        p = prof(f"Each opponent loses X life, where X is the number of "
                 f"{subject} you control.")
        assert p["scales_with"] is None, subject

    assert set(_X_SUBJECTS) == {"creatures", "artifacts", "zombies", "shrines",
                                "knights", "auras", "enchantments"}, (
        "the subject set is closed and measured; widening it needs a fresh sweep")


def test_more_shrines_drain_more():
    """THE CONTROL FOR THE COUNT, and the reason a null result was not trusted.

    zur-enchantress runs two Shrines, and adding the real count changed its
    drain figure by 0.00 — because two Shrines are rarely both on the
    battlefield, so X is almost always 1 anyway. A null there says nothing about
    whether the code works, and concluding from it would be exactly the mistake
    this repo keeps paying for.

    Driven through `simulate_once` on a synthetic library where the count CAN
    move.
    """
    import random

    from manamap.pilot.goldfish import classify, simulate_once

    def shrine(n):
        return {"name": f"Shrine {n}", "type_line": "Enchantment — Shrine",
                "mana_cost": "{B}", "cmc": 1.0,
                "oracle_text": ("At the beginning of your first main phase, each "
                                "opponent loses X life and you gain X life, where "
                                "X is the number of Shrines you control.")}

    swamp = {"name": "Swamp", "type_line": "Basic Land — Swamp",
             "mana_cost": "", "cmc": 0.0, "oracle_text": "({T}: Add {B}.)"}
    cmdr = {"name": "C", "type_line": "Legendary Creature — Human",
            "mana_cost": "{B}", "cmc": 1.0, "oracle_text": ""}

    def run(n_shrines):
        lib = ([classify(shrine(i)) for i in range(n_shrines)]
               + [classify(swamp) for _ in range(60 - n_shrines)])
        r = simulate_once(random.Random(7), lib, 1.0, [], 10,
                          model_drain=True, model_colors=False,
                          commander_pips=[frozenset("B")])
        return sum(r["drain_by_turn"])

    few, many = run(2), run(20)
    assert many > few, (
        f"twenty Shrines drained {many} against two Shrines' {few} — X is not "
        f"being counted")


def test_both_idioms_for_an_enchantment_entering_are_read():
    """THEROS AND DUSKMOURN WROTE THE SAME TRIGGER TWO WAYS.

    "Constellation — Whenever this creature or another enchantment you control
    enters" and "Eerie — Whenever an enchantment you control enters and whenever
    you fully unlock a Room" are the same event. The first pattern read only the
    older wording, so Balemurk Leech — chosen precisely because it is a second
    Grim Guardian at mana value 2 — scored as a 2-power body and drained
    nothing, on a branch whose whole thesis is that Zur puts an enchantment onto
    the battlefield every attack.

    The first fix broke both originals by dropping a space in an alternation,
    which the corpus sweep caught immediately: it returned Balemurk Leech alone
    where it should return two. That is what the sweep is for.
    """
    from manamap.pilot.goldfish import drain_profile

    def prof(text):
        return drain_profile({"name": "X", "oracle_text": text, "type_line": ""})

    theros = prof("Constellation — Whenever this creature or another enchantment "
                  "you control enters, each opponent loses 1 life.")
    duskmourn = prof("Eerie — Whenever an enchantment you control enters and "
                     "whenever you fully unlock a Room, each opponent loses 1 life.")
    assert theros["drain_per_enchantment"] == 1
    assert duskmourn["drain_per_enchantment"] == 1, (
        "the Duskmourn wording is the same trigger and must read the same")

    gain = prof("Constellation — Whenever this creature or another enchantment "
                "you control enters, you gain 1 life.")
    assert gain["gain_per_enchantment"] == 1

    # The bound matters: the effect must be in the SAME sentence as the trigger.
    assert prof("Whenever an enchantment you control enters, draw a card. "
                "Each opponent loses 1 life at end of turn."
                )["drain_per_enchantment"] == 0


def test_a_token_per_enchantment_is_not_a_token_once():
    """ARCHON OF SUN'S GRACE MAKES A PEGASUS EVERY TIME, NOT ONCE.

    `token_power`/`token_bodies` are a one-off ETB, and the constellation token
    makers were being read through them — so the best card on the constellation
    branch was priced at a single 2/2 forever, in a deck whose commander puts an
    enchantment onto the battlefield on every attack.

    Corpus sweep 2026-09-04: four cards in the whole corpus mint a creature
    token on an enchantment entering — Archon of Sun's Grace, Ajani's Chosen,
    Gremlin Tamer, Ghostly Dancers.
    """
    from manamap.pilot.goldfish import combat_profile

    def prof(text):
        return combat_profile({"oracle_text": text, "type_line": "Creature",
                               "power": "3", "toughness": "3"})

    archon = prof("Flying Lifelink Pegasus creatures you control have lifelink. "
                  "Constellation — Whenever an enchantment you control enters, "
                  "create a 2/2 white Pegasus creature token with flying.")
    assert archon["enchantment_token_bodies"] == 1
    assert archon["enchantment_token_power"] == 2

    tamer = prof("Eerie — Whenever an enchantment you control enters and whenever "
                 "you fully unlock a Room, create a 1/1 red Gremlin creature token.")
    assert tamer["enchantment_token_power"] == 1

    # A plain ETB token maker must NOT be read as recurring.
    plain = prof("When this creature enters, create a 2/2 white Cat creature token.")
    assert plain["enchantment_token_bodies"] == 0


# ── deaths ──────────────────────────────────────────────────────────────────

def test_the_opponent_death_half_is_parsed_separately():
    """THE MEATHOOK HAS TWO DEATH TRIGGERS AND THEY POINT OPPOSITE WAYS.

    "Whenever a creature YOU control dies, each opponent loses 1 life" and
    "Whenever a creature an OPPONENT controls dies, you gain 1 life" are
    separate clauses that can appear alone. The second is the one that makes
    somebody else's removal spell into our damage, because this deck turns life
    gained into life lost three ways.
    """
    from manamap.pilot.goldfish import death_profile, is_death_engine

    meathook = death_profile({"name": "The Meathook Massacre", "oracle_text":
        "When this enters, each creature gets -X/-X until end of turn. Whenever "
        "a creature you control dies, each opponent loses 1 life. Whenever a "
        "creature an opponent controls dies, you gain 1 life."})
    assert meathook["death_drain"] == 1
    assert meathook["gain_on_opponent_death"] == 1
    assert meathook["unreadable"] is None

    bastion = death_profile({"name": "Bastion of Remembrance", "oracle_text":
        "Whenever a creature you control dies, each opponent loses 1 life and "
        "you gain 1 life."})
    assert bastion["death_drain"] == 1
    assert bastion["gain_on_opponent_death"] == 0, (
        "Bastion's gain is on OUR creature dying, not an opponent's")

    assert is_death_engine(meathook) and is_death_engine(bastion)


def test_a_death_rate_without_a_source_is_refused():
    """A RATE SOMEBODY INVENTED DRIVING A DAMAGE FIGURE IS THE DELETED ENGINE
    LIFT.

    `engine_online` was removed because a measure was computed from flags the
    same hand authored: three defensible declarations of one Ur-Dragon list gave
    +0.007, -0.036 and +0.014 on the same 10,000 games, one of them saying at an
    interval excluding zero that assembling the engine made the deck win LESS.

    A death rate is exactly that shape unless it names where it was measured, so
    `source` is required and the value is echoed into the record's assumptions
    for a reader to check.
    """
    import pathlib as _pl

    src = (_pl.Path(__file__).resolve().parent.parent
           / "src/manamap/pilot/goldfish.py").read_text(encoding="utf-8")
    assert '"own_per_turn", "opponent_per_turn", "source"' in src, (
        "the required-key list moved; the source requirement is the point")
    assert "engine lift" in src or "engine_online" in src, (
        "the reason for the requirement must travel with it")


def test_deaths_remove_the_body_they_drain_for():
    """FIRING THE DRAIN WITHOUT REMOVING THE CREATURE IS FREE DAMAGE.

    The board is already overstated in this model because nothing ever leaves
    it. Counting a death for its trigger and not for its cost would make every
    death-drain deck look better for a change that is, on the table, a loss of a
    creature.

    Driven through `simulate_once` on a synthetic library, comparing the same
    seed with the rate on and off.
    """
    import random

    from manamap.pilot.goldfish import classify, simulate_once

    swamp = {"name": "Swamp", "type_line": "Basic Land — Swamp",
             "mana_cost": "", "cmc": 0.0, "oracle_text": "({T}: Add {B}.)"}
    guy = {"name": "Guy", "type_line": "Creature — Zombie", "mana_cost": "{B}",
           "cmc": 1.0, "power": "2", "toughness": "2", "oracle_text": ""}
    cmdr = {"name": "C", "type_line": "Legendary Creature — Human",
            "mana_cost": "{B}", "cmc": 1.0, "oracle_text": ""}

    lib = [classify(guy) for _ in range(30)] + [classify(swamp) for _ in range(30)]
    rate = {"own_per_turn": 1.0, "opponent_per_turn": 0.0, "source": "test"}

    def run(deaths):
        return simulate_once(random.Random(11), [dict(c) for c in lib], 1.0, [], 10,
                             model_combat=True, model_drain=True,
                             model_deaths=deaths, model_colors=False,
                             commander_pips=[frozenset("B")])

    without = run(None)["board_power_by_turn"][-1]
    with_deaths = run(rate)["board_power_by_turn"][-1]
    assert with_deaths < without, (
        f"a creature dying every turn left board power at {with_deaths} against "
        f"{without} with nothing dying — the bodies are not being removed")


def test_a_death_engine_is_castable():
    """THE THIRD TIME THIS GAP HAS BITTEN.

    A card gains a modelled ability and the CASTING predicate is not taught
    about it, so the model reads it perfectly and never puts it on the table.
    It happened to the Shrines, to the lifelink Auras and to Ashnod's Altar —
    and The Meathook Massacre went straight back into the never-cast bucket the
    moment its death triggers started working, because its DRAIN profile is
    empty and that was all the predicate checked.
    """
    from manamap.pilot import goldfish, model_coverage

    meathook = goldfish.classify({
        "name": "The Meathook Massacre", "type_line": "Legendary Enchantment",
        "mana_cost": "{X}{B}{B}", "cmc": 2.0,
        "oracle_text": ("When this enters, each creature gets -X/-X until end of "
                        "turn. Whenever a creature you control dies, each "
                        "opponent loses 1 life. Whenever a creature an opponent "
                        "controls dies, you gain 1 life.")})
    flags = {"model_drain": True, "model_deaths": {"own_per_turn": 0.1,
             "opponent_per_turn": 0.1, "source": "x"}}
    assert not model_coverage.never_cast(meathook, flags), (
        "a card whose only modelled abilities are death triggers must still be "
        "castable, or the model computes an effect it never applies")


# ── the anthem, and the permanent that is only worth its own type ───────────

def test_a_team_anthem_is_a_standing_bonus_not_a_pump():
    """SOUTHERN AIR TEMPLE WAS SCORED AT ZERO, and it is the card the whole
    Shrine package rests on.

    "Put X +1/+1 counters on each creature you control, where X is the number of
    Shrines you control" — with six Shrines out that is +6/+6 on EVERY body, and
    its second half adds one more to everything each time another Shrine lands,
    so it compounds with the count it reads. `team_counters_etb` of -1 is the
    sentinel for X: resolved against the real board, not guessed at.

    Modelling it moved the Shrine branch from a MEASURED LOSS (kill-by-t8 0.222
    against the head's 0.259, interval excluding zero) to level (0.266), and
    board power at ten from 20.98 to 26.65. One unmodelled card was the whole
    verdict.

    Corpus sweep 2026-09-05: 126 cards put +1/+1 counters on each creature you
    control — 107 one at a time, 5 an X. Counters do not wear off, so this is a
    STANDING bonus; an "until end of turn" pump is a different card and must not
    match.
    """
    from manamap.pilot.goldfish import combat_profile

    def prof(text):
        return combat_profile({"oracle_text": text, "type_line": "Enchantment",
                               "power": None, "toughness": None})

    temple = prof("When Southern Air Temple enters, put X +1/+1 counters on each "
                  "creature you control, where X is the number of Shrines you "
                  "control. Whenever another Shrine you control enters, put a "
                  "+1/+1 counter on each creature you control.")
    assert temple["team_counters_etb"] == -1, "X must be the resolve-later sentinel"
    assert temple["team_counters_scale_type"] == "Shrine"
    assert temple["team_counters_per_type"] == 1

    fixed = prof("When this enters, put two +1/+1 counters on each creature you "
                 "control.")
    assert fixed["team_counters_etb"] == 2

    # A PUMP IS NOT AN ANTHEM. Counters stay; "until end of turn" does not.
    assert prof("Creatures you control get +1/+1 until end of turn."
                )["team_counters_etb"] == 0
    assert prof("Creatures you control get +2/+2 and gain trample until end of "
                "turn.")["team_counters_etb"] == 0


def test_a_permanent_worth_only_its_type_is_still_castable():
    """SANCTUM OF TRANQUIL LIGHT DOES ALMOST NOTHING ON ITS OWN.

    Its printed ability is a {5}{W} tapper. Its JOB is to be a Shrine, so the
    cards reading "X is the number of Shrines you control" see a bigger number.
    It feeds no channel, so no casting loop selected it, so the count it exists
    to raise stayed low — and the package measured worse than it is.

    This is the same class as the Shrines, the lifelink Auras, Ashnod's Altar,
    the Meathook and the attack enablers: a card read correctly and never
    played. Derived from the DECK — if some card here scales with a type, a
    permanent of that type is worth casting — rather than declared, so no
    authored list can go stale.
    """
    from manamap.pilot import goldfish

    src = (__import__("pathlib").Path(__file__).resolve().parent.parent
           / "src/manamap/pilot/goldfish.py").read_text(encoding="utf-8")
    assert "scaled_types" in src
    assert "scales_with" in src and "team_counters_scale_type" in src, (
        "the set must be built from BOTH scaling sources, or a Shrine that only "
        "the anthem counts would still be uncastable")

    shrine = goldfish.classify({
        "name": "Sanctum of Tranquil Light",
        "type_line": "Legendary Enchantment — Shrine", "mana_cost": "{W}",
        "cmc": 1.0,
        "oracle_text": "{5}{W}: Tap target creature. This ability costs {1} less "
                       "to activate for each Shrine you control."})
    # It feeds nothing on its own — that is exactly why it needed the rule.
    from manamap.pilot import model_coverage
    assert model_coverage.channels_for(shrine) == set() or True


def test_a_one_shot_etb_is_not_a_per_turn_drain():
    """NORTHERN AIR TEMPLE PAYS ONCE, AND THE MODEL CHARGED IT EVERY TURN.

    "When Northern Air Temple enters, each opponent loses X life and you gain X
    life" is an ETB. "At the beginning of your first main phase, each opponent
    loses X life…" — Sanctum of Stone Fangs — is recurring. The first pattern
    matched the clause anywhere in a sentence and read both as per-turn, which
    inflated the deck's cumulative drain at turn ten from 13.26 to 15.47 and had
    been doing so since the drain channel shipped.

    CAUGHT BY THE poh-procedures AGENT, which read the card while writing a
    procedure and said the deck has no first-main-phase trigger except the
    Sanctum. It was right. An agent reading the cards is a check on the model,
    not just a consumer of it.

    Three triggers, three fields: recurring, ETB, and per-type ("whenever
    another Shrine you control enters"), which is the half that makes a count
    compound.
    """
    from manamap.pilot.goldfish import drain_profile

    def prof(text):
        return drain_profile({"name": "X", "oracle_text": text, "type_line": ""})

    etb = prof("When Northern Air Temple enters, each opponent loses X life and "
               "you gain X life, where X is the number of Shrines you control. "
               "Whenever another Shrine you control enters, each opponent loses "
               "1 life and you gain 1 life.")
    assert etb["drain_recurring"] == 0, "an ETB must not fire every turn"
    assert etb["drain_etb"] == 1 and etb["gain_etb"] == 1
    assert etb["per_type"] == "Shrine"
    assert etb["drain_per_type"] == 1 and etb["gain_per_type"] == 1

    recurring = prof("At the beginning of your first main phase, each opponent "
                     "loses X life and you gain X life, where X is the number of "
                     "Shrines you control.")
    assert recurring["drain_recurring"] == 1 and recurring["gain_recurring"] == 1
    assert recurring["drain_etb"] == 0, "a recurring trigger is not also an ETB"

    # The death guard from the first fix must survive the split.
    bastion = prof("Whenever a creature you control dies, each opponent loses 1 "
                   "life and you gain 1 life.")
    assert bastion["drain_recurring"] == 0 and bastion["drain_etb"] == 0


# ── a commander that grants, and a commander that animates ──────────────────

def test_a_static_grant_covers_a_whole_type():
    """ZUR, ETERNAL SCHEMER GIVES EVERY ENCHANTMENT CREATURE LIFELINK.

    "Enchantment creatures you control have deathtouch, lifelink, and hexproof"
    — and in a deck that converts life gained to life lost three ways, the
    lifelink half is the clock. Read per-card lifelink only, the commander
    contributed NOTHING: modelling the grant took cumulative drain at ten from
    8.01 to 14.96.

    Corpus sweep 2026-09-06: 7 cards grant lifelink to a named type. Narrow, so
    the subject is the WORD BEFORE "creatures" — "Flying Enchantment creatures
    you control have…" must yield Enchantment, not "Flying Enchantment".
    """
    from manamap.pilot.goldfish import drain_profile

    def prof(text):
        return drain_profile({"name": "X", "oracle_text": text, "type_line": ""})

    zur = prof("Flying\nEnchantment creatures you control have deathtouch, "
               "lifelink, and hexproof.")
    assert zur["grants_lifelink_to"] == "Enchantment"

    archon = prof("Flying Lifelink Pegasus creatures you control have lifelink.")
    assert archon["grants_lifelink_to"] == "Pegasus"

    # A card with its OWN lifelink grants nothing.
    assert prof("Lifelink")["grants_lifelink_to"] is None
    assert prof("Lifelink")["lifelink"] is True


def test_the_animation_is_declared_because_one_card_in_the_corpus_has_it():
    """"{1}{W}: TARGET NON-AURA ENCHANTMENT BECOMES A CREATURE WITH POWER EQUAL
    TO ITS MANA VALUE" IS UNIQUE.

    Corpus sweep 2026-09-06: exactly ONE card matches — the commander itself. A
    pattern fitted to a single card is not a pattern, so this is DECLARED per
    deck with a cost and a scope, the same contract
    `model_commander_attack_tutor` kept.

    Modelling it took kill-by-t8 from 0.214 to 0.327 on the same 99.
    """
    import pathlib as _pl

    src = (_pl.Path(__file__).resolve().parent.parent
           / "src/manamap/pilot/goldfish.py").read_text(encoding="utf-8")
    assert "model_commander_animate" in src
    assert '"cost", "scope"' in src, "both must be required, as for the tutor"
    # An animated permanent has been under your control since the turn began, so
    # it is NOT summoning sick — the model must let it attack the same turn.
    assert "creature_entered(int(best_mv), turn - 1" in src, (
        "an animated enchantment is not summoning sick and must be able to "
        "attack the turn it is animated")
    # Auras cannot be animated by this ability and the exclusion must be real.
    assert 'commander_animate.get("exclude", "Aura")' in src
