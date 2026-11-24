"""Tokens into value, and which clock wins (2026-11-24, Edgar).

The pilot: "tokens into value... the tokens are the fodder for the engine... life gain, life
drain, card draw". The goldfish read Edgar as a combat deck because its attacks meet no
blockers, and reported nothing about the conversion the deck is built on. Three readings:

* FODDER: creature tokens made, tokens that died while a death payoff was out to pay on them,
  and the mana a free outlet makes off what it eats (Ashnod's Altar: 2).
* CLOCK: the life lost to anything but the attack, and what ended each game won by turn 10 --
  combat, non-combat, or a combo line's own clock.
* A COMBO LINE WHOLE: the turn a known line first had every piece out, apart from whether its
  finisher was there.

Proved by re-introducing the bugs: count no token in `creature_entered` and the fodder tests
fail; fold the sweep's drains back into the attack (`_swing = dealt`) and the attack exceeds
the damage it is part of.
"""
import random

import pytest

from manamap.pilot import diagnostic, goldfish
from manamap.pilot.goldfish_profiles import sac_mana_profile

from conftest import requires_data, requires_deck

ARTIST = "Whenever another creature you control dies, each opponent loses 1 life and you gain 1 life."
MAKER = "When this creature enters, create two 1/1 white Soldier creature tokens."


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def _doc(makers):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Artist", ARTIST, 2, "Creature — Vampire", quantity=1, power="0", toughness="1"),
             _c("Maker" if makers else "Plain Body", MAKER if makers else "", 2,
                "Creature — Human", quantity=20, power="2", toughness="2")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    return {"cards": cards}


def _games(doc, iterations=300, **kw):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(11)
    return [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_drain=True,
                                   model_sacrifice=True,
                                   model_deaths={"own_per_turn": 1.0, "opponent_per_turn": 0.0},
                                   **kw)
            for _ in range(iterations)]


# ── the outlet's mana ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, mana", [
    ("Sacrifice a creature: Add {C}{C}.", 2),                       # Ashnod's Altar
    ("Sacrifice a creature: Add one mana of any color.", 1),        # Phyrexian Altar
    ("Sacrifice another creature: Scry 1.", 0),                     # Viscera Seer
    ("{T}: Add {C}.\n{T}, Sacrifice a creature: Add {B}{B}.", 0),   # Phyrexian Tower: costed
])
def test_the_outlet_mana_reads_from_the_text(text, mana):
    assert sac_mana_profile({"oracle_text": text}) == mana


@requires_data
def test_the_corpus_sweep_is_three_cards():
    from manamap.pilot import card_pool
    hits = sorted(n for n, t in card_pool.corpus_oracle().items()
                  if t and sac_mana_profile({"oracle_text": t}))
    assert hits == ["Ashnod's Altar", "Everythingamajig", "Phyrexian Altar"], hits


# ── fodder ────────────────────────────────────────────────────────────────────

def test_tokens_made_and_converted_are_counted_and_ordered():
    games = _games(_doc(True))
    assert any(r["tokens_made_by_turn"][-1] for r in games)
    assert any(r["tokens_converted_by_turn"][-1] for r in games)
    for r in games:
        made, conv = r["tokens_made_by_turn"], r["tokens_converted_by_turn"]
        assert all(a <= b for a, b in zip(made, made[1:])), "cumulative"
        assert conv[-1] <= made[-1], "a token converted is a token that was made"


def test_the_fodder_series_are_absent_without_the_sacrifice_channel():
    library, _ = goldfish.build_library(_doc(True))
    r = goldfish.simulate_once(random.Random(3), library, 3, [], 10, model_combat=True)
    assert "tokens_converted_by_turn" not in r
    assert diagnostic.fodder([r]) is None and diagnostic.clock([r]) is None
    key = ("fodder", "tokens_converted_by_turn", "8")
    assert diagnostic.PER_GAME[key](r) is None


# ── which clock ───────────────────────────────────────────────────────────────

def _r(kill_turn, kill_by, swing, total):
    return {"kill_turn": kill_turn, "kill_by": kill_by,
            "combat_only_by_turn": swing, "total_damage_by_turn": total}


def test_the_kill_source_is_whichever_dealt_more():
    assert goldfish.kill_source(_r(3, "life", [10, 10, 10], [10, 12, 20])) == "combat"
    assert goldfish.kill_source(_r(3, "life", [2, 2, 2], [10, 15, 20])) == "noncombat"
    assert goldfish.kill_source(_r(3, "combo", [0, 0, 0], [0, 0, 0])) == "combo"
    assert goldfish.kill_source(_r(None, None, [0], [0])) is None


def test_noncombat_life_is_the_total_less_the_attack():
    r = _r(None, None, [5, 5, 5], [6, 9, 12])
    assert goldfish.noncombat_life(r, 2) == 5
    assert goldfish.noncombat_life(r, 3) == 12


def test_the_sweeps_drain_is_not_filed_as_the_attack():
    """Ashnod's Altar eats the tokens after combat while the Artist is out: those drains ride
    in the turn's damage, and the attack alone must read less than it."""
    doc = _doc(True)
    doc["cards"].insert(1, _c("Altar", "Sacrifice a creature: Add {C}{C}.", 3, "Artifact",
                              quantity=6))
    doc["cards"][-1]["quantity"] -= 6
    games = _games(doc, iterations=200)
    assert any(s < d for r in games
               for s, d in zip(r["combat_only_by_turn"], r["damage_by_turn"]))
    assert any(r["sac_mana_by_turn"][-1] for r in games), "the Altar made mana off what it ate"


def test_the_attack_never_exceeds_the_damage_it_is_part_of():
    for r in _games(_doc(True), iterations=100):
        assert len(r["combat_only_by_turn"]) == len(r["damage_by_turn"])
        assert all(s <= d + 1e-9 for s, d in zip(r["combat_only_by_turn"], r["damage_by_turn"]))


# ── on the deck it was built for ──────────────────────────────────────────────

@requires_deck
def test_edgar_reports_its_fodder_and_its_clock():
    d = goldfish.run("edgar-vampires", iterations=800, seed=7, quiet=True, _band=False,
                     with_results=True)
    m, rows = d["metrics"], d["_results"]
    assert m["fodder"]["mean_tokens_made_by_turn"]["8"] > 0
    assert m["fodder"]["mean_tokens_converted_by_turn"]["8"] > 0
    # the shares of what won sum to the kill rate by turn ten
    assert sum(m["clock"]["kill_source_by_turn_10"].values()) == pytest.approx(
        m["combat"]["kill_by_turn_rate"]["10"], abs=0.003)
    # Ashnod's Altar makes two a creature, and only off what the outlet sacrificed
    for r in rows:
        assert r["sac_mana_by_turn"][-1] <= 2 * r["sacrifices_by_turn"][-1]


@requires_deck
def test_a_combo_line_is_whole_at_least_as_often_as_it_kills():
    from manamap.pilot.common import deck_dir
    if not (deck_dir("edgar-vampires") / "branches" / "threat-v1" / "cards.json").exists():
        pytest.skip("threat-v1 is not on disk")
    rows = goldfish.run("edgar-vampires", iterations=1500, seed=7, quiet=True, _band=False,
                        branch="threat-v1", with_results=True)["_results"]
    assert "combo_assembled_turn" in rows[0]
    whole = sum(1 for r in rows if r["combo_assembled_turn"] is not None)
    killed = sum(1 for r in rows if r["kill_by"] == "combo")
    assert whole >= killed


# ── items 4-5 (2026-11-24): the opponents' deaths, typed attack triggers, life gained ─────

from manamap.pilot.goldfish_profiles import cast_token_profile, combat_profile, death_profile

NOBLE = "Whenever this creature or another creature dies, target player loses 1 life and you gain 1 life."
ZULAPORT = ("Whenever this creature or another creature you control dies, each opponent loses 1 "
            "life and you gain 1 life.")
KALITAS = ("Lifelink\nIf a nontoken creature an opponent controls would die, instead exile that "
           "card and create a tapped 2/2 black Zombie creature token.")
SEEKER = "Whenever a Vampire you control attacks, each opponent loses 1 life and you gain 1 life."
CAPTAIN = "Whenever this creature attacks, put a +1/+1 counter on each Vampire you control."


@pytest.mark.parametrize("text, anyone", [
    (NOBLE, True),
    (ZULAPORT, False),
    ("Whenever this creature or another Zombie dies, each opponent loses 1 life.", False),
    ("Whenever a creature dies, that creature's controller may draw a card.", False),
])
def test_a_death_payoff_knows_whose_creature_it_pays_on(text, anyone):
    assert death_profile({"oracle_text": text, "name": "X"})["death_any_controller"] is anyone


def test_the_typed_readers_read_their_cards():
    assert death_profile({"oracle_text": KALITAS, "name": "K"})["opp_death_token"] == {
        "power": 2, "toughness": 2, "type": "Zombie"}
    edgar = ("Eminence — Whenever you cast another Vampire spell, if Edgar Markov is in the "
             "command zone or on the battlefield, create a 1/1 black Vampire creature token.")
    assert cast_token_profile({"oracle_text": edgar})["token_type"] == "Vampire"
    body = {"name": "B", "type_line": "Creature — Vampire", "power": "2", "toughness": "2",
            "keywords": [], "cmc": 3}
    assert combat_profile(dict(body, oracle_text=SEEKER))["attack_drain_typed"] == {
        "type": "Vampire", "drain": 1, "gain": 1}
    assert combat_profile(dict(body, oracle_text=CAPTAIN))["attack_counters_typed"] == {
        "type": "Vampire", "n": 1}
    finneas = ("Whenever this creature attacks, put a +1/+1 counter on each other creature you "
               "control that's a token or a Rabbit.")
    assert combat_profile(dict(body, oracle_text=finneas))["attack_counters_typed"] is None


def _typed_doc(payoff_text, body_type, payoff_type="Creature — Human"):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Payoff", payoff_text, 2, payoff_type, quantity=10, power="1", toughness="3"),
             _c("Body", "", 2, "Creature — " + body_type, quantity=40, power="2", toughness="2")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    return {"cards": cards}


def _mean(rs, f):
    return sum(f(r) for r in rs) / len(rs)


def test_an_any_creature_payoff_drains_on_the_opponents_deaths():
    """Only opponents' creatures die here: Noble pays, Zulaport does not."""
    kw = dict(model_deaths={"own_per_turn": 0.0, "opponent_per_turn": 1.0})
    lib_n, _ = goldfish.build_library(_typed_doc(NOBLE, "Human"))
    lib_z, _ = goldfish.build_library(_typed_doc(ZULAPORT, "Human"))
    run = lambda lib: (lambda rng: [goldfish.simulate_once(
        rng, lib, 3, [], 10, model_combat=True, model_drain=True, model_sacrifice=True, **kw)
        for _ in range(200)])(random.Random(5))
    drained = lambda rs: _mean(rs, lambda r: sum(r["drain_by_turn"]))
    assert drained(run(lib_n)) > 0
    assert drained(run(lib_z)) == 0


def test_kalitas_turns_the_opponents_deaths_into_our_tokens():
    kw = dict(model_deaths={"own_per_turn": 0.0, "opponent_per_turn": 1.0})
    lib_k, _ = goldfish.build_library(_typed_doc(KALITAS, "Human"))
    lib_0, _ = goldfish.build_library(_typed_doc("", "Human"))
    run = lambda lib: (lambda rng: [goldfish.simulate_once(
        rng, lib, 3, [], 10, model_combat=True, model_drain=True, model_sacrifice=True, **kw)
        for _ in range(200)])(random.Random(5))
    made = lambda rs: _mean(rs, lambda r: r["tokens_made_by_turn"][-1])
    assert made(run(lib_k)) > made(run(lib_0)) + 1


def test_sanctum_seeker_drains_per_attacking_vampire_and_the_gain_is_counted():
    vamp = _games(_typed_doc(SEEKER, "Vampire"), 200)
    human = _games(_typed_doc(SEEKER, "Human"), 200)
    noncombat = lambda rs: _mean(rs, lambda r: goldfish.noncombat_life(r, 10))
    gained = lambda rs: _mean(rs, lambda r: r["life_gained_by_turn"][-1])
    assert noncombat(vamp) > noncombat(human) + 1
    assert gained(vamp) > gained(human)
    for r in vamp:
        g = r["life_gained_by_turn"]
        assert all(a <= b for a, b in zip(g, g[1:])), "cumulative"


def test_a_typed_attack_counter_grows_that_type_only():
    vamp = _games(_typed_doc(CAPTAIN, "Vampire"), 200)
    human = _games(_typed_doc(CAPTAIN, "Human"), 200)
    power = lambda rs: _mean(rs, lambda r: r["board_power_by_turn"][-1])
    assert power(vamp) > power(human) + 2


def test_the_life_and_hand_rows_are_absent_off_a_fodder_deck():
    library, _ = goldfish.build_library(_doc(True))
    r = goldfish.simulate_once(random.Random(3), library, 3, [], 10, model_combat=True,
                               model_drain=True)
    for key in (("clock", "life_gained_by_turn", "8"), ("clock", "hand_size_by_turn", "8")):
        assert diagnostic.PER_GAME[key](r) is None


# ── item 6 (2026-11-24): the draw the goldfish could not read ─────────────────────────────

from manamap.pilot.goldfish_profiles import first_main_modes_profile, subtypes_of
from manamap.pilot.goldfish import draw_is_unread

BMC = ("At the beginning of your first main phase, choose one or more —\n"
       "• Sell Contraband — Create a Treasure token. You lose 1 life.\n"
       "• Buy Information — Draw a card. You lose 2 life.\n"
       "• Hire a Mercenary — Create a 3/2 colorless Shapeshifter creature token with changeling. "
       "You lose 3 life. (It is every creature type.)")
CHAMPION = ("When this creature enters, you draw X cards and you lose X life, where X is the "
            "number of Vampires you control.")
CLAMP = "Equipped creature gets +1/-1.\nWhenever equipped creature dies, draw two cards.\nEquip {1}"


def test_a_card_that_makes_a_changeling_is_not_one():
    assert subtypes_of("Enchantment", BMC) == frozenset()
    shifter = subtypes_of("Creature — Shapeshifter", "Changeling (This card is every creature type.)")
    assert "Shapeshifter" in shifter


def test_black_market_connections_reads_its_three_modes():
    assert first_main_modes_profile({"oracle_text": BMC}) == {
        "treasure": 1, "draw": 1, "token_power": 3, "token_toughness": 2}


def test_the_three_draws_are_read_and_say_so():
    kw = dict(model_discard=False, model_combat=True, model_deaths=True)
    for name, text, tl in (("BMC", BMC, "Enchantment"),
                           ("Champion", CHAMPION, "Creature — Vampire Knight"),
                           ("Clamp", CLAMP, "Artifact — Equipment")):
        assert not draw_is_unread({"name": name, "oracle_text": text, "type_line": tl}, **kw), name


def _modes_doc(with_bmc):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("BMC" if with_bmc else "Plain Enchantment", BMC if with_bmc else "", 3,
                "Enchantment", quantity=10),
             _c("Body", "", 2, "Creature — Human", quantity=30, power="2", toughness="2")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    return {"cards": cards}


def _plain_games(doc, n=200, **kw):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(9)
    return [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_draw=True,
                                   model_treasures=True, model_sacrifice=True, **kw)
            for _ in range(n)]


def test_black_market_connections_draws_makes_treasure_and_a_body_every_turn():
    on, off = _plain_games(_modes_doc(True)), _plain_games(_modes_doc(False))
    extra = lambda rs: _mean(rs, lambda r: r["drawn_extra_by_turn"][-1])
    made = lambda rs: _mean(rs, lambda r: r["tokens_made_by_turn"][-1])
    assert extra(on) > extra(off) + 1
    assert made(on) > made(off) + 1


def test_champion_of_dusk_draws_for_every_vampire_tokens_included():
    """The bodies are Humans; only the TOKENS they make differ, so the gap is the tokens."""
    def doc(token_type):
        maker = f"When this creature enters, create two 1/1 black {token_type} creature tokens."
        cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
                 _c("Champion", CHAMPION, 5, "Creature — Vampire Knight", quantity=10,
                    power="4", toughness="4"),
                 _c("Body", maker, 2, "Creature — Human", quantity=30, power="2",
                    toughness="2")]
        cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
        return {"cards": cards}
    vamp, human = _plain_games(doc("Vampire")), _plain_games(doc("Human"))
    extra = lambda rs: _mean(rs, lambda r: r["drawn_extra_by_turn"][-1])
    assert extra(human) > 0, "Champion counts itself"
    assert extra(vamp) > extra(human) + 0.5
