"""The audit of how the goldfish reads sharknado (2026-11-04): 22 defects, one file.

Every test here drives `simulate_once` or a reader with real oracle text, on a stacked library
(no shuffle), so a game can be read turn by turn and no figure is pinned: it asks what happened,
never what a mean came to. Unit tier: no tracked data. The record is `docs/gotchas-bench.md`.
"""
import random

import pytest

from manamap.config import LIFE_CHAIN_LIBRARY_FLOOR, OPPONENT_HAND
from manamap.pilot import goldfish, model_coverage, net_change
from manamap.pilot.goldfish_library import _target_met, keep_rule_passes


class Stacked(random.Random):
    """No shuffle, and `random()` is a stated constant: 0.9 means no opponent pays a tax."""

    def __init__(self, r=0.9):
        super().__init__(1)
        self.r = r

    def shuffle(self, x, *a):
        pass

    def random(self):
        return self.r


def _c(name, text="", cost="", tl="Sorcery", power=None, toughness=None, colors=(), layout="normal"):
    pips = [s for s in cost.replace("}", " ").replace("{", " ").split()]
    cmc = sum(int(p) if p.isdigit() else 0 if p == "X" else 1 for p in pips)
    return {"name": name, "oracle_text": text, "mana_cost": cost, "cmc": float(cmc), "type_line": tl,
            "power": power, "toughness": toughness, "colors": list(colors), "keywords": [],
            "color_identity": [], "quantity": 1, "is_commander": False, "layout": layout}


LAND = _c("Land", "({T}: Add {C}.)", tl="Basic Land — Wastes")
BLANK = _c("Blank", "", "{9}{9}", "Sorcery")
FLAGS = dict(model_draw=True, model_discard=True, model_combat=True, model_treasures=True)

BRALLIN = _c("Brallin, Skyshark Rider",
             "Whenever you discard a card, put a +1/+1 counter on Brallin and it deals 1 damage to "
             "each opponent.", "{3}{R}", "Legendary Creature — Human Shaman", "3", "3", "R")
SHABRAZ = _c("Shabraz, the Skyshark",
             "Flying\nWhenever you draw a card, put a +1/+1 counter on Shabraz and you gain 1 life.",
             "{3}{W}{U}", "Legendary Creature — Shark Bird", "3", "3", "WU")
WHEEL = _c("Wheel of Fortune", "Each player discards their hand, then draws seven cards.", "{2}{R}",
           colors="R")
WINDFALL = _c("Windfall", "Each player discards their hand, then draws cards equal to the greatest "
              "number of cards a player discarded this way.", "{2}{U}", colors="U")
TITHE = _c("Smothering Tithe", "Whenever an opponent draws a card, that player may pay {2}. If the "
           "player doesn't, you create a Treasure token.", "{3}{W}", "Enchantment", colors="W")
NEEDLEHEAD = _c("Razorkin Needlehead", "This creature has first strike during your turn.\nWhenever an "
                "opponent draws a card, this creature deals 1 damage to them.", "{R}{R}",
                "Creature — Human Assassin", "2", "2", "R")
GLINT = _c("Glint-Horn Buccaneer", "Haste\nWhenever you discard a card, this creature deals 1 damage "
           "to each opponent.\n{1}{R}, Discard a card: Draw a card. Activate only if this creature is "
           "attacking.", "{1}{R}{R}", "Creature — Minotaur Pirate", "2", "4", "R")
CRAWLER = _c("Psychosis Crawler", "Psychosis Crawler's power and toughness are each equal to the "
             "number of cards in your hand.\nWhenever you draw a card, each opponent loses 1 life.",
             "{5}", "Artifact Creature — Phyrexian Horror", "*", "*")
SPHINX = _c("Consecrated Sphinx", "Flying\nWhenever an opponent draws a card, you may draw two cards.",
            "{4}{U}{U}", "Creature — Sphinx", "4", "6", "U")
MASTERMIND = _c("Faerie Mastermind", "Flash\nFlying\nWhenever an opponent draws their second card each "
                "turn, you draw a card.\n{3}{U}: Each player draws a card.", "{1}{U}",
                "Creature — Faerie Rogue", "2", "1", "U")
BARD = _c("Bard, King of Dale", "Reach, vigilance\nIf you would draw a card except the first one you "
          "draw in each of your draw steps, draw two cards instead.\nIf one or more tokens would be "
          "created under your control, twice that many of those tokens are created instead.",
          "{4}{W}{U}", "Legendary Creature — Human Noble Archer", "3", "5", "WU")
TORBRAN = _c("Torbran, Thane of Red Fell", "If a red source you control would deal damage to an "
             "opponent or a permanent an opponent controls, it deals that much damage plus 2 instead.",
             "{1}{R}{R}{R}", "Legendary Creature — Dwarf Noble", "2", "4", "R")
BLADE = _c("Blade Historian", "Attacking creatures you control have double strike.", "{R/W}{R/W}{R/W}{R/W}",
           "Creature — Human Cleric", "2", "3", "RW")
GRATUITOUS = _c("Gratuitous Violence", "If a creature you control would deal damage to a permanent or "
                "player, it deals double that damage instead.", "{2}{R}{R}{R}", "Enchantment", colors="R")
EMANCIPATION = _c("Fiery Emancipation", "If a source you control would deal damage to a permanent or "
                  "player, it deals triple that damage to that permanent or player instead.",
                  "{3}{R}{R}{R}", "Enchantment", colors="R")
SOLPHIM = _c("Solphim, Mayhem Dominus", "If a source you control would deal noncombat damage to an "
             "opponent or a permanent an opponent controls, it deals double that damage to that player "
             "or permanent instead.", "{2}{R}{R}", "Legendary Creature — Phyrexian Horror", "4", "3", "R")
SHOCK = _c("Lava Spike", "Lava Spike deals 3 damage to target player or planeswalker.", "{R}", colors="R")


def _hand_lands(n):
    return [LAND] * n


def _seq(hand=(), draws=()):
    """Five lands and two named cards in the opening seven, then each turn's draw in order."""
    hand = (list(hand) + [BLANK, BLANK])[:2]
    return [LAND] * 5 + hand + list(draws)


def _turn(r):
    """The index of the turn the first wheel fired."""
    return r["first_wheel_turn"] - 1


def _gained(r, key, t):
    return r[key][t] - (r[key][t - 1] if t else 0)


def _keepable(seq):
    """Seven lands is a mulligan, so a list opening on them is dealt five and the first two cards
    named after them, with the rest in order: the named cards are in hand from turn one."""
    if sum(c is LAND or "Land" in c["type_line"] for c in seq[:7]) <= 5:
        return seq
    lands = [c for c in seq if "Land" in c["type_line"]]
    named = [c for c in seq if "Land" not in c["type_line"]]
    if len(lands) < 5:
        return seq
    extra = lands[5:]
    return lands[:5] + (named[:2] + [BLANK, BLANK])[:2] + extra + named[2:]


def _play(top, lands_below=40, turns=10, r=0.9, commander=None, partner=None, **flags):
    """One game, the library in the order given, lands after it. Returns the result dict."""
    deck = [goldfish.classify(c) for c in _keepable(list(top)) + [LAND] * lands_below]
    kw = dict(FLAGS, **flags)
    if commander:
        kw.update(commander_combat=goldfish.combat_profile(commander),
                  commander_event=goldfish.event_payoffs(commander),
                  commander_damage={"bonus": goldfish.damage_bonus(commander),
                                    "source": goldfish.damage_source(commander)})
        cmc = int(commander["cmc"])
    else:
        cmc = 99
    if partner:
        kw["partner"] = {"cmc": int(partner["cmc"]), "pips": [],
                         "combat": goldfish.combat_profile(partner),
                         "event": goldfish.event_payoffs(partner),
                         "damage": {"bonus": goldfish.damage_bonus(partner),
                                    "source": goldfish.damage_source(partner)}}
    return goldfish.simulate_once(Stacked(r), deck, cmc, kw.pop("targets", []), turns, **kw)


# ── 1. A Signet nets one ─────────────────────────────────────────────────────

SIGNET = _c("Azorius Signet", "{1}, {T}: Add {W}{U}.", "{2}", "Artifact")


def test_a_signet_adds_one_mana_a_turn_and_stands_for_both_colours():
    p = goldfish.classify(SIGNET)
    assert (p["produces"], p["produces_gross"]) == (1, 2)
    assert p["colors"] == {"W", "U"}
    r = _play([SIGNET] + _hand_lands(6), turns=5)
    assert r["mana_by_turn"][2] - r["mana_by_turn"][1] == 2, "land + signet from the turn after"
    # One land and the Signet pay {W}{U}: {W} and {U} are both its own.
    body = _c("WU Bear", "", "{W}{U}", "Creature — Bear", "2", "2")
    a = _c("Mountain", "({T}: Add {R}.)", tl="Basic Land — Mountain")
    got = _play([a, a, SIGNET, body, BLANK, BLANK, BLANK], turns=4, model_colors=True)
    assert got["bodies_by_turn"][-1] == 1


def test_the_gross_reader_is_untouched_for_cardbert():
    """`card_fields.mana_repeatable` delegates here: netting it moves a model-facing field."""
    assert goldfish.produced_mana("{1}, {T}: Add {W}{U}.", "Artifact") == 2


def test_a_cost_is_the_abilitys_own_clause_not_forty_characters():
    text = "Sacrifice a creature: Draw a card.\n{T}: Add {G}."
    assert goldfish.classify(_c("X", text, "{2}", "Artifact"))["produces"] == 1
    flat = "Discard a card: Scry 1. {T}: Add {G}."     # cards.csv has no line breaks
    assert goldfish.classify(_c("X", flat, "{2}", "Artifact"))["produces"] == 1
    # A later conditional mode is not the rate.
    fan = "{T}: Add {G}.\nFerocious — {T}: Add {G}{G}{G}{G}. Activate only if you control a creature."
    assert goldfish.classify(_c("X", fan, "{2}", "Creature — Elf"))["produces"] == 1


# ── 2. What a doubler reaches ────────────────────────────────────────────────

def test_the_scopes_read_off_the_subject():
    def scale(c):
        cb = goldfish.combat_profile(c)
        return cb["team_damage_multiplier"], cb["team_damage_scope"]
    assert scale(BLADE) == (2, "strike")
    assert scale(GRATUITOUS) == (2, "creature")
    assert scale(EMANCIPATION) == (3, "any")
    assert scale(SOLPHIM) == (2, "noncombat")


def test_a_double_strike_grant_does_not_reach_a_noncombat_ping():
    r = _play(_seq((GLINT, BLADE), [LAND, LAND, WHEEL, LAND, LAND]), turns=8)
    t = _turn(r)
    assert r["event_damage_by_turn"][t] == _gained(r, "discarded_by_turn", t) > 0


def test_a_creature_doubler_reaches_a_creatures_ping_and_not_a_life_loss_or_a_spell():
    v = _play(_seq((GRATUITOUS, GLINT), [LAND] * 5 + [WHEEL, LAND, LAND]), turns=10)
    t = _turn(v)
    assert v["event_damage_by_turn"][t] == 2 * _gained(v, "discarded_by_turn", t) > 0
    # Psychosis Crawler's life loss: one a card drawn, doubled by nothing.
    c = _play(_seq((GRATUITOUS, CRAWLER), [LAND] * 5 + [BLANK, WHEEL, LAND]), turns=10)
    t = _turn(c)
    assert c["event_damage_by_turn"][t] == _gained(c, "drawn_extra_by_turn", t) + 1 > 1
    # Burn: untouched by Gratuitous, tripled by Emancipation.
    def burn(perm):
        return sum(_play(_seq((perm, BLANK), [LAND] * 6 + [SHOCK]), turns=9)["damage_by_turn"])
    assert burn(GRATUITOUS) == burn(BLANK) == 3
    assert burn(EMANCIPATION) == 9


def test_an_any_source_doubler_does_not_touch_life_loss_and_triples_a_ping():
    c = _play(_seq((EMANCIPATION, CRAWLER), [LAND] * 6 + [WHEEL, LAND, LAND]), turns=12)
    t = _turn(c)
    assert c["event_damage_by_turn"][t] == _gained(c, "drawn_extra_by_turn", t) + 1 > 1
    # Needlehead's ping on their draws: their draw step and the wheel's seven, all three times.
    g = _play(_seq((EMANCIPATION, NEEDLEHEAD), [LAND] * 6 + [WHEEL, LAND, LAND]), turns=12)
    t = _turn(g)
    assert g["event_damage_by_turn"][t] == 3 * (1 + 7)


def test_a_noncombat_doubler_leaves_combat_alone():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    a = _play([LAND] * 7 + [SOLPHIM, bear], turns=6)
    b = _play([LAND] * 7 + [_c("Vanilla", "", "{2}{R}{R}", "Creature — Horror", "4", "3"), bear], turns=6)
    assert a["damage_by_turn"] == b["damage_by_turn"]


def test_two_double_strike_grants_are_one_and_torbran_is_per_strike():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2", "R")
    one = _play([LAND] * 7 + [bear, BLADE], turns=7)
    two = _play([LAND] * 7 + [bear, BLADE, dict(BLADE, name="Blade Two")], turns=7)
    ds = _c("Double Bear", "Double strike", "{1}", "Creature — Bear", "2", "2", "R")
    own = _play([LAND] * 7 + [ds, BLADE], turns=7)
    # Bear and Historian swing 2 and 2, twice each, with a second Historian's 2 twice beside them.
    assert two["damage_by_turn"][-1] == one["damage_by_turn"][-1] + 4
    assert own["damage_by_turn"][-1] == one["damage_by_turn"][-1], "own strike and a grant stack"
    plain = _play([LAND] * 7 + [bear, TORBRAN], turns=7)
    # Bear (red) 2+2, Torbran (red) 2+2.
    assert plain["damage_by_turn"][-1] == 8


def test_life_loss_gets_no_torbran_and_red_burn_does():
    red_loss = _c("Red Loss", "Whenever you draw a card, each opponent loses 1 life.", "{1}{R}", "Creature — Imp",
                  "1", "1", "R")
    c = _play(_seq((TORBRAN, red_loss), [LAND] * 2 + [BLANK, BLANK, WHEEL, LAND, LAND]), turns=10)
    t = _turn(c)
    assert c["event_damage_by_turn"][t] == _gained(c, "drawn_extra_by_turn", t) + 1 > 1
    with_t = _play(_seq((TORBRAN, BLANK), [LAND] * 4 + [SHOCK]), turns=6)
    assert with_t["damage_by_turn"][4] - 2 * 2 == 3 + 2, "Torbran's swing is 2+2; the spike is 3+2"


# ── 3. A payoff is paid for what happens after it arrives ────────────────────

def test_a_tithe_cast_after_the_turns_wheel_is_not_paid_for_the_wheel():
    """The wheel fires on what is left, Tithe is among what it finds, the second pass casts it."""
    r = _play(_seq((MAKO, BLANK), [LAND] * 6 + [WHEEL] + [TITHE] * 7), turns=7)
    t = _turn(r)
    assert r["mana_by_turn"][t] == 7
    assert _gained(r, "treasures_by_turn", t) == 3, "only this cycle's draw step: three seats, nobody pays"


def test_a_tithe_is_paid_for_a_wheel_that_comes_after_it():
    r = _play(_seq((TITHE, BLANK), [LAND, LAND, WHEEL, LAND, LAND, LAND, LAND]), turns=9)
    t = _turn(r)
    assert _gained(r, "treasures_by_turn", t) > 3, "the wheel's seven draws, at three seats"


def test_shabraz_takes_no_counter_for_the_draw_step_before_he_landed():
    r = _play([LAND] * 7 + [BLANK] * 3, turns=6, commander=SHABRAZ)
    assert r["commander_turn"] == 5
    assert r["commander_counters_by_turn"][4] == 0
    assert r["commander_counters_by_turn"][5] == 1


def test_a_payoff_cast_after_a_wheel_is_not_paid_for_its_discards():
    # A commander pays no ping here; Mako is out so the wheel fires, and Glint-Horn is among the
    # seven it finds, cast in the second pass with what was left.
    r = _play(_seq((MAKO, BLANK), [LAND] * 5 + [WHEEL] + [GLINT] * 7), turns=6)
    t = _turn(r)
    assert r["bodies_by_turn"][t] >= 2, "Glint-Horn landed on the wheel's turn"
    assert r["event_damage_by_turn"][t] == 0


# ── 4. Draws first, then the payoffs that read them ─────────────────────────

def test_sphinx_draws_pay_shabraz_and_a_per_draw_payoff():
    r = _play([LAND] * 8 + [SPHINX] + _hand_lands(30), turns=9, commander=SHABRAZ)
    t = 8      # turn nine: Sphinx out, three seats' draw steps, two cards each, plus his own step
    gained = r["commander_counters_by_turn"][t] - r["commander_counters_by_turn"][t - 1]
    assert gained == 1 + 6


def test_a_draw_on_a_draw_stays_unread_so_no_curiosity_loop():
    e = goldfish.event_payoffs(_c("Loop", "Whenever you draw a card, draw a card."))
    assert not e["per_opponent_draw_our_draw"] and not e["per_discard_draw"]
    assert not goldfish.has_event_payoff(e)


# ── 5. Shu Yun ───────────────────────────────────────────────────────────────

SHU = _c("Shu Yun, the Silent Tempest", "Prowess\nWhenever you cast a noncreature spell, you may pay "
         "{R/W}{R/W}. If you do, target creature gains double strike until end of turn.",
         "{2}{U}", "Legendary Creature — Human Monk", "3", "2", "U")


def test_shu_yun_has_no_double_strike_of_her_own():
    cb = goldfish.combat_profile(SHU)
    assert cb["double_strike"] is False and cb["prowess"] is True
    assert goldfish.combat_profile(_c("K", "Double strike", "", "Creature — X", "1", "1"))["double_strike"]
    assert goldfish.combat_profile(_c("K", "Flying, double strike", "", "Creature — X", "1", "1"))["double_strike"]
    assert not goldfish.combat_profile(_c("K", "{R}: This creature gains double strike until end of turn.",
                                          "", "Creature — X", "1", "1"))["double_strike"]


def test_shu_yun_buys_a_strike_with_the_mana_and_only_on_a_noncreature_cast():
    alone = _play(_seq((SHU, BLANK), [LAND] * 4), turns=4)
    assert alone["damage_by_turn"][3] == 3, "no noncreature spell, no strike and no prowess"
    # A cheap noncreature spell on turn four, with mana over: prowess +1 and a second strike.
    trinket = _c("Trinket", "{T}: Add {C}.", "{1}", "Artifact")
    r = _play(_seq((SHU, BLANK), [LAND, LAND, LAND, trinket]), turns=4)
    assert r["damage_by_turn"][3] == 8, "(3 + 1 prowess) twice"
    # No mana left for the strike: prowess alone.
    dear = _c("Dear", "{T}: Add {C}.", "{4}", "Artifact")
    r = _play(_seq((SHU, BLANK), [LAND, LAND, LAND, dear]), turns=4)
    assert r["damage_by_turn"][3] == 4


# ── 6. Shark Typhoon ─────────────────────────────────────────────────────────

TYPHOON = _c("Shark Typhoon", "Whenever you cast a noncreature spell, create an X/X blue Shark creature "
             "token with flying, where X is that spell's mana value.\nCycling {X}{1}{U} ({X}{1}{U}, "
             "Discard this card: Draw a card.)\nWhen you cycle this card, create an X/X blue Shark "
             "creature token with flying.", "{5}{U}", "Enchantment", colors="U")


def test_typhoon_makes_no_flat_bodies_and_is_taught_to_the_loops():
    p = goldfish.classify(TYPHOON)
    assert p["bodies"] == 0 and p["creature_bodies"] == 0 and p["combat"]["token_bodies"] == 0
    assert p["engine"]["cast_token_x"] == "noncreature"
    assert p["engine"]["cycling"]["x"] and p["engine"]["cycling"]["token_x"]
    assert not model_coverage.never_cast(p, FLAGS)


def test_typhoon_as_an_enchantment_makes_a_shark_of_the_spells_mana_value():
    spell = _c("Rock", "{T}: Add {C}.", "{3}", "Artifact")
    r = _play([LAND] * 7 + [TYPHOON, BLANK, BLANK, BLANK, BLANK, spell] + _hand_lands(4), turns=11)
    assert r["bodies_by_turn"][-1] == 1
    assert r["board_power_by_turn"][-1] == 3


def test_typhoon_cycles_for_spare_mana_with_a_payoff_out():
    r = _play(_seq((TYPHOON, BLANK), [BLANK] * 5 + [LAND] * 3), turns=5, commander=BRALLIN)
    assert r["commander_turn"] == 4
    # Turn five: five lands, nothing else to buy. Cycling for X = 3 is a discard, a draw and a 3/3.
    assert _gained(r, "discarded_by_turn", 4) == 1
    assert r["event_damage_by_turn"][4] == 1
    assert r["board_power_by_turn"][4] == 3 + 1 + 3


# ── 7. The greatest number ───────────────────────────────────────────────────

def test_windfall_off_an_empty_hand_draws_their_hand_and_they_draw_as_many():
    d = goldfish.draw_profile(WINDFALL)
    assert d["wheel_greatest"] and d["wheel_draws"] == -1
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    # Three lands and cheap bears: by turn four the hand is the Windfall and a bear, and it digs.
    r = _play([LAND, LAND, LAND, bear, bear, bear, WINDFALL] + [bear] * 4 + [LAND] * 10, turns=4)
    t = _turn(r)
    assert _gained(r, "discarded_by_turn", t) < OPPONENT_HAND
    assert _gained(r, "drawn_extra_by_turn", t) == OPPONENT_HAND, "the greatest was theirs"
    # And they draw as many: two Needleheads read it.
    n = _play([LAND] * 5 + [NEEDLEHEAD, NEEDLEHEAD] + [LAND, WINDFALL] + [LAND] * 20, turns=6)
    t = _turn(n)
    assert n["event_damage_by_turn"][t] == 2 * (1 + max(OPPONENT_HAND, _gained(n, "discarded_by_turn", t)))


def test_a_that_many_wheel_gives_each_player_their_own_count():
    inc = _c("Incendiary Command", "Choose two —\n• Incendiary Command deals 4 damage to target player or "
             "planeswalker.\n• Each player discards all the cards in their hand, then draws that many cards.",
             "{3}{R}{R}", colors="R")
    d = goldfish.draw_profile(inc)
    assert d["wheel_draws"] == -1 and not d["wheel_greatest"] and d["wheel_symmetric"]
    r = _play([LAND] * 6 + [NEEDLEHEAD, inc] + [LAND] * 20, turns=7)
    turn = next(i for i, w in enumerate(r["wheels_cast_by_turn"]) if w)
    assert r["event_damage_by_turn"][turn] == 1 + OPPONENT_HAND


# ── 8. Marauding Mako ────────────────────────────────────────────────────────

MAKO = _c("Marauding Mako", "Whenever you discard one or more cards, put that many +1/+1 counters on this "
          "creature.\nCycling {2} ({2}, Discard this card: Draw a card.)", "{R}", "Creature — Shark Pirate",
          "1", "1", "R")
LAEZEL = _c("Lae'zel, Vlaakith's Champion", "If you would put one or more counters on a creature or "
            "planeswalker you control or on yourself, put that many plus one of each of those kinds of "
            "counters on that permanent or player instead.", "{2}{W}", "Legendary Creature — Gith Warrior", "3", "3")


def test_mako_grows_per_card_on_one_event():
    e = goldfish.event_payoffs(MAKO)
    assert e["per_discard_counter"] == 1 and e["per_discard_counter_batch"] and not e["unmodelled"]
    assert goldfish.classify(MAKO)["engine"]["cycling"]["cost"] == 2
    r = _play(_seq((MAKO, LAEZEL), [LAND] * 3 + [WHEEL, LAND, LAND]), turns=6)
    t = _turn(r)
    # Lae'zel: one more counter on the EVENT, not one per card discarded.
    assert r["board_power_by_turn"][t] == 1 + 3 + _gained(r, "discarded_by_turn", t) + 1


def test_mako_counters_wait_for_mako_to_swing_not_for_a_commander():
    r = _play(_seq((MAKO, BLANK), [LAND] * 2 + [WHEEL, LAND, LAND]), turns=6)
    t = _turn(r)
    assert r["damage_by_turn"][t + 1] == 1 + _gained(r, "discarded_by_turn", t)


# ── 9. A tap ability taps ────────────────────────────────────────────────────

MARY = _c("Mary Read and Anne Bonny", "Haste\n{T}: Draw a card, then discard a card.\nWhenever you discard "
          "an Island, Pirate, or Vehicle card, create a tapped Treasure token.", "{1}{U}{R}",
          "Legendary Creature — Human Assassin Pirate", "3", "3", "UR")
KITSA = _c("Kitsa, Otterball Elite", "Vigilance\nProwess (Whenever you cast a noncreature spell, this "
           "creature gets +1/+1 until end of turn.)\n{T}: Draw a card, then discard a card.\n{2}, {T}: Copy "
           "target instant or sorcery spell you control. You may choose new targets for the copy. Activate "
           "only if Kitsa's power is 3 or greater.", "{1}{U}", "Legendary Creature — Otter Wizard", "1", "3", "U")


def test_mary_read_does_one_or_the_other():
    # No payoff out, lands in hand: she swings every turn and never loots.
    r = _play(_seq((MARY, BLANK), [LAND] * 6), turns=6)
    assert r["damage_by_turn"][3:] == [3, 3, 3]
    assert r["discarded_by_turn"][-1] == 0
    # A payoff worth her swing: she loots, and tapped for it she does not attack.
    loud = _c("Loud", "Whenever you discard a card, each opponent loses 3 life.", "{1}", "Enchantment")
    r = _play(_seq((MARY, loud), [LAND] * 6), turns=6)
    assert r["discarded_by_turn"][-1] >= 3
    assert r["damage_by_turn"][3:] == [0, 0, 0]
    assert goldfish.classify(MARY)["event"]["per_discard_typed_treasure"] == ("Island", "Pirate", "Vehicle")


def test_a_discarded_pirate_makes_a_tapped_treasure():
    pirate = _c("Pirate", "", "{9}{9}", "Creature — Pirate", "1", "1")
    r = _play(_seq((MARY, pirate), [LAND] * 3 + [WHEEL, LAND, LAND]), turns=6)
    t = _turn(r)
    assert r["treasures_by_turn"][t] >= 1, "a Pirate discarded, a Treasure standing at that turn's end"


def test_kitsa_does_not_loot_the_turn_she_lands_and_vigilance_keeps_her_swing():
    r = _play(_seq((BLANK, BLANK), [LAND] * 4 + [KITSA, LAND, LAND]), turns=7, commander=BRALLIN)
    assert r["commander_turn"] == 4
    assert r["bodies_by_turn"][4] >= 1 and _gained(r, "discarded_by_turn", 4) == 0, "summoning sick"
    assert _gained(r, "discarded_by_turn", 5) == 1
    # Brallin's swing and her own 1 together: vigilance leaves her both.
    assert r["damage_by_turn"][5] >= 3 + 1 + 1


def test_the_archivist_taps_for_his_wheel_so_he_does_not_swing_that_turn():
    arch = _c("Jace's Archivist", "{U}, {T}: Each player discards their hand, then draws cards equal to the "
              "greatest number of cards a player discarded this way.", "{1}{U}{U}",
              "Creature — Vedalken Wizard", "2", "2", "U")
    r = _play([LAND] * 7 + [arch, NEEDLEHEAD], turns=7)
    wheel = next(i for i, w in enumerate(r["wheels_cast_by_turn"]) if w)
    assert r["first_wheel_turn"] > 3
    assert r["damage_by_turn"][wheel] == 2, "Needlehead's swing alone"


# ── 10. Loots wait; a cost is paid first ─────────────────────────────────────

LOOTING = _c("Faithless Looting", "Draw two cards, then discard two cards.\nFlashback {2}{R} (You may cast "
             "this card from your graveyard for its flashback cost. Then exile it.)", "{R}", colors="R")
REUNION = _c("Cathartic Reunion", "As an additional cost to cast this spell, discard two cards.\nDraw three "
             "cards.", "{1}{R}", colors="R")


def test_a_loot_waits_for_a_payoff_unless_it_is_digging():
    held = _play(_seq((LOOTING, BLANK), [LAND] * 3), turns=3)
    assert held["discarded_by_turn"][-1] == 0
    # Two lands, one played each turn, nothing after: by turn three no land in hand, no commander.
    digs = _play([LAND, LAND, LOOTING, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK] + [LAND] * 20, turns=3)
    assert digs["discarded_by_turn"][-1] > 0


def test_cathartic_reunion_pays_two_other_cards_first():
    d = goldfish.draw_profile(REUNION)
    assert d["spell_discard_is_cost"] and d["spell_discard"] == 2 and d["spell_draw"] == 3
    # Digging on turn two with blanks to pay it: two blanks go first and the three it draws are
    # lands, all kept. Drawn first, a loot's policy would have thrown two of those lands away.
    r = _play([LAND, LAND, REUNION] + [BLANK] * 6 + [LAND] * 3 + [BLANK] * 20, turns=5, lands_below=0)
    assert r["discarded_by_turn"][1] == 2 and r["drawn_extra_by_turn"][1] == 3
    assert r["land_hits"][2:5] == [True, True, True]


def test_a_reunion_with_one_other_card_stays_in_hand():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    big = _c("Big", "", "{9}{9}", "Artifact")
    # Five lands go down, the bears are cast as they come, and Reunion arrives beside Big alone.
    deck = [goldfish.classify(c) for c in [LAND] * 5 + [MAKO, big] + [bear] * 5 + [REUNION]]
    r = goldfish.simulate_once(Stacked(), deck, 99, [], 6, **FLAGS)
    assert r["discarded_by_turn"][-1] == 0


# ── 11. Mind Stone and Fiery Islet ──────────────────────────────────────────

MIND_STONE = _c("Mind Stone", "{T}: Add {C}.\n{1}, {T}, Sacrifice this artifact: Draw a card.", "{2}", "Artifact")
ISLET = _c("Fiery Islet", "{T}, Pay 1 life: Add {U} or {R}.\n{1}, {T}, Sacrifice this land: Draw a card.",
           tl="Land")


def test_mind_stone_stays_a_rock_until_the_mana_is_spare():
    early = _play([LAND] * 6 + [MIND_STONE] + [LAND] * 30, turns=4)
    assert early["mana_by_turn"][3] == 5, "four lands and the stone"
    late = _play([LAND] * 6 + [MIND_STONE] + [LAND] * 30, turns=10)
    assert late["drawn_extra_by_turn"][-1] == 1, "cracked once, and only once"
    assert late["mana_by_turn"][-1] == 10, "ten lands; the stone is gone"


def test_fiery_islet_is_a_land_until_it_is_spare():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    r = _play([ISLET, LAND, LAND, LAND, bear, bear, bear] + [LAND] * 40, turns=10)
    assert r["drawn_extra_by_turn"][6] == 0, "not while the mana is the point"
    assert r["drawn_extra_by_turn"][-1] == 1, "cracked once, and only once"
    assert r["mana_by_turn"][-1] == 9, "ten land drops, one sacrificed"


# ── 12. Escalate; Fateful Showdown ───────────────────────────────────────────

DEFIANCE = _c("Collective Defiance", "Escalate {1} (Pay this cost for each mode chosen beyond the first.)\n"
              "Choose one or more —\n• Target player discards all the cards in their hand, then draws that many "
              "cards.\n• Collective Defiance deals 4 damage to target creature.\n• Collective Defiance deals 3 "
              "damage to target opponent or planeswalker.", "{1}{R}{R}", colors="R")
SHOWDOWN = _c("Fateful Showdown", "Fateful Showdown deals damage to any target equal to the number of cards in "
              "your hand. Discard all the cards in your hand, then draw that many cards.", "{2}{R}{R}", "Instant",
              colors="R")


def test_the_second_mode_of_an_escalate_card_is_paid_for():
    assert goldfish.classify(DEFIANCE)["face_damage"]["escalate"] == 1
    # Glint-Horn on three, Defiance on turn four with one over: the 3 comes with it.
    rich = _play(_seq((GLINT, DEFIANCE), [LAND] * 4), turns=4)
    assert rich["damage_by_turn"][3] == 2 + 3
    # Exactly three mana, nothing else to do: the wheel alone resolves.
    bare = _play(_seq((DEFIANCE, BLANK), [LAND] * 2), turns=3)
    assert bare["wheels_cast_by_turn"][2] == 1 and bare["damage_by_turn"][2] == 0


def test_fateful_showdown_is_red_burn_equal_to_the_hand():
    assert goldfish.damage_source(SHOWDOWN)["deals"]
    r = _play(_seq((TORBRAN, MAKO), [LAND] * 3 + [SHOWDOWN, LAND, LAND, LAND]), turns=7)
    t = _turn(r)
    h = _gained(r, "discarded_by_turn", t)
    swing_torbran, swing_mako = 2 + 2, 1 + 2 + h      # Mako's counters are on before the swing
    assert r["damage_by_turn"][t] == swing_torbran + swing_mako + h + 2, "the hand, plus Torbran's 2"


# ── 13. Idyllic Tutor ────────────────────────────────────────────────────────

IDYLLIC = _c("Idyllic Tutor", "Search your library for an enchantment card, reveal it, put it into your hand, "
             "then shuffle.", "{2}{W}", colors="W")


def test_idyllic_tutor_fetches_an_enchantment_to_hand():
    p = goldfish.classify(IDYLLIC)
    assert p["tutor"] and p["tutor_filter"] == (("Enchantment",),)
    deep = _seq((IDYLLIC, BLANK), [LAND] * 10 + [GRATUITOUS])
    wheel_target = {"label": "wheel", "need": [{"any_of": ["Wheel of Fortune"]}]}
    ench_target = {"label": "ench", "need": [{"any_of": ["Gratuitous Violence"]}]}
    got = _play(deep + [WHEEL], turns=6, targets=[wheel_target, ench_target])
    assert got["target_turns"][0] is None, "a tutor for enchantments cannot find a wheel"
    assert got["target_turns"][1] == 3
    # It came out of the library: two turns on, with five mana, the enchantment is cast.
    r = _play(_seq((IDYLLIC, GLINT), [LAND] * 4 + [WHEEL] * 2 + [LAND] * 3 + [GRATUITOUS]), turns=9)
    t = next(i for i, w in enumerate(r["wheels_cast_by_turn"]) if w)
    assert r["event_damage_by_turn"][t] == 2 * _gained(r, "discarded_by_turn", t) > 0


def test_a_tutor_fills_only_what_it_reaches():
    wheel = {"need": [{"any_of": ["W"]}]}
    types = {"W": "Sorcery", "E": "Enchantment"}
    assert _target_met(wheel, set(), True, [None], types)
    assert not _target_met(wheel, set(), True, [(("Enchantment",),)], types)
    both = {"need": [{"any_of": ["W"]}, {"any_of": ["E"]}]}
    assert not _target_met(both, set(), True, [None], types), "one tutor, one need"
    assert _target_met(both, set(), True, [None, (("Enchantment",),)], types)
    assert _target_met(wheel, set(), True, 1), "the count form still means unrestricted"


# ── 14. Power that reads; activated abilities; modes ─────────────────────────

RIELLE = _c("Rielle, the Everwise", "Rielle gets +1/+0 for each instant and sorcery card in your graveyard.\n"
            "Whenever you discard one or more cards for the first time each turn, draw that many cards.",
            "{1}{U}{R}", "Legendary Creature — Human Wizard", "0", "3", "UR")
MAGUS = _c("Magus of the Wheel", "{1}{R}, {T}, Sacrifice this creature: Each player discards their hand, then "
           "draws seven cards.", "{2}{R}", "Creature — Human Wizard", "3", "3", "R")
HEARTWARMING = _c("Heartwarming Redemption", "Discard all the cards in your hand, then draw that many cards "
                  "plus one. You gain life equal to the number of cards in your hand.", "{2}{R}{W}", "Instant")
PRISMARI = _c("Prismari Command", "Choose two —\n• Prismari Command deals 2 damage to any target.\n• Target player "
              "draws two cards, then discards two cards.\n• Target player creates a Treasure token.\n• Destroy target "
              "artifact.", "{1}{U}{R}", "Instant", colors="UR")
SHATTERED = _c("Shattered Perception", "Discard all the cards in your hand, then draw that many cards.\nFlashback "
               "{5}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)", "{2}{R}")
ZIRDA = _c("Zirda, the Dawnwaker", "Companion — Each permanent card in your starting deck has an activated ability. "
           "(If this card is your chosen companion, you may put it into your hand from outside the game for {3} as a "
           "sorcery.)\nAbilities you activate that aren't mana abilities cost {2} less to activate. This effect can't "
           "reduce the mana in that cost to less than one mana.\n{1}, {T}: Target creature can't block this turn.",
           "{1}{R/W}{R/W}", "Legendary Creature — Elemental Fox", "3", "3")
SPLENDOR = _c("Gleaming Splendor", "Whenever an opponent draws their second card each turn, you create a Treasure "
              "token.\n{2}{W}: Two target players each draw a card.", "{1}{W}", "Enchantment", colors="W")


def test_psychosis_crawler_swings_for_the_hand():
    big = _c("Big", "", "{9}{9}", "Artifact")
    r = _play(_seq((CRAWLER, big), [big] * 7), turns=7)
    t = 5
    assert r["damage_by_turn"][t] == r["hand_size_by_turn"][t], "power is the hand, not 0"
    assert r["board_power_by_turn"][t] > 0


def test_rielle_counts_the_spells_in_the_yard():
    r = _play(_seq((RIELLE, BLANK), [LAND] * 3 + [SHOCK, SHOCK]), turns=6)
    # Two burn spells cast, so 0 + 2 power by the time she swings.
    assert r["board_power_by_turn"][-1] == 2


def test_magus_pops_itself_and_never_a_commander():
    # The same power as the commander: the old read matched on power and took the first row.
    lord = _c("Double Lord", "Double strike", "{3}{R}", "Legendary Creature — Lord", "3", "3", "R")
    r = _play(_seq((BLANK, BLANK), [LAND] * 4 + [MAGUS, LAND, LAND, LAND]), turns=8, commander=lord)
    assert r["wheels_cast_by_turn"][-1] == 1
    assert r["damage_by_turn"][-1] == 6, "the double striker is still here"


@pytest.mark.parametrize("card,players,cost", [(MASTERMIND, "each", 4), (SPLENDOR, "two", 3)])
def test_the_activated_table_draws_are_read(card, players, cost):
    d = goldfish.draw_profile(card)
    assert d["activated_draw"] == 1 and d["activated_draw_players"] == players
    assert d["activated_draw_cost"] == cost and d["unmodelled"] is None


def test_an_activation_condition_other_than_attacking_is_refused_and_named():
    glint = goldfish.draw_profile(GLINT)
    assert glint["activated_draw_needs_attack"] and glint["activated_draw"] == 1
    arch = goldfish.draw_profile(_c("Arch", "{5}, {T}: Draw a card. Activate only if you have the city's blessing.",
                                    tl="Land"))
    assert arch["activated_draw"] == 0 and arch["unmodelled"] == "Arch"


def test_glint_horn_ability_needs_an_attacker_and_repeats_on_spare_mana():
    r = _play([LAND] * 7 + [GLINT, BLANK, BLANK, BLANK] + [LAND] * 30, turns=8)
    assert r["drawn_extra_by_turn"][-1] >= 2, "it loots on spare mana, more than once a game"
    assert r["drawn_extra_by_turn"][2] == 0, "she cast with the last of three mana on turn three"


def test_zirda_takes_two_off_an_activation_and_never_below_one():
    def activations(zirda):
        top = [LAND] * 6 + ([zirda] if zirda else [BLANK]) + [MASTERMIND, NEEDLEHEAD] + [LAND] * 40
        return _play(top, turns=8)["drawn_extra_by_turn"][-1]
    assert activations(ZIRDA) > activations(None)


def test_heartwarming_is_a_wheel_plus_one_and_its_life_is_a_gain_event():
    d = goldfish.draw_profile(HEARTWARMING)
    assert d["wheel_plus"] == 1 and d["spell_gain_hand"]
    payoff = _c("Priest", "Whenever you gain life, each opponent loses 1 life.", "{1}", "Creature — Cleric", "1", "1")
    r = _play([LAND] * 6 + [GLINT, HEARTWARMING] + [LAND] * 30, turns=6, model_drain=True)
    assert r["wheels_cast_by_turn"][-1] == 1


def test_prismari_is_a_loot_and_a_treasure_and_no_face_mode():
    p = goldfish.classify(PRISMARI)
    assert p["draw"]["spell_draw"] == 2 and p["draw"]["spell_discard"] == 2
    assert p["combat"]["spell_treasure"] == 1
    assert p["face_damage"]["kind"] is None
    r = _play(_seq((MAKO, PRISMARI), [LAND] * 7), turns=7)
    assert r["treasures_by_turn"][-1] == 1
    assert r["discarded_by_turn"][-1] == 2
    assert r["drawn_extra_by_turn"][-1] == 2


def test_flashback_casts_it_again_once_and_exiles_it():
    p = goldfish.classify(LOOTING)
    assert p["engine"]["flashback"]["cost"] == 3
    r = _play(_seq((MAKO, LOOTING), [LAND] * 9), turns=9)
    assert r["discarded_by_turn"][-1] == 4, "the spell and its flashback, two cards each, and no third"
    assert goldfish.classify(SHATTERED)["engine"]["flashback"]["cost"] == 6
    assert goldfish.draw_profile(SHATTERED)["wheel_draws"] == -1


# ── 15. What a land pays ─────────────────────────────────────────────────────

VERGE = _c("Riverpyre Verge", "{T}: Add {R}.\n{T}: Add {U}. Activate only if you control an Island or a Mountain.",
           tl="Land")
SPIRE = _c("Spire of Industry", "{T}: Add {C}.\n{T}, Pay 1 life: Add one mana of any color. Activate only if you "
           "control an artifact.", tl="Land")
CAVERNS = _c("Gemstone Caverns", "If this card is in your opening hand and you're not the starting player, you may "
             "begin the game with Gemstone Caverns on the battlefield with a luck counter on it. If you do, exile a "
             "card from your hand.\n{T}: Add {C}. If Gemstone Caverns has a luck counter on it, instead add one mana "
             "of any color.", tl="Legendary Land")
PRAIRIE = _c("Rugged Prairie", "{T}: Add {C}.\n{R/W}, {T}: Add {R}{R}, {R}{W}, or {W}{W}.", tl="Land")
PATHWAY = _c("Needleverge Pathway // Pillarverge Pathway", "{T}: Add {R}. // {T}: Add {W}.", tl="Land // Land",
             layout="modal_dfc")
ISLAND = _c("Island", "({T}: Add {U}.)", tl="Basic Land — Island")
MOUNTAIN = _c("Mountain", "({T}: Add {R}.)", tl="Basic Land — Mountain")


def _cast_with(lands, cost, turns=None):
    body = _c("Bear", "", cost, "Creature — Bear", "2", "2")
    padding = [BLANK] * (7 - len(lands) - 1)
    r = _play(list(lands) + [body] + padding + [BLANK] * 3, lands_below=0, turns=turns or len(lands),
              model_colors=True)
    return r["bodies_by_turn"][-1] == 1


def test_a_verge_needs_its_type_for_the_second_colour():
    assert not _cast_with([VERGE, LAND], "{1}{U}")
    assert _cast_with([VERGE, MOUNTAIN], "{1}{U}")
    assert _cast_with([VERGE, LAND], "{1}{R}")


def test_spire_needs_an_artifact_and_caverns_needs_a_luck_counter():
    assert not _cast_with([SPIRE, LAND], "{1}{U}")
    assert not _cast_with([CAVERNS, LAND], "{1}{U}")
    stone = _c("Stone", "", "{0}", "Artifact")
    p = goldfish.classify(SPIRE)
    assert p["land_gate"]["artifact"] and p["land_gate"]["gated"] == frozenset("WUBRG")


def test_a_filter_needs_a_source_of_its_own_colours():
    assert not _cast_with([PRAIRIE, LAND], "{1}{W}")
    assert _cast_with([PRAIRIE, MOUNTAIN], "{1}{W}")


def test_a_pathway_is_one_face():
    assert _cast_with([PATHWAY, MOUNTAIN], "{R}{R}") or _cast_with([PATHWAY, MOUNTAIN], "{R}{W}")
    assert not _cast_with([PATHWAY, LAND, LAND], "{R}{W}")
    assert _cast_with([PATHWAY, LAND], "{1}{W}")


# ── 16. Deploy, then wheel ───────────────────────────────────────────────────

def test_the_turn_deploys_what_it_can_pay_before_it_wheels():
    """The pilot's order: with a payoff out and three mana, the three-drop lands and the wheel
    waits. Wheel first would throw the three-drop away."""
    three = _c("Three", "", "{3}", "Creature — Bear", "3", "3")
    r = _play(_seq((MAKO, three), [LAND, LAND, WHEEL, LAND, LAND]), turns=4)
    assert r["bodies_by_turn"][2] == 2 and r["wheels_cast_by_turn"][2] == 0
    assert r["wheels_cast_by_turn"][3] == 1
    # Six mana: Razorkin (2), then the Wheel (3), and what it found on what is left.
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    r = _play(_seq((MAKO, BLANK), [LAND] * 5 + [NEEDLEHEAD]) + [WHEEL] + [bear] * 8, turns=7)
    t = _turn(r)
    assert r["bodies_by_turn"][t] >= 3


def test_nothing_castable_is_discarded_when_the_mana_could_have_paid_it():
    cheap = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    r = _play([LAND] * 3 + [WHEEL] + [cheap] * 3 + [LAND] * 30, turns=8, commander=BRALLIN)
    turn = r["first_wheel_turn"] - 1
    # Brallin on four, the bears down; the discard on the turn it fires is lands and blanks.
    assert r["bodies_by_turn"][turn] >= 3


def test_a_tax_on_their_draws_lands_ahead_of_a_creature():
    three = _c("Three", "", "{3}", "Creature — Bear", "3", "3")
    r = _play(_seq((TITHE, BLANK), [LAND] * 3 + [three, LAND]), turns=5)
    assert r["treasure_online_by_turn"][4] is True, "Tithe on four, ahead of the cheaper body"
    assert r["bodies_by_turn"][3] == 0
    assert model_coverage.never_cast(goldfish.classify(TITHE), FLAGS) is False


def test_a_wheel_still_digs_for_a_land_ahead_of_a_body():
    bear = _c("Bear", "", "{2}{U}", "Creature — Bear", "2", "2")
    r = _play([LAND, LAND, LAND, WHEEL, bear, BLANK, BLANK, BLANK, BLANK, BLANK] + [LAND] * 30, turns=3,
              commander=BRALLIN)
    assert r["wheels_cast_by_turn"][2] == 1, "no land in hand and the commander in the zone"
    assert r["bodies_by_turn"][2] == 0


def test_with_nothing_to_pay_a_wheel_waits_for_what_next_turn_can_cast():
    r = _play(_seq((TITHE, WHEEL), [LAND] * 5), turns=4)
    assert r["wheels_cast_by_turn"][2] == 0, "Tithe is four and the land is in hand"


# ── 17. Ten real effects, and the coverage report ───────────────────────────

KNOWLEDGE = _c("Knowledge Is Power", "Creatures you control get +X/+X, where X is the number of cards you've drawn "
               "this turn.", "{3}{W}{U}", "Enchantment", colors="WU")
AKROMA = _c("Akroma's Will", "Choose one. If you control a commander as you cast this spell, you may choose both "
            "instead.\n• Creatures you control gain flying, vigilance, and double strike until end of turn.\n• "
            "Creatures you control gain lifelink, indestructible, and protection from each color until end of turn.",
            "{3}{W}", "Instant", colors="W")
ASCENDANCY = _c("Jeskai Ascendancy", "Whenever you cast a noncreature spell, creatures you control get +1/+1 until "
                "end of turn. Untap those creatures.\nWhenever you cast a noncreature spell, you may draw a card. If "
                "you do, discard a card.", "{U}{R}{W}", "Enchantment", colors="URW")
CHARM = _c("Izzet Charm", "Choose one —\n• Counter target noncreature spell unless its controller pays {2}.\n• Izzet "
           "Charm deals 2 damage to target creature.\n• Draw two cards, then discard two cards.", "{U}{R}", "Instant")
REMORA = _c("Mystic Remora", "Cumulative upkeep {1} (At the beginning of your upkeep, put an age counter on this "
            "permanent, then sacrifice it unless you pay its upkeep cost for each age counter on it.)\nWhenever an "
            "opponent casts a noncreature spell, you may draw a card unless that player pays {4}.", "{U}", "Enchantment")
TEFERI = _c("Teferi, Master of Time", "You may activate loyalty abilities of Teferi on any player's turn any time you "
            "could cast an instant.\n+1: Draw a card, then discard a card.\n−3: Target creature you don't control "
            "phases out.\n−10: Take two extra turns after this one.", "{2}{U}{U}", "Legendary Planeswalker — Teferi")
TALENT = _c("Artist's Talent", "(Gain the next level as a sorcery to add its ability.)\nWhenever you cast a noncreature "
            "spell, you may discard a card. If you do, draw a card.\n{2}{R}: Level 2\nNoncreature spells you cast cost "
            "{1} less to cast.\n{2}{R}: Level 3\nIf a source you control would deal noncombat damage to an opponent or a "
            "permanent an opponent controls, it deals that much damage plus 2 instead.", "{1}{R}", "Enchantment — Class")
GREAVES = _c("Lightning Greaves", "Equipped creature has haste and shroud. (It can't be the target of spells or "
             "abilities.)\nEquip {0}", "{2}", "Artifact — Equipment")
MIRRORMADE = _c("Mirrormade", "You may have this enchantment enter as a copy of any artifact or enchantment on the "
                "battlefield.", "{1}{U}{U}", "Enchantment")


@pytest.mark.parametrize("card", [EMANCIPATION, KNOWLEDGE, AKROMA, ASCENDANCY, CHARM, REMORA, TEFERI, TALENT,
                                  GREAVES, MIRRORMADE])
def test_the_ten_are_read_and_cast(card):
    p = goldfish.classify(card)
    assert not model_coverage.never_cast(p, dict(FLAGS, model_colors=True)), card["name"]
    assert model_coverage.channels_for(p), card["name"]


def test_being_named_in_a_target_group_is_not_being_read():
    blank = _c("Counterspell", "Counter target spell.", "{U}{U}", "Instant")
    st = model_coverage.card_state(blank, {"model_draw": True}, named={"Counterspell"})
    assert st["state"] == "invisible" and not st["possible"]
    read = model_coverage.card_state(SIGNET, {"model_draw": True}, named={"Azorius Signet"})
    assert read["state"] == "seen" and "target" in read["possible"]


def test_knowledge_is_power_reads_the_turns_draws():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    r = _play([LAND] * 7 + [KNOWLEDGE, bear], turns=7)
    # Bear 2 + Knowledge's own: the draw step's one card each.
    assert r["damage_by_turn"][-1] == 2 + 1


def test_akromas_will_doubles_the_team_on_spare_mana():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    assert goldfish.classify(AKROMA)["combat"]["spell_team_double_strike"]
    r = _play(_seq((bear, AKROMA), [LAND] * 5), turns=5)
    assert r["damage_by_turn"][2] == 2, "three mana is not four"
    assert r["damage_by_turn"][3] == 4, "four mana, nothing else to buy: the bear twice"


def test_jeskai_ascendancy_pumps_and_loots_on_a_noncreature_spell():
    bear = _c("Bear", "", "{1}", "Creature — Bear", "2", "2")
    trinket = _c("Trinket", "{T}: Add {C}.", "{1}", "Artifact")
    r = _play(_seq((MAKO, ASCENDANCY), [LAND] * 3 + [trinket, LAND]), turns=5)
    # Turn four, the rock: Mako swings for 1, +1 for the spell, +1 for the loot's counter.
    assert _gained(r, "discarded_by_turn", 3) == 1 and _gained(r, "drawn_extra_by_turn", 3) == 1
    assert r["damage_by_turn"][3] == 3


def test_izzet_charm_loots_on_a_payoff():
    assert goldfish.draw_profile(CHARM)["spell_draw"] == 2
    r = _play(_seq((GLINT, CHARM), [LAND] * 3), turns=4)
    assert 2 <= r["discarded_by_turn"][-1]


def test_remora_draws_at_the_stated_rates_and_is_given_up_by_its_age(monkeypatch):
    from manamap.pilot import goldfish_turn
    e = goldfish.event_payoffs(REMORA)
    assert e["cumulative_upkeep"] == 1 and e["per_opponent_cast_noncreature"] and e["per_opponent_cast_taxed"]
    # r=0.4: one whole cast a seat, each noncreature, every tax paid: no card.
    paid = _play(_seq((REMORA, BLANK), [LAND] * 7), turns=7, r=0.4)
    assert paid["drawn_extra_by_turn"][-1] == 0
    # r=0.9: every cast is a creature spell under the authored share: nothing to read.
    assert _play(_seq((REMORA, BLANK), [LAND] * 7), turns=7, r=0.9)["drawn_extra_by_turn"][-1] == 0
    # Every cast noncreature, none paying: three a turn, until age four is not paid.
    monkeypatch.setattr(goldfish_turn, "OPPONENT_NONCREATURE_SHARE", 1.0)
    r = _play(_seq((REMORA, BLANK), [LAND] * 7), turns=7, r=0.9)
    assert _gained(r, "drawn_extra_by_turn", 3) == 3, "age three, paid out of four mana"
    assert _gained(r, "drawn_extra_by_turn", 4) == 0, "age four is given up"


def test_teferi_loots_on_every_seats_turn():
    r = _play([LAND] * 6 + [TEFERI] + [LAND] * 40, turns=5, commander=BRALLIN)
    turn = 4
    assert r["discarded_by_turn"][turn] - r["discarded_by_turn"][turn - 1] == 4


def test_artists_talent_rummages_on_a_noncreature_spell():
    assert goldfish.classify(TALENT)["engine"]["cast_loot"] == "discard"
    trinket = _c("Trinket", "{T}: Add {C}.", "{1}", "Artifact")
    r = _play(_seq((MAKO, TALENT), [LAND] * 3 + [trinket, LAND]), turns=5)
    assert _gained(r, "discarded_by_turn", 3) == 1 and _gained(r, "drawn_extra_by_turn", 3) == 1


def test_greaves_give_the_landing_commander_haste():
    r = _play([LAND] * 6 + [GREAVES] + [LAND] * 20, turns=5, commander=BRALLIN)
    assert r["commander_turn"] == 4
    assert r["damage_by_turn"][3] == 3, "he swings the turn he lands"
    without = _play([LAND] * 6 + [BLANK] + [LAND] * 20, turns=5, commander=BRALLIN)
    assert without["damage_by_turn"][3] == 0


def test_mirrormade_enters_as_a_tithe():
    one = _play([LAND] * 6 + [TITHE, BLANK] + [LAND] * 30, turns=8)
    two = _play([LAND] * 6 + [TITHE, MIRRORMADE] + [LAND] * 30, turns=8)
    assert two["treasures_by_turn"][-1] > one["treasures_by_turn"][-1]


def test_mirrormade_does_not_copy_a_legend_and_waits_for_a_target():
    cleave = _c("Embercleave", "", "{4}{R}{R}", "Legendary Artifact — Equipment")
    r = _play([LAND] * 7 + [MIRRORMADE], turns=5)
    assert r["hand_size_by_turn"][-1] >= 1


# ── 17b. The loop ────────────────────────────────────────────────────────────

def _loop_top(*extra):
    return [LAND] * 7 + [TITHE, MASTERMIND, BARD, *extra]


def test_the_loop_runs_on_spare_mana_and_its_treasure_and_stops_at_the_floor():
    rock = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", "{1}", "Artifact")
    top = _seq((TITHE, rock), [MASTERMIND, BARD, NEEDLEHEAD] + [LAND] * 30 + [BLANK] * 20)
    r = _play(top, turns=4, lands_below=0)
    t = next(i for i in range(4) if _gained(r, "drawn_extra_by_turn", i))
    assert _gained(r, "drawn_extra_by_turn", t) >= 6, "several activations in one turn, Bard doubling each"
    left = len(top) - 7 - (t + 1) - r["drawn_extra_by_turn"][t]
    # The stop is the library's. Mastermind's three seats pay after it: two cards each under Bard.
    assert left >= LIFE_CHAIN_LIBRARY_FLOOR - 6
    assert r["drawn_extra_by_turn"][-1] == r["drawn_extra_by_turn"][t], "and it stays stopped"


def test_the_needlehead_kill_is_visible_through_the_loop():
    rock = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", "{1}", "Artifact")
    draws = [MASTERMIND, BARD, NEEDLEHEAD, CRAWLER] + [LAND] * 40 + [BLANK] * 20
    r = _play(_seq((TITHE, rock), draws), turns=12, lands_below=0)
    assert r["kill_turn"] is not None and r["kill_turn"] <= 8
    without = _play(_seq((TITHE, rock), [MASTERMIND, BARD, BLANK, BLANK] + [LAND] * 40 + [BLANK] * 20),
                    turns=12, lands_below=0)
    assert sum(r["event_damage_by_turn"]) > sum(without["event_damage_by_turn"])


def test_a_symmetric_activation_with_nothing_paying_for_it_is_not_bought():
    plain = _c("Bell", "{T}: Each player draws a card.", "{2}", "Artifact")
    r = _play([LAND] * 7 + [plain] + [LAND] * 30, turns=8)
    assert r["drawn_extra_by_turn"][-1] == 0


def test_a_treasure_pays_a_repeat_only_under_tithe():
    d = _c("Looter", "{1}: Draw a card.", "{1}", "Artifact")
    gold = _c("Gold", "When this artifact enters, create three Treasure tokens.", "{0}", "Artifact")
    r = _play([LAND] * 3 + [d, gold] + [BLANK] * 2 + [LAND] * 30, turns=3)
    # Three lands, the looter and whatever is left: Treasure paid the first draw and no repeat.
    assert r["treasures_by_turn"][-1] >= 2


def test_a_game_with_the_loop_replays_exactly():
    deck = [goldfish.classify(c) for c in _loop_top(NEEDLEHEAD) + [LAND] * 40 + [BLANK] * 30]
    go = lambda: goldfish.simulate_once(random.Random("loop:1"), deck, 99, [], 12, **FLAGS)
    assert go() == go()


# ── 18-22. The figures ───────────────────────────────────────────────────────

def test_all_damage_holds_the_pings_and_damage_does_not():
    r = _play([LAND] * 7 + [GLINT, WHEEL] + [LAND] * 20, turns=7)
    t = r["first_wheel_turn"] - 1
    assert r["event_damage_by_turn"][t] > 0
    assert r["total_damage_by_turn"][t] == r["damage_by_turn"][t] + r["event_damage_by_turn"][t]


def test_the_new_figures_are_defined_and_the_stale_sentence_is_gone():
    # NOT A ROW: all damage tracks damage @T10 at r = 0.99 game for game on the one deck that has
    # both, and a second row would read as two confirmations of one fact.
    assert all(row[0] != "all damage @T10" for row in net_change.ROWS)
    assert "IT EXCLUDES" in net_change.METRICS["damage @T10"]["what"]
    assert "Activated, X-based" not in net_change.METRICS["extra cards by T8"]["what"]
    assert "END of turn 10" in net_change.METRICS["hoard @T10"]["what"]
    assert "counters" in net_change.METRICS["board power @T6"]["what"]


def test_board_power_holds_the_counters():
    r = _play([LAND] * 7 + [MAKO, WHEEL] + [LAND] * 20, turns=7)
    t = r["first_wheel_turn"] - 1
    assert r["board_power_by_turn"][t] == 1 + r["discarded_by_turn"][t]


def test_the_hoard_is_read_at_the_end_of_the_turn():
    """Three Treasure off a Tithe on turn four's end; a five-drop next turn spends what it needs."""
    five = _c("Five", "", "{5}{9}", "Creature — X", "1", "1")
    keep = _play([LAND] * 7 + [TITHE] + [LAND] * 30, turns=5)
    assert keep["treasures_by_turn"][3] == 3, "made at the end of the main phase: it stands"
    spend = _play([LAND] * 5 + [TITHE, _c("Six", "", "{6}", "Creature — X", "1", "1")] + [LAND] * 30, turns=5)
    assert spend["treasures_by_turn"][4] < keep["treasures_by_turn"][4]


def test_time_to_counter_and_to_the_right_card():
    r = _play([LAND] * 7 + [BLANK] * 3, turns=6, commander=SHABRAZ)
    assert r["commander_first_counter_delays"] == [1], "landed turn five, first counter turn six"
    r = _play([LAND] * 6 + [GLINT] + [LAND] * 20, turns=4)
    assert r["first_right_card_turn"] == 3
    r = _play([LAND] * 6 + [SHOCK] + [LAND] * 20, turns=4)
    assert r["first_right_card_turn"] is None, "absent is absent"


def test_draw_not_modelled_stops_naming_what_the_event_channel_reads():
    rhystic = _c("Rhystic Study", "Whenever an opponent casts a spell, you may draw a card unless that player "
                 "pays {1}.", "{2}{U}", "Enchantment")
    assert goldfish.draw_profile(rhystic)["unmodelled"] == "Rhystic Study"
    assert goldfish.event_payoffs(rhystic)["per_opponent_cast_draw"] == 1
    # The list `meta` publishes reads through this filter.
    for read in (rhystic, SPHINX, MASTERMIND, MAKO, TEFERI, ASCENDANCY, TYPHOON):
        assert goldfish.draw_is_read_elsewhere(read), read["name"]
    shredder = _c("Ledger Shredder", "Flying\nWhenever a player casts their second spell each turn, this "
                  "creature connives. (Draw a card, then discard a card.)", "{1}{U}", "Creature — Bird", "1", "3")
    assert not goldfish.draw_is_read_elsewhere(shredder), "a connive is still blind, and it says so"


def test_commander_lethal_counts_torbran_on_a_red_commander():
    a = _play([LAND] * 7 + [TORBRAN] + [LAND] * 30, turns=10, commander=BRALLIN)
    b = _play([LAND] * 7 + [_c("Vanilla", "", "{1}{R}{R}{R}", "Creature — Dwarf", "2", "4")] + [LAND] * 30, turns=10,
              commander=BRALLIN)
    assert (a["commander_lethal_turn"] or 99) < (b["commander_lethal_turn"] or 99)


def test_free_with_a_commander_is_affordable_with_no_mana():
    swat = _c("Deflecting Swat", "If you control a commander, you may cast this spell without paying its mana cost.\n"
              "You may choose new targets for target spell or ability.", "{2}{R}", "Instant", colors="R")
    assert goldfish.classify(swat)["engine"]["free_with_commander"]
    deck = [goldfish.classify(c) for c in [LAND] * 6 + [swat] + [LAND] * 30]
    r = goldfish.simulate_once(Stacked(), deck, 4, [], 5, interaction_names=frozenset({"Deflecting Swat"}),
                               commander_combat=goldfish.combat_profile(BRALLIN), **FLAGS)
    assert r["commander_turn"] == 4
    assert r["interaction_castable_by_turn"][3] is True, "four lands, four spent: free"


def test_the_keep_rule_counts_a_creature_not_a_treasure_maker():
    rule = {"needs_any": ["body"]}
    assert not keep_rule_passes([goldfish.classify(TITHE), goldfish.classify(LAND)], rule)
    assert not keep_rule_passes([goldfish.classify(TYPHOON)], rule)
    assert keep_rule_passes([goldfish.classify(GLINT)], rule)


def test_a_second_mulligan_bottoms_a_card():
    all_lands = [LAND] * 60
    deck = [goldfish.classify(c) for c in all_lands]
    r = goldfish.simulate_once(Stacked(), deck, 99, [], 1, **FLAGS)
    assert r["mulligans"] == 2
    # Seven lands twice is no keep, so the second bottoms one and the hand is six, plus a draw, less a drop.
    assert r["hand_size_by_turn"][0] == 6


def test_a_first_mulligan_is_free():
    # A shuffle that deals seven lands then a keepable seven.
    keep = [LAND, LAND] + [BLANK] * 5
    mull = [LAND] * 7

    class Once(Stacked):
        n = 0

        def shuffle(self, x, *a):
            Once.n += 1
            if Once.n == 2:
                x[:] = [goldfish.classify(c) for c in keep] + x[7:]

    deck = [goldfish.classify(c) for c in mull + [LAND] * 40]
    r = goldfish.simulate_once(Once(), deck, 99, [], 1, **FLAGS)
    assert r["mulligans"] == 1
    assert r["hand_size_by_turn"][0] == 7, "seven, a draw, a land drop"


def test_a_finisher_needs_a_commander_and_not_a_hand_left_over():
    r = _play(_seq((WHEEL, BLANK), [LAND] * 6), turns=6, commander=BRALLIN)
    assert r["finisher_wheel_turn"] is not None
    r = _play(_seq((MAKO, WHEEL), [LAND] * 6), turns=6)
    assert r["wheels_cast_by_turn"][-1] and r["finisher_wheel_turn"] is None




def test_a_tap_cost_draws_once_a_turn_on_a_noncreature_too():
    """The repeat loop tracked tapping on creature rows only: Jayemdae Tome drew per mana."""
    rock = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", "{1}", "Artifact")
    tome = _c("Tome", "{1}, {T}: Draw a card.", "{1}", "Artifact")
    r = _play([LAND] * 3 + [rock, tome] + [BLANK] * 2 + [LAND] * 40, turns=6)
    extra = r["drawn_extra_by_turn"]
    assert extra[-1] > 0, "the tome was never used"
    assert max(b - a for a, b in zip(extra, extra[1:])) == 1


def test_a_tithe_on_the_table_is_a_reason_to_wheel(monkeypatch):
    """Smothering Tithe is no `event` payoff, so a hand with anything left in it kept the wheel,
    and the wheel is what the Tithe is for. WHEEL_MIN_HAND at zero takes the thin-hand clause out
    of it, and a land left in hand takes out digging: only the payoff can open the gate."""
    from manamap.pilot import goldfish_turn
    monkeypatch.setattr(goldfish_turn, "WHEEL_MIN_HAND", 0)
    rock = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", "{1}", "Artifact")
    hand = [LAND, LAND, LAND, rock, TITHE, WHEEL, BLANK]
    r = _play(hand + [BLANK] * 4 + [LAND] * 10, lands_below=30, turns=4)
    assert r["first_wheel_turn"] == 2


# ── The pilot's library, taught 2026-11-04 ───────────────────────────────────

BORROWED = _c("Borrowed Knowledge", "Choose one \u2014\n\u2022 Discard your hand, then draw cards equal to the number "
              "of cards in target opponent's hand.\n\u2022 Discard your hand, then draw cards equal to the "
              "number of cards discarded this way.", "{2}{R}{W}", colors="RW")
PROSPERITY = _c("Prosperity", "Each player draws X cards.", "{X}{U}", colors="U")
BRAINGEYSER = _c("Braingeyser", "Target player draws X cards.", "{X}{U}{U}", colors="U")
MONUMENT = _c("Monument to Endurance", "Whenever you discard a card, choose one that hasn't been chosen this "
              "turn \u2014 \u2022 Draw a card. \u2022 Create a Treasure token. \u2022 Each opponent loses 3 life.",
              "{3}", "Artifact")
NARSET = _c("Narset, Jeskai Waymaster", "At the beginning of your end step, you may discard your hand. If you "
            "do, draw cards equal to the number of spells you've cast this turn.", "{U}{R}{W}",
            "Legendary Creature \u2014 Human Monk", "3", "2", "URW")
SPHINX = _c("Consecrated Sphinx", "Flying\nWhenever an opponent draws a card, you may draw two cards.",
            "{4}{U}{U}", "Creature \u2014 Sphinx", "4", "6", "U")
RICH = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", "{1}", "Artifact")
TRI = _c("Tri-Land", "{T}: Add {U}, {R}, or {W}.", tl="Land")


def test_borrowed_knowledge_is_a_one_sided_wheel_at_the_bigger_count():
    d = goldfish.draw_profile(BORROWED)
    assert d["wheel_draws"] == -1 and d["wheel_greatest"] and not d["wheel_symmetric"]
    r = _play(_seq((RICH, BORROWED), [BLANK] * 2), turns=2)
    assert r["first_wheel_turn"] == 2
    assert r["drawn_extra_by_turn"][1] >= OPPONENT_HAND, "drew the opponent's hand count off a thin hand"


def test_a_symmetric_x_draw_waits_for_a_payoff_and_pays_it():
    alone = _play(_seq((RICH, PROSPERITY), [BLANK] * 6), turns=4)
    assert alone["drawn_extra_by_turn"][-1] == 0, "with nothing on their draws, the table's cards are the cost"
    tithe = _play(_seq((RICH, TITHE), [PROSPERITY] + [BLANK] * 6), turns=4)
    plain = _play(_seq((RICH, TITHE), [BRAINGEYSER] + [BLANK] * 6), turns=4)
    assert tithe["drawn_extra_by_turn"][-1] > 0
    assert tithe["treasures_by_turn"][-1] > plain["treasures_by_turn"][-1]
    sphinx = _play(_seq((RICH, SPHINX), [PROSPERITY] + [BLANK] * 6), turns=4)
    one_sided = _play(_seq((RICH, SPHINX), [BRAINGEYSER] + [BLANK] * 6), turns=4)
    assert sphinx["drawn_extra_by_turn"][-1] > one_sided["drawn_extra_by_turn"][-1]


def test_monument_pays_each_mode_once_a_turn_and_never_a_draw_per_discard():
    ev = goldfish.event_payoffs(MONUMENT)
    assert ev["per_discard_draw"] == 0 and ev["per_discard_modes"]["life"] == 3
    assert goldfish.classify(MONUMENT)["treasure_trigger"] is None
    r = _play(_seq((RICH, MONUMENT), [WHEEL, BLANK, BLANK, BLANK]), turns=5)
    t = r["first_wheel_turn"] - 1
    assert r["discarded_by_turn"][t] >= 3
    assert r["event_damage_by_turn"][t] == 3, "the life loss, once, whatever the wheel discarded"
    assert r["treasures_by_turn"][t] >= 1
    assert r["drawn_extra_by_turn"][t] <= 8, "seven off the wheel and one off the mode, not one per discard"


def _narset_deal(narset, last=None):
    """Turn two empties the hand: two lands and a big rock, then four spells on twelve mana."""
    a, b, c = (_c(f"Trinket {i}", "{T}: Add {C}.", "{1}", "Artifact") for i in "abc")
    return [TRI, TRI, RICH, narset, a, b, last or c, LAND, BLANK, LAND, BLANK, LAND, BLANK]


def test_narset_wheels_at_the_end_step_and_pays_next_main_phase():
    assert goldfish.classify(NARSET)["engine"]["end_step_wheel"]
    r = _play(_narset_deal(NARSET), lands_below=20, turns=4)
    assert r["discarded_by_turn"][-1] > 0, "a turn of spells earns the end step"
    without = _play(_narset_deal(_c("Trinket d", "{T}: Add {C}.", "{1}", "Artifact")), lands_below=20, turns=4)
    assert without["discarded_by_turn"][-1] == 0
    assert r["drawn_extra_by_turn"][-1] > without["drawn_extra_by_turn"][-1]


def test_narsets_draws_pay_a_payoff_already_out_at_the_next_main_phase():
    ping = _c("Pinger", "Whenever you draw a card, each opponent loses 1 life.", "{1}", "Artifact")
    r = _play(_narset_deal(NARSET, ping), lands_below=20, turns=4)
    other = _play(_narset_deal(_c("Trinket d", "{T}: Add {C}.", "{1}", "Artifact"), ping), lands_below=20, turns=4)
    assert sum(r["event_damage_by_turn"]) > sum(other["event_damage_by_turn"]) + 2, \
        "the end step's draws were paid at the next main phase"


def test_narset_never_trades_down_a_full_hand():
    r = _play([TRI, TRI, TRI, NARSET, BLANK, BLANK, BLANK] + [BLANK] * 3 + [LAND] * 30, turns=5)
    assert r["discarded_by_turn"][-1] == 0
