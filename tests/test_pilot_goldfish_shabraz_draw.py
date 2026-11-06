"""The six draw shapes a Shabraz-first Sharknado runs on (2026-11-06).

The branch `shabraz-v1` put sixteen cards in, and the goldfish read the draw on
almost none of them. Worse, the combat draws it DID read went straight from the
deck to the hand: no doubler touched them and no per-draw payoff saw them, so
Shabraz, whose whole text is "whenever you draw a card, put a +1/+1 counter on
Shabraz", never grew off a Sword of Fire and Ice hit. The shapes:

  1. a combat draw is a draw (through `draw_n`, carried to the next settlement)
     and an Equipment's trigger waits for its equip;
  2. an attack trigger that needs company (Sokka; battalion);
  3. a cast trigger on every instant or sorcery (Archmage of Runes);
  4. draw N, then put two back (Brainstorm, Brainsurge);
  5. a payoff's tenth counter (Shang-Chi and the Ten Rings);
  6. an opponent's FIRST noncreature spell each turn (Esper Sentinel).

The profile tests drive the production readers on the printed text. The
behaviour tests run the goldfish twice on the same seeds, once with the clause
and once with it struck, so each one fails if its fix is reverted.
"""

import copy

import pytest

from manamap.pilot import goldfish
from manamap.pilot.goldfish_profiles import (
    SPELL_CAST_GATE, combat_profile, draw_profile, event_payoffs)
from conftest import requires_deck


def _card(name, text, type_line, mana_cost="{2}", power=None, toughness=None):
    return {"name": name, "oracle_text": text, "type_line": type_line,
            "mana_cost": mana_cost, "power": power, "toughness": toughness}


SWORD_FI = _card(
    "Sword of Fire and Ice",
    "Equipped creature gets +2/+2 and has protection from red and from blue.\n"
    "Whenever equipped creature deals combat damage to a player, this Equipment deals 2 "
    "damage to any target and you draw a card.\nEquip {2}", "Artifact — Equipment", "{3}")
MASK = _card(
    "Mask of Memory",
    "Whenever equipped creature deals combat damage to a player, you may draw two cards. "
    "If you do, discard a card.\nEquip {1} ({1}: Attach to target creature you control. "
    "Equip only as a sorcery.)", "Artifact — Equipment", "{2}")
SOKKA = _card("Sokka, Lateral Strategist",
              "Vigilance\nWhenever Sokka and at least one other creature attack, draw a card.",
              "Legendary Creature — Human Warrior Ally", "{1}{W/U}{W/U}", "2", "2")
WILLIE = _card(
    "Willie Lumpkin, Postman",
    "Willie Lumpkin can't be blocked.\nWhenever Willie Lumpkin deals combat damage to an "
    "opponent, you draw a card and that player may draw a card. If they do, that player "
    "can't attack you or permanents you control during their next turn.",
    "Legendary Creature — Human Citizen", "{W}{U}", "1", "1")
ARCHMAGE = _card("Archmage of Runes",
                 "Instant and sorcery spells you cast cost {1} less to cast.\n"
                 "Whenever you cast an instant or sorcery spell, draw a card.",
                 "Creature — Giant Wizard", "{3}{U}{U}", "3", "6")
BRAINSURGE = _card("Brainsurge", "Draw four cards, then put two cards from your hand on top "
                   "of your library in any order.", "Instant", "{2}{U}")
SHANG_CHI = _card("Shang-Chi and the Ten Rings",
                  "First strike\nWhenever you draw a card, put a +1/+1 counter on Shang-Chi.\n"
                  "When the tenth +1/+1 counter is put on Shang-Chi, you draw five cards and "
                  "gain 5 life.", "Legendary Creature — Human Warrior Hero", "{1}{W}{U}", "2", "2")
SENTINEL = _card("Esper Sentinel",
                 "Whenever an opponent casts their first noncreature spell each turn, draw a "
                 "card unless that player pays {X}, where X is this creature's power.",
                 "Artifact Creature — Human Soldier", "{W}", "1", "1")


# ── the readers ──────────────────────────────────────────────────────────

def test_an_equipment_trigger_waits_for_its_equip():
    sw, mk = combat_profile(SWORD_FI), combat_profile(MASK)
    assert (sw["attack_draw"], sw["attack_damage"], sw["equip_cost"]) == (1, 2, 2)
    # Mask's draw is a loot: two in, one out.
    assert (mk["attack_draw"], mk["attack_discard"], mk["equip_cost"]) == (2, 1, 1)
    # A creature's own trigger has no equip.
    assert combat_profile(WILLIE)["equip_cost"] is None


def test_to_an_opponent_reads_as_to_a_player():
    assert combat_profile(WILLIE)["attack_draw"] == 1


def test_an_attack_trigger_that_needs_company():
    p = combat_profile(SOKKA)
    assert (p["attack_draw"], p["attack_needs_company"]) == (1, 1)
    battalion = combat_profile(_card(
        "Paladin", "Whenever this creature and at least two other creatures attack, draw a card.",
        "Creature — Human Knight", "{2}{W}", "2", "2"))
    assert battalion["attack_needs_company"] == 2


def test_a_cast_trigger_on_every_instant_or_sorcery():
    d = draw_profile(ARCHMAGE)
    assert (d["cast_draw"], d["cast_draw_gate"]) == (1, SPELL_CAST_GATE)
    # Not a type-line substring, so no permanent cast can fire it.
    assert SPELL_CAST_GATE not in ARCHMAGE["type_line"]


def test_draw_four_then_put_two_back():
    d = draw_profile(BRAINSURGE)
    assert (d["spell_draw"], d["spell_put_back"]) == (4, 2)


def test_a_draw_with_a_price_the_model_cannot_charge_is_left_unread():
    for name, text in (("Breakthrough", "Draw four cards, then choose X cards in your hand "
                                        "and discard the rest."),
                       ("Meditate", "Draw four cards. You skip your next turn.")):
        d = draw_profile(_card(name, text, "Sorcery", "{2}{U}"))
        assert d["spell_draw"] == 0 and d["unmodelled"] == name


def test_the_tenth_counter():
    e = event_payoffs(SHANG_CHI)
    assert (e["per_draw_counter"], e["tenth_counter_draw"], e["tenth_counter_gain"]) == (1, 5, 5)


def test_an_opponents_first_noncreature_spell_each_turn():
    e = event_payoffs(SENTINEL)
    assert e["per_opponent_cast_draw"] == 1
    assert e["per_opponent_cast_first"] and e["per_opponent_cast_noncreature"]
    assert e["per_opponent_cast_taxed"]
    # Rhystic Study is unchanged: every spell, no cap.
    rs = event_payoffs(_card("Rhystic Study", "Whenever an opponent casts a spell, you may draw "
                             "a card unless that player pays {1}.", "Enchantment", "{2}{U}"))
    assert rs["per_opponent_cast_draw"] == 1 and not rs["per_opponent_cast_first"]


# ── the behaviour, paired on the same seeds ─────────────────────────────

#: Sharknado with the shapes swapped in for cards that do not interact with them.
_SWAPS = [("Prosperity", "Sword of Fire and Ice"), ("Xorn", "Leitmotif Composer"),
          ("Lightning Greaves", "Mask of Memory"),
          ("Mirrormade", "Willie Lumpkin, Postman"), ("Thought Vessel", "Sokka, Lateral Strategist"),
          ("Psychosis Crawler", "Archmage of Runes"), ("Cathartic Reunion", "Brainsurge"),
          ("Gleaming Splendor", "Shang-Chi and the Ten Rings"),
          ("Razorkin Needlehead", "Esper Sentinel")]


@pytest.fixture(scope="module")
def doc():
    from manamap.pilot import try_swap
    d, _, _, _ = try_swap.apply_swaps("sharknado", None, _SWAPS)
    return d


def _strike(doc, name, old, new):
    d = copy.deepcopy(doc)
    hit = 0
    for c in d["cards"]:
        if c["name"] == name:
            assert old in c["oracle_text"], f"{name}: {old!r} is not in its text"
            c["oracle_text"] = c["oracle_text"].replace(old, new)
            hit += 1
    assert hit, f"{name} is not in the doc — the arms are identical"
    return d


def _isolated(doc, keep):
    """EVERY OTHER NONLAND CARD BLANKED, so the cards in `keep` are the deck's only draw
    beyond the draw step. On the full list a struck clause also moves everything the
    missing card would have found, and the old bug passed most of these tests that way.
    The lands keep their text (it is their mana) and so do the commanders."""
    d = copy.deepcopy(doc)
    for c in d["cards"]:
        if c["name"] in keep or c.get("is_commander") or "Land" in (c.get("type_line") or ""):
            continue
        c["oracle_text"] = ""
    return d


def _run(d, n=1500):
    return goldfish.run("sharknado", doc=d, iterations=n, seed=3, quiet=True,
                        _band=False)["metrics"]


def _extra(m, t="8"):
    return m["mean_extra_cards_drawn_by_turn"][t]


_SPELLS = {"Brainstorm", "Brainsurge"}


@requires_deck
@pytest.mark.parametrize("name,old,new,keep,t", [
    ("Sword of Fire and Ice", " and you draw a card", "", set(), "8"),
    ("Mask of Memory", "you may draw two cards. If you do, discard a card.", "nothing happens.",
     set(), "8"),
    ("Leitmotif Composer", "Whenever this creature deals combat damage to a player, draw a card.\n",
     "", set(), "8"),
    ("Willie Lumpkin, Postman", "you draw a card and that player may draw a card",
     "nothing happens", set(), "8"),
    ("Sokka, Lateral Strategist", ", draw a card.", ", scry 1.", {"Willie Lumpkin, Postman"}, "8"),
    ("Archmage of Runes", "Whenever you cast an instant or sorcery spell, draw a card.", "",
     _SPELLS, "8"),
    ("Shang-Chi and the Ten Rings", "\nWhen the tenth +1/+1 counter is put on Shang-Chi, you "
     "draw five cards and gain 5 life.", "",
     _SPELLS | {"Archmage of Runes", "Sword of Fire and Ice", "Leitmotif Composer",
                "Willie Lumpkin, Postman"}, "10"),
    ("Esper Sentinel", "draw a card unless", "scry 1 unless", set(), "8"),
])
def test_each_draw_is_a_draw_the_deck_makes(doc, name, old, new, keep, t):
    """PROVED BY STRIKING THE CLAUSE, on a deck where it is the only draw. Before
    2026-11-06 a combat draw never reached `drawn_extra` and a bodiless combat trigger
    was never cast or registered, so the combat cards read equal either way."""
    d = _isolated(doc, keep | {name})
    assert _extra(_run(d), t) > _extra(_run(_strike(d, name, old, new)), t)


@requires_deck
def test_a_combat_draw_grows_shabraz(doc):
    """THE BUG ITSELF: the draws a hit makes are paid to every per-draw payoff, and
    Shabraz is one. Before 2026-11-06 a combat draw paid him nothing, so these read equal."""
    combat = ("Sword of Fire and Ice", "Leitmotif Composer", "Willie Lumpkin, Postman")
    d = _isolated(doc, set(combat))
    # BRALLIN'S DISCARD COUNTERS ARE COMMANDER COUNTERS TOO, and a fuller hand feeds them
    # through the hand limit, which is how the old bug passed this test. Struck, so the
    # counters are Shabraz's alone.
    d = _strike(d, "Brallin, Skyshark Rider",
                "Whenever you discard a card, put a +1/+1 counter on Brallin and it deals 1 "
                "damage to each opponent.", "")
    struck = d
    for name, old in zip(combat, (" and you draw a card",
                                  "Whenever this creature deals combat damage to a player, "
                                  "draw a card.\n",
                                  "you draw a card and that player may draw a card")):
        struck = _strike(struck, name, old, "nothing happens")
    # THE COUNTERS FOLLOW THE CARDS: at least half the extra cards the hits drew by turn
    # eight are counters on Shabraz by turn ten (one per card; the carry lands them at the
    # next settlement). Without the carry the cards are counted and pay nobody: measured
    # 2026-11-06, +1.39 cards and +0.15 counters, against +1.56 counters with it.
    a, b = _run(d), _run(struck)
    ctr = lambda m: m["discard"]["mean_commander_counters_by_turn"]["10"]
    gained = _extra(a) - _extra(b)
    assert gained > 0.5, "the hits drew nothing — the test has no arm"
    assert ctr(a) - ctr(b) >= 0.5 * gained


@requires_deck
def test_an_equipment_draws_only_once_it_is_equipped(doc):
    """An equip nobody can pay is an equipment that never triggers. Before 2026-11-06
    the equip was never charged, so the two arms read the same."""
    d = _isolated(doc, {"Sword of Fire and Ice"})
    dear = _strike(d, "Sword of Fire and Ice", "Equip {2}", "Equip {30}")
    assert _extra(_run(d)) > _extra(_run(dear))


@requires_deck
def test_put_back_leaves_two_fewer_in_hand(doc):
    d = _isolated(doc, {"Brainsurge"})
    plain = _strike(d, "Brainsurge", ", then put two cards from your hand on top of your "
                    "library in any order", "")
    hand8 = lambda m: m["mean_hand_size_by_turn"]["8"]
    assert hand8(_run(d)) < hand8(_run(plain))


# ── the kit on the commander (2026-11-06) ───────────────────────────────

HELM = _card("Champion's Helm", "Equipped creature gets +2/+2.\nAs long as equipped creature is "
             "legendary, it has hexproof.\nEquip {1}", "Artifact — Equipment", "{3}")
CLEAVE = _card("Embercleave", "Flash\nThis spell costs {1} less to cast for each attacking "
               "creature you control.\nWhen Embercleave enters, attach it to target creature you "
               "control.\nEquipped creature gets +1/+1 and has double strike and trample.\n"
               "Equip {3}", "Legendary Artifact — Equipment", "{4}{R}{R}")


def test_an_equipments_stats_are_read():
    h, e = combat_profile(HELM), combat_profile(CLEAVE)
    assert (h["equip_power"], h["equip_double_strike"], h["equip_cost"]) == (2, False, 1)
    assert (e["equip_power"], e["equip_double_strike"], e["attach_on_enter"]) == (1, True, True)
    # Protection-only Equipment carries no stats and is not cast for them.
    boots = combat_profile(_card("Swiftfoot Boots", "Equipped creature has hexproof and haste.\n"
                                 "Equip {1}", "Artifact — Equipment", "{2}"))
    assert (boots["equip_power"], boots["equip_double_strike"], boots["equip_cost"]) == (0, False, None)


_KIT_SWAPS = [("Prosperity", "Champion's Helm"), ("Xorn", "Embercleave")]


@pytest.fixture(scope="module")
def kit_doc():
    from manamap.pilot import try_swap
    d, _, _, _ = try_swap.apply_swaps("sharknado", None, _KIT_SWAPS)
    return _isolated(d, {"Champion's Helm", "Embercleave"})


def _lethal10(m):
    return m["commander_lethal"]["games_by_turn"]["10"]


@requires_deck
def test_the_kit_rides_on_the_commander(kit_doc):
    """The same cards cast on the same seeds, the bonus smaller: +2/+2 against +1/+1 on the
    Helm, double strike against vigilance on Embercleave. Before 2026-11-06 an Equipment's
    stats reached nobody, so the arms read equal."""
    smaller = _strike(kit_doc, "Champion's Helm", "gets +2/+2", "gets +1/+1")
    no_ds = _strike(kit_doc, "Embercleave", "double strike and trample", "vigilance and trample")
    full = _lethal10(_run(kit_doc, 3000))
    assert full > _lethal10(_run(smaller, 3000))
    assert full > _lethal10(_run(no_ds, 3000))
