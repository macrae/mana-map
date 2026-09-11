"""Gishath, Sun's Avatar: the goldfish learns the deck's engines (2026-09-11).

The commander connects for seven, reveals seven, and every Dinosaur among them
enters free. The goldfish read a 7-power body and flagged the trigger
unreadable; it also read Mirari's Wake as nothing, Rishkar's Expertise as
nothing, Earthshaker Dreadmaw as one card and Ghalta and Mavren as unreadable.
Forge reads all of them and its AI uses the trigger (42 resolutions in 40
games), so Forge was the only instrument that saw the deck's win condition.

Declared per deck (`model_commander_combat_reveal`, one card in the corpus)
with a connect rate MEASURED from a Forge run and named; the four parsed
shapes are swept and locked here.
"""

import copy
import json
import random

import pytest

from manamap.pilot import goldfish

from conftest import requires_data, requires_deck


def _card(name, text, cmc=3, type_line="Creature — Dinosaur", power="4",
          toughness="4"):
    return {"name": name, "type_line": type_line, "cmc": cmc, "mana_cost": "",
            "oracle_text": text, "quantity": 1, "is_commander": False,
            "power": power, "toughness": toughness}


# ── 1. The four parsed shapes ─────────────────────────────────────────────

def test_draw_equal_to_greatest_power_is_read_and_not_unmodelled():
    d = goldfish.draw_profile(_card("Rishkar's Expertise",
        "Draw cards equal to the greatest power among creatures you control.\n"
        "You may cast a spell with mana value 5 or less from your hand without "
        "paying its mana cost.", cmc=6, type_line="Sorcery", power=None, toughness=None))
    assert d["spell_draw_greatest_power"] is True and d["spell_draw"] == 0
    assert not d["unmodelled"]


def test_draw_per_type_on_entry_replaces_the_one_card_read():
    d = goldfish.draw_profile(_card("Earthshaker Dreadmaw",
        "Trample\nWhen this creature enters, draw a card for each other Dinosaur "
        "you control.", cmc=6))
    assert d["etb_draw_per_type"] == "Dinosaur"
    assert d["etb_draw"] == 0, "the per-type draw must not also count as one card"
    assert not d["unmodelled"]


@pytest.mark.parametrize("name,text,bonus", [
    ("Mirari's Wake", "Creatures you control get +1/+1.\nWhenever you tap a land "
     "for mana, add one mana of any type that land produced.", 1),
    ("Zendikar Resurgent", "Whenever you tap a land for mana, add an additional "
     "one mana of any type that land produced.\nWhenever you cast a creature "
     "spell, draw a card.", 1),
    ("Sol Ring", "{T}: Add {C}{C}.", 0),
])
def test_a_land_mana_bonus_is_read(name, text, bonus):
    c = goldfish.classify(_card(name, text, cmc=5, type_line="Enchantment",
                                power=None, toughness=None))
    assert c["land_mana_bonus"] == bonus


def test_ghaltas_token_scales_and_utvaras_does_not():
    ghalta = goldfish.combat_profile(_card("Ghalta and Mavren",
        "Trample\nWhenever you attack, choose one —\n• Create a tapped and "
        "attacking X/X green Dinosaur creature token with trample, where X is "
        "the greatest power among other attacking creatures you control.\n"
        "• Create a number of 1/1 white Vampire creature tokens with lifelink "
        "equal to the number of other attacking creatures you control.", cmc=6))
    assert ghalta["attack_token_scales"] is True and ghalta["attack_token_bodies"] >= 1
    assert not ghalta["unreadable"]
    utvara = goldfish.combat_profile(_card("Utvara Hellkite",
        "Flying\nWhenever a Dragon you control attacks, create a 6/6 red Dragon "
        "creature token with flying.", cmc=8, type_line="Creature — Dragon"))
    assert utvara["attack_token_scales"] is False and utvara["attack_token_power"] == 6


@requires_data
def test_the_four_sweeps_are_locked():
    from manamap.pilot import card_pool
    g = p = s = b = 0
    for _, row in card_pool.load_frame().iterrows():
        text = row.get("oracle_text") or ""
        if not isinstance(text, str):
            continue
        card = {"name": row["name"], "oracle_text": text, "type_line": str(row["type_line"]),
                "mana_cost": row.get("mana_cost") or "", "cmc": row.get("cmc") or 0,
                "power": row.get("power"), "toughness": row.get("toughness")}
        d = goldfish.draw_profile(card)
        g += bool(d["spell_draw_greatest_power"])
        p += bool(d["etb_draw_per_type"])
        s += bool(goldfish.combat_profile(card)["attack_token_scales"])
        b += bool("Land" not in card["type_line"] and goldfish._LAND_MANA_BONUS_RE.search(text))
    # 3 "greatest power among creatures you control" + 4 "the power of target
    # creature you control" (Soul's Majesty) -- the same channel, the biggest body.
    assert (g, p, s, b) == (7, 10, 1, 4), (g, p, s, b)


# ── 2. The declared reveal ────────────────────────────────────────────────

def _dino_deck():
    return {"cards": [
        {"name": "Gishath", "type_line": "Legendary Creature — Dinosaur Avatar", "cmc": 8,
         "mana_cost": "{5}{R}{G}{W}", "oracle_text": "Vigilance, trample, haste",
         "quantity": 1, "is_commander": True, "power": "7", "toughness": "6"},
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
         "oracle_text": "", "quantity": 40, "power": None, "toughness": None},
        dict(_card("Raptor", "", cmc=4), quantity=40),
        dict(_card("Rock", "{T}: Add {C}.", cmc=2, type_line="Artifact", power=None,
                   toughness=None), quantity=19),
    ]}


def _run(reveal, iterations=150):
    library, commanders = goldfish.build_library(_dino_deck())
    rng = random.Random(3)
    cc = goldfish.combat_profile(commanders[0])
    results = [goldfish.simulate_once(rng, library, 8, [], 10, model_combat=True,
                                      commander_combat=cc, commander_reveal=reveal)
               for _ in range(iterations)]
    return goldfish.aggregate(results, [], 10, False, True, commander_reveal=reveal)


def test_the_reveal_puts_bodies_onto_the_battlefield_and_reports_itself():
    declared = {"type": "Dinosaur", "connects_per_attack": 1.0, "source": "test"}
    on = _run(declared)
    off = _run(None)
    assert on["commander_reveal"]["declared"] and on["commander_reveal"]["mean_fired"] > 0
    assert on["commander_reveal"]["mean_bodies"] > 0
    assert "commander_reveal" not in off, "absent means absent"
    assert on["combat"]["mean_damage_by_turn"]["10"] > off["combat"]["mean_damage_by_turn"]["10"]
    # A rate of zero fires nothing: the rate is honoured, not the declaration.
    zero = _run({"type": "Dinosaur", "connects_per_attack": 0.0, "source": "test"})
    assert zero["commander_reveal"]["mean_fired"] == 0


def test_the_reveal_is_typed():
    wrong_type = _run({"type": "Dragon", "connects_per_attack": 1.0, "source": "test"})
    assert wrong_type["commander_reveal"]["mean_fired"] > 0
    assert wrong_type["commander_reveal"]["mean_bodies"] == 0


@requires_deck
def test_a_declaration_without_a_rate_or_source_is_refused():
    doc = json.load(open("data/decks/gishath/goldfish_targets.json"))
    bad = copy.deepcopy(doc)
    bad["model_commander_combat_reveal"] = {"type": "Dinosaur"}
    with pytest.raises(SystemExit, match="connects_per_attack"):
        goldfish.run("gishath", quiet=True, iterations=10, _targets_doc=bad)


@requires_deck
def test_gishath_reads_its_own_trigger_and_it_is_worth_something():
    """Prove by re-introducing the gap: the declared reveal removed, the
    commander is a 7-power body again."""
    doc = json.load(open("data/decks/gishath/goldfish_targets.json"))
    blind = copy.deepcopy(doc); blind.pop("model_commander_combat_reveal")
    without = goldfish.run("gishath", quiet=True, iterations=2000, seed=5, _targets_doc=blind
                           )["metrics"]["combat"]["kill_by_turn_rate"]["8"]
    seen = goldfish.run("gishath", quiet=True, iterations=2000, seed=5
                        )["metrics"]["combat"]["kill_by_turn_rate"]["8"]
    assert seen > without, (seen, without)


# ── 3. The land-mana bonus moves mana, from the turn after ────────────────

def test_a_land_mana_bonus_doubles_land_mana_next_turn():
    deck = {"cards": [
        {"name": "Cmd", "type_line": "Legendary Creature — Dinosaur", "cmc": 9,
         "mana_cost": "", "oracle_text": "", "quantity": 1, "is_commander": True,
         "power": "7", "toughness": "6"},
        {"name": "Forest", "type_line": "Basic Land — Forest", "cmc": 0,
         "oracle_text": "", "quantity": 60, "power": None, "toughness": None},
        dict(_card("Wake", "Whenever you tap a land for mana, add one mana of any "
                   "type that land produced.", cmc=1, type_line="Enchantment",
                   power=None, toughness=None), quantity=39),
    ]}
    library, _ = goldfish.build_library(deck)
    rng = random.Random(1)
    r = goldfish.aggregate([goldfish.simulate_once(rng, library, 9, [], 8)
                            for _ in range(100)], [], 8)
    mana = r["mean_available_mana_by_turn"]
    assert mana["6"] > 2 * 4.5, mana


# ── 4. The second batch: typed entry triggers, casts that fire things, a
#      spell that hits for power, a tutor onto the battlefield ─────────────

def test_a_typed_entry_trigger_names_its_type_and_a_generic_one_does_not():
    tempest = goldfish.combat_profile(_card("Dragon Tempest",
        "Whenever a creature you control with flying enters, it gains haste until "
        "end of turn.\nWhenever a Dragon you control enters, it deals X damage to any "
        "target, where X is the number of Dragons you control.", cmc=2,
        type_line="Enchantment", power=None, toughness=None))
    assert tempest["etb_type_gate"] == "Dragon"
    terror = goldfish.combat_profile(_card("Terror of the Peaks",
        "Flying\nWhenever another creature enters under your control, Terror of the "
        "Peaks deals damage equal to that creature's power to any target.", cmc=5,
        type_line="Creature — Dragon"))
    assert terror["etb_type_gate"] is None
    echoes = goldfish.combat_profile(_card("Molten Echoes",
        "As this enchantment enters, choose a creature type.\nWhenever a nontoken "
        "creature you control of the chosen type enters, create a token that's a "
        "copy of that creature. That token gains haste. Exile it at the beginning "
        "of the next end step.", cmc=4, type_line="Enchantment", power=None, toughness=None))
    assert echoes["etb_copy"] and echoes["etb_type_gate"] == "chosen" and echoes["etb_nontoken_only"]


def test_the_cast_shapes_are_read():
    beanstalk = goldfish.draw_profile(_card("Up the Beanstalk",
        "When this enchantment enters and whenever you cast a spell with mana value "
        "5 or greater, draw a card.", cmc=2, type_line="Enchantment", power=None, toughness=None))
    assert beanstalk["cast_draw"] == 1 and beanstalk["cast_draw_gate_mv"] == 5
    bestiary = goldfish.draw_profile(_card("Lifecrafter's Bestiary",
        "At the beginning of your upkeep, scry 1.\nWhenever you cast a creature "
        "spell, you may pay {G}. If you do, draw a card.", cmc=3, type_line="Artifact",
        power=None, toughness=None))
    assert bestiary["cast_draw"] == 1 and bestiary["cast_draw_gate"] == "Creature" and bestiary["cast_draw_cost"] == 1
    unsealing = goldfish.combat_profile(_card("Sarkhan's Unsealing",
        "Whenever you cast a creature spell with power 4, 5, or 6, this enchantment "
        "deals 4 damage to any target.\nWhenever you cast a creature spell with power "
        "7 or greater, this enchantment deals 4 damage to each opponent and each "
        "creature they control.", cmc=4, type_line="Enchantment", power=None, toughness=None))
    assert unsealing["cast_damage"] == 4 and unsealing["cast_damage_power_min"] == 4
    ignition = goldfish.combat_profile(_card("Chandra's Ignition",
        "Target creature you control deals damage equal to its power to each other "
        "creature and each opponent.", cmc=5, type_line="Sorcery", power=None, toughness=None))
    assert ignition["spell_damage_greatest_power"] is True
    order = goldfish.classify(_card("Savage Order",
        "As an additional cost to cast this spell, sacrifice a creature with power 4 "
        "or greater.\nSearch your library for a Dinosaur creature card, put it onto "
        "the battlefield, then shuffle. It gains indestructible until your next turn.",
        cmc=4, type_line="Sorcery", power=None, toughness=None))
    assert order["tutor"] and order["tutor_to_battlefield"] == "Dinosaur" and order["tutor_needs_body"]


@requires_data
def test_the_second_batch_sweeps_are_locked():
    from manamap.pilot import card_pool
    c = dict(gate=0, chosen=0, spell_dmg=0, cast_dmg=0, mv=0, pay=0, tutor_bf=0)
    for _, row in card_pool.load_frame().iterrows():
        text = row.get("oracle_text") or ""
        if not isinstance(text, str):
            continue
        mc = row.get("mana_cost")
        card = {"name": row["name"], "oracle_text": text, "type_line": str(row["type_line"]),
                "mana_cost": mc if isinstance(mc, str) else "", "cmc": row.get("cmc") or 0,
                "power": row.get("power"), "toughness": row.get("toughness")}
        p = goldfish.classify(card); cb = p["combat"]; d = p["draw"]
        c["gate"] += bool(cb["etb_type_gate"] and cb["etb_type_gate"] != "chosen")
        c["chosen"] += cb["etb_type_gate"] == "chosen"
        c["spell_dmg"] += bool(cb["spell_damage_greatest_power"])
        c["cast_dmg"] += bool(cb["cast_damage"])
        c["mv"] += bool(d["cast_draw_gate_mv"])
        c["pay"] += bool(d["cast_draw_cost"])
        c["tutor_bf"] += p["tutor_to_battlefield"] is not None
    assert c == dict(gate=144, chosen=1, spell_dmg=1, cast_dmg=1, mv=3, pay=1, tutor_bf=3), c


def _sim(cards, model_draw=False, iterations=200, seed=9):
    deck = {"cards": [
        {"name": "Cmd", "type_line": "Legendary Creature — Dinosaur Avatar", "cmc": 8,
         "mana_cost": "", "oracle_text": "Haste", "quantity": 1, "is_commander": True,
         "power": "7", "toughness": "6"},
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
         "oracle_text": "", "quantity": 40, "power": None, "toughness": None},
    ] + cards}
    library, commanders = goldfish.build_library(deck)
    rng = random.Random(seed)
    cc = goldfish.combat_profile(commanders[0])
    rs = [goldfish.simulate_once(rng, library, 8, [], 10, model_combat=True, model_draw=model_draw,
                                 commander_combat=cc) for _ in range(iterations)]
    return goldfish.aggregate(rs, [], 10, False, True, model_draw)


def test_a_typed_trigger_does_not_fire_on_the_wrong_type():
    tempest = dict(_card("Tempest", "Whenever a Dragon you control enters, it deals 3 "
                         "damage to any target.", cmc=2, type_line="Enchantment",
                         power=None, toughness=None), quantity=20)
    birds = dict(_card("Bird", "", cmc=2, type_line="Creature — Bird", power="1", toughness="1"), quantity=39)
    dragons = dict(_card("Drake", "", cmc=2, type_line="Creature — Dragon", power="1", toughness="1"), quantity=39)
    with_birds = _sim([tempest, birds])["combat"]["mean_damage_by_turn"]["6"]
    with_dragons = _sim([tempest, dragons])["combat"]["mean_damage_by_turn"]["6"]
    assert with_dragons > with_birds + 5, (with_dragons, with_birds)


def test_a_cast_fires_at_the_commander_door_too():
    beanstalk = dict(_card("Beanstalk", "Whenever you cast a spell with mana value 5 or "
                           "greater, draw a card.", cmc=2, type_line="Enchantment",
                           power=None, toughness=None), quantity=20)
    filler = dict(_card("Rock", "{T}: Add {C}.", cmc=2, type_line="Artifact", power=None, toughness=None), quantity=39)
    on = _sim([beanstalk, filler], model_draw=True)["mean_extra_cards_drawn_by_turn"]["10"]
    off = _sim([dict(filler, quantity=59)], model_draw=True)["mean_extra_cards_drawn_by_turn"]["10"]
    assert on > off, (on, off)     # the eight-mana commander is the only 5+ spell


def test_a_spell_that_deals_power_hits_for_the_biggest_body():
    ignition = dict(_card("Ignition", "Target creature you control deals damage equal to "
                          "its power to each other creature and each opponent.", cmc=5,
                          type_line="Sorcery", power=None, toughness=None), quantity=20)
    filler = dict(_card("Rock", "{T}: Add {C}.", cmc=2, type_line="Artifact", power=None, toughness=None), quantity=39)
    on = _sim([ignition, filler])["combat"]["mean_damage_by_turn"]["10"]
    off = _sim([dict(filler, quantity=59)])["combat"]["mean_damage_by_turn"]["10"]
    # The spell needs a body to aim; the eight-mana commander is the only one,
    # so it fires late and once per copy drawn after that -- small, but real.
    assert on > off + 1, (on, off)


def test_a_tutor_onto_the_battlefield_fetches_the_biggest_of_its_type():
    order = dict(_card("Order", "As an additional cost to cast this spell, sacrifice a "
                       "creature with power 4 or greater.\nSearch your library for a "
                       "Dinosaur creature card, put it onto the battlefield, then shuffle.",
                       cmc=4, type_line="Sorcery", power=None, toughness=None), quantity=10)
    fodder = dict(_card("Fodder", "", cmc=3, power="4", toughness="4"), quantity=30)
    titan = dict(_card("Titan", "", cmc=9, power="12", toughness="12"), quantity=19)
    on = _sim([order, fodder, titan])["combat"]["mean_board_power_by_turn"]["7"]
    off = _sim([dict(fodder, quantity=40), titan])["combat"]["mean_board_power_by_turn"]["7"]
    assert on > off, (on, off)
