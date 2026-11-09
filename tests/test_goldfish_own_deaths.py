"""Every death of ours pays Meren's experience (2026-11-07).

"Whenever another creature you control dies, you get an experience counter." Until this
date the model counted a death only where its own outlets sacrificed one, so Sakura-Tribe
Elder, Spore Frog, Cankerbloom and Haywire Mite sacrificing themselves, Plaguecrafter's
edict, Skullclamp and the body a sacrifice-tutor costs all left her counter where it was:
45% of games reached turn ten with no experience at all.

`own_death_profile` reads the three shapes from the card's own text; the turn loop kills
the body through `_deaths_after_combat`, the one helper that now pays the counter.
"""

import json

import pytest

from manamap.pilot import goldfish, goldfish_library, model_coverage
from manamap.pilot.common import deck_file
from manamap.pilot.goldfish_profiles import own_death_profile

from conftest import requires_deck, requires_dump

SLUG = "meren-recursion"


def _card(name, text, type_line):
    return {"name": name, "oracle_text": text, "type_line": type_line}


def test_the_three_shapes_read_from_the_text():
    elder = own_death_profile(_card("Sakura-Tribe Elder", "Sacrifice this creature: Search your "
                                    "library for a basic land card, put that card onto the "
                                    "battlefield tapped, then shuffle.", "Creature — Snake Shaman"))
    assert elder["self_sac"] and elder["self_sac_cost"] == 0
    mite = own_death_profile(_card("Haywire Mite", "{G}, Sacrifice this creature: Exile target "
                                   "noncreature artifact or noncreature enchantment.",
                                   "Artifact Creature — Insect"))
    assert mite["self_sac"] and mite["self_sac_cost"] == 1
    crafter = own_death_profile(_card("Plaguecrafter", "When this creature enters, each player "
                                      "sacrifices a creature or planeswalker of their choice. Each "
                                      "player who can't discards a card.", "Creature — Human Shaman"))
    assert crafter["self_edict"] and not crafter["self_sac"]
    clamp = own_death_profile(_card("Skullclamp", "Equipped creature gets +1/-1.\nWhenever "
                                    "equipped creature dies, draw two cards.\nEquip {1}",
                                    "Artifact — Equipment"))
    assert clamp["clamp"] == {"toughness_minus": 1, "draw": 2, "equip": 1}


def test_an_equipment_that_pays_nothing_on_the_death_is_not_a_clamp():
    """The corpus sweep matched Barbed Battlegear (+4/-1, no death clause) beside Skullclamp."""
    gear = own_death_profile(_card("Barbed Battlegear", "Equipped creature gets +4/-1.\nEquip {2}",
                                   "Artifact — Equipment"))
    assert gear["clamp"] is None


def test_a_trigger_that_sacrifices_is_not_an_activated_cost():
    """"Sacrifice this creature" inside an effect (no colon after it) is not a choice the pilot
    makes; only a COST is read."""
    p = own_death_profile(_card("Ball Lightning", "Trample, haste\nAt the beginning of the end "
                                "step, sacrifice this creature.", "Creature — Elemental"))
    assert not p["self_sac"]


def test_the_clamp_is_cast_and_its_mirror_agrees():
    prof = {"is_land": False, "bodies": 0, "produces": 0, "tutor": False, "reduces": None,
            "sac_outlet": None, "death": {}, "own_death": {"clamp": {"toughness_minus": 1,
                                                                     "draw": 2, "equip": 1}}}
    assert model_coverage.never_cast(dict(prof), {"model_commander_recursion": True}) is False
    assert model_coverage.never_cast(dict(prof), {}) is True


def _xp(monkeypatch=None, blank=None, n=1500, doc=None):
    if blank is not None:
        real = goldfish_library.own_death_profile

        def _without(card):
            p = real(card)
            return {**p, **blank}
        monkeypatch.setattr(goldfish_library, "own_death_profile", _without)
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    m = goldfish.run(SLUG, iterations=n, seed=7, quiet=True, _band=False, _targets_doc=t,
                     doc=doc)["metrics"]
    return m["commander_recursion"]["mean_experience_by_turn"]["10"], m


@requires_deck
@pytest.mark.parametrize("shape,blank", [
    ("self-sacrifice", {"self_sac": False}),
    ("edict", {"self_edict": False}),
    ("clamp", {"clamp": None}),
])
def test_each_death_source_pays_experience(monkeypatch, shape, blank):
    """PROVED BY REMOVING EACH SHAPE: her experience at turn ten falls without it.
    The list must carry the shape: Plaguecrafter, the only edict, left in V18 (2026-11-09),
    so the edict case puts it back in for a removal spell the goldfish does not read."""
    from manamap.pilot import try_swap
    doc = None
    have = {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}
    if shape == "edict" and "Plaguecrafter" not in have:
        doc = try_swap.apply_swaps(SLUG, None, [("Assassin's Trophy", "Plaguecrafter")])[0]
    on, _ = _xp(doc=doc)
    off, _ = _xp(monkeypatch, blank, doc=doc)
    assert on > off, (shape, on, off)


@requires_deck
def test_the_clamp_draws_on_the_death(monkeypatch):
    _, on = _xp()
    _, off = _xp(monkeypatch, {"clamp": None})
    assert on["mean_extra_cards_drawn_by_turn"]["10"] > off["mean_extra_cards_drawn_by_turn"]["10"]


# ── the bodies that come back every turn (2026-11-07) ─────────────────────────────────────

from manamap.pilot.goldfish_profiles import recurring_body_profile  # noqa: E402


def test_the_recurring_shapes_read_from_the_text():
    snake = recurring_body_profile(_card("Ophiomancer", "At the beginning of each upkeep, if you "
                                         "control no Snakes, create a 1/1 black Snake creature "
                                         "token with deathtouch.", "Creature — Human Shaman"))
    assert snake["upkeep"] == {"each": True, "only_if_none": True, "n": 1, "power": 1,
                               "toughness": 1}
    faerie = recurring_body_profile(_card("Bitterblossom", "At the beginning of your upkeep, you "
                                          "lose 1 life and create a 1/1 black Faerie Rogue "
                                          "creature token with flying.", "Kindred Enchantment — Faerie"))
    assert faerie["upkeep"]["each"] is False and faerie["upkeep"]["n"] == 1
    jadar = recurring_body_profile(_card("Jadar, Ghoulcaller of Nephalia", "At the beginning of "
                                         "your end step, if you control no creatures with "
                                         "decayed, create a 2/2 black Zombie creature token with "
                                         "decayed.", "Legendary Creature — Human Wizard"))
    assert jadar["end_step"]["only_if_none"] and jadar["end_step"]["power"] == 2
    skel = recurring_body_profile(_card("Reassembling Skeleton", "{1}{B}: Return this card from "
                                        "your graveyard to the battlefield tapped.",
                                        "Creature — Skeleton Warrior"))
    assert skel["return_cost"] == 2
    ghast = recurring_body_profile(_card("Bloodghast", "This creature can't block.\nLandfall — "
                                         "Whenever a land you control enters, you may return this "
                                         "card from your graveyard to the battlefield.",
                                         "Creature — Vampire Spirit"))
    assert ghast["return_landfall"]


def _xp_with(monkeypatch, swaps, blank=None, n=1500):
    from manamap.pilot import try_swap
    if blank is not None:
        real = goldfish_library.recurring_body_profile
        monkeypatch.setattr(goldfish_library, "recurring_body_profile",
                            lambda card: {**real(card), **blank})
    doc = try_swap.apply_swaps(SLUG, None, swaps)[0] if swaps else None
    m = goldfish.run(SLUG, iterations=n, seed=7, quiet=True, _band=False, doc=doc)["metrics"]
    return m["commander_recursion"]["mean_experience_by_turn"]["10"]


@requires_deck
@requires_dump
@pytest.mark.parametrize("card,blank", [
    ("Ophiomancer", {"upkeep": None}),
    ("Jadar, Ghoulcaller of Nephalia", {"end_step": None}),
    ("Reassembling Skeleton", {"return_cost": None}),
])
def test_a_body_that_comes_back_pays_experience(monkeypatch, card, blank):
    """PROVED BY REMOVING THE SHAPE: the same list, the same seeds, fewer counters."""
    # the card may already be in the list (Ophiomancer and Jadar went in on 2026-11-07)
    have = {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}
    swaps = [] if card in have else [("Assassin's Trophy", card)]
    assert _xp_with(monkeypatch, swaps) > _xp_with(monkeypatch, swaps, blank), card


# ── the reanimator half and the big bodies' arrival (2026-11-07) ──────────────────────────

from manamap.pilot.goldfish_profiles import graveyard_profile  # noqa: E402


def test_the_graveyard_shapes_read_from_the_text():
    entomb = graveyard_profile(_card("Entomb", "Search your library for a card, put that card "
                                     "into your graveyard, then shuffle.", "Instant"))
    assert entomb["to_yard"] == 1 and not entomb["reanimate"]
    buried = graveyard_profile(_card("Buried Alive", "Search your library for up to three "
                                     "creature cards, put them into your graveyard, then "
                                     "shuffle.", "Sorcery"))
    assert buried["to_yard"] == 3 and buried["to_yard_creatures_only"]
    ad = graveyard_profile(_card("Animate Dead", "Enchant creature card in a graveyard\nWhen "
                                 "this Aura enters, if it's on the battlefield, it loses \"enchant "
                                 "creature card in a graveyard\" and gains \"enchant creature put "
                                 "onto the battlefield with this Aura.\" Return enchanted creature "
                                 "card to the battlefield under your control and attach this Aura "
                                 "to it.", "Enchantment — Aura"))
    assert ad["reanimate"] and not ad["temporary"]
    feet = graveyard_profile(_card("Footsteps of the Goryo", "Return target creature card from "
                                   "your graveyard to the battlefield. Sacrifice that creature at "
                                   "the beginning of the next end step.", "Sorcery — Arcane"))
    assert feet["reanimate"] and feet["temporary"]
    unearth = graveyard_profile(_card("Unearth", "Return target creature card with mana value 3 or "
                                      "less from your graveyard to the battlefield.", "Sorcery"))
    assert unearth["reanimate"] and unearth["reanimate_max_mv"] == 3, "the cap is a limit, kept"
    persist = graveyard_profile(_card("Persist", "Return target nonlegendary creature card from "
                                      "your graveyard to the battlefield with a -1/-1 counter on "
                                      "it.", "Sorcery"))
    assert persist["reanimate"] and persist["reanimate_nonlegendary"]
    vic = graveyard_profile(_card("Victimize", "Choose two target creature cards in your "
                                  "graveyard. Sacrifice a creature. If you do, return the chosen "
                                  "cards to the battlefield tapped.", "Sorcery"))
    assert vic["reanimate"] and vic["reanimate_n"] == 2 and vic["reanimate_sac"]
    tutor = graveyard_profile(_card("Demonic Tutor", "Search your library for a card, put that "
                                    "card into your hand, then shuffle.", "Sorcery"))
    assert tutor["to_yard"] == 0, "a tutor to HAND is not a loader"


def _routes(monkeypatch=None, blank=None, n=1500):
    if blank is not None:
        real = goldfish_library.graveyard_profile
        monkeypatch.setattr(goldfish_library, "graveyard_profile", lambda c: {**real(c), **blank})
    from manamap.pilot import goldfish as g
    out = []
    orig = g.simulate_once

    def w(*a, **k):
        r = orig(*a, **k)
        out.append(r)
        return r
    monkeypatch.setattr(g, "simulate_once", w)
    g.run(SLUG, iterations=n, seed=7, quiet=True, _band=False)
    return [route for r in out for _mv, _t, route in r["big_arrivals"]]


@requires_deck
def test_a_reanimation_spell_brings_a_big_body_back(monkeypatch):
    """PROVED BY REMOVING THE SHAPE: with no card read as a reanimator, no arrival is."""
    on = _routes(monkeypatch)
    assert on.count("reanimate") > 0
    with pytest.MonkeyPatch.context() as mp:
        off = _routes(mp, {"reanimate": False, "temporary": False})
    assert off.count("reanimate") == 0


@requires_deck
def test_the_recursion_rows_are_absent_on_a_deck_without_meren():
    from manamap.pilot import diagnostic
    doc = json.loads(deck_file("gishath", "cards.json").read_text())
    assert "recursion" not in diagnostic.run_on(doc, "gishath", iterations=50, seed=1, quiet=True)


def test_the_loader_shapes_read_from_the_text():
    surv = graveyard_profile(_card("Survival of the Fittest", "{G}, Discard a creature card: "
                                   "Search your library for a creature card, reveal it, put it "
                                   "into your hand, then shuffle.", "Enchantment"))
    assert surv["discard_tutor"] == {"cost": 1, "taps": False}
    shaman = graveyard_profile(_card("Fauna Shaman", "{G}, {T}, Discard a creature card: Search "
                                     "your library for a creature card, reveal it, put it into "
                                     "your hand, then shuffle.", "Creature — Elf Shaman"))
    assert shaman["discard_tutor"] == {"cost": 1, "taps": True}
    grave = graveyard_profile(_card("Unmarked Grave", "Search your library for a nonlegendary "
                                    "card, put that card into your graveyard, then shuffle.",
                                    "Sorcery"))
    assert grave["to_yard"] == 1 and grave["to_yard_nonlegendary"]
    orders = graveyard_profile(_card("Jarad's Orders", "Search your library for up to two "
                                     "creature cards and reveal them. Put one into your hand and "
                                     "the other into your graveyard. Then shuffle.", "Sorcery"))
    assert orders["to_yard"] == 1 and orders["to_yard_also_hand"]
    lore = graveyard_profile(_card("Oriq Loremage", "{T}: Search your library for a card, put "
                                   "it into your graveyard, then shuffle.", "Creature — Human Warlock"))
    assert lore["to_yard_tap"], "a tap loader loads every turn, not once"


@requires_deck
def test_survival_puts_creatures_in_the_graveyard_for_meren(monkeypatch):
    """PROVED BY REMOVING THE SHAPE: without the discard tutor read, Meren's end step finds
    fewer big bodies, so fewer of them arrive by her route."""
    on = _routes(monkeypatch)
    with pytest.MonkeyPatch.context() as mp:
        off = _routes(mp, {"discard_tutor": None})
    assert on.count("meren") > off.count("meren")


def test_hogaak_and_mazirek_read_from_the_text():
    hog = own_death_profile(_card("Hogaak, Arisen Necropolis", "You can't spend mana to cast "
                                  "this spell.\nConvoke, delve\nTrample", "Legendary Creature — Avatar"))
    assert hog["no_mana"] and hog["convoke"] and hog["delve"] and not hog["cast_from_yard"]
    hog2 = own_death_profile(_card("Hogaak, Arisen Necropolis", "You can't spend mana to cast "
                                   "this spell.\nConvoke, delve\nYou may cast this card from your "
                                   "graveyard.\nTrample", "Legendary Creature — Avatar"))
    assert hog2["cast_from_yard"]
    maz = own_death_profile(_card("Mazirek, Kraul Death Priest", "Flying\nWhenever a player "
                                  "sacrifices another permanent, put a +1/+1 counter on each "
                                  "creature you control.", "Legendary Creature — Insect Shaman"))
    assert maz["sac_team_counter"]


@requires_deck
@requires_dump
def test_hogaak_is_cast_by_convoke_and_delve_not_mana(monkeypatch):
    """Under the rule Hogaak lands on turns the mana could not pay seven; read as an ordinary
    7-drop (the rule removed) it never does. PROVED BY REMOVING THE RULE."""
    from manamap.pilot import try_swap
    # the card may be in the list already (Hogaak went in with V6, 2026-11-07)
    have = {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}
    doc = None if "Hogaak, Arisen Necropolis" in have else try_swap.apply_swaps(
        SLUG, None, [("Assassin's Trophy", "Hogaak, Arisen Necropolis")])[0]

    def cast_mana(mp):
        out = []
        orig = goldfish.simulate_once

        def w(*a, **k):
            r = orig(*a, **k)
            out.append(r)
            return r
        mp.setattr(goldfish, "simulate_once", w)
        goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False, doc=doc)
        return [r["mana_by_turn"][t - 1] for r in out
                for (mv, t, route), n in zip(r["big_arrivals"], r["big_arrival_names"])
                if n.startswith("Hogaak") and route == "cast"]
    with pytest.MonkeyPatch.context() as mp:
        ruled = cast_mana(mp)
    with pytest.MonkeyPatch.context() as mp:
        real = goldfish_library.own_death_profile
        mp.setattr(goldfish_library, "own_death_profile", lambda c: {**real(c), "no_mana": False})
        free = cast_mana(mp)
    assert ruled and any(m < 7 for m in ruled), "convoke and delve never paid for it"
    # `mana_by_turn` is REPEATABLE mana: a Dark Ritual or a Treasure on top of five lands pays
    # seven, so a hard cast can sit under 7 here (2026-11-09: 3 of 159 once tutors fetched
    # Hogaak as a combo piece). The rule's signature is the SHARE of casts under seven.
    low = lambda xs: sum(m < 7 for m in xs) / len(xs)
    assert free and low(free) < 0.05, "a hard cast needs seven mana"
    assert low(ruled) > low(free) + 0.2, (low(ruled), low(free))


@requires_deck
@requires_dump
def test_mazirek_grows_the_board_on_every_sacrifice(monkeypatch):
    """PROVED BY REMOVING THE TRIGGER: the same games, a smaller board by turn ten."""
    from manamap.pilot import try_swap
    # the card may be in the list already (Hogaak went in with V6, 2026-11-07)
    have = {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}
    doc = None if "Mazirek, Kraul Death Priest" in have else try_swap.apply_swaps(
        SLUG, None, [("Assassin's Trophy", "Mazirek, Kraul Death Priest")])[0]

    def power(mp):
        m = goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False, doc=doc)["metrics"]
        return m["combat"]["mean_board_power_by_turn"]["10"] if "mean_board_power_by_turn" in m.get("combat", {}) \
            else m["mean_damage_by_turn"]["10"]
    on = power(monkeypatch)
    with pytest.MonkeyPatch.context() as mp:
        real = goldfish_library.own_death_profile
        mp.setattr(goldfish_library, "own_death_profile", lambda c: {**real(c), "sac_team_counter": False})
        off = power(mp)
    assert on > off, (on, off)


def test_the_x_team_pumps_read_from_the_text():
    from manamap.pilot.goldfish_profiles import team_pump_x
    hoof = team_pump_x(_card("Craterhoof Behemoth", "Haste\nWhen this creature enters, creatures "
                             "you control gain trample and get +X/+X until end of turn, where X "
                             "is the number of creatures you control.", "Creature — Beast"))
    assert hoof == {"when": "enters", "x": "creatures"}
    stamp = team_pump_x(_card("Overwhelming Stampede", "Until end of turn, creatures you control "
                              "gain trample and get +X/+X, where X is the greatest power among "
                              "creatures you control.", "Sorcery"))
    assert stamp == {"when": "spell", "x": "greatest_power"}


@requires_deck
def test_craterhoof_pumps_the_team(monkeypatch):
    """PROVED BY REMOVING THE PUMP: the same games, less damage by turn ten."""
    have = {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}
    from manamap.pilot import try_swap
    doc = None if "Craterhoof Behemoth" in have else try_swap.apply_swaps(
        SLUG, None, [("Assassin's Trophy", "Craterhoof Behemoth")])[0]

    def dmg(mp):
        return goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False,
                            doc=doc)["metrics"]["combat"]["mean_damage_by_turn"]["10"]
    on = dmg(monkeypatch)
    with pytest.MonkeyPatch.context() as mp:
        real = goldfish_library.own_death_profile
        mp.setattr(goldfish_library, "own_death_profile", lambda c: {**real(c), "team_pump_x": None})
        off = dmg(mp)
    assert on > off, (on, off)


# ── the experience economy: the PLAYER'S counters (2026-11-08) ────────────────────────────

from manamap.pilot.goldfish_profiles import experience_profile  # noqa: E402


def test_the_experience_shapes_read_from_the_text():
    toph = experience_profile(_card(
        "Toph, Earthbending Master", "Landfall — Whenever a land you control enters, you get an "
        "experience counter. Whenever you attack, earthbend X, where X is the number of "
        "experience counters you have.", "Legendary Creature — Human Warrior Ally"))
    assert toph["landfall_experience"] == 1
    con = experience_profile(_card(
        "Winding Constrictor", "If one or more counters would be put on an artifact or creature "
        "you control, that many plus one of each of those kinds of counters are put on it "
        "instead.\nIf you would get one or more counters, you get that many plus one of each of "
        "those kinds of counters instead.", "Creature — Snake"))
    assert con["player_plus"] == 1 and con["player_times"] == 1
    sage = experience_profile(_card(
        "Evolution Sage", "Landfall — Whenever a land you control enters, proliferate. (Choose "
        "any number of permanents and/or players, then give each another counter of each kind "
        "already there.)", "Creature — Elf Druid"))
    assert sage["proliferate"] == "landfall"
    yawg = experience_profile(_card(
        "Yawgmoth, Thran Physician", "Protection from Humans\nPay 1 life, Sacrifice another "
        "creature: Put a -1/-1 counter on up to one target creature and draw a card.\n{B}{B}, "
        "Discard a card: Proliferate.", "Legendary Creature — Human Cleric"))
    assert yawg["proliferate"] == "activated"
    assert yawg["proliferate_cost"] == {"generic": 0, "pips": ["B", "B"], "tap": False,
                                        "discard": True, "life": 0}
    assert toph["earthbend"] == "experience"
    misty = experience_profile(_card(
        "Misty Rainforest", "{T}, Pay 1 life, Sacrifice this land: Search your library for a "
        "Forest or Island card, put it onto the battlefield, then shuffle.", "Land"))
    assert misty["fetch_landfall"]
    talent = experience_profile(_card(
        "Innkeeper's Talent", "Level 3: If you would put one or more counters on a permanent or "
        "player, put twice that many of each of those kinds of counters on that permanent or "
        "player instead.", "Enchantment — Class"))
    assert talent["player_times"] == 1                  # a level nothing buys


def _xp_exp(monkeypatch, swaps, blank=None, n=1500):
    from manamap.pilot import try_swap
    if blank is not None:
        real = goldfish_library.experience_profile
        monkeypatch.setattr(goldfish_library, "experience_profile",
                            lambda card: {**real(card), **blank})
    doc = try_swap.apply_swaps(SLUG, None, swaps)[0] if swaps else None
    m = goldfish.run(SLUG, iterations=n, seed=7, quiet=True, _band=False, doc=doc)["metrics"]
    return m["commander_recursion"]["mean_experience_by_turn"]["7"]


def _have():
    return {line.split(" ", 1)[1].strip() for line in
            deck_file(SLUG, "decklist.txt").read_text().splitlines() if line[:1].isdigit()}


@requires_deck
@requires_dump
@pytest.mark.parametrize("card,blank", [
    ("Winding Constrictor", {"player_plus": 0}),
    ("Toph, Earthbending Master", {"landfall_experience": 0}),
    ("Evolution Sage", {"proliferate": None}),
    ("Yawgmoth, Thran Physician", {"proliferate": None, "proliferate_cost": None}),
])
def test_each_experience_source_pays(monkeypatch, card, blank):
    """PROVED BY REMOVING THE SHAPE: the same list, the same seeds, fewer counters at T7."""
    swaps = [] if card in _have() else [("Assassin's Trophy", card)]
    assert _xp_exp(monkeypatch, swaps) > _xp_exp(monkeypatch, swaps, blank), card


@requires_deck
def test_the_economy_is_inert_without_merens_declaration():
    """Gated on `model_commander_recursion`: a deck without it plays the games it played."""
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    t.pop("model_commander_recursion", None)
    m = goldfish.run(SLUG, iterations=300, seed=7, quiet=True, _band=False, _targets_doc=t)["metrics"]
    assert "commander_recursion" not in m


def test_read_proliferate_names_what_the_turn_loop_applies():
    """`model_coverage.READ_PROLIFERATE` is a claim about the loop: each timing it names is
    branched on there, and an unapplied one (activated) is not in it."""
    from conftest import simulator_source
    src = simulator_source()
    for kind in model_coverage.READ_PROLIFERATE:
        assert f'_xp.get("proliferate") == "{kind}"' in src, kind
    assert "cast" not in model_coverage.READ_PROLIFERATE and "other" not in model_coverage.READ_PROLIFERATE


@requires_deck
@requires_dump
def test_earthbend_adds_damage(monkeypatch):
    """PROVED BY REMOVING THE SHAPE: Toph's earthbent land swings for her experience."""
    from manamap.pilot import try_swap
    swaps = [] if "Toph, Earthbending Master" in _have() else [("Assassin's Trophy",
                                                               "Toph, Earthbending Master")]
    doc = try_swap.apply_swaps(SLUG, None, swaps)[0] if swaps else None

    def dmg():
        m = goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False, doc=doc)["metrics"]
        return m["combat"]["mean_damage_by_turn"]["10"]
    on = dmg()
    real = goldfish_library.experience_profile
    monkeypatch.setattr(goldfish_library, "experience_profile",
                        lambda card: {**real(card), "earthbend": None})
    assert on > dmg()


# ── an assembled infinite ends the game (`model_combos`, 2026-11-09) ─────────────────────

@requires_deck
def test_an_assembled_infinite_with_a_finisher_is_a_kill(monkeypatch):
    """The lines are Spellbook's; a line whose pieces are on the battlefield and which can
    finish ends the game. PROVED BY GIVING EVERY LINE A FINISHER: the combo kill rate rises,
    and with the flag off it is absent."""
    from manamap.pilot import goldfish_turn
    t = json.loads(deck_file(SLUG, "goldfish_targets.json").read_text())
    t["model_combos"] = True
    base = goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False,
                        _targets_doc=t)["metrics"]["combat"].get("kill_by_combo_rate") or 0
    real = goldfish_turn.simulate_once

    def finishing(*a, **k):
        if k.get("combo_lines"):
            k["combo_lines"] = [{**ln, "produces": ln["produces"] + ("Infinite damage",)}
                                for ln in k["combo_lines"]]
        return real(*a, **k)
    monkeypatch.setattr(goldfish, "simulate_once", finishing)
    forced = goldfish.run(SLUG, iterations=1500, seed=7, quiet=True, _band=False,
                          _targets_doc=t)["metrics"]["combat"]["kill_by_combo_rate"]
    assert forced > base
    monkeypatch.undo()
    t["model_combos"] = False
    off = goldfish.run(SLUG, iterations=300, seed=7, quiet=True, _band=False, _targets_doc=t)
    assert "kill_by_combo_rate" not in off["metrics"]["combat"]
