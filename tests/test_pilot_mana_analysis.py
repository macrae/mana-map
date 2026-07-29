"""Sources Say's deterministic core: classes, sources, producer kinds."""

from manamap.pilot.mana_analysis import land_classes, nonland_producer_kind
from manamap.pilot.manabase import land_colors


def test_tapped_snow_dual_carries_all_its_classes():
    card = {"name": "Alpine Meadow", "type_line": "Snow Land — Mountain Plains",
            "oracle_text": "({T}: Add {R} or {W}.)\nThis land enters tapped."}
    assert {"snow", "tapped"} <= land_classes(card)
    assert land_colors(card) == {"R", "W"}


def test_add_or_clause_counts_both_colours():
    card = {"name": "Clifftop Retreat", "type_line": "Land",
            "oracle_text": "Clifftop Retreat enters tapped unless you control "
                           "a Mountain or a Plains.\n{T}: Add {R} or {W}.",
            "color_identity": []}
    assert land_colors(card) == {"R", "W"}


def test_fetch_land_classified():
    card = {"name": "Windswept Heath", "type_line": "Land",
            "oracle_text": "{T}, Pay 1 life, Sacrifice this land: Search your "
                           "library for a Forest or Plains card..."}
    assert "fetch" in land_classes(card)


def test_producer_kinds_follow_the_type_line():
    rock = {"name": "Arcane Signet", "type_line": "Artifact",
            "oracle_text": "{T}: Add one mana of any color in your "
                           "commander's color identity."}
    dork = {"name": "Llanowar Elves", "type_line": "Creature — Elf Druid",
            "oracle_text": "{T}: Add {G}."}
    ritual = {"name": "Seething Song", "type_line": "Instant",
              "oracle_text": "Add {R}{R}{R}{R}{R}."}
    plain = {"name": "Bear", "type_line": "Creature — Bear", "oracle_text": ""}
    assert nonland_producer_kind(rock) == "ramp:rock"
    assert nonland_producer_kind(dork) == "ramp:dork"
    assert nonland_producer_kind(ritual) == "ramp:ritual"
    assert nonland_producer_kind(plain) is None


def test_restricted_mana_produces_nothing():
    card = {"name": "Haven", "type_line": "Land", "color_identity": [],
            "oracle_text": "{T}: Add one mana of any color. Spend this mana "
                           "only to cast a Dragon creature spell."}
    assert land_colors(card) == set()
