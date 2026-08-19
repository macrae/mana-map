"""mana-analysis's deterministic core: classes, sources, producer kinds."""

from conftest import requires_deck

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


# ── Copies, not entries: the bug that published "18 lands" for a 33-land deck ──


def _deck(tmp_path, monkeypatch, cards):
    import json
    decks = tmp_path / "decks"
    base = decks / "test-deck"
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "cards.json").write_text(json.dumps(
        {"deck": "test-deck", "decklist_sha256": "abc", "cards": cards}))
    return base


def _land(name, oracle, quantity=1, type_line="Basic Land — Island"):
    return {"name": name, "type_line": type_line, "oracle_text": oracle,
            "quantity": quantity, "is_commander": False,
            "color_identity": []}


def test_basic_land_copies_all_count_as_sources(tmp_path, monkeypatch):
    """Eleven Islands are eleven blue sources, not one entry."""
    from manamap.pilot import common
    from manamap.pilot.mana_analysis import analyze

    _deck(tmp_path, monkeypatch, [
        {"name": "Cmd", "type_line": "Legendary Creature", "oracle_text": "",
         "quantity": 1, "is_commander": True,
         "color_identity": ["U"], "mana_cost": "{1}{U}", "cmc": 2.0},
        _land("Island", "({T}: Add {U}.)", quantity=11),
        _land("Reliquary Tower", "{T}: Add {C}.", type_line="Land"),
    ])
    common.clear_memo()
    result = analyze("test-deck")

    assert result["lands"]["total"] == 12      # copies
    assert result["lands"]["entries"] == 2     # distinct cards
    assert result["sources"]["lands"]["U"] == 11
    # The table stays one row per distinct land, carrying its copy count.
    island = [r for r in result["lands"]["list"] if r["name"] == "Island"][0]
    assert island["copies"] == 11


def test_tapped_ratio_uses_copies_in_the_denominator(tmp_path, monkeypatch):
    """A lone tapped land beside 11 basics is 1-in-12, not 1-in-2."""
    from manamap.pilot import common
    from manamap.pilot.mana_analysis import analyze

    _deck(tmp_path, monkeypatch, [
        {"name": "Cmd", "type_line": "Legendary Creature", "oracle_text": "",
         "quantity": 1, "is_commander": True,
         "color_identity": ["U"], "mana_cost": "{1}{U}", "cmc": 2.0},
        _land("Island", "({T}: Add {U}.)", quantity=11),
        _land("Sunken Hollow", "This land enters tapped.\n{T}: Add {U}.",
              type_line="Land"),
    ])
    common.clear_memo()
    result = analyze("test-deck")

    assert result["lands"]["enters_tapped"] == 1
    assert result["lands"]["total"] == 12
    # 1/12 is under the one-in-three budget, so no note fires.
    assert not [n for n in result["notes"] if "enter tapped" in n]


@requires_deck
def test_tracked_mana_analysis_artifacts_are_current():
    """Every committed mana_analysis.json must match a fresh computation.

    The published-18-lands bug shipped because nothing compared the artifact to
    the code that makes it. This is that comparison.
    """
    import json
    from pathlib import Path
    from manamap.pilot.mana_analysis import analyze

    for path in sorted(Path("data/decks").glob("*/mana_analysis.json")):
        tracked = json.loads(path.read_text())
        assert tracked == analyze(path.parent.name), (
            f"{path} is stale — re-run `manamap pilot mana-analysis "
            f"{path.parent.name}`")
