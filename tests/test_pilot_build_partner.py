"""A Partner pair is two commanders, the union of their identities, and a 98.

Built with one commander, sharknado (Shabraz, the Skyshark with Brallin,
Skyshark Rider, 2026-09-10) came out Azorius with the discard half of the deck
missing, because identity came from one name and the validator's allowance for a
second commander had no writer. Now the builder carries a `partner`, and every
reader that wants "the commander" still gets one name.
"""

import json

import pytest

from manamap.pilot import build_deck, validate_build
from conftest import requires_data, requires_deck


def test_the_brief_carries_the_partner(tmp_path, monkeypatch):
    monkeypatch.setattr(build_deck, "DECKS_DIR", tmp_path)
    path, doc = build_deck.scaffold_brief("zz", "Shabraz, the Skyshark",
                                          partner="Brallin, Skyshark Rider")
    assert doc["partner"] == "Brallin, Skyshark Rider"
    assert json.loads(path.read_text())["commander"] == "Shabraz, the Skyshark"
    # No partner, no key: a reader must not find an empty one and treat it as a name.
    _, solo = build_deck.scaffold_brief("zy", "Zur the Enchanter")
    assert "partner" not in solo


def test_both_commanders_render_and_both_are_counted():
    plan = {"commander": "Shabraz, the Skyshark", "partner": "Brallin, Skyshark Rider",
            "slots": [{"name": "Windfall", "role": "draw"}], "land_counts": {"Island": 2}}
    text = build_deck.decklist_text(plan)
    assert text.splitlines()[:2] == ["1 Shabraz, the Skyshark *CMDR*", "1 Brallin, Skyshark Rider *CMDR*"]
    assert validate_build.deck_card_names(plan) == [
        "Shabraz, the Skyshark", "Brallin, Skyshark Rider", "Windfall", "Island", "Island"]


@requires_data
def test_a_partner_without_the_ability_is_refused():
    """Two legendaries is not a pair; the rule that allows the second is Partner."""
    df = build_deck.load_frame()
    shabraz = df[df["name"] == "Shabraz, the Skyshark"].iloc[0].to_dict()
    with pytest.raises(build_deck.BriefError, match="no Partner ability"):
        build_deck.resolve_partner(df, {"partner": "Zur the Enchanter"}, shabraz)
    with pytest.raises(build_deck.BriefError, match="not in cards.csv"):
        build_deck.resolve_partner(df, {"partner": "Nobody, the Nonexistent"}, shabraz)
    brallin = build_deck.resolve_partner(df, {"partner": "Brallin, Skyshark Rider"}, shabraz)
    assert brallin["name"] == "Brallin, Skyshark Rider"
    assert build_deck.resolve_partner(df, {}, shabraz) is None


@requires_deck
def test_sharknado_is_a_jeskai_98_plus_two():
    plan = json.load(open(build_deck.DECKS_DIR / "sharknado" / "build_plan.json"))
    assert plan["partner"] == "Brallin, Skyshark Rider"
    assert plan["color_identity"] == ["R", "U", "W"]
    names = validate_build.deck_card_names(plan)
    assert len(names) == 100 and names.count("Brallin, Skyshark Rider") == 1
    assert "Brallin, Skyshark Rider" not in {s["name"] for s in plan["slots"]}
    assert sum(v for k, v in plan["role_budget"].items() if k != "lands") == len(plan["slots"])


def test_a_land_in_the_library_goes_to_the_mana_base_first():
    """The pilot typed Steam Vents; the chooser must start from it, not from
    whatever covers the most colours alphabetically."""
    from manamap.pilot import manabase
    vents = {"name": "Steam Vents", "type_line": "Land — Island Mountain",
             "oracle_text": "As Steam Vents enters, you may pay 2 life. If you don't, it enters tapped."}
    junk = {"name": "Aaa Rainbow", "type_line": "Land", "oracle_text": "{T}: Add one mana of any color."}
    spells = [{"name": "Lightning Bolt", "mana_cost": "{R}", "type_line": "Instant"},
              {"name": "Counterspell", "mana_cost": "{U}{U}", "type_line": "Instant"}]
    lands, _ = manabase.build(spells, [junk], 3, basics={
        "U": {"name": "Island", "type_line": "Basic Land — Island", "oracle_text": ""},
        "R": {"name": "Mountain", "type_line": "Basic Land — Mountain", "oracle_text": ""}}, keep=[vents])
    names = [l["name"] for l in lands]
    assert names[0] == "Steam Vents" and len(names) == 3
    assert names.count("Steam Vents") == 1, "a kept land must not be picked twice"
