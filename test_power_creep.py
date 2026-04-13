"""Tests for power creep / obsolescence detection (power_creep.py)."""

import pandas as pd
import pytest

from power_creep import (
    color_requirement_subset,
    find_strictly_better,
    parse_color_requirement,
    parse_stat,
    parse_tags_set,
)


# ── parse_stat ──


def test_parse_stat_number():
    assert parse_stat("3") == 3.0
    assert parse_stat("0") == 0.0


def test_parse_stat_star():
    assert parse_stat("*") is None


def test_parse_stat_none():
    assert parse_stat(None) is None
    assert parse_stat("") is None


# ── parse_color_requirement ──


def test_parse_color_requirement():
    assert parse_color_requirement("{2}{W}{W}") == {"W": 2}
    assert parse_color_requirement("{U}{U}{U}") == {"U": 3}
    assert parse_color_requirement("{1}{B}") == {"B": 1}


def test_parse_color_requirement_colorless():
    assert parse_color_requirement("{3}") == {}
    assert parse_color_requirement("{C}") == {}


def test_parse_color_requirement_hybrid():
    result = parse_color_requirement("{W/U}")
    assert result.get("W", 0) == 0.5
    assert result.get("U", 0) == 0.5


def test_parse_color_requirement_empty():
    assert parse_color_requirement("") == {}
    assert parse_color_requirement(None) == {}


# ── color_requirement_subset ──


def test_color_subset_same():
    assert color_requirement_subset("{1}{W}", "{1}{W}") is True


def test_color_subset_easier():
    assert color_requirement_subset("{2}{W}{W}", "{1}{W}") is True


def test_color_subset_harder():
    assert color_requirement_subset("{1}{W}", "{1}{W}{W}") is False


def test_color_subset_different_color():
    assert color_requirement_subset("{1}{W}", "{1}{U}") is False


# ── parse_tags_set ──


def test_parse_tags_set():
    assert parse_tags_set("etb, draw, counterspell") == {"etb", "draw", "counterspell"}
    assert parse_tags_set("") == set()
    assert parse_tags_set(None) == set()


# ── find_strictly_better ──


def make_card(name, supertype, cmc, mana_cost, tags, power=None, toughness=None, released_at="2020-01-01"):
    return {
        "name": name,
        "supertype": supertype,
        "cmc": cmc,
        "mana_cost": mana_cost,
        "mechanical_tags": tags,
        "power": power,
        "toughness": toughness,
        "released_at": released_at,
        "edhrec_rank": None,
    }


def test_strictly_better_lower_cmc():
    """Cancel -> Counterspell: same effect, lower cost."""
    df = pd.DataFrame([
        make_card("Cancel", "Instant", 3, "{1}{U}{U}", "counterspell", released_at="2007-01-01"),
        make_card("Counterspell", "Instant", 2, "{U}{U}", "counterspell", released_at="2021-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Cancel" in result
    replacements = [r["name"] for r in result["Cancel"]["obsoleted_by"]]
    assert "Counterspell" in replacements


def test_strictly_better_creature_stats():
    """A 2/2 for 2 is obsoleted by a 3/3 for 2 with same abilities."""
    df = pd.DataFrame([
        make_card("Old Bear", "Creature", 2, "{1}{G}", "evasion_flying", power="2", toughness="2", released_at="2000-01-01"),
        make_card("New Bear", "Creature", 2, "{1}{G}", "evasion_flying", power="3", toughness="3", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Old Bear" in result
    assert result["Old Bear"]["obsoleted_by"][0]["name"] == "New Bear"
    advantages = result["Old Bear"]["obsoleted_by"][0]["advantages"]
    assert "Better Power" in advantages
    assert "Better Toughness" in advantages


def test_not_strictly_better_different_supertype():
    """Different supertypes are not comparable."""
    df = pd.DataFrame([
        make_card("Draw Instant", "Instant", 3, "{2}{U}", "draw", released_at="2000-01-01"),
        make_card("Draw Creature", "Creature", 2, "{1}{U}", "draw", power="2", toughness="2", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Draw Instant" not in result


def test_not_strictly_better_harder_color():
    """Card with harder color requirement is not strictly better."""
    df = pd.DataFrame([
        make_card("Easy Card", "Instant", 2, "{1}{U}", "draw", released_at="2000-01-01"),
        make_card("Hard Card", "Instant", 2, "{U}{U}", "draw", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Easy Card" not in result


def test_not_strictly_better_fewer_tags():
    """Card with fewer abilities is not strictly better."""
    df = pd.DataFrame([
        make_card("Multi Effect", "Instant", 3, "{2}{U}", "draw, counterspell", released_at="2000-01-01"),
        make_card("Single Effect", "Instant", 2, "{1}{U}", "draw", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Multi Effect" not in result


def test_not_strictly_better_older():
    """Older cards don't obsolete newer cards."""
    df = pd.DataFrame([
        make_card("New Card", "Instant", 3, "{2}{U}", "draw", released_at="2022-01-01"),
        make_card("Old Card", "Instant", 2, "{1}{U}", "draw", released_at="2000-01-01"),
    ])
    result = find_strictly_better(df)
    assert "New Card" not in result


def test_strictly_better_additional_ability():
    """Card with additional abilities (superset tags) is strictly better."""
    df = pd.DataFrame([
        make_card("Base Card", "Creature", 3, "{2}{W}", "etb", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Better Card", "Creature", 3, "{2}{W}", "etb, draw", power="2", toughness="2", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Base Card" in result
    advantages = result["Base Card"]["obsoleted_by"][0]["advantages"]
    assert any("Additional" in a for a in advantages)


def test_vanilla_cards_excluded():
    """Vanilla cards (no tags) should not appear in results."""
    df = pd.DataFrame([
        make_card("Vanilla A", "Creature", 3, "{2}{G}", "", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Vanilla B", "Creature", 2, "{1}{G}", "", power="3", toughness="3", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Vanilla A" not in result


def test_identical_cards_not_strictly_better():
    """Identical card from later set should not be strictly better (needs advantage)."""
    df = pd.DataFrame([
        make_card("Same Card", "Instant", 2, "{1}{U}", "draw", released_at="2000-01-01"),
        make_card("Same Card Reprint", "Instant", 2, "{1}{U}", "draw", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Same Card" not in result


def test_lands_excluded():
    """Land cards should not be compared."""
    df = pd.DataFrame([
        make_card("Old Land", "Land", 0, "", "ramp, tap_ability", released_at="2000-01-01"),
        make_card("New Land", "Land", 0, "", "ramp, tap_ability, lifegain", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Old Land" not in result


def test_max_5_replacements():
    """Results capped at 5 replacements per card."""
    cards = [make_card("Old Draw", "Instant", 5, "{4}{U}", "draw", released_at="2000-01-01")]
    for i in range(10):
        cards.append(make_card(
            f"Better Draw {i}", "Instant", 4, "{3}{U}", "draw",
            released_at=f"202{i}-01-01"
        ))
    df = pd.DataFrame(cards)
    result = find_strictly_better(df)
    if "Old Draw" in result:
        assert len(result["Old Draw"]["obsoleted_by"]) <= 5


# ── parse_stat modifier rejection ──


def test_parse_stat_modifier_plus():
    """Augment card stats like '+2' should return None."""
    assert parse_stat("+2") is None
    assert parse_stat("+1") is None


def test_parse_stat_modifier_minus():
    """Negative modifier stats like '-1' should return None."""
    assert parse_stat("-1") is None
    assert parse_stat("-3") is None


# ── Empty mana cost exclusion ──


def test_empty_mana_cost_excluded():
    """Cards with empty mana cost should not obsolete or be obsoleted."""
    df = pd.DataFrame([
        make_card("Real Card", "Creature", 2, "{1}{G}", "etb", power="2", toughness="2", released_at="2000-01-01"),
        make_card("No Cost Card", "Creature", 0, "", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Real Card" not in result
