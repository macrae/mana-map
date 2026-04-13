"""Tests for synergy detection (synergy.py)."""

import json
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from synergy import build_card_tags, build_synergy_graph, build_tag_index


# ── Fixtures ──


def make_df(cards):
    """Build a DataFrame from list of (name, mechanical_tags) tuples."""
    return pd.DataFrame(cards, columns=["name", "mechanical_tags"])


# ── build_tag_index ──


def test_build_tag_index_basic():
    df = make_df([
        ("Card A", "etb, draw"),
        ("Card B", "sacrifice, death_trigger"),
        ("Card C", "etb"),
    ])
    idx = build_tag_index(df)
    assert "Card A" in idx["etb"]
    assert "Card C" in idx["etb"]
    assert "Card A" in idx["draw"]
    assert "Card B" in idx["sacrifice"]


def test_build_tag_index_empty_tags():
    df = make_df([("Card A", ""), ("Card B", None)])
    idx = build_tag_index(df)
    assert all(len(v) == 0 for v in idx.values())


# ── build_card_tags ──


def test_build_card_tags():
    df = make_df([("Card A", "etb, draw"), ("Card B", "")])
    ct = build_card_tags(df)
    assert ct["Card A"] == {"etb", "draw"}
    assert ct["Card B"] == set()


# ── build_synergy_graph ──


@patch("synergy.load_combo_partners", return_value={})
def test_blink_etb_synergy(mock_combo):
    """Cards with blink should synergize with ETB cards."""
    df = make_df([
        ("Restoration Angel", "blink"),
        ("Mulldrifter", "etb, draw"),
        ("Thragtusk", "etb, lifegain"),
        ("Lightning Bolt", "removal"),
    ])
    graph = build_synergy_graph(df)

    # Restoration Angel should have Mulldrifter and Thragtusk as synergy partners
    assert "Restoration Angel" in graph
    partner_names = [p["partner"] for p in graph["Restoration Angel"]]
    assert "Mulldrifter" in partner_names
    assert "Thragtusk" in partner_names
    assert "Lightning Bolt" not in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_sac_death_synergy(mock_combo):
    """Sacrifice outlets should synergize with death trigger cards."""
    df = make_df([
        ("Viscera Seer", "sacrifice, tap_ability"),
        ("Blood Artist", "death_trigger"),
        ("Zulaport Cutthroat", "death_trigger"),
    ])
    graph = build_synergy_graph(df)

    assert "Viscera Seer" in graph
    partner_names = [p["partner"] for p in graph["Viscera Seer"]]
    assert "Blood Artist" in partner_names
    assert "Zulaport Cutthroat" in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_synergy_is_bidirectional(mock_combo):
    """Both sides of a synergy rule should find each other."""
    df = make_df([
        ("Blink Card", "blink"),
        ("ETB Card", "etb"),
    ])
    graph = build_synergy_graph(df)

    assert "Blink Card" in graph
    assert "ETB Card" in graph
    assert graph["Blink Card"][0]["partner"] == "ETB Card"
    assert graph["ETB Card"][0]["partner"] == "Blink Card"


@patch("synergy.load_combo_partners", return_value={})
def test_synergy_labels(mock_combo):
    """Synergy labels should describe the rule that matched."""
    df = make_df([
        ("Blinker", "blink"),
        ("ETB Creature", "etb"),
    ])
    graph = build_synergy_graph(df)

    synergies = graph["Blinker"][0]["synergies"]
    assert "Blink + ETB" in synergies


@patch("synergy.load_combo_partners", return_value={})
def test_multiple_synergy_rules(mock_combo):
    """A card matching multiple rules with a partner should have higher score."""
    df = make_df([
        ("Multi Card", "blink, sacrifice"),
        ("ETB Death Card", "etb, death_trigger"),
        ("ETB Only Card", "etb"),
    ])
    graph = build_synergy_graph(df)

    assert "Multi Card" in graph
    partners = {p["partner"]: p for p in graph["Multi Card"]}
    # ETB Death Card matches both blink+etb AND sac+death = score 2
    assert partners["ETB Death Card"]["score"] >= 2
    # ETB Only Card matches only blink+etb = score 1
    assert partners["ETB Only Card"]["score"] >= 1
    # Higher score should be ranked first
    assert graph["Multi Card"][0]["partner"] == "ETB Death Card"


@patch("synergy.load_combo_partners", return_value={"Card A": ["Card B"]})
def test_excludes_combo_partners(mock_combo):
    """Known combo partners should be excluded from synergy results."""
    df = make_df([
        ("Card A", "blink"),
        ("Card B", "etb"),
        ("Card C", "etb"),
    ])
    graph = build_synergy_graph(df)

    assert "Card A" in graph
    partner_names = [p["partner"] for p in graph["Card A"]]
    assert "Card B" not in partner_names  # excluded: combo partner
    assert "Card C" in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_no_self_synergy(mock_combo):
    """A card should not synergize with itself."""
    df = make_df([
        ("Self Card", "blink, etb"),
    ])
    graph = build_synergy_graph(df)

    if "Self Card" in graph:
        partner_names = [p["partner"] for p in graph["Self Card"]]
        assert "Self Card" not in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_top_10_limit(mock_combo):
    """Synergy results should be capped at 10 per card."""
    cards = [("Blinker", "blink")]
    # Add 15 ETB cards
    for i in range(15):
        cards.append((f"ETB Card {i}", "etb"))
    df = make_df(cards)
    graph = build_synergy_graph(df)

    assert "Blinker" in graph
    assert len(graph["Blinker"]) <= 10


@patch("synergy.load_combo_partners", return_value={})
def test_no_tags_no_synergies(mock_combo):
    """Cards with no tags should have no synergies."""
    df = make_df([
        ("Vanilla Creature", ""),
        ("ETB Card", "etb"),
    ])
    graph = build_synergy_graph(df)

    assert "Vanilla Creature" not in graph


@patch("synergy.load_combo_partners", return_value={})
def test_tokens_anthem_synergy(mock_combo):
    """Token generators should synergize with anthem effects."""
    df = make_df([
        ("Krenko", "tokens"),
        ("Goblin Chieftain", "anthem"),
    ])
    graph = build_synergy_graph(df)

    assert "Krenko" in graph
    partner_names = [p["partner"] for p in graph["Krenko"]]
    assert "Goblin Chieftain" in partner_names

    # Check label
    synergies = graph["Krenko"][0]["synergies"]
    assert any("Tokens" in s for s in synergies)


@patch("synergy.load_combo_partners", return_value={})
def test_synergy_graph_json_serializable(mock_combo):
    """Ensure the output can be serialized to JSON."""
    df = make_df([
        ("Blinker", "blink"),
        ("ETB Card", "etb"),
    ])
    graph = build_synergy_graph(df)
    output = json.dumps(graph, separators=(",", ":"))
    parsed = json.loads(output)
    assert "Blinker" in parsed


# ── New rule coverage tests ──


@patch("synergy.load_combo_partners", return_value={})
def test_no_duplicate_rule_scoring(mock_combo):
    """Tokens+anthem should score exactly 1, not 2 (no duplicate rule)."""
    df = make_df([
        ("Token Maker", "tokens"),
        ("Anthem Lord", "anthem"),
    ])
    graph = build_synergy_graph(df)

    assert "Token Maker" in graph
    partner = graph["Token Maker"][0]
    assert partner["partner"] == "Anthem Lord"
    assert partner["score"] == 1
    assert len(partner["synergies"]) == 1


@patch("synergy.load_combo_partners", return_value={})
def test_bounce_etb_synergy(mock_combo):
    """Bounce cards should find ETB partners."""
    df = make_df([
        ("Man-o'-War", "bounce, etb"),
        ("Mulldrifter", "etb, draw"),
        ("Vanilla Bear", ""),
    ])
    graph = build_synergy_graph(df)

    assert "Man-o'-War" in graph
    partner_names = [p["partner"] for p in graph["Man-o'-War"]]
    assert "Mulldrifter" in partner_names
    assert "Vanilla Bear" not in partner_names
    # Check label
    mulldrifter_entry = [p for p in graph["Man-o'-War"] if p["partner"] == "Mulldrifter"][0]
    assert "Bounce + ETB" in mulldrifter_entry["synergies"]


@patch("synergy.load_combo_partners", return_value={})
def test_evasion_damage_trigger_synergy(mock_combo):
    """Flying cards should find damage_trigger partners."""
    df = make_df([
        ("Flyer", "evasion_flying"),
        ("Damage Dealer", "damage_trigger"),
        ("Vanilla Card", "draw"),
    ])
    graph = build_synergy_graph(df)

    assert "Flyer" in graph
    partner_names = [p["partner"] for p in graph["Flyer"]]
    assert "Damage Dealer" in partner_names
    assert "Vanilla Card" not in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_equipment_attack_trigger_synergy(mock_combo):
    """Equipment should find attack_trigger partners."""
    df = make_df([
        ("Sword of X", "equipment"),
        ("Attacker", "attack_trigger"),
        ("Lifegainer", "lifegain"),
    ])
    graph = build_synergy_graph(df)

    assert "Sword of X" in graph
    partner_names = [p["partner"] for p in graph["Sword of X"]]
    assert "Attacker" in partner_names
    assert "Lifegainer" not in partner_names


@patch("synergy.load_combo_partners", return_value={})
def test_aura_protection_synergy(mock_combo):
    """Aura cards should find protection partners."""
    df = make_df([
        ("Rancor", "aura, evasion_trample"),
        ("Hexproof Guy", "protection"),
        ("Random Card", "draw"),
    ])
    graph = build_synergy_graph(df)

    assert "Rancor" in graph
    partner_names = [p["partner"] for p in graph["Rancor"]]
    assert "Hexproof Guy" in partner_names
    assert "Random Card" not in partner_names
