"""Tests for synergy detection (synergy.py)."""

import json
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data

from manamap.analysis.synergy import (
    build_card_tags,
    build_playability,
    build_synergy_graph,
    build_tag_index,
)


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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
def test_synergy_labels(mock_combo):
    """Synergy labels should describe the rule that matched."""
    df = make_df([
        ("Blinker", "blink"),
        ("ETB Creature", "etb"),
    ])
    graph = build_synergy_graph(df)

    synergies = graph["Blinker"][0]["synergies"]
    assert "Blink + ETB" in synergies


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={"Card A": ["Card B"]})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
def test_no_self_synergy(mock_combo):
    """A card should not synergize with itself."""
    df = make_df([
        ("Self Card", "blink, etb"),
    ])
    graph = build_synergy_graph(df)

    if "Self Card" in graph:
        partner_names = [p["partner"] for p in graph["Self Card"]]
        assert "Self Card" not in partner_names


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
def test_no_tags_no_synergies(mock_combo):
    """Cards with no tags should have no synergies."""
    df = make_df([
        ("Vanilla Creature", ""),
        ("ETB Card", "etb"),
    ])
    graph = build_synergy_graph(df)

    assert "Vanilla Creature" not in graph


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


@patch("manamap.analysis.synergy.load_combo_partners", return_value={})
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


# ── playability ranking ─────────────────────────────────────────────────


class TestBuildPlayability:
    def test_absent_column_is_tolerated(self):
        """The unit fixtures carry only names and tags, and real callers must not be
        able to break by omitting a column."""
        df = make_df([("A", "blink"), ("B", "etb")])
        play = build_playability(df, ["A", "B"], {"A": 0, "B": 1})
        assert list(play) == [0.0, 0.0]

    def test_popular_beats_obscure(self):
        df = pd.DataFrame({"name": ["Top", "Mid", "Obscure", "Unranked"],
                           "mechanical_tags": ["blink"] * 4,
                           "edhrec_rank": [1, 5000, 30000, None]})
        names = ["Top", "Mid", "Obscure", "Unranked"]
        play = build_playability(df, names, {n: i for i, n in enumerate(names)})
        assert play[0] > play[1] > play[2] > play[3]
        assert play[3] == 0.0, "an unranked card must not outrank a ranked one"

    def test_stays_below_one(self):
        """The guarantee the whole ranking rests on: playability can never outrank a
        full score step, so a 2-rule match always beats a 1-rule match."""
        df = pd.DataFrame({"name": ["Top"], "mechanical_tags": ["blink"],
                           "edhrec_rank": [1]})
        assert build_playability(df, ["Top"], {"Top": 0})[0] < 1.0


@requires_data
def test_synergy_partners_are_playable():
    """The ranking gate.

    Partners used to be tie-broken by embedding similarity, which for a
    *complementary* relation is backwards — it surfaces cards resembling the anchor
    rather than cards that play with it. Measured on the shipped graph, partners were
    in the top 2,000 most-played 7.0% of the time against a 6.3% corpus baseline:
    barely above chance. Skullclamp recommended Playable Delusionary Hydra.

    A score tier is usually large (median 70 cards, p90 1,529), so the tiebreak decides
    almost everything. After ranking by playability: median rank 10,713 -> 1,472 and
    top-2,000 share 7.0% -> 60.2%.

    This exists so that cannot silently regress, the way the embedding collapse did.
    """
    import json

    import numpy as np

    from manamap.config import OUTPUT_CSV_PATH, SYNERGY_GRAPH_PATH

    cards = pd.read_csv(OUTPUT_CSV_PATH, low_memory=False)
    rank = dict(zip(cards["name"], pd.to_numeric(cards["edhrec_rank"], errors="coerce")))
    with open(SYNERGY_GRAPH_PATH, encoding="utf-8") as fh:
        graph = json.load(fh)

    ranks = [rank[p["partner"]]
             for entry in list(graph.values())[:4000]
             for p in entry
             if rank.get(p["partner"]) == rank.get(p["partner"])]
    ranks = np.array([r for r in ranks if r == r])

    share = (ranks < 2000).mean()
    assert share > 0.40, (
        f"only {share:.1%} of synergy partners are top-2,000 cards — the playability "
        f"tiebreak has regressed (it was 7.0% with the similarity tiebreak, 60.2% after)"
    )
    assert np.median(ranks) < 4000, f"median partner rank {np.median(ranks):.0f} is too deep"
