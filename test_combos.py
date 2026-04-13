"""Tests for combo data processing (process_combos.py)."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from process_combos import (
    build_combo_graph,
    extract_card_names,
    extract_color_identity,
    extract_produces,
    load_known_cards,
)


# ── Fixtures ──


def make_combo(card_names, identity="", produces=None):
    """Helper to build a combo variant dict matching Commander Spellbook format."""
    uses = [{"card": {"name": name}} for name in card_names]
    prods = [{"feature": {"name": p}} for p in (produces or [])]
    return {"uses": uses, "identity": identity, "produces": prods}


# ── extract_card_names ──


def test_extract_card_names_basic():
    combo = make_combo(["Sol Ring", "Dramatic Reversal", "Isochron Scepter"])
    assert extract_card_names(combo) == ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"]


def test_extract_card_names_empty():
    assert extract_card_names({}) == []
    assert extract_card_names({"uses": []}) == []


def test_extract_card_names_missing_card_field():
    combo = {"uses": [{"card": {}}, {"card": {"name": "Lightning Bolt"}}]}
    assert extract_card_names(combo) == ["Lightning Bolt"]


def test_extract_card_names_strips_whitespace():
    combo = make_combo(["  Sol Ring  ", "Lightning Bolt"])
    names = extract_card_names(combo)
    assert names == ["Sol Ring", "Lightning Bolt"]


# ── extract_color_identity ──


def test_extract_color_identity():
    assert extract_color_identity({"identity": "wub"}) == "WUB"
    assert extract_color_identity({"identity": "r"}) == "R"
    assert extract_color_identity({}) == ""


# ── extract_produces ──


def test_extract_produces():
    combo = make_combo(["A", "B"], produces=["Infinite mana", "Infinite storm count"])
    assert extract_produces(combo) == ["Infinite mana", "Infinite storm count"]


def test_extract_produces_empty():
    assert extract_produces({}) == []
    assert extract_produces({"produces": []}) == []


# ── load_known_cards ──


def test_load_known_cards():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame({"name": ["Sol Ring", "Lightning Bolt", "Counterspell"]})
        df.to_csv(f.name, index=False)
        cards = load_known_cards(Path(f.name))
    assert cards == {"Sol Ring", "Lightning Bolt", "Counterspell"}


# ── build_combo_graph ──


def test_build_combo_graph_basic():
    known = {"Sol Ring", "Dramatic Reversal", "Isochron Scepter"}
    combos = [
        make_combo(
            ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"],
            identity="u",
            produces=["Infinite colorless mana"],
        )
    ]
    partners, combo_list = build_combo_graph(combos, known)

    # Each card should partner with the other two
    assert set(partners["Sol Ring"]) == {"Dramatic Reversal", "Isochron Scepter"}
    assert set(partners["Dramatic Reversal"]) == {"Sol Ring", "Isochron Scepter"}
    assert set(partners["Isochron Scepter"]) == {"Sol Ring", "Dramatic Reversal"}

    assert len(combo_list) == 1
    assert combo_list[0]["cards"] == ["Sol Ring", "Dramatic Reversal", "Isochron Scepter"]
    assert combo_list[0]["ci"] == "U"
    assert combo_list[0]["produces"] == ["Infinite colorless mana"]


def test_build_combo_graph_filters_unknown_cards():
    known = {"Sol Ring", "Lightning Bolt"}
    combos = [
        make_combo(["Sol Ring", "Unknown Card That Doesnt Exist"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)
    assert len(partners) == 0
    assert len(combo_list) == 0


def test_build_combo_graph_skips_single_card_combos():
    known = {"Sol Ring"}
    combos = [make_combo(["Sol Ring"])]
    partners, combo_list = build_combo_graph(combos, known)
    assert len(partners) == 0
    assert len(combo_list) == 0


def test_build_combo_graph_multiple_combos():
    known = {"A", "B", "C", "D"}
    combos = [
        make_combo(["A", "B"], produces=["Effect 1"]),
        make_combo(["C", "D"], produces=["Effect 2"]),
        make_combo(["A", "C"], produces=["Effect 3"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)

    assert len(combo_list) == 3
    # A partners with B and C
    assert set(partners["A"]) == {"B", "C"}
    # B only partners with A
    assert set(partners["B"]) == {"A"}
    # C partners with A and D
    assert set(partners["C"]) == {"A", "D"}


def test_build_combo_graph_deduplicates_partners():
    known = {"A", "B", "C"}
    combos = [
        make_combo(["A", "B"], produces=["Effect 1"]),
        make_combo(["A", "B", "C"], produces=["Effect 2"]),
    ]
    partners, combo_list = build_combo_graph(combos, known)

    # A-B partnership appears in both combos but should be deduplicated
    assert "B" in partners["A"]
    assert partners["A"].count("B") == 1  # sorted list, each entry once


def test_build_combo_graph_partners_are_sorted():
    known = {"Z", "M", "A"}
    combos = [make_combo(["Z", "M", "A"])]
    partners, _ = build_combo_graph(combos, known)

    assert partners["Z"] == ["A", "M"]
    assert partners["M"] == ["A", "Z"]
    assert partners["A"] == ["M", "Z"]


def test_combo_graph_json_serializable():
    """Ensure the output can be serialized to JSON."""
    known = {"Sol Ring", "Dramatic Reversal"}
    combos = [make_combo(["Sol Ring", "Dramatic Reversal"], identity="u", produces=["Infinite mana"])]
    partners, combo_list = build_combo_graph(combos, known)

    graph = {"partners": partners, "combos": combo_list}
    output = json.dumps(graph, separators=(",", ":"))
    parsed = json.loads(output)
    assert "partners" in parsed
    assert "combos" in parsed
