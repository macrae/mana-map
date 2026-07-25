"""Tests for combo data processing (process_combos.py)."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from manamap.ingest.process_combos import (
    bracket_summary,
    build_card_index,
    build_combo_graph,
    extract_bracket,
    extract_card_names,
    extract_color_identity,
    extract_produces,
    load_known_cards,
)


# ── Fixtures ──


def make_combo(card_names, identity="", produces=None, bracket_tag=None,
               mana_value_needed=None, popularity=None):
    """Helper to build a combo variant dict matching Commander Spellbook format."""
    uses = [{"card": {"name": name}} for name in card_names]
    prods = [{"feature": {"name": p}} for p in (produces or [])]
    return {
        "uses": uses,
        "identity": identity,
        "produces": prods,
        "bracketTag": bracket_tag,
        "manaValueNeeded": mana_value_needed,
        "popularity": popularity,
    }


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


# ── extract_bracket ──


@pytest.mark.parametrize("tag,expected", [
    ("E", 1), ("C", 2), ("O", 2), ("P", 3), ("S", 3), ("R", 4),
])
def test_extract_bracket_maps_spellbook_letters(tag, expected):
    bracket, banned = extract_bracket({"bracketTag": tag})
    assert bracket == expected
    assert banned is False


def test_extract_bracket_flags_banned():
    bracket, banned = extract_bracket({"bracketTag": "B"})
    assert bracket is None
    assert banned is True


def test_extract_bracket_unknown_letter_is_none_not_one():
    """An unrecognized tag must not read as bracket 1 — that under-reports a floor."""
    bracket, banned = extract_bracket({"bracketTag": "X"})
    assert bracket is None
    assert banned is False


def test_extract_bracket_missing_tag():
    assert extract_bracket({}) == (None, False)


# ── enriched combo records ──


def test_build_combo_graph_carries_bracket_fields():
    known = {"A", "B"}
    combos = [make_combo(["A", "B"], bracket_tag="R", mana_value_needed=4, popularity=1200)]
    _, combo_list = build_combo_graph(combos, known)

    assert combo_list[0]["bracket"] == 4
    assert combo_list[0]["mana_value_needed"] == 4
    assert combo_list[0]["popularity"] == 1200
    assert "banned" not in combo_list[0]


def test_build_combo_graph_keeps_banned_combos_flagged():
    """Format-agnostic by design: banned combos are flagged, never dropped."""
    known = {"A", "B"}
    combos = [make_combo(["A", "B"], bracket_tag="B")]
    partners, combo_list = build_combo_graph(combos, known)

    assert len(combo_list) == 1
    assert combo_list[0]["banned"] is True
    assert combo_list[0]["bracket"] is None
    assert set(partners["A"]) == {"B"}


# ── build_card_index ──


def test_build_card_index_maps_names_to_combo_indices():
    known = {"A", "B", "C"}
    combos = [make_combo(["A", "B"]), make_combo(["B", "C"])]
    _, combo_list = build_combo_graph(combos, known)
    index = build_card_index(combo_list)

    assert index["A"] == [0]
    assert index["B"] == [0, 1]
    assert index["C"] == [1]


def test_build_card_index_deduplicates_repeated_names():
    index = build_card_index([{"cards": ["A", "A", "B"]}])
    assert index["A"] == [0]


def test_build_card_index_empty():
    assert build_card_index([]) == {}


# ── bracket_summary ──


def test_bracket_summary_counts_by_bracket_and_banned():
    known = {"A", "B", "C", "D"}
    combos = [
        make_combo(["A", "B"], bracket_tag="E"),
        make_combo(["C", "D"], bracket_tag="E"),
        make_combo(["A", "C"], bracket_tag="R"),
        make_combo(["B", "D"], bracket_tag="B"),
    ]
    _, combo_list = build_combo_graph(combos, known)

    assert bracket_summary(combo_list) == {"1": 2, "4": 1, "banned": 1}
