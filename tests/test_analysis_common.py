"""Tests for shared analysis helpers (analysis/common.py)."""

import numpy as np
import pandas as pd

from manamap.analysis.common import (
    build_name_index,
    color_identity_mask,
    parse_color_identity,
    parse_tag_set,
    top_k_similar,
)


def _df():
    return pd.DataFrame({
        "name": ["Sol Ring", "Lightning Bolt", "Counterspell", "Vindicate"],
        "color_identity": ["", "R", "U", "W, B"],
    })


def _unit_rows(vectors):
    """L2-normalize rows the way the pipeline does at build time."""
    arr = np.array(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# ── parse_color_identity ──


def test_parse_color_identity_comma_space_format():
    assert parse_color_identity("W, B") == {"W", "B"}


def test_parse_color_identity_single():
    assert parse_color_identity("R") == {"R"}


def test_parse_color_identity_empty_is_colorless_not_missing():
    assert parse_color_identity("") == set()
    assert parse_color_identity(None) == set()
    assert parse_color_identity(float("nan")) == set()


# ── build_name_index ──


def test_build_name_index_maps_positions():
    assert build_name_index(_df()) == {
        "Sol Ring": 0, "Lightning Bolt": 1, "Counterspell": 2, "Vindicate": 3,
    }


def test_build_name_index_duplicate_names_last_wins():
    df = pd.DataFrame({"name": ["Garbage Elemental", "Garbage Elemental"]})
    assert build_name_index(df)["Garbage Elemental"] == 1


# ── color_identity_mask ──


def test_color_identity_mask_subset_only():
    mask = color_identity_mask(_df(), {"R"})
    # Sol Ring (colorless) and Lightning Bolt (R) fit; Counterspell and Vindicate don't
    assert list(mask) == [True, True, False, False]


def test_color_identity_mask_colorless_fits_everything():
    mask = color_identity_mask(_df(), set())
    assert list(mask) == [True, False, False, False]


def test_color_identity_mask_multicolor_commander():
    mask = color_identity_mask(_df(), {"W", "B", "U"})
    assert list(mask) == [True, False, True, True]


def test_color_identity_mask_is_case_insensitive():
    assert list(color_identity_mask(_df(), {"r"})) == [True, True, False, False]


# ── top_k_similar ──


def test_top_k_similar_ranks_by_dot_product():
    emb = _unit_rows([[1, 0], [0.9, 0.1], [0.5, 0.5], [0, 1]])
    result = top_k_similar(emb, 0, k=2)
    assert [i for i, _ in result] == [1, 2]
    assert result[0][1] > result[1][1]


def test_top_k_similar_excludes_self():
    emb = _unit_rows([[1, 0], [0, 1]])
    assert [i for i, _ in top_k_similar(emb, 0, k=5)] == [1]


def test_top_k_similar_respects_mask():
    emb = _unit_rows([[1, 0], [0.9, 0.1], [0.5, 0.5], [0, 1]])
    mask = np.array([True, False, True, True])
    # Row 1 is the nearest neighbour but masked out
    assert [i for i, _ in top_k_similar(emb, 0, k=2, mask=mask)] == [2, 3]


def test_top_k_similar_k_larger_than_pool():
    emb = _unit_rows([[1, 0], [0, 1]])
    assert len(top_k_similar(emb, 0, k=99)) == 1


def test_top_k_similar_empty_mask_returns_empty():
    emb = _unit_rows([[1, 0], [0, 1]])
    assert top_k_similar(emb, 0, k=3, mask=np.zeros(2, dtype=bool)) == []


def test_top_k_similar_returns_plain_python_types():
    """Results are JSON-serialized by callers — numpy scalars break that."""
    emb = _unit_rows([[1, 0], [0.9, 0.1]])
    idx, score = top_k_similar(emb, 0, k=1)[0]
    assert isinstance(idx, int)
    assert isinstance(score, float)


# ── parse_tag_set (existing behaviour, guarded) ──


def test_parse_tag_set_basic():
    assert parse_tag_set("etb, draw, ramp") == {"etb", "draw", "ramp"}


def test_parse_tag_set_empty():
    assert parse_tag_set("") == set()
    assert parse_tag_set(None) == set()
