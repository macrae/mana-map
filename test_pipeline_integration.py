"""Integration tests: validate that pipeline outputs are consistent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import (
    ABILITY_EMBEDDINGS_BIN_PATH,
    ABILITY_EMBEDDINGS_PATH,
    ABILITY_MODEL_PATH,
    ABILITY_PROJECTION_PATH,
    CARD_FEATURES_PATH,
    CARD_METADATA_PATH,
    COMBO_GRAPH_PATH,
    COLOR_VECTORS_PATH,
    EMBEDDINGS_BIN_PATH,
    EMBEDDINGS_PATH,
    MECHANICAL_TAG_DIM,
    MECHANICAL_TAG_NAMES,
    MECHANICAL_TAGS_PATH,
    MODEL_PATH,
    OBSOLESCENCE_INDEX_PATH,
    OUTPUT_CSV_PATH,
    PROJECTION_PATH,
    SYNERGY_GRAPH_PATH,
    TEXT_EMBEDDINGS_PATH,
)


# ── Skip if pipeline hasn't been run ──


def requires_file(path):
    return pytest.mark.skipif(
        not Path(path).exists(),
        reason=f"{path} not found — run pipeline first"
    )


# ── Card count consistency ──


@requires_file(OUTPUT_CSV_PATH)
class TestCardCountConsistency:
    """All pipeline outputs should have the same card count."""

    @pytest.fixture(autouse=True)
    def load_card_count(self):
        self.df = pd.read_csv(OUTPUT_CSV_PATH)
        self.n = len(self.df)
        assert self.n > 0, "cards.csv is empty"

    @requires_file(TEXT_EMBEDDINGS_PATH)
    def test_text_embeddings_count(self):
        emb = np.load(TEXT_EMBEDDINGS_PATH)
        assert emb.shape[0] == self.n

    @requires_file(CARD_FEATURES_PATH)
    def test_card_features_count(self):
        features = dict(np.load(CARD_FEATURES_PATH))
        assert features["supertype"].shape[0] == self.n
        assert features["keywords"].shape[0] == self.n

    @requires_file(MECHANICAL_TAGS_PATH)
    def test_mechanical_tags_in_features(self):
        features = dict(np.load(CARD_FEATURES_PATH))
        assert "mechanical_tags" in features
        assert features["mechanical_tags"].shape == (self.n, MECHANICAL_TAG_DIM)

    @requires_file(MECHANICAL_TAGS_PATH)
    def test_mechanical_tags_file(self):
        tags = np.load(MECHANICAL_TAGS_PATH)
        assert tags.shape[0] == self.n
        assert tags.shape[1] == MECHANICAL_TAG_DIM

    @requires_file(COLOR_VECTORS_PATH)
    def test_color_vectors_count(self):
        vecs = np.load(COLOR_VECTORS_PATH)
        assert vecs.shape == (self.n, 5)

    @requires_file(EMBEDDINGS_PATH)
    def test_embeddings_count(self):
        emb = np.load(EMBEDDINGS_PATH)
        assert emb.shape == (self.n, 128)

    @requires_file(CARD_METADATA_PATH)
    def test_metadata_count(self):
        meta = pd.read_csv(CARD_METADATA_PATH)
        assert len(meta) == self.n

    @requires_file(PROJECTION_PATH)
    def test_projection_count(self):
        with open(PROJECTION_PATH, "r") as f:
            records = json.load(f)
        assert len(records) == self.n

    @requires_file(EMBEDDINGS_BIN_PATH)
    def test_embeddings_bin_size(self):
        expected = self.n * 128 * 4
        actual = EMBEDDINGS_BIN_PATH.stat().st_size
        assert actual == expected


# ── Mechanical tags quality ──


@requires_file(MECHANICAL_TAGS_PATH)
class TestMechanicalTagQuality:
    """Mechanical tags should cover >= 70% of non-land cards."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        self.df = pd.read_csv(OUTPUT_CSV_PATH)

    def test_mechanical_tags_column_exists(self):
        assert "mechanical_tags" in self.df.columns

    def test_tag_coverage_non_land(self):
        non_land = self.df[self.df["supertype"] != "Land"]
        tagged = non_land["mechanical_tags"].fillna("").str.len() > 0
        coverage = tagged.sum() / len(non_land)
        assert coverage >= 0.70, f"Tag coverage {coverage:.2%} < 70%"


# ── Ability model outputs ──


@requires_file(ABILITY_MODEL_PATH)
class TestAbilityModel:

    @requires_file(ABILITY_EMBEDDINGS_PATH)
    def test_ability_embeddings_shape(self):
        emb = np.load(ABILITY_EMBEDDINGS_PATH)
        default_emb = np.load(EMBEDDINGS_PATH)
        assert emb.shape == default_emb.shape

    @requires_file(ABILITY_PROJECTION_PATH)
    def test_ability_projection_exists(self):
        with open(ABILITY_PROJECTION_PATH, "r") as f:
            records = json.load(f)
        assert len(records) > 0

    @requires_file(ABILITY_EMBEDDINGS_BIN_PATH)
    def test_ability_bin_size(self):
        default_size = EMBEDDINGS_BIN_PATH.stat().st_size
        ability_size = ABILITY_EMBEDDINGS_BIN_PATH.stat().st_size
        assert ability_size == default_size


# ── Synergy graph ──


@requires_file(SYNERGY_GRAPH_PATH)
class TestSynergyGraph:

    @pytest.fixture(autouse=True)
    def load_graph(self):
        with open(SYNERGY_GRAPH_PATH, "r") as f:
            self.graph = json.load(f)

    def test_graph_not_empty(self):
        assert len(self.graph) > 0

    def test_entries_have_required_fields(self):
        for card_name, partners in list(self.graph.items())[:10]:
            assert isinstance(partners, list)
            for p in partners:
                assert "partner" in p
                assert "score" in p
                assert "synergies" in p
                assert isinstance(p["synergies"], list)
                assert p["score"] > 0

    def test_max_10_partners(self):
        for card_name, partners in self.graph.items():
            assert len(partners) <= 10


# ── Obsolescence index ──


@requires_file(OBSOLESCENCE_INDEX_PATH)
class TestObsolescenceIndex:

    @pytest.fixture(autouse=True)
    def load_index(self):
        with open(OBSOLESCENCE_INDEX_PATH, "r") as f:
            self.index = json.load(f)

    def test_index_not_empty(self):
        assert len(self.index) > 0

    def test_entries_have_required_fields(self):
        for card_name, data in list(self.index.items())[:10]:
            assert "obsoleted_by" in data
            for rep in data["obsoleted_by"]:
                assert "name" in rep
                assert "advantages" in rep
                assert isinstance(rep["advantages"], list)
                assert len(rep["advantages"]) > 0

    def test_entries_have_similarity(self):
        """Obsolescence entries should include similarity scores."""
        for card_name, data in list(self.index.items())[:10]:
            for rep in data["obsoleted_by"]:
                assert "similarity" in rep, f"Missing similarity for {card_name} -> {rep['name']}"
                assert 0.0 <= rep["similarity"] <= 1.0

    def test_count_bounds(self):
        """With similarity gate + tiered thresholds, expect 5,000-16,000 flagged cards."""
        count = len(self.index)
        assert count >= 5000, f"Too few flagged cards: {count}"
        assert count <= 16000, f"Too many flagged cards: {count}"


# ── Combo graph ──


@requires_file(COMBO_GRAPH_PATH)
class TestComboGraph:

    def test_combo_graph_structure(self):
        with open(COMBO_GRAPH_PATH, "r") as f:
            graph = json.load(f)
        assert "partners" in graph
        assert "combos" in graph
        assert len(graph["partners"]) > 0


# ── All existing tests still pass ──


@requires_file(OUTPUT_CSV_PATH)
def test_csv_has_base_required_columns():
    """Check columns that exist in the original pipeline."""
    df = pd.read_csv(OUTPUT_CSV_PATH)
    required = [
        "oracle_id", "name", "layout", "type_line", "supertype",
        "mana_cost", "cmc", "colors", "color_identity", "primary_color",
        "oracle_text", "keywords", "embedding_text",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


@requires_file(MECHANICAL_TAGS_PATH)
def test_csv_has_mechanical_tags_column():
    """Check mechanical_tags column exists (requires re-running extract.py)."""
    df = pd.read_csv(OUTPUT_CSV_PATH)
    assert "mechanical_tags" in df.columns
