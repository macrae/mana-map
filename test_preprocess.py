"""Unit tests for preprocess.py feature encoding functions."""

import numpy as np
import pandas as pd
import pytest

from preprocess import (
    build_color_identity_vocab,
    build_top_keywords,
    build_vocab_index,
    encode_categorical,
    encode_keywords_multihot,
    normalize_cmc,
    normalize_edhrec_rank,
    parse_color_vectors,
)


class TestBuildVocabIndex:
    def test_supertype_roundtrip(self):
        vocab = ["Artifact", "Creature", "Enchantment", "Land"]
        index = build_vocab_index(vocab)
        assert index["Artifact"] == 0
        assert index["Creature"] == 1
        assert index["Land"] == 3
        assert len(index) == 4

    def test_rarity_roundtrip(self):
        vocab = ["common", "mythic", "rare", "uncommon"]
        index = build_vocab_index(vocab)
        for v in vocab:
            assert v in index
        assert len(index) == 4


class TestEncodeCategorical:
    def test_known_values(self):
        vocab = {"Creature": 0, "Land": 1, "Instant": 2}
        series = pd.Series(["Creature", "Land", "Instant", "Creature"])
        result = encode_categorical(series, vocab)
        np.testing.assert_array_equal(result, [0, 1, 2, 0])

    def test_unknown_bucket(self):
        vocab = {"Creature": 0, "Land": 1}
        series = pd.Series(["Creature", "Planeswalker"])
        result = encode_categorical(series, vocab)
        assert result[0] == 0
        assert result[1] == 2  # len(vocab) = unknown bucket

    def test_nan_handling(self):
        vocab = {"Creature": 0}
        series = pd.Series(["Creature", None]).fillna("Unknown")
        result = encode_categorical(series, vocab)
        assert result[0] == 0
        assert result[1] == 1  # "Unknown" not in vocab → unknown bucket


class TestBuildColorIdentityVocab:
    def test_colorless_at_index_0(self):
        df = pd.DataFrame({"color_identity": ["W", "U", "", "W, U"]})
        vocab = build_color_identity_vocab(df)
        assert vocab[0] == ""

    def test_all_uniques_present(self):
        df = pd.DataFrame({"color_identity": ["W", "U", "", "W, U", "B"]})
        vocab = build_color_identity_vocab(df)
        assert set(vocab) == {"", "B", "U", "W", "W, U"}

    def test_nan_treated_as_colorless(self):
        df = pd.DataFrame({"color_identity": [None, "R", "G"]})
        vocab = build_color_identity_vocab(df)
        assert "" in vocab


class TestNormalizeCmc:
    def test_zero(self):
        result = normalize_cmc(pd.Series([0.0]))
        assert result[0] == pytest.approx(0.0)

    def test_sixteen(self):
        result = normalize_cmc(pd.Series([16.0]))
        assert result[0] == pytest.approx(1.0)

    def test_gleemax_clipped(self):
        result = normalize_cmc(pd.Series([1_000_000.0]))
        assert result[0] == pytest.approx(1.0)

    def test_negative_clip(self):
        result = normalize_cmc(pd.Series([-1.0]))
        assert result[0] == pytest.approx(0.0)


class TestNormalizeEdhrecRank:
    def test_nan_fill(self):
        series = pd.Series([100.0, np.nan, 200.0])
        result = normalize_edhrec_rank(series)
        assert not np.isnan(result).any()

    def test_output_range(self):
        series = pd.Series([1.0, 100.0, 10000.0, 50000.0])
        result = normalize_edhrec_rank(series)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestParseColorVectors:
    def test_mono_red(self):
        df = pd.DataFrame({"color_identity": ["R"]})
        result = parse_color_vectors(df)
        # WUBRG: R is index 3
        expected = np.array([[0, 0, 0, 1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_azorius(self):
        df = pd.DataFrame({"color_identity": ["W, U"]})
        result = parse_color_vectors(df)
        expected = np.array([[1, 1, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_colorless(self):
        df = pd.DataFrame({"color_identity": [""]})
        result = parse_color_vectors(df)
        expected = np.zeros((1, 5), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_five_color(self):
        df = pd.DataFrame({"color_identity": ["W, U, B, R, G"]})
        result = parse_color_vectors(df)
        expected = np.ones((1, 5), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_shape(self):
        df = pd.DataFrame({"color_identity": ["R", "G", ""]})
        result = parse_color_vectors(df)
        assert result.shape == (3, 5)


class TestKeywordEncoding:
    def test_top_n(self):
        df = pd.DataFrame({"keywords": ["Flying", "Flying", "Trample", "Haste, Trample"]})
        top = build_top_keywords(df, top_n=2)
        # Flying appears 2x, Trample 2x, Haste 1x
        assert len(top) == 2
        assert "Flying" in top
        assert "Trample" in top

    def test_multihot(self):
        top = ["Flying", "Trample", "Haste"]
        df = pd.DataFrame({"keywords": ["Flying, Trample", "Haste"]})
        result = encode_keywords_multihot(df, top)
        np.testing.assert_array_equal(result[0], [1, 1, 0])
        np.testing.assert_array_equal(result[1], [0, 0, 1])

    def test_unknown_keywords_ignored(self):
        top = ["Flying"]
        df = pd.DataFrame({"keywords": ["Flying, Menace"]})
        result = encode_keywords_multihot(df, top)
        np.testing.assert_array_equal(result[0], [1])

    def test_nan_keywords(self):
        top = ["Flying"]
        df = pd.DataFrame({"keywords": [None, "Flying"]})
        result = encode_keywords_multihot(df, top)
        np.testing.assert_array_equal(result[0], [0])
        np.testing.assert_array_equal(result[1], [1])
