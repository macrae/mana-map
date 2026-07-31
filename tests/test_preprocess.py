"""Unit tests for preprocess.py feature encoding functions."""

import numpy as np
import pandas as pd
import pytest

from manamap.ingest.preprocess import (
    assert_vocab_fits,
    build_color_identity_features,
    build_color_identity_vocab,
    build_top_keywords,
    build_vocab_index,
    encode_categorical,
    encode_keywords_multihot,
    normalize_cmc,
    normalize_edhrec_rank,
    normalize_power_toughness,
    parse_color_vectors,
    parse_mana_pips,
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


class TestKeywordVocabExcludesEmptyString:
    """The dead-slot bug: half the corpus has no keywords at all.

    `"".split(",")` is `[""]`, so without a guard the empty string outranks Flying
    five to one, takes index 0, and `encode_keywords_multihot` never sets that
    column because it skips falsy strings — a permanently zero feature and 49 real
    keywords in a nominal 50.
    """

    def test_empty_string_is_not_a_keyword(self):
        df = pd.DataFrame({"keywords": ["", "", "", "", "Flying"]})
        assert build_top_keywords(df, top_n=5) == ["Flying"]

    def test_empty_rows_do_not_displace_real_keywords(self):
        df = pd.DataFrame({"keywords": [""] * 20 + ["Flying", "Trample"]})
        top = build_top_keywords(df, top_n=2)
        assert set(top) == {"Flying", "Trample"}

    def test_no_dead_column_survives_encoding(self):
        df = pd.DataFrame({"keywords": ["", "Flying", "Trample"]})
        top = build_top_keywords(df, top_n=2)
        assert encode_keywords_multihot(df, top).sum(axis=0).min() > 0


class TestEdhrecRankIsFixedScale:
    """A per-run min-max made the same card's feature differ between runs."""

    def test_same_rank_same_value_regardless_of_corpus(self):
        small = normalize_edhrec_rank(pd.Series([1.0, 100.0]))
        large = normalize_edhrec_rank(pd.Series([1.0, 100.0, 31000.0]))
        assert small[0] == pytest.approx(large[0])
        assert small[1] == pytest.approx(large[1])

    def test_clips_rather_than_exceeding_one(self):
        assert normalize_edhrec_rank(pd.Series([10**9])).max() <= 1.0


class TestPowerToughness:
    def test_creature_stats_scaled(self):
        df = pd.DataFrame({"power": ["3"], "toughness": ["3"]})
        result = normalize_power_toughness(df)
        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(0.2)  # 3/15
        assert result[0, 2] == 1.0

    def test_noncreature_has_no_stats_flag(self):
        df = pd.DataFrame({"power": [None], "toughness": [None]})
        assert normalize_power_toughness(df)[0, 2] == 0.0

    def test_variable_power_is_not_a_noncreature(self):
        """`*` is a creature whose power is variable, not the absence of a creature."""
        df = pd.DataFrame({"power": ["*"], "toughness": ["*"]})
        result = normalize_power_toughness(df)
        assert result[0, 2] == 1.0, "has_stats must stay set for */1+* creatures"
        assert result[0, 0] == 0.0

    def test_giant_stats_clip(self):
        df = pd.DataFrame({"power": ["99"], "toughness": ["99"]})
        assert normalize_power_toughness(df)[0, 0] == pytest.approx(1.0)


class TestManaPips:
    """Only scalar CMC used to reach the model, so {U}{U} and {2} were identical."""

    def test_double_blue_differs_from_generic_two(self):
        df = pd.DataFrame({"mana_cost": ["{U}{U}", "{2}"]})
        pips = parse_mana_pips(df)
        assert not np.array_equal(pips[0], pips[1])
        assert pips[0][1] == pytest.approx(0.5)   # 2 blue / scale 4
        assert pips[1][5] == pytest.approx(0.5)   # 2 generic / scale 4

    def test_hybrid_counts_for_both_colours(self):
        pips = parse_mana_pips(pd.DataFrame({"mana_cost": ["{W/U}"]}))
        assert pips[0][0] > 0 and pips[0][1] > 0

    def test_mixed_cost(self):
        pips = parse_mana_pips(pd.DataFrame({"mana_cost": ["{2}{W}{W}"]}))
        assert pips[0][0] == pytest.approx(0.5)
        assert pips[0][5] == pytest.approx(0.5)

    def test_no_cost(self):
        assert parse_mana_pips(pd.DataFrame({"mana_cost": [""]})).sum() == 0.0


class TestColorIdentityFeatures:
    """Multi-hot so subset structure exists: the string categorical made
    'B, G' and 'B, G, R' unrelated indices."""

    def test_golgari_is_a_subset_of_jund(self):
        df = pd.DataFrame({"color_identity": ["B, G", "B, G, R"]})
        feats = build_color_identity_features(df)
        golgari, jund = feats[0][:5], feats[1][:5]
        assert np.all(jund >= golgari), "Jund must contain Golgari's colours"

    def test_colour_count(self):
        df = pd.DataFrame({"color_identity": ["W, U, B, R, G", ""]})
        feats = build_color_identity_features(df)
        assert feats[0][5] == pytest.approx(1.0)
        assert feats[1][5] == pytest.approx(0.0)


class TestVocabCapacityGuard:
    """Fail here, not with an index error hours into training."""

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="config caps it"):
            assert_vocab_fits("layout", ["a", "b", "c"], capacity=3)

    def test_exactly_at_capacity_is_allowed(self):
        assert_vocab_fits("layout", ["a", "b"], capacity=3)
