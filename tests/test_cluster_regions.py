"""Unit tests for cluster_regions.py naming and utility functions."""

import json
from collections import Counter

import numpy as np
import pytest

from manamap.config import REGIONS_ABILITY_PATH, REGIONS_DEFAULT_PATH
from manamap.analysis.cluster_regions import (
    assign_parents,
    compute_centroid,
    compute_span,
    name_cluster_ability,
    name_cluster_colortype,
)


# ── Color+Type naming ───────────────────────────────────────────────────


class TestNameClusterColorType:

    def test_dominant_color_and_type(self):
        colors = ["W"] * 50 + ["U"] * 30 + ["B"] * 20
        types = ["Creature"] * 40 + ["Instant"] * 35 + ["Sorcery"] * 25
        label, short = name_cluster_colortype(colors, types)
        assert label == "White Creatures"
        assert short == "White"

    def test_color_only(self):
        colors = ["R"] * 60 + ["G"] * 20 + ["B"] * 20
        types = ["Creature"] * 25 + ["Instant"] * 25 + ["Sorcery"] * 25 + ["Enchantment"] * 25
        label, short = name_cluster_colortype(colors, types)
        assert label == "Red"
        assert short == "Red"

    def test_type_only(self):
        colors = ["W"] * 20 + ["U"] * 20 + ["B"] * 20 + ["R"] * 20 + ["G"] * 20
        types = ["Instant"] * 80 + ["Sorcery"] * 20
        label, short = name_cluster_colortype(colors, types)
        assert label == "Instants"
        assert short == "Instants"

    def test_guild_name(self):
        colors = ["W"] * 30 + ["U"] * 25 + ["B"] * 15 + ["R"] * 15 + ["G"] * 15
        types = ["Creature"] * 25 + ["Instant"] * 25 + ["Sorcery"] * 25 + ["Enchantment"] * 25
        label, short = name_cluster_colortype(colors, types)
        assert label == "Azorius"
        assert short == "Azorius"

    def test_guild_with_dominant_type(self):
        colors = ["W"] * 30 + ["U"] * 25 + ["B"] * 15 + ["R"] * 15 + ["G"] * 15
        types = ["Creature"] * 40 + ["Instant"] * 20 + ["Sorcery"] * 20 + ["Enchantment"] * 20
        label, short = name_cluster_colortype(colors, types)
        assert "Azorius" in label
        assert "Creatures" in label
        assert short == "Azorius"

    def test_empty_cluster(self):
        label, short = name_cluster_colortype([], [])
        assert label == "Unknown"
        assert short == "Unknown"

    def test_multicolor_dominant(self):
        colors = ["Multicolor"] * 50 + ["W"] * 15 + ["U"] * 15 + ["B"] * 10 + ["R"] * 10
        types = ["Creature"] * 60 + ["Enchantment"] * 40
        label, short = name_cluster_colortype(colors, types)
        assert "Gold" in label

    def test_colorless_dominant(self):
        colors = ["Colorless"] * 50 + ["W"] * 15 + ["U"] * 15 + ["B"] * 10 + ["R"] * 10
        types = ["Artifact"] * 50 + ["Creature"] * 50
        label, short = name_cluster_colortype(colors, types)
        assert "Colorless" in label

    def test_fallback_no_dominance(self):
        # No color >=40% and no type >=30%, no guild pair
        colors = ["W"] * 20 + ["U"] * 20 + ["B"] * 20 + ["R"] * 20 + ["Colorless"] * 20
        types = ["Creature"] * 20 + ["Instant"] * 20 + ["Sorcery"] * 20 + ["Enchantment"] * 20 + ["Artifact"] * 20
        label, short = name_cluster_colortype(colors, types)
        assert isinstance(label, str) and len(label) > 0

    def test_type_pluralization(self):
        """Types that don't end in 's' should get pluralized."""
        colors = ["W"] * 50 + ["U"] * 50
        types = ["Enchantment"] * 80 + ["Instant"] * 20
        label, _ = name_cluster_colortype(colors, types)
        assert "Enchantments" in label

    def test_sorcery_pluralization(self):
        """Sorcery should become Sorceries, not Sorcerys."""
        colors = ["R"] * 80 + ["G"] * 20
        types = ["Sorcery"] * 80 + ["Instant"] * 20
        label, _ = name_cluster_colortype(colors, types)
        assert "Sorceries" in label
        assert "Sorcerys" not in label


# ── Ability naming ──────────────────────────────────────────────────────


class TestNameClusterAbility:

    def test_single_dominant_tag(self):
        tags = ["blink,etb"] * 60 + ["draw"] * 40
        global_freq = {"blink": 0.05, "etb": 0.15, "draw": 0.10}
        label, short = name_cluster_ability(tags, global_freq)
        assert "Blink" in label
        assert short == "Blink"

    def test_two_strong_tags(self):
        tags = ["sacrifice,death_trigger"] * 80 + ["draw"] * 20
        global_freq = {"sacrifice": 0.05, "death_trigger": 0.05, "draw": 0.10}
        label, short = name_cluster_ability(tags, global_freq)
        # Should have both tags since they're both overrepresented
        assert "&" in label or "Sacrifice" in label

    def test_tfidf_common_tag_needs_high_freq(self):
        """A globally common tag (etb) needs very high cluster freq to dominate."""
        # etb is very common globally (0.30), so even if 50% of cluster has it,
        # a rarer tag that appears equally often should score higher
        tags = ["etb,blink"] * 50 + ["etb"] * 50
        global_freq = {"etb": 0.30, "blink": 0.02}
        label, short = name_cluster_ability(tags, global_freq)
        assert "Blink" in label  # blink has higher TF-IDF

    def test_rare_tag_below_presence_threshold(self):
        """A tag present in <10% of the cluster should not be used for naming."""
        # 1000 cards: only 50 have "storm" (5%), 950 have nothing
        tags = ["storm"] * 50 + [""] * 950
        global_freq = {"storm": 0.005}
        colors = ["R"] * 1000
        types = ["Creature"] * 1000
        label, short = name_cluster_ability(tags, global_freq, colors, types)
        # Storm at 5% should fail the 10% minimum → falls back to color/type
        assert "Storm" not in label
        assert "Red" in label

    def test_empty_tags_with_color_fallback(self):
        tags = [""] * 50
        global_freq = {"etb": 0.15}
        colors = ["R"] * 50
        types = ["Creature"] * 50
        label, short = name_cluster_ability(tags, global_freq, colors, types)
        assert "Red" in label

    def test_empty_tags_no_fallback(self):
        tags = [""] * 50
        global_freq = {"etb": 0.15}
        label, short = name_cluster_ability(tags, global_freq)
        assert label == "Vanilla"

    def test_nan_tags(self):
        tags = [float("nan")] * 50
        global_freq = {"etb": 0.15}
        label, short = name_cluster_ability(tags, global_freq)
        assert label == "Vanilla"

    def test_empty_cluster(self):
        label, short = name_cluster_ability([], {})
        assert label == "Unknown"

    def test_tag_display_names_used(self):
        tags = ["graveyard_matters"] * 80 + ["draw"] * 20
        global_freq = {"graveyard_matters": 0.03, "draw": 0.10}
        label, short = name_cluster_ability(tags, global_freq)
        assert "Graveyard" in label


# ── Centroid and span computation ───────────────────────────────────────


class TestGeometry:

    def test_centroid_simple(self):
        cx, cy = compute_centroid(np.array([0.0, 10.0]), np.array([0.0, 10.0]))
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_centroid_negative(self):
        cx, cy = compute_centroid(np.array([-5.0, 5.0]), np.array([-3.0, 3.0]))
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_span_width_dominant(self):
        xs = np.array([0.0, 20.0, 10.0])
        ys = np.array([0.0, 5.0, 2.0])
        span, width, height = compute_span(xs, ys)
        assert span == pytest.approx(20.0)
        assert width == pytest.approx(20.0)
        assert height == pytest.approx(5.0)

    def test_span_height_dominant(self):
        xs = np.array([0.0, 3.0, 1.0])
        ys = np.array([-10.0, 10.0, 0.0])
        span, width, height = compute_span(xs, ys)
        assert span == pytest.approx(20.0)
        assert width == pytest.approx(3.0)
        assert height == pytest.approx(20.0)

    def test_span_keeps_both_axes_so_a_filament_is_distinguishable(self):
        """A 20x1 streak and a 20x20 cloud used to serialise identically.

        `span` is max(w, h) and drives the viz's label culling, so it stays —
        but collapsing to it discarded the aspect ratio, which is the only
        thing in the artifact that could tell a road from a region.
        """
        streak_span, streak_w, streak_h = compute_span(
            np.array([0.0, 20.0]), np.array([0.0, 1.0]))
        cloud_span, cloud_w, cloud_h = compute_span(
            np.array([0.0, 20.0]), np.array([0.0, 20.0]))
        assert streak_span == cloud_span == pytest.approx(20.0)
        assert streak_h / streak_w < 0.1
        assert cloud_h / cloud_w == pytest.approx(1.0)


# ── Parent assignment ───────────────────────────────────────────────────


class TestParentAssignment:

    def test_assigns_closest_parent(self):
        l0 = [
            {"id": "l0_0", "cx": 0.0, "cy": 0.0},
            {"id": "l0_1", "cx": 20.0, "cy": 20.0},
        ]
        l1 = [
            {"id": "l1_0", "cx": 1.0, "cy": 1.0},
            {"id": "l1_1", "cx": 19.0, "cy": 19.0},
            {"id": "l1_2", "cx": 10.0, "cy": 10.0},  # equidistant-ish
        ]
        assign_parents(l0, l1)
        assert l1[0]["parent"] == "l0_0"
        assert l1[1]["parent"] == "l0_1"
        assert "parent" in l1[2]  # assigned something

    def test_empty_l0(self):
        l1 = [{"id": "l1_0", "cx": 5.0, "cy": 5.0}]
        assign_parents([], l1)
        assert "parent" not in l1[0]

    def test_empty_l1(self):
        """No assertion at all, once. "should not raise" is a real property —
        assert it, and assert the l0 list came back untouched."""
        l0 = [{"id": "l0_0", "cx": 0.0, "cy": 0.0}]
        assign_parents(l0, [])
        assert l0 == [{"id": "l0_0", "cx": 0.0, "cy": 0.0}]


# ── Output format ───────────────────────────────────────────────────────


REGION_PATHS = [REGIONS_DEFAULT_PATH, REGIONS_ABILITY_PATH]


class TestOutputFormat:
    """THE SCHEMA, ASSERTED AGAINST THE TRACKED ARTIFACTS.

    These four tests used to build a region dict with a local helper and then
    assert that the helper's own keys were present. Zero production code was
    touched; `cluster_regions` could stop emitting `short` entirely and all four
    stayed green. The artifacts are tracked — `regions_default.json` and
    `regions_ability.json` — so the schema can be checked against what actually
    ships, which is the only version anything reads.
    """

    L0_FIELDS = ("id", "level", "label", "short", "cx", "cy", "span", "count",
                 "top_tags")

    def _regions(self, path):
        if not path.exists():
            pytest.skip(f"{path.name} not generated (run `manamap cluster-regions`)")
        return json.loads(path.read_text(encoding="utf-8"))["regions"]

    @pytest.mark.parametrize("path", REGION_PATHS, ids=lambda p: p.stem)
    def test_every_l0_region_carries_the_required_fields(self, path):
        regions = self._regions(path)
        l0 = [r for r in regions if r["level"] == 0]
        assert l0, f"{path.name} has no level-0 regions"
        for region in l0:
            missing = [f for f in self.L0_FIELDS if f not in region]
            assert not missing, f"{region.get('id')} lacks {missing}"

    @pytest.mark.parametrize("path", REGION_PATHS, ids=lambda p: p.stem)
    def test_every_l1_region_names_a_parent_that_exists(self, path):
        regions = self._regions(path)
        ids = {r["id"] for r in regions}
        l1 = [r for r in regions if r["level"] == 1]
        assert l1, f"{path.name} has no level-1 regions"
        for region in l1:
            assert region.get("parent") in ids, (
                f"{region['id']} names parent {region.get('parent')!r}, which "
                f"is not a region in this file")

    @pytest.mark.parametrize("path", REGION_PATHS, ids=lambda p: p.stem)
    def test_the_short_label_is_a_shortening(self, path):
        """`short` is what the map draws when a name will not fit. A `short`
        longer than its own `label` is not a shortening."""
        checked = 0
        for region in self._regions(path):
            if "short" not in region:
                continue
            checked += 1
            assert isinstance(region["short"], str) and region["short"]
            assert len(region["short"]) <= len(region["label"]), region["id"]
        assert checked > 10, f"only {checked} regions carry a short label"

    @pytest.mark.parametrize("path", REGION_PATHS, ids=lambda p: p.stem)
    def test_the_artifact_round_trips_through_json(self, path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert json.loads(json.dumps(doc)) == doc


# ── Membership (the tracked artifacts) ──────────────────────────────────


@pytest.mark.parametrize("path", REGION_PATHS, ids=lambda p: p.stem)
class TestMembership:
    """The per-card region assignment, which used to be computed and discarded.

    Without it nothing in the repo could answer "which region is this card in" —
    the viz could draw a region's name but never its members. These arrays are
    positional over cards.csv row order, so they inherit the index-alignment
    invariant: `membership.l0[i]` describes `cards.csv[i]`.
    """

    def _load(self, path):
        if not path.exists():
            pytest.skip(f"{path.name} not generated (run `manamap cluster-regions`)")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_membership_covers_every_card(self, path):
        doc = self._load(path)
        n = doc["meta"]["card_count"]
        assert len(doc["membership"]["l0"]) == n
        assert len(doc["membership"]["l1"]) == n

    def test_every_label_indexes_a_real_region(self, path):
        doc = self._load(path)
        ids = {r["id"] for r in doc["regions"]}
        for level in (0, 1):
            for cid in set(doc["membership"][f"l{level}"]):
                if cid == -1:
                    continue          # noise is a real answer, not a gap
                assert f"l{level}_{cid}" in ids

    def test_membership_counts_match_the_stored_counts(self, path):
        """The two halves of the artifact must agree, or a drill lights up the
        wrong number of cards from the same file that labelled the region."""
        doc = self._load(path)
        by_id = {r["id"]: r for r in doc["regions"]}
        for level in (0, 1):
            observed = Counter(doc["membership"][f"l{level}"])
            for cid, seen in observed.items():
                if cid == -1:
                    continue
                assert by_id[f"l{level}_{cid}"]["count"] == seen

    def test_regions_record_both_axes(self, path):
        doc = self._load(path)
        for region in doc["regions"]:
            assert "w" in region and "h" in region
            assert region["span"] == pytest.approx(max(region["w"], region["h"]), abs=0.11)
