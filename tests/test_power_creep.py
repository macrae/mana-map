"""Tests for power creep / obsolescence detection (power_creep.py)."""

import pytest

from conftest import requires_data

import numpy as np
import pandas as pd

from manamap.analysis.power_creep import (
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
        make_card("Cancel", "Instant", 3, "{1}{U}{U}", "counterspell, draw", released_at="2007-01-01"),
        make_card("Counterspell", "Instant", 2, "{U}{U}", "counterspell, draw", released_at="2021-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Cancel" in result
    replacements = [r["name"] for r in result["Cancel"]["compare_with"]]
    assert "Counterspell" in replacements


def test_strictly_better_creature_stats():
    """A 2/2 for 2 is obsoleted by a 3/3 for 2 with same abilities."""
    df = pd.DataFrame([
        make_card("Old Bear", "Creature", 2, "{1}{G}", "evasion_flying, etb", power="2", toughness="2", released_at="2000-01-01"),
        make_card("New Bear", "Creature", 2, "{1}{G}", "evasion_flying, etb", power="3", toughness="3", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Old Bear" in result
    assert result["Old Bear"]["compare_with"][0]["name"] == "New Bear"
    advantages = result["Old Bear"]["compare_with"][0]["gains"]
    assert "more power" in advantages
    assert "more toughness" in advantages


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
        make_card("Base Card", "Creature", 3, "{2}{W}", "etb, lifegain", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Better Card", "Creature", 3, "{2}{W}", "etb, lifegain, draw", power="2", toughness="2", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Base Card" in result
    # The extra tag is named PLAINLY now, not wrapped in "Additional: …" — and
    # only a tag whose valence is a GAIN lands in `gains` at all. `draw` is one;
    # `discard` would land in `costs` and dock the score.
    gains = result["Base Card"]["compare_with"][0]["gains"]
    assert "draw" in gains, gains


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
    cards = [make_card("Old Draw", "Instant", 5, "{4}{U}", "draw, counterspell", released_at="2000-01-01")]
    for i in range(10):
        cards.append(make_card(
            f"Better Draw {i}", "Instant", 4, "{3}{U}", "draw, counterspell",
            released_at=f"202{i}-01-01"
        ))
    df = pd.DataFrame(cards)
    result = find_strictly_better(df)
    if "Old Draw" in result:
        assert len(result["Old Draw"]["compare_with"]) <= 5


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
        make_card("Real Card", "Creature", 2, "{1}{G}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("No Cost Card", "Creature", 0, "", "etb, draw, tokens", power="3", toughness="3", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df)
    assert "Real Card" not in result


# ── Similarity gate tests ──


def test_similarity_gate_blocks_dissimilar():
    """Low similarity should prevent obsolescence detection."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    # Create fake embeddings pointing in very different directions
    embs = np.zeros((2, 128), dtype=np.float32)
    embs[0, 0] = 1.0  # Card A points along dim 0
    embs[1, 1] = 1.0  # Card B points along dim 1 (orthogonal -> sim = 0)
    result = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.75)
    assert "Card A" not in result


def test_similarity_gate_allows_similar():
    """High similarity should allow obsolescence detection."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    # Create fake embeddings pointing in the same direction (sim = 1.0)
    embs = np.zeros((2, 128), dtype=np.float32)
    embs[0, 0] = 1.0
    embs[1, 0] = 1.0
    result = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.75)
    assert "Card A" in result


def test_min_tags_excludes_single_tag():
    """Single-tag cards should be excluded with min_tags=2."""
    df = pd.DataFrame([
        make_card("One Tag", "Instant", 3, "{2}{U}", "draw", released_at="2000-01-01"),
        make_card("Also One Tag", "Instant", 2, "{1}{U}", "draw", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df, min_tags=2)
    assert "One Tag" not in result


def test_min_tags_allows_multi_tag():
    """Multi-tag cards should be included with min_tags=2."""
    df = pd.DataFrame([
        make_card("Two Tags", "Instant", 3, "{2}{U}", "draw, counterspell", released_at="2000-01-01"),
        make_card("Better Two", "Instant", 2, "{1}{U}", "draw, counterspell", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df, min_tags=2)
    assert "Two Tags" in result


def test_fallback_without_embeddings():
    """Without embeddings, similarity gate should be skipped (still works)."""
    df = pd.DataFrame([
        make_card("Old Card", "Instant", 3, "{2}{U}", "draw, counterspell", released_at="2000-01-01"),
        make_card("New Card", "Instant", 2, "{1}{U}", "draw, counterspell", released_at="2022-01-01"),
    ])
    result = find_strictly_better(df, ability_embeddings=None)
    assert "Old Card" in result


def test_similarity_score_in_output():
    """Output records should have similarity field when embeddings provided."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    embs = np.ones((2, 128), dtype=np.float32)  # identical -> sim = 1.0
    result = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.5)
    assert "Card A" in result
    assert "similarity" in result["Card A"]["compare_with"][0]


def test_sort_by_similarity():
    """Results should be sorted by similarity descending."""
    df = pd.DataFrame([
        make_card("Base", "Instant", 4, "{3}{U}", "draw, counterspell", released_at="2000-01-01"),
        make_card("Close Match", "Instant", 3, "{2}{U}", "draw, counterspell", released_at="2022-01-01"),
        make_card("Far Match", "Instant", 3, "{2}{U}", "draw, counterspell", released_at="2023-01-01"),
    ])
    embs = np.zeros((3, 128), dtype=np.float32)
    embs[0, 0] = 1.0   # Base
    embs[1, 0] = 0.95; embs[1, 1] = 0.31  # Close Match (high sim)
    embs[2, 0] = 0.8; embs[2, 1] = 0.6    # Far Match (lower sim)
    result = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.5)
    if "Base" in result and len(result["Base"]["compare_with"]) >= 2:
        sims = [r["similarity"] for r in result["Base"]["compare_with"]]
        assert sims == sorted(sims, reverse=True)


def test_configurable_threshold():
    """Custom threshold should be respected."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    # Embeddings with moderate similarity (~0.5)
    embs = np.zeros((2, 128), dtype=np.float32)
    embs[0, 0] = 1.0
    embs[1, 0] = 0.5; embs[1, 1] = 0.866  # cos sim ≈ 0.5

    # High threshold should block
    result_high = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.9)
    assert "Card A" not in result_high

    # Low threshold should allow
    result_low = find_strictly_better(df, ability_embeddings=embs, similarity_threshold=0.3)
    assert "Card A" in result_low


def test_single_tag_uses_higher_threshold():
    """1-tag cards should use single_tag_threshold, not similarity_threshold."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    # Similarity = 0.95 — above base threshold (0.75) but below single-tag threshold (0.98)
    embs = np.zeros((2, 128), dtype=np.float32)
    embs[0, 0] = 1.0
    embs[1, 0] = 0.95; embs[1, 1] = 0.3122  # cos sim ≈ 0.95

    # With default tiered thresholds: 1-tag needs 0.98, so 0.95 is blocked
    result = find_strictly_better(df, ability_embeddings=embs,
                                  similarity_threshold=0.75, single_tag_threshold=0.98)
    assert "Card A" not in result

    # Lower single-tag threshold allows it
    result2 = find_strictly_better(df, ability_embeddings=embs,
                                   similarity_threshold=0.75, single_tag_threshold=0.90)
    assert "Card A" in result2


def test_multi_tag_uses_base_threshold():
    """2+-tag cards should use similarity_threshold, not the stricter single_tag_threshold."""
    df = pd.DataFrame([
        make_card("Card A", "Creature", 3, "{2}{U}", "etb, draw", power="2", toughness="2", released_at="2000-01-01"),
        make_card("Card B", "Creature", 2, "{1}{U}", "etb, draw", power="3", toughness="3", released_at="2022-01-01"),
    ])
    # Similarity = 0.85 — above base threshold (0.75) but below single-tag threshold (0.98)
    embs = np.zeros((2, 128), dtype=np.float32)
    embs[0, 0] = 1.0
    embs[1, 0] = 0.85; embs[1, 1] = 0.5268  # cos sim ≈ 0.85

    # 2-tag card uses base threshold (0.75), so 0.85 passes
    result = find_strictly_better(df, ability_embeddings=embs,
                                  similarity_threshold=0.75, single_tag_threshold=0.98)
    assert "Card A" in result


# ── The 2026-08 repair: read the card, and publish a degree ──────────────
#
# The audit that prompted it, over all 22,753 pairs the index then published:
#   15.5%  a COST reported as an advantage (`Additional: discard`)
#   22.9%  the replacement ADDS a restriction the original lacks
#   29.9%  of ability-pairs, the replacement's ability costs MORE
#    8.2%  the replacement is commander-illegal
#   30.8%  played LESS than the card it claimed to outclass
#   82.0%  share a real functional role — the RETRIEVAL half was fine
#
# The module read tags, cost, stats and a date and never opened `oracle_text`.

from manamap.analysis import power_creep as pc


def _rec(name, text="", cmc=0.0, mana_cost="", power=None, toughness=None,
         rank=None, tribes=None, restrictions=None, activation=None):
    return {"name": name, "text": text, "cmc": cmc, "mana_cost": mana_cost,
            "power": power, "toughness": toughness, "edhrec_rank": rank,
            "effective_cost": pc.effective_cost(cmc, mana_cost),
            "tribes": tribes if tribes is not None else set(),
            "restrictions": (restrictions if restrictions is not None
                             else pc.restrictions(text)),
            "activation": activation}


def test_phyrexian_mana_is_read_as_the_discount_it_is():
    """Dismember is a one-mana spell, and the index compared it on mana value 3.

    `preprocess.parse_mana_pips` counts `{B/P}` as a full black pip and says so
    deliberately — that is right for CASTABILITY, where the pip is payable with
    mana. It is wrong for a COST COMPARISON. Same one-concept-two-questions
    split as `cast_pips` vs `manabase.count_pips`.
    """
    assert pc.effective_cost(3.0, "{1}{B/P}{B/P}") == 1.0
    assert pc.effective_cost(2.0, "{1}{B}") == 2.0
    # NaN is a float and NaN is TRUTHY, so `mana_cost or ""` does not catch a
    # missing cell — the trap every other pandas read in this module guards.
    assert pc.effective_cost(2.0, float("nan")) == 2.0


def test_a_tribal_gate_is_seen_where_the_tag_vocabulary_is_blind():
    """THE FAILURE THE CAVEAT IS ACTUALLY ABOUT.

    `MECHANICAL_TAGS["death_trigger"]` is `when .* dies`, and the `.*` sits
    exactly where the subject noun lives — so "whenever a Goblin you control
    dies" and "whenever another creature you control dies" produce byte-identical
    tag sets. The gate was the substring the regex threw away.
    """
    types = {"Goblin", "Vampire", "Elf"}
    assert pc.tribal_gates(
        "Whenever a Goblin you control dies, each opponent loses 1 life.",
        types) == {"Goblin"}
    assert pc.tribal_gates(
        "Whenever another creature you control dies, each opponent loses 1 life.",
        types) == set()
    # THE HARD CASE, and it must NOT fire: the Vampire is who gets the counter,
    # not what the trigger is gated on.
    assert pc.tribal_gates(
        "Whenever this creature or another creature dies, put a +1/+1 counter "
        "on each Vampire you control.", types) == set()


def test_a_tag_has_a_sign():
    """`Additional: discard` was published as a reason to make the swap, so
    "discard a card for hexproof" read as an upgrade over unconditional
    hexproof. `context` is the honest default and is reported as a DIFFERENCE."""
    gains, costs, context = pc.valence({"draw", "discard", "tokens"})
    assert gains == ["draw"] and costs == ["discard"] and context == ["tokens"]


def test_the_strength_ranks_the_documented_failures_below_a_clean_pair():
    """CALIBRATION, against the four cases `pool_facts` documents.

    The weights are not free parameters: each is anchored here, so one cannot be
    tuned until these stop meaning anything.
    """
    plain = _rec("A", "Whenever another creature you control dies, drain 1.",
                 cmc=3.0, rank=500)
    clean = _rec("B", "Whenever another creature you control dies, drain 1.",
                 cmc=2.0, rank=400)
    good = pc.obsolescence_strength(plain, clean, ["costs 1 less"], [], None, None)

    tribal = _rec("C", "Whenever a Goblin you control dies, drain 1.",
                  cmc=2.0, rank=400, tribes={"Goblin"})
    narrow = pc.obsolescence_strength(plain, tribal, ["costs 1 less"], [], None, None)

    costly = _rec("D", "Whenever another creature dies, drain 1.",
                  cmc=2.0, rank=400)
    charged = pc.obsolescence_strength(plain, costly, ["costs 1 less"],
                                       ["discard"], None, None)

    paid = pc.obsolescence_strength(_rec("E", cmc=2.0, rank=500, activation=0),
                                    _rec("F", cmc=2.0, rank=400, activation=2),
                                    ["more toughness"], [], 0, 2)

    assert good > narrow, "a tribal gate must cost more than it gains"
    assert good > charged, "a card that charges you must rank below one that does not"
    assert good > paid, "a free ability replaced by a paid one is not an upgrade"
    assert narrow < 0.35, f"a tribal narrowing scored {narrow}"
    for score in (good, narrow, charged, paid):
        assert 0.0 <= score <= 1.0


@requires_data
def test_the_index_publishes_a_degree_and_it_separates():
    """The score must rank the pairs with a detectable problem BELOW the rest,
    or it is decoration. Measured at the repair: 0.271 against 0.630."""
    from manamap.analysis import eval_obsolescence
    got = eval_obsolescence.collect()
    if got is None:
        pytest.skip("requires the card corpus")
    strength = got["strength"]
    if strength["separation"] is None:
        pytest.skip("no scored pairs on this index")
    assert strength["separation"] > 0.15, (
        f"the score separates the bad pairs from the good by only "
        f"{strength['separation']} — it is not carrying its own weight")
    # And retrieval must not have been traded away for precision: the embedding
    # was already finding cards that do the same job 82% of the time.
    assert got["role_agreement"]["share"] > 0.70, got["role_agreement"]
