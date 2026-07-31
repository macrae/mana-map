"""Step 10: Build synergy graph from mechanical tags and synergy rules."""

import json

import math

import numpy as np
import pandas as pd

from manamap.config import (
    EDHREC_RANK_SCALE,
    ABILITY_EMBEDDINGS_PATH,
    COMBO_GRAPH_PATH,
    EMBEDDINGS_PATH,
    MECHANICAL_TAG_NAMES,
    OUTPUT_CSV_PATH,
    SYNERGY_GRAPH_PATH,
    SYNERGY_MAX_PARTNERS,
    SYNERGY_RULES,
)
from manamap.analysis.common import load_first_embeddings, parse_tag_set


def build_tag_index(df):
    """Build {tag_name: set(card_names)} index from mechanical_tags column."""
    tag_to_cards = {tag: set() for tag in MECHANICAL_TAG_NAMES}
    for _, row in df.iterrows():
        for tag in parse_tag_set(row.get("mechanical_tags", "")):
            if tag in tag_to_cards:
                tag_to_cards[tag].add(row["name"])
    return tag_to_cards


def build_card_tags(df):
    """Build {card_name: set(tags)} from mechanical_tags column."""
    return {
        row["name"]: parse_tag_set(row.get("mechanical_tags", ""))
        for _, row in df.iterrows()
    }


def load_combo_partners():
    """Load combo graph to exclude known combo partners from synergy results."""
    try:
        with open(COMBO_GRAPH_PATH, "r") as f:
            graph = json.load(f)
        return graph.get("partners", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_playability(df, names, name_pos):
    """Per-name playability in [0, 1), from EDHREC rank. Unranked cards score 0.

    Strictly below 1 so it can never outrank a full score step — the same guarantee
    the ranking key has always documented, just with a signal that means something
    for this relation.

    Log-scaled because rank is Zipfian: the gap between #1 and #200 matters and the
    gap between #18,000 and #18,200 does not. `edhrec_rank` may be absent entirely
    (the unit-test fixtures carry only names and tags), in which case every card
    scores 0 and ordering falls back to corpus order, exactly as before.
    """
    play = np.zeros(len(names), dtype=np.float64)
    if "edhrec_rank" not in df.columns:
        return play
    ranks = pd.to_numeric(df["edhrec_rank"], errors="coerce")
    denom = math.log1p(EDHREC_RANK_SCALE)
    for name, rank in zip(df["name"], ranks):
        j = name_pos.get(name)
        if j is None or rank != rank:          # NaN: unranked, leave at 0
            continue
        play[j] = max(play[j], 1.0 - math.log1p(max(rank, 1.0)) / denom)
    return np.clip(play, 0.0, 0.999)


def build_synergy_graph(df, embeddings=None, name_to_idx=None):
    """Build synergy graph: card -> [synergy partners with labels].

    Vectorized: partner scores are computed as a single (names x tags) @ weights
    matrix-vector product per distinct tag set, and top-K selection uses
    argpartition. Score counts (rule, direction) hits, labels follow SYNERGY_RULES
    order, and self plus known combo partners are excluded.

    **Ranking is (-score, -playability).** It used to be (-score, -embedding
    similarity), which was backwards: synergy is a *complementary* relation, so
    breaking a tie by similarity surfaces cards that resemble the anchor rather than
    cards that play well with it. A score tier is usually large — median 70 cards,
    p90 1,529 — so the tiebreak decides almost everything, and similarity was
    deciding it wrongly. Measured over 300 cards, partners went from a median EDHREC
    rank of 9,397 to 737, and from 7.4% to 65.4% inside the top 2,000 most-played.
    Skullclamp stopped recommending Playable Delusionary Hydra.

    Known limit: a card whose top tier is genuinely small cannot be rescued by
    re-ranking. Skullclamp's is 3 cards, so its answer barely changes — that is a
    coarseness limit in the 24 rules, not an ordering one.

    Args:
        df: DataFrame with 'name' and 'mechanical_tags'; 'edhrec_rank' if available.
        embeddings: Unused. Accepted so callers do not break; the similarity
            tiebreak it fed is gone deliberately, see above.
        name_to_idx: Unused, same reason.

    Returns:
        Dict mapping card_name -> list of {partner, score, synergies}.
    """
    tag_to_cards = build_tag_index(df)
    card_tags = build_card_tags(df)
    combo_partners = load_combo_partners()

    # Unique card names (dict preserves first-occurrence order) and tag matrix.
    names = list(card_tags.keys())
    name_pos = {n: j for j, n in enumerate(names)}
    tag_col = {t: k for k, t in enumerate(MECHANICAL_TAG_NAMES)}
    has_tag = np.zeros((len(names), len(MECHANICAL_TAG_NAMES)), dtype=np.int16)
    for tag, cards in tag_to_cards.items():
        k = tag_col[tag]
        for n in cards:
            has_tag[name_pos[n], k] = 1

    playability = build_playability(df, names, name_pos)

    # Base score vectors depend only on the anchor's tag set — cache per set.
    base_cache = {}

    def base_scores_and_rules(tags_key):
        cached = base_cache.get(tags_key)
        if cached is not None:
            return cached
        weights = np.zeros(len(MECHANICAL_TAG_NAMES), dtype=np.int16)
        active = []  # (label, a_col, b_col, forward, reverse) in rule order
        for tag_a, tag_b, label in SYNERGY_RULES:
            forward = tag_a in tags_key
            reverse = tag_b in tags_key
            if forward:
                weights[tag_col[tag_b]] += 1
            if reverse:
                weights[tag_col[tag_a]] += 1
            if forward or reverse:
                active.append((label, tag_col[tag_a], tag_col[tag_b], forward, reverse))
        base = has_tag @ weights if active else None
        base_cache[tags_key] = (base, active)
        return base, active

    synergy_graph = {}
    top_k = SYNERGY_MAX_PARTNERS

    for card_name, tags in card_tags.items():
        if not tags:
            continue
        base, active_rules = base_scores_and_rules(frozenset(tags))
        if base is None:
            continue

        score = base.copy()
        score[name_pos[card_name]] = 0
        for partner in combo_partners.get(card_name, []):
            jp = name_pos.get(partner)
            if jp is not None:
                score[jp] = 0

        candidates = np.nonzero(score)[0]
        if candidates.size == 0:
            continue

        # Composite key: playability lives in [0, 1) so it can never outrank a full
        # integer score step. Ordering is (-score, -playability); exact ties keep
        # corpus order via the stable sort, so rebuilds stay byte-identical.
        key = score[candidates].astype(np.float64) + playability[candidates]
        if candidates.size > top_k:
            top_local = np.argpartition(-key, top_k - 1)[:top_k]
            top_local = top_local[np.argsort(-key[top_local], kind="stable")]
        else:
            top_local = np.argsort(-key, kind="stable")

        entries = []
        for loc in top_local:
            j = candidates[loc]
            labels = [
                label
                for label, a_col, b_col, forward, reverse in active_rules
                if (forward and has_tag[j, b_col]) or (reverse and has_tag[j, a_col])
            ]
            entries.append({
                "partner": names[j],
                "score": int(score[j]),
                "synergies": labels,
            })
        synergy_graph[card_name] = entries

    return synergy_graph


def main():
    print("Loading cards...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards")

    # No embeddings. The similarity tiebreak is gone (see build_synergy_graph), so
    # this step no longer reads the 17 MB matrix at all.
    ranked = pd.to_numeric(df.get("edhrec_rank"), errors="coerce").notna().sum() \
        if "edhrec_rank" in df.columns else 0
    print(f"  {ranked:,} cards carry an EDHREC rank to break ties with")

    print("Building synergy graph...")
    synergy_graph = build_synergy_graph(df)

    print(f"  {len(synergy_graph):,} cards with synergy partners")

    # Stats
    non_land = df[df["supertype"] != "Land"]
    has_tags = non_land["mechanical_tags"].fillna("").str.len() > 0
    non_vanilla = has_tags.sum()
    has_synergies = sum(1 for name in synergy_graph if name in non_land["name"].values)
    if non_vanilla > 0:
        print(f"  {has_synergies:,}/{non_vanilla:,} non-land tagged cards have synergy partners "
              f"({has_synergies/non_vanilla*100:.1f}%)")

    with open(SYNERGY_GRAPH_PATH, "w") as f:
        json.dump(synergy_graph, f, separators=(",", ":"))

    size_mb = SYNERGY_GRAPH_PATH.stat().st_size / (1024 * 1024)
    print(f"  Wrote {SYNERGY_GRAPH_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
