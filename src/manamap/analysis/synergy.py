"""Step 10: Build synergy graph from mechanical tags and synergy rules."""

import json

import numpy as np
import pandas as pd

from manamap.config import (
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


def build_synergy_graph(df, embeddings=None, name_to_idx=None):
    """Build synergy graph: card -> [synergy partners with labels].

    Vectorized: partner scores are computed as a single (names x tags) @ weights
    matrix-vector product per distinct tag set, and top-K selection uses
    argpartition. Semantics match the original per-pair loop: score counts
    (rule, direction) hits, labels follow SYNERGY_RULES order, self and known
    combo partners are excluded, ranking is by (-score, -embedding similarity).

    Args:
        df: DataFrame with 'name' and 'mechanical_tags' columns.
        embeddings: Optional (N, 128) embedding array for tiebreaking.
        name_to_idx: Optional {name: index} map.

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

    # Per-name embedding rows for the similarity tiebreaker (zero vector -> sim 0.0,
    # matching the original's 0.0 fallback for names missing from name_to_idx).
    emb_unique = None
    if embeddings is not None and name_to_idx is not None:
        emb_unique = np.zeros((len(names), embeddings.shape[1]), dtype=np.float32)
        for j, n in enumerate(names):
            i = name_to_idx.get(n)
            if i is not None:
                emb_unique[j] = embeddings[i]

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

        if emb_unique is not None:
            sims = emb_unique[candidates] @ emb_unique[name_pos[card_name]]
        else:
            sims = np.zeros(candidates.size, dtype=np.float32)

        # Composite key: cosine sim scaled into [0, 0.5) can never outrank a
        # full integer score step, so ordering is (-score, -sim).
        key = score[candidates].astype(np.float64) + (sims.astype(np.float64) + 1.0) / 4.0
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

    # Load ability embeddings for tiebreaking (fall back to color+type)
    embeddings, loaded_path = load_first_embeddings(ABILITY_EMBEDDINGS_PATH, EMBEDDINGS_PATH)
    name_to_idx = None
    if embeddings is None:
        print("  No embeddings found — skipping similarity tiebreaking")
    else:
        name_to_idx = {name: i for i, name in enumerate(df["name"])}
        if loaded_path == ABILITY_EMBEDDINGS_PATH:
            print("  Loaded ability embeddings for similarity tiebreaking")
        else:
            print("  Loaded color+type embeddings for similarity tiebreaking (ability embeddings not found)")

    print("Building synergy graph...")
    synergy_graph = build_synergy_graph(df, embeddings, name_to_idx)

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
