"""Step 10: Build synergy graph from mechanical tags and synergy rules."""

import json

import numpy as np
import pandas as pd

from config import (
    ABILITY_EMBEDDINGS_PATH,
    COMBO_GRAPH_PATH,
    EMBEDDINGS_PATH,
    MECHANICAL_TAG_NAMES,
    OUTPUT_CSV_PATH,
    SYNERGY_GRAPH_PATH,
    SYNERGY_MAX_PARTNERS,
    SYNERGY_RULES,
)
from mechanical_tags import tag_oracle_text


def build_tag_index(df):
    """Build {tag_name: set(card_names)} index from mechanical_tags column."""
    tag_to_cards = {tag: set() for tag in MECHANICAL_TAG_NAMES}
    for _, row in df.iterrows():
        tags_str = row.get("mechanical_tags", "")
        if not tags_str or pd.isna(tags_str):
            continue
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        for tag in tags:
            if tag in tag_to_cards:
                tag_to_cards[tag].add(row["name"])
    return tag_to_cards


def build_card_tags(df):
    """Build {card_name: set(tags)} from mechanical_tags column."""
    card_tags = {}
    for _, row in df.iterrows():
        tags_str = row.get("mechanical_tags", "")
        if not tags_str or pd.isna(tags_str):
            card_tags[row["name"]] = set()
        else:
            card_tags[row["name"]] = {t.strip() for t in tags_str.split(",") if t.strip()}
    return card_tags


def load_combo_partners():
    """Load combo graph to exclude known combo partners from synergy results."""
    try:
        with open(COMBO_GRAPH_PATH, "r") as f:
            graph = json.load(f)
        return graph.get("partners", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def compute_embedding_similarity(name_to_idx, embeddings, idx_a, idx_b):
    """Compute cosine similarity between two cards by index."""
    a = embeddings[idx_a]
    b = embeddings[idx_b]
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def build_synergy_graph(df, embeddings=None, name_to_idx=None):
    """Build synergy graph: card -> [synergy partners with labels].

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

    # Build synergy candidates for each card
    synergy_graph = {}

    for card_name, tags in card_tags.items():
        if not tags:
            continue

        # Find all matching synergy rules where this card has tag_A
        partner_scores = {}  # partner_name -> {score, synergies}

        for tag_a, tag_b, label in SYNERGY_RULES:
            if tag_a in tags:
                # Find all cards with tag_b
                for partner in tag_to_cards.get(tag_b, set()):
                    if partner == card_name:
                        continue
                    # Exclude known combo partners
                    if partner in combo_partners.get(card_name, []):
                        continue

                    if partner not in partner_scores:
                        partner_scores[partner] = {"score": 0, "synergies": []}
                    partner_scores[partner]["score"] += 1
                    if label not in partner_scores[partner]["synergies"]:
                        partner_scores[partner]["synergies"].append(label)

            # Also check reverse: if this card has tag_B, it synergizes with tag_A cards
            if tag_b in tags:
                for partner in tag_to_cards.get(tag_a, set()):
                    if partner == card_name:
                        continue
                    if partner in combo_partners.get(card_name, []):
                        continue

                    if partner not in partner_scores:
                        partner_scores[partner] = {"score": 0, "synergies": []}
                    partner_scores[partner]["score"] += 1
                    if label not in partner_scores[partner]["synergies"]:
                        partner_scores[partner]["synergies"].append(label)

        if not partner_scores:
            continue

        # Sort by rule count, then embedding similarity as tiebreaker
        ranked = []
        for partner, info in partner_scores.items():
            emb_sim = 0.0
            if embeddings is not None and name_to_idx is not None:
                idx_a = name_to_idx.get(card_name)
                idx_b = name_to_idx.get(partner)
                if idx_a is not None and idx_b is not None:
                    emb_sim = compute_embedding_similarity(
                        name_to_idx, embeddings, idx_a, idx_b
                    )
            ranked.append({
                "partner": partner,
                "score": info["score"],
                "synergies": info["synergies"],
                "_emb_sim": emb_sim,
            })

        ranked.sort(key=lambda x: (-x["score"], -x["_emb_sim"]))

        # Keep top partners
        top = ranked[:SYNERGY_MAX_PARTNERS]
        synergy_graph[card_name] = [
            {"partner": r["partner"], "score": r["score"], "synergies": r["synergies"]}
            for r in top
        ]

    return synergy_graph


def main():
    print("Loading cards...")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    print(f"  {len(df):,} cards")

    # Load ability embeddings for tiebreaking (fall back to color+type)
    embeddings = None
    name_to_idx = None
    try:
        embeddings = np.load(ABILITY_EMBEDDINGS_PATH)
        name_to_idx = {name: i for i, name in enumerate(df["name"])}
        print(f"  Loaded ability embeddings for similarity tiebreaking")
    except FileNotFoundError:
        try:
            embeddings = np.load(EMBEDDINGS_PATH)
            name_to_idx = {name: i for i, name in enumerate(df["name"])}
            print(f"  Loaded color+type embeddings for similarity tiebreaking (ability embeddings not found)")
        except FileNotFoundError:
            print(f"  No embeddings found — skipping similarity tiebreaking")

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
