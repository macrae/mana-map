"""Shared analysis utilities: tag parsing, similarity search, embedding loading."""

import numpy as np
import pandas as pd

WUBRG = ("W", "U", "B", "R", "G")


def parse_tag_set(tags_str):
    """Parse a comma-separated mechanical_tags string into a set of tags."""
    if not tags_str or pd.isna(tags_str):
        return set()
    return {t.strip() for t in str(tags_str).split(",") if t.strip()}


def parse_color_identity(value):
    """Parse a cards.csv color_identity cell ("B, G, R") into a set.

    Empty/NaN means colorless, which is a subset of every identity — not a
    missing value.
    """
    if not value or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {c.strip().upper() for c in str(value).split(",") if c.strip()}


def build_name_index(df):
    """Map card name → row index.

    Names are not unique (Un-set variants collide), so this is last-write-wins
    and must never be used where positional identity matters. Embeddings index
    positionally: embeddings[i] is cards.csv row i.
    """
    return {name: i for i, name in enumerate(df["name"])}


def color_identity_mask(df, identity):
    """Boolean array: True where a card's color identity fits inside `identity`.

    This is the Commander legality constraint, not a color filter — a colorless
    card fits every commander.
    """
    allowed = {c.upper() for c in identity}
    return np.array(
        [parse_color_identity(v) <= allowed for v in df["color_identity"]],
        dtype=bool,
    )


def top_k_similar(embeddings, idx, k, mask=None):
    """Top-k most similar rows to `idx`, excluding itself.

    Rows are L2-normalized at build time, so cosine is a plain dot product.
    `mask` is an optional boolean array of eligible rows. Returns
    [(row_index, score)] sorted by descending score.
    """
    scores = embeddings @ embeddings[idx]
    eligible = np.ones(len(scores), dtype=bool) if mask is None else mask.copy()
    eligible[idx] = False
    if not eligible.any():
        return []

    # argpartition over the eligible subset, then sort just that slice
    candidates = np.flatnonzero(eligible)
    k = min(k, len(candidates))
    subset = scores[candidates]
    top = candidates[np.argpartition(subset, -k)[-k:]]
    return [(int(i), float(scores[i])) for i in top[np.argsort(-scores[top])]]


def cosine_similarity(a, b):
    """Cosine similarity between two vectors (0.0 if either has zero norm)."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_first_embeddings(*paths):
    """Load the first readable .npy among paths.

    Returns (array, path) for the first that loads, or (None, None) if none do.
    """
    for path in paths:
        try:
            return np.load(path), path
        except FileNotFoundError:
            continue
    return None, None
