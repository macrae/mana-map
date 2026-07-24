"""Shared analysis utilities: tag parsing, cosine similarity, embedding loading."""

import numpy as np
import pandas as pd


def parse_tag_set(tags_str):
    """Parse a comma-separated mechanical_tags string into a set of tags."""
    if not tags_str or pd.isna(tags_str):
        return set()
    return {t.strip() for t in str(tags_str).split(",") if t.strip()}


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
