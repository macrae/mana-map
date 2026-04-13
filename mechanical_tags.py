"""Mechanical tag extraction from MTG oracle text via regex patterns."""

import re

import numpy as np
import pandas as pd

from config import MECHANICAL_TAG_NAMES, MECHANICAL_TAGS


# Pre-compile patterns for performance
_COMPILED_TAGS = {
    tag: re.compile(pattern, re.IGNORECASE)
    for tag, pattern in MECHANICAL_TAGS.items()
}


def tag_oracle_text(text):
    """Extract mechanical tags from oracle text.

    Args:
        text: Oracle text string (may be empty/None).

    Returns:
        Sorted list of tag name strings that matched.
    """
    if not text or not isinstance(text, str):
        return []
    tags = []
    for tag, pattern in _COMPILED_TAGS.items():
        if pattern.search(text):
            tags.append(tag)
    return sorted(tags)


def tag_oracle_text_from_row(row):
    """Extract tags from a DataFrame row, combining oracle_text and keywords.

    Also checks the type_line for Equipment/Aura supertypes.
    """
    text = str(row.get("oracle_text", "") or "")
    keywords = str(row.get("keywords", "") or "")
    type_line = str(row.get("type_line", "") or "")

    # Append keywords as pseudo-oracle text so keyword-based tags fire
    combined = text
    if keywords:
        combined += " " + keywords
    if type_line:
        combined += " " + type_line

    return tag_oracle_text(combined)


def encode_tags_multihot(df, tag_names=None):
    """Encode mechanical_tags column into (N, num_tags) float32 multi-hot array.

    Args:
        df: DataFrame with a 'mechanical_tags' column (comma-separated tag strings).
        tag_names: Ordered list of tag names. Defaults to MECHANICAL_TAG_NAMES.

    Returns:
        (N, len(tag_names)) float32 numpy array.
    """
    if tag_names is None:
        tag_names = MECHANICAL_TAG_NAMES

    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    n = len(df)
    dim = len(tag_names)
    result = np.zeros((n, dim), dtype=np.float32)

    for i, tag_str in enumerate(df["mechanical_tags"].fillna("")):
        if tag_str:
            for tag in tag_str.split(", "):
                tag = tag.strip()
                if tag in tag_to_idx:
                    result[i, tag_to_idx[tag]] = 1.0

    return result
