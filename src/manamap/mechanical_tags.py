"""Mechanical tags: regex patterns over oracle text -> a fixed retrieval vocabulary.

`MECHANICAL_TAGS` (config.py) maps a tag to a pattern; `tag_oracle_text` returns the
tags a text matches and `encode_tags_multihot` makes the model input. Shared by step 2
(extract) and the analysis steps, so a card is tagged one way everywhere.

FROZEN. The tag list is model-facing — the ability model's input width is the tag
count — so editing `MECHANICAL_TAGS` invalidates `model_ability.pt` (retrain steps
3-5). What job a card does in a 99 is `ROLE_PATTERNS`, a separate dict, because roles
change often and tags must not.
"""

import re

import numpy as np

from manamap.config import MECHANICAL_TAG_NAMES, MECHANICAL_TAGS


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
