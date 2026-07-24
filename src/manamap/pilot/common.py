"""Shared pilot helpers: rule-ID validation, deck paths, rules-DB loading."""

import json
import re

import numpy as np

from manamap.config import DECKS_DIR, RULES_EMBEDDINGS_PATH, RULES_INDEX_PATH

# The single source of truth for valid citation IDs: a numbered CR rule
# ("601", "601.2", "601.2a") or a glossary term ("glossary:storm").
RULE_ID_RE = re.compile(r"^\d{3}(\.\d+[a-z]?)?$|^glossary:[a-z0-9'-]+$")


def deck_dir(slug):
    """Return the deck directory for a slug, or fail with an actionable message."""
    path = DECKS_DIR / slug
    if not path.is_dir():
        raise FileNotFoundError(
            f"No deck directory for '{slug}'. Create {path}/decklist.txt first "
            f"(one '1 Card Name' per line, commander marked with a 'Commander:' "
            f"section header or a trailing *CMDR*)."
        )
    return path


def load_deck_cards(slug):
    """Load a deck's cards.json, failing with a pointer to fetch-deck if absent."""
    path = deck_dir(slug) / "cards.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `manamap pilot fetch-deck {slug}` first."
        )
    with open(path) as f:
        return json.load(f)


def load_rules_db():
    """Load the rules DB. Returns (rules, order, embeddings).

    rules: {rule_id: {"text", "section", "parent"}}
    order: list of rule_ids, aligned so embeddings[i] embeds order[i]
    embeddings: (N, 384) float32, rows L2-normalized at build time
    """
    if not RULES_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{RULES_INDEX_PATH} not found — run `manamap pilot download-rules` "
            f"then `manamap pilot build-rules-db` first."
        )
    with open(RULES_INDEX_PATH) as f:
        index = json.load(f)
    embeddings = np.load(RULES_EMBEDDINGS_PATH)
    order = index["order"]
    if len(order) != embeddings.shape[0]:
        raise ValueError(
            f"Rules DB inconsistent: index has {len(order)} chunks but embeddings "
            f"have {embeddings.shape[0]} rows — re-run `manamap pilot build-rules-db`."
        )
    return index["rules"], order, embeddings
