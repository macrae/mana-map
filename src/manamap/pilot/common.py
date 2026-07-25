"""Shared pilot helpers: rule-ID validation, deck paths, rules/strategy-DB loading."""

import hashlib
import json
import re

import numpy as np

from manamap.config import (
    DECKS_DIR,
    RULES_EMBEDDINGS_PATH,
    RULES_INDEX_PATH,
    STRATEGY_DOC_PATH,
    STRATEGY_EMBEDDINGS_PATH,
    STRATEGY_INDEX_PATH,
)

# The single source of truth for valid citation IDs: a numbered CR rule
# ("601", "601.2", "601.2a") or a glossary term ("glossary:storm").
RULE_ID_RE = re.compile(r"^\d{3}(\.\d+[a-z]?)?$|^glossary:[a-z0-9'-]+$")

# Strategy-section IDs ("strategy:tempo", "strategy:multiplayer.politics").
# Prefix-dispatched alongside RULE_ID_RE so one citation validator serves both.
STRATEGY_ID_RE = re.compile(r"^strategy:[a-z0-9-]+(\.[a-z0-9-]+){0,2}$")


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


def strategy_doc_sha256():
    """sha256 of the strategy doc — the doc/DB staleness handshake."""
    return hashlib.sha256(STRATEGY_DOC_PATH.read_bytes()).hexdigest()


def load_strategy_db():
    """Load the strategy DB. Returns (sections, order, embeddings).

    sections: {section_id: {"title", "text", "section", "parent", "sources"}}
    order: list of section_ids, aligned so embeddings[i] embeds order[i]
    embeddings: (N, 384) float32, rows L2-normalized at build time

    Fails hard if strategy.md was edited after the DB was built.
    """
    if not STRATEGY_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{STRATEGY_INDEX_PATH} not found — run `manamap pilot build-strategy-db` "
            f"first (requires {STRATEGY_DOC_PATH})."
        )
    with open(STRATEGY_INDEX_PATH) as f:
        index = json.load(f)
    embeddings = np.load(STRATEGY_EMBEDDINGS_PATH)
    order = index["order"]
    if len(order) != embeddings.shape[0]:
        raise ValueError(
            f"Strategy DB inconsistent: index has {len(order)} chunks but embeddings "
            f"have {embeddings.shape[0]} rows — re-run `manamap pilot build-strategy-db`."
        )
    if STRATEGY_DOC_PATH.exists() and index["meta"].get("doc_sha256") != strategy_doc_sha256():
        raise ValueError(
            f"{STRATEGY_DOC_PATH} was edited after the DB was built — "
            f"re-run `manamap pilot build-strategy-db`."
        )
    return index["sections"], order, embeddings
