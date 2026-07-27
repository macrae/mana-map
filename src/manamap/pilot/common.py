"""Shared pilot helpers: rule-ID validation, deck paths, rules/strategy-DB loading."""

import atexit
import hashlib
import json
import re
import sys

import numpy as np

from manamap.config import (
    CARD_ROLES_PATH,
    COMBO_DETAILS_PATH,
    DECKS_DIR,
    RULES_EMBEDDINGS_PATH,
    RULES_INDEX_PATH,
    STRATEGY_DOC_PATH,
    STRATEGY_EMBEDDINGS_PATH,
    STRATEGY_INDEX_PATH,
    SYNERGY_GRAPH_PATH,
)

# The single source of truth for valid citation IDs: a numbered CR rule
# ("601", "601.2", "601.2a") or a glossary term ("glossary:storm").
RULE_ID_RE = re.compile(r"^\d{3}(\.\d+[a-z]?)?$|^glossary:[a-z0-9'-]+$")

# Strategy-section IDs ("strategy:tempo", "strategy:multiplayer.politics").
# Prefix-dispatched alongside RULE_ID_RE so one citation validator serves both.
STRATEGY_ID_RE = re.compile(r"^strategy:[a-z0-9-]+(\.[a-z0-9-]+){0,2}$")

# Decklist section headers, lowercased with any trailing colon stripped. Shared by
# the parser (fetch_deck) and the writer (build_deck), which must agree: the writer
# has to recognise a sideboard block to preserve it, and if these two sets drifted
# a rebuild would silently delete the section the parser can still read.
COMMANDER_SECTION_MARKERS = frozenset({"commander", "commanders"})
MAIN_SECTION_MARKERS = frozenset({"deck", "mainboard", "main"})
SIDEBOARD_SECTION_MARKERS = frozenset({"sideboard", "side", "maybeboard", "considering"})


# ── Memoized artifact loading ───────────────────────────────────────────
#
# The global graphs are 1.9–27.8 MB and several modules parse the same file in
# one process (deck-facts alone used to parse card_roles.json twice and pay
# 0.42 s for a synergy parse it printed one number from). Same key discipline
# as agent_cache._SHA_MEMO: (mtime_ns, size) per path, so a rewrite in-process
# is picked up and the stale copy is evicted rather than accumulated.
#
# Treat every returned object as READ-ONLY — callers share one parse.

_JSON_MEMO = {}


def load_json_memo(path):
    """Parse a JSON file once per (mtime_ns, size) per process. Read-only."""
    stat = path.stat()  # propagates FileNotFoundError, same as open() did
    sig = (stat.st_mtime_ns, stat.st_size)
    hit = _JSON_MEMO.get(str(path))
    if hit is not None and hit[0] == sig:
        return hit[1]
    with open(path) as f:
        doc = json.load(f)
    _JSON_MEMO[str(path)] = (sig, doc)
    return doc


def clear_memo():
    """Drop all memoized parses (test teardown; mirrors _SHA_MEMO.clear())."""
    _JSON_MEMO.clear()
    _STRATEGY_SHA_MEMO.clear()


# Freeing tens of MB of parsed JSON via refcounting is nearly free; letting it
# survive to interpreter shutdown costs ~0.7 s of teardown GC per CLI call
# (measured on deck-facts). Clear the memo before Python starts tearing the
# world down so the win from one-parse-per-process isn't paid back at exit.
atexit.register(clear_memo)


def load_card_roles():
    """card_roles.json's roles map, parsed once per process. Read-only."""
    return load_json_memo(CARD_ROLES_PATH)["roles"]


def load_combo_details():
    """combo_details.json ({combos, by_card, meta}), parsed once. Read-only."""
    return load_json_memo(COMBO_DETAILS_PATH)


def load_synergy_graph():
    """synergy_graph.json (name → top-10 shortlist), parsed once. Read-only."""
    return load_json_memo(SYNERGY_GRAPH_PATH)


def load_json(path, default=None):
    """Parse `path` if it exists, else `default`. For small per-deck artifacts."""
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def mainboard(cards):
    """The cards actually in the deck — everything not behind a SIDEBOARD marker."""
    return [c for c in cards if not c.get("is_sideboard")]


def sideboard(cards):
    """The cards behind a SIDEBOARD marker, accessories included."""
    return [c for c in cards if c.get("is_sideboard")]


def is_land(card):
    """Land by front-face type line — the face that comes down on the battlefield."""
    return "Land" in str(card.get("type_line", "")).split(" // ")[0]


def checker_passed(doc):
    """The publication gate: only a checker-passed artifact may be stated as fact.

    One predicate, three consumers (renderer, cache, newsstand) — if the gate
    ever changes shape, it changes for all of them at once.
    """
    return (doc.get("checker") or {}).get("verdict") == "pass"


def try_load_rules_db():
    """Best-effort rules for citation checks: real rules, or {} when the DB is absent.

    Validators degrade gracefully — an absent rules DB skips rule-text
    verification rather than failing the artifact.
    """
    try:
        rules, _, _ = load_rules_db()
        return rules
    except (FileNotFoundError, ValueError):
        return {}


def report_errors(fail_label, errors, ok_line=None):
    """The shared validator CLI tail: FAIL + list + exit 1, else the OK line.

    Pass `ok_line=None` when the OK message derefs the validated document —
    composing it eagerly would crash on exactly the malformed input the FAIL
    branch exists for. Print it after this returns instead.
    """
    if errors:
        print(f"FAIL {fail_label} ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    if ok_line is not None:
        print(ok_line)


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


_STRATEGY_SHA_MEMO = {}


def strategy_doc_sha256():
    """sha256 of the strategy doc — the doc/DB staleness handshake.

    Memoized on (mtime_ns, size): a whole-deck cache-status scan asks for this
    once per strategy-consuming routine.
    """
    stat = STRATEGY_DOC_PATH.stat()
    sig = (stat.st_mtime_ns, stat.st_size)
    hit = _STRATEGY_SHA_MEMO.get("doc")
    if hit is not None and hit[0] == sig:
        return hit[1]
    digest = hashlib.sha256(STRATEGY_DOC_PATH.read_bytes()).hexdigest()
    _STRATEGY_SHA_MEMO["doc"] = (sig, digest)
    return digest


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
