"""Shared pilot helpers: rule-ID validation, deck paths, rules/strategy-DB loading."""

import atexit
import hashlib
import json
import os
import pathlib
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


_MTIME_MEMO = {}


def mtime_memo(path, key, build, absent=None):
    """`build()`'s result, recomputed only when `path` changes on disk.

    The same (mtime_ns, size) discipline as `load_json_memo`, for derived values
    rather than raw parses — a name set, a flags dict, a digest.

    It exists because the hand-rolled copies did not all get it right. Two gated
    on truthiness instead (`if not _MEMO: ...`), which never notices a rewrite:
    correct for a CLI process that exits in seconds, wrong for pytest, where one
    process regenerates `cards.csv` and every later assertion in that session is
    answered from the pre-edit copy. The failure is silent and reads as a stale
    fixture rather than a caching bug.

    `key` is explicit rather than derived from `build` because two different
    builders legitimately read one file — `cards.csv` backs both a name set and
    an oracle map — and a key taken from `__qualname__` would collide the moment
    someone passed a lambda.

    A missing file returns `absent` and caches nothing, so a fresh clone that
    later generates the artifact picks it up without restarting.
    """
    try:
        stat = path.stat()
    except OSError:
        return absent
    sig = (stat.st_mtime_ns, stat.st_size)
    hit = _MTIME_MEMO.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    value = build()
    _MTIME_MEMO[key] = (sig, value)
    return value


def clear_memo():
    """Drop all memoized parses (test teardown; mirrors _SHA_MEMO.clear())."""
    _JSON_MEMO.clear()
    _MTIME_MEMO.clear()
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


# ── The opened-line status vocabulary ────────────────────────────────────
#
# A claim that a card "opens a line" is a candidate until a stack artifact passes
# the checker. Two validators share this vocabulary — considering and diagnosis —
# so it lives here rather than in either of them. It used to live in
# `validate_sideboard.py`, which made a retired sideboard validator a load-bearing
# library for The Short List; `validate_upgrade_watch.py` was the third consumer
# and went with the sideboard.

# The magic string five agent prompts already use for an unproven line. Kept
# identical so one grep finds every candidate claim in the repo.
UNVERIFIED_STATUS = "needs a stack scenario"
# An opened line may claim this only by pointing at a stack artifact whose checker
# verdict is `pass` and which actually names the line's cards — the claim is
# re-checked against the file, never trusted.
VERIFIED_STATUS = "verified"


def check_verified_line(i, line, deck_path, label="opens_lines"):
    """A `verified` claim is re-checked against the stack artifact, never trusted."""
    rel = line.get("stack_artifact")
    if not rel:
        return [f"{label} {i}: status {VERIFIED_STATUS!r} requires a "
                f"`stack_artifact` path to a checker-passed stack"]
    if deck_path is None:
        return [f"{label} {i}: status {VERIFIED_STATUS!r} cannot be confirmed "
                f"without the deck directory"]
    path = deck_path / rel
    if not path.exists():
        return [f"{label} {i}: stack_artifact {rel!r} does not exist"]
    with open(path) as f:
        artifact = json.load(f)
    if artifact.get("checker", {}).get("verdict") != "pass":
        return [f"{label} {i}: stack_artifact {rel!r} has checker verdict "
                f"{artifact.get('checker', {}).get('verdict')!r}, not 'pass' — "
                f"only a passing stack verifies a line"]
    body = json.dumps(artifact, ensure_ascii=False)
    missing = [c for c in line.get("cards", []) if c not in body]
    if missing:
        return [f"{label} {i}: stack_artifact {rel!r} never mentions "
                f"{missing} — it cannot verify this line"]
    return []




def resolve_out_path(out, slug, command, ext=".json"):
    """Where a per-deck view should be written — slug-scoped, or a loud error.

    Concurrent deck agents share one scratchpad directory, and they were writing
    per-deck views to generic names: `audit.json`, `aud.json`, `audit2.json`. The
    second write silently replaced the first, so an agent working heliod would read
    goblin-storm's numbers under a heliod invocation and reason from them. It hit
    seven agents across two sessions, and **every instance was caught by an agent
    noticing an implausible figure, which is not a control** — the failure is
    silent and the wrong deck's numbers are exactly the shape the reader expects.

    The CLI cannot police a shell redirect (`deck-audit X > audit.json`), so this
    does the two things it can: makes the SAFE path the easy one, and makes the
    unsafe one fail loudly.

      * a DIRECTORY (existing, or a trailing slash) auto-names
        `<command>-<slug><ext>` inside it — collisions become impossible;
      * a bare filename stays relative to the deck's own directory, as before;
      * an explicit path elsewhere must carry the slug in its basename, or this
        raises. That is the case that used to corrupt silently.

    `ext` is the auto-naming suffix only; it never rewrites a name the caller
    chose. Most views are JSON, but `diagnosis-report` emits markdown, and a
    directory that auto-named its report `.json` would be lying about the bytes.
    """
    text = str(out)
    path = pathlib.Path(text)
    if text.endswith(("/", os.sep)) or path.is_dir():
        return path / f"{command}-{slug}{ext}"
    if "/" not in text and os.sep not in text:
        return deck_dir(slug) / text
    if slug not in path.name:
        raise SystemExit(
            f"refusing to write {command} for {slug!r} to {path} — the filename does "
            f"not contain the slug, and concurrent deck agents share one scratch "
            f"directory. A generic name is how one deck's view silently overwrites "
            f"another's. Pass a directory (auto-names {command}-{slug}{ext}), or a "
            f"filename containing {slug!r}."
        )
    return path


def expand_copies(cards):
    """One element per PHYSICAL card, not per decklist entry.

    cards.json stores basics as a single entry with `quantity: N` — eleven
    Islands are one entry. Any statistic about the library (how many lands, how
    many blue sources, what the hypergeometric draw looks like) is a question
    about copies, and counting entries silently halves it. Use this before
    counting anything the shuffler would see; use the raw list only where the
    question really is per distinct card (artist authorship, role coverage).
    """
    out = []
    for card in cards:
        out.extend([card] * int(card.get("quantity") or 1))
    return out


def is_land(card):
    """Land by front-face type line — the face that comes down on the battlefield."""
    return "Land" in str(card.get("type_line", "")).split(" // ")[0]


def commander_rejection(row):
    """Why this card can't start in the command zone, or None if it can.

    One predicate, two callers with different needs: `build_deck` raises with
    the reason, `pool_facts` just filters a box of cards down to its legal
    commanders. Written twice, these drift — and the failure is silent in the
    direction that matters, since a filter that is quietly wrong about
    "can be your commander" simply omits the card from a shortlist.

    `row` is anything with .get: a cards.csv row, a cards.json record.
    """
    front = str(row.get("type_line", "") or "").split(" // ")[0]
    text = str(row.get("oracle_text", "") or "").lower()
    if not ("Legendary" in front and "Creature" in front) and "can be your commander" not in text:
        return ("not a legal commander (needs to be a legendary creature, "
                "or say 'can be your commander')")
    if row.get("legal_commander") != "legal":
        return "not legal in Commander"
    return None


def checker_passed(doc):
    """The publication gate: only a checker-passed artifact may be stated as fact.

    One predicate, three consumers (renderer, cache, newsstand) — if the gate
    ever changes shape, it changes for all of them at once.
    """
    return (doc.get("checker") or {}).get("verdict") == "pass"


def withheld(doc):
    """Explicitly withheld from publication — the board is no longer reachable.

    Separate from `presentable` because a decision scenario carries no checker
    block, so it can be withheld without ever having been checker-passed.
    """
    return doc.get("presentable", True) is False


def presentable(doc):
    """Can this verified artifact be shown as something THIS deck can do?

    Passing the checker makes a scenario a true rules answer forever. It does
    not make the board reachable: cut a card off the board and the artifact
    still holds as a finding while describing a game this 99 can no longer
    produce. Yawgmoth's stacks 001 and 007 resolve a Blowfly Infestation board
    and Blowfly was cut; 013's board carries Necroskitter and Carnifex Demon.

    So publication needs two gates, not one — `checker_passed` for "is it true"
    and this for "is it ours". Set `presentable: false` with a
    `presentable_note` saying which card left. The flag is top-level on purpose:
    `scenario:self` fingerprints only `{title, scenario}`, so marking one costs
    no re-verification.

    One predicate, three consumers (the renderer, the newsstand manifest and the
    deck dossier) — if the gate ever changes shape, it changes for all of them.
    """
    return checker_passed(doc) and not withheld(doc)


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
