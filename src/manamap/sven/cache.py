"""Sven's two caches, and the one idea both of them rest on.

A cached answer that is WRONG is worse than no cache at all — this bench's whole
claim is that every figure is one you can re-derive, and an answer served from
before a decklist changed quietly breaks that. So both caches here are keyed on
a SIGNATURE OF THE FILES THE ANSWER DEPENDED ON, never on the question alone and
never on a clock.

**Coarse on purpose.** A tool that reads one file inside `data/decks/heliod/`
declares the whole directory as its dependency. That over-invalidates: editing
`heliod/log.jsonl` drops a cached answer about heliod's mana base, which did not
need dropping. It is the right direction to be wrong in, and `tests/conftest.py`
already made this exact call for the regenerate-and-compare cache:

    Inputs are named EXPLICITLY, never auto-discovered. Naming a directory
    over-invalidates and cannot be wrong; auto-discovery could silently serve a
    stale pass.

Tracing real reads (an `open()` audit hook, say) would be finer and would be a
second thing to keep correct, with a silent-staleness failure mode. Directory
signatures are boring and cannot lie.

TWO CACHES, DIFFERENT LIFETIMES:

- `FactCache` — in-process, unbounded within a signature, evicted on change.
  Holds tool results. Dies with the daemon. This is the one that makes a
  follow-up question instant.
- `answer_get` / `answer_put` — on disk under `SVEN_CACHE_DIR`, LRU-bounded,
  survives a restart. Holds whole rendered answers. Gitignored, and deliberately
  NOT a tracked artifact, so it needs no validator and no `deck_status` row.

Both are dropped wholesale when `CACHE_VERSION` moves. Bump it whenever the
shape of what is stored changes, or when Sven's charter changes enough that his
old wording would misrepresent him.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from manamap.config import DATA_DIR

#: Bump to invalidate every cached fact and answer everywhere. The precedent is
#: `config.AGENT_CACHE_VERSION`, which exists for the same reason.
CACHE_VERSION = 1

#: Gitignored. An answer cache is a convenience, never evidence — nothing in the
#: repo may read it to learn a fact, and no validator gates it.
SVEN_CACHE_DIR = Path(os.environ.get("MANAMAP_SVEN_CACHE", DATA_DIR / ".sven-cache"))

#: How many answers to keep. Small: the hit rate lives in the first handful of
#: repeated questions, and a cache big enough to need real eviction policy is a
#: cache that has outgrown being a convenience.
ANSWER_CACHE_MAX = 256

#: Files whose churn must never invalidate an answer. `.agent-cache.json` moves
#: when an unrelated routine is recorded; `.DS_Store` is noise. Mirrors
#: `conftest._DIGEST_SKIP_FILES`, which learned this the same way.
_SKIP_FILES = frozenset({".agent-cache.json", ".DS_Store"})
_SKIP_DIRS = frozenset({"__pycache__", ".agent-out", ".pytest_cache", "logs"})


def _stat_sig(path):
    """`(mtime_ns, size)` for one file, or None if it is not there.

    Absence is part of the key: a tool that answered "there is no diagnosis for
    this deck" must be invalidated when one appears.
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def signature(paths):
    """A digest over the current state of `paths` — files or directories.

    `logs/` is skipped inside a directory walk on purpose. A running Forge batch
    rewrites its logs continuously, and a signature that moved every few seconds
    would make every answer about that deck uncacheable for hours — exactly when
    the pilot is asking about it most.
    """
    parts = []
    for path in sorted({Path(p) for p in paths}, key=str):
        if path.is_dir():
            for child in sorted(path.rglob("*"), key=str):
                if child.is_dir():
                    continue
                if child.name in _SKIP_FILES:
                    continue
                if _SKIP_DIRS & set(child.relative_to(path).parts):
                    continue
                parts.append((str(child), _stat_sig(child)))
        else:
            parts.append((str(path), _stat_sig(path)))
    blob = json.dumps([CACHE_VERSION, parts], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class FactCache:
    """Tool results, in this process, keyed on what the tool read.

    Not an `lru_cache`: the key has to include a file signature that changes
    underneath us, and a stale entry must be REPLACED rather than accumulated —
    the same discipline as `common.mtime_memo`, and the reason
    `agent_cache._SHA_MEMO` is a known leak (it never evicts, and it is absent
    from `common.clear_memo`).
    """

    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    def get_or_call(self, key, deps, build):
        """`build()`'s result, recomputed only when `deps` change on disk."""
        sig = signature(deps)
        hit = self._store.get(key)
        if hit is not None and hit[0] == sig:
            self.hits += 1
            return hit[1]
        self.misses += 1
        value = build()
        self._store[key] = (sig, value)
        return value

    def clear(self):
        self._store.clear()

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "rate": round(self.hits / total, 3) if total else None,
                "entries": len(self._store)}


def _answer_key(question, touched_signature, model):
    """One answer's identity: what was asked, of what state, by which model.

    `model` is in the key because Haiku's answer and Sonnet's answer to the same
    question are different artifacts, and an escalated turn must not be served
    later from the cheap one's cache entry.
    """
    norm = " ".join((question or "").lower().split())
    blob = json.dumps([CACHE_VERSION, norm, touched_signature, model], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def answer_get(question, touched_signature, model):
    """A previously rendered answer, or None. Never raises."""
    path = SVEN_CACHE_DIR / f"{_answer_key(question, touched_signature, model)}.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    try:                                    # touch for LRU; failure is harmless
        os.utime(path, None)
    except OSError:
        pass
    return doc


def answer_put(question, touched_signature, model, answer, meta=None):
    """Store one answer. Best-effort: a cache write must never fail a turn."""
    key = _answer_key(question, touched_signature, model)
    doc = {"question": question, "answer": answer, "model": model,
           "signature": touched_signature, "at": time.time(),
           "cache_version": CACHE_VERSION, **(meta or {})}
    try:
        SVEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (SVEN_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        _evict()
    except OSError:
        pass
    return key


def _evict():
    """Keep the newest `ANSWER_CACHE_MAX` entries, by access time."""
    try:
        entries = sorted(SVEN_CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for path in entries[:-ANSWER_CACHE_MAX] if len(entries) > ANSWER_CACHE_MAX else []:
        try:
            path.unlink()
        except OSError:
            pass


def purge():
    """Drop every stored answer. Returns how many went."""
    gone = 0
    for path in SVEN_CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
            gone += 1
        except OSError:
            pass
    return gone
