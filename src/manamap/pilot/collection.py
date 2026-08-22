"""The pilot's physical collection: the ONE reader of `COLLECTION_DIR`, and every
view taken from it.

`COLLECTION_DIR` existed for months with exactly one dereference in the whole repo
(`deck_history._owned_index`), while `pool_facts` required every caller to type the
paths and `card_search` had no idea the box existed. So "does he own this?" was
answered by hand, in a throwaway script, roughly ten times in one session — and by
two *different* parsers that disagree:

  * `deck_history._owned_index`  — membership only, both faces, NOT resolved against
    the corpus (a typo in a box file became a card you own), re-globbed on every call.
  * `pool_facts.read_sources`    — copy counts and per-file provenance, front-face
    mapped onto `cards.csv`'s joined `"A // B"` form, re-parsed on every call.

Same nine files, two answers. This module is the single reader; `pool_facts` keeps
its own path-driven entry point because *that* answers a different question ("analyse
these boxes I am pointing at"), and this one answers "what does the pilot have".

**Ownership means a BOX, and deck membership deliberately does not count.** The first
version of this module defaulted to "in a box OR sleeved in a tracked deck", reasoning
that unsleeving is a decision rather than a purchase. `validate-recon` refuted it
immediately: `data/decks/` holds build plans as well as assembled decks, and nothing
distinguishes them. `kinnan` was built from the WHOLE FORMAT as a deterministic
baseline — the pilot does not own its commander, let alone Tropical Island or Walking
Ballista — so counting its 99 made 99 unowned cards read as owned, and a recon that
correctly said "buy this" failed the gate.

`include_decks=True` is still available for a caller who wants the union and knows
what it contains, but it is not the default and no gate uses it. If the bench ever
learns which decks are physically sleeved (`issue.json` status is the nearest thing,
and it only marks the dead ones), this becomes answerable properly.

Memoized on a signature over every file AND the directory listing, following
`common._RULES_DB_MEMO` rather than `mtime_memo`: that keys on a single path, and a
*new* `.txt` appearing changes the answer without changing any existing file's mtime.
"""

import json

from manamap.config import COLLECTION_DIR, DECKS_DIR
from manamap.pilot.common import expand_faces

# {key: (signature, value)} — see the module docstring for why not `mtime_memo`.
# Registered in `common.clear_memo()` so test teardown drops it with the others.
_COLLECTION_MEMO = {}


def _files():
    """The box files, sorted. An absent directory is not an error: no collection
    means no ownership claim, which is the contract `COLLECTION_DIR` is declared
    under in config.py."""
    if not COLLECTION_DIR.is_dir():
        return []
    return sorted(COLLECTION_DIR.glob("*.txt"))


def _signature():
    """(path, mtime_ns, size) per file — the directory listing included, because a
    box added or removed changes the answer while every existing file is untouched."""
    sig = []
    for path in _files():
        try:
            st = path.stat()
        except OSError:                      # pragma: no cover — raced deletion
            continue
        sig.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _memo(key, build):
    sig = _signature()
    hit = _COLLECTION_MEMO.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    value = build()
    _COLLECTION_MEMO[key] = (sig, value)
    return value


def _build_index():
    from manamap.pilot.fetch_deck import parse_decklist

    index = {}
    for path in _files():
        try:
            entries = parse_decklist(path.read_text())
        except (OSError, ValueError):
            continue
        for entry in entries:
            for face in expand_faces(entry["name"]):
                index.setdefault(face, set()).add(path.stem)
    return index


def owned_index():
    """`{name-or-face: {box file stem}}` — membership with provenance.

    Both faces are indexed because a decklist may name either, and an ownership
    answer that only matches the joined form silently says no to every DFC. This is
    the shape `deck_history.pending` needs: it reports the box a card came from, and
    an ownership claim nobody can source is not evidence.
    """
    return _memo("index", _build_index)


def _build_deck_names():
    names = set()
    if not DECKS_DIR.is_dir():
        return names
    for path in sorted(DECKS_DIR.glob("*/cards.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for card in doc.get("cards") or []:
            name = card.get("name")
            if name:
                names |= expand_faces(name)
    return names


def deck_names():
    """Every card name sleeved into a tracked deck, faces expanded.

    Deliberately NOT memoized on the collection signature — it reads a different set
    of files, and keying it on the box's mtimes would serve a stale answer every time
    a deck changed without the boxes changing. It is a few small JSON reads.
    """
    return _build_deck_names()


def owned_names(include_decks=False):
    """Every name the pilot has in a box.

    `include_decks=True` adds every card in every tracked deck's 99 — which includes
    decks that exist only as build plans, so it overstates ownership. See the module
    docstring; no gate uses it.
    """
    names = set(owned_index())
    if include_decks:
        names |= deck_names()
    return names


def owns(name, include_decks=False):
    """Does the pilot have this card, under either face?"""
    have = owned_names(include_decks)
    return bool(expand_faces(name) & have)


def sources_for(name):
    """The box file stems holding this card, or an empty set. Faces resolved."""
    index = owned_index()
    out = set()
    for face in expand_faces(name):
        out |= index.get(face, set())
    return out


def summary():
    """`{files, distinct_in_boxes, distinct_including_decks}` — for a report line."""
    index = owned_index()
    return {
        "files": [p.name for p in _files()],
        "distinct_in_boxes": len(index),
        "distinct_including_decks": len(owned_names(include_decks=True)),
    }
