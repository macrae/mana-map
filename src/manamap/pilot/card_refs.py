"""Pilot: which deck cards does an artifact actually reference?

The cache's `cards:semantic` token is one digest over the whole deck, so a
one-land swap MISSes every routine — ~330k tokens of agents re-answering
questions whose inputs did not meaningfully change. The refs extractor is the
other half of the fix: at record time each routine stores the set of card
names its artifact references, and a later card change invalidates only the
routines whose refs it intersects.

The matcher errs CONSERVATIVE by construction — a false positive costs one
unnecessary regeneration, a false negative would publish stale prose:

  * full card name as a substring (covers possessives: "Gishath's");
  * each face of a double-faced name ("Monster Manual // Zoological Study"
    matches on either face);
  * each *distinctive token* of a name — alphabetic tokens of six or more
    characters that are not tribal/common words — so the prose habit of
    writing "the Forerunner ping web" still pins `Forerunner of the Empire`.

All matching is case-insensitive substring search over the artifact's
canonical JSON. Pure functions, no I/O.
"""

import json
import re


def canonical_json(obj):
    """Same canonical form agent_cache uses; local copy avoids a circular import."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

# Words that appear in many card names or constantly in prose without naming a
# specific card. Stopwording only reduces FALSE POSITIVES — full-name matching
# always applies, so a card whose every token is stopworded still matches on
# its full name.
TOKEN_STOPWORDS = frozenset({
    "avatar", "ancient", "creature", "dragon", "vampire", "dinosaur", "goblin",
    "spirit", "knight", "legion", "master", "temple", "primal", "sacred",
    "mountain", "island", "forest", "swamp", "plains", "command", "commander",
    "battle", "battlefield", "protection", "hidden", "shadow", "blood",
})

MIN_TOKEN_LEN = 6

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def name_probes(name):
    """The lowercase substrings whose presence marks a reference to `name`."""
    probes = {name.lower()}
    faces = [f.strip() for f in name.split(" // ")] if " // " in name else [name]
    for face in faces:
        probes.add(face.lower())
        for token in _TOKEN_RE.findall(face):
            token_l = token.lower().strip("'")
            if len(token_l) >= MIN_TOKEN_LEN and token_l not in TOKEN_STOPWORDS:
                probes.add(token_l)
    return probes


def text_refs(text, deck_names):
    """Deck card names referenced anywhere in `text` (case-insensitive)."""
    haystack = text.lower()
    hits = set()
    for name in deck_names:
        if any(probe in haystack for probe in name_probes(name)):
            hits.add(name)
    return hits


def artifact_card_refs(doc, deck_names):
    """Referenced deck cards across a whole artifact document."""
    return sorted(text_refs(canonical_json(doc), deck_names))


def artifact_card_refs_by_key(doc, keys, deck_names):
    """Per-owned-key refs for artifacts shared between routines.

    `manual_prose.json`'s `card_roles` is keyed by exact card names; those
    keys serialize into the JSON, so the plain matcher covers them too.
    """
    return {
        key: sorted(text_refs(canonical_json(doc.get(key)), deck_names))
        for key in sorted(keys)
    }


def deck_card_names(deck_doc):
    """All card names in a cards.json document (main + sideboard)."""
    return sorted({c["name"] for c in deck_doc.get("cards", []) if c.get("name")})
