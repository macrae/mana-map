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
from collections import Counter
from functools import lru_cache


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
    # Creature types. A resolution quoting a type line ("Zombie Cleric",
    # "Phyrexian Horror") is not naming a card, and `cleric` alone was enough
    # to make six stacks look like they referenced Starscape Cleric.
    "cleric", "wizard", "warrior", "soldier", "zombie", "skeleton", "elemental",
    "horror", "shaman", "assassin", "samurai", "warlock", "treefolk",
    "aetherborn", "insect", "scarab", "snake",
    # Rules vocabulary. Stack prose is made of these words; none of them names
    # a card on its own, and full-name matching still covers any card that
    # happens to contain one.
    "artifact", "enchantment", "instant", "sorcery", "permanent", "permanents",
    "graveyard", "library", "opponent", "opponents", "sacrifice", "sacrificed",
    "trigger", "triggers", "triggered", "ability", "abilities", "damage",
    "counter", "counters", "target", "targets", "untapped", "activate",
    "activated", "resolve", "resolves", "player", "players",
})

MIN_TOKEN_LEN = 6

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _distinctive_tokens(name):
    """The candidate tokens of one name, before ambiguity is considered."""
    faces = [f.strip() for f in name.split(" // ")] if " // " in name else [name]
    out = set()
    for face in faces:
        for token in _TOKEN_RE.findall(face):
            token_l = token.lower().strip("'")
            if len(token_l) >= MIN_TOKEN_LEN and token_l not in TOKEN_STOPWORDS:
                out.add(token_l)
    return out


@lru_cache(maxsize=32)
def ambiguous_tokens(deck_names):
    """Tokens carried by MORE THAN ONE card in this deck.

    A token two deck cards share distinguishes neither of them, so counting it
    for either is a false positive by construction — and the cost is not
    theoretical. `yawgmoth` is a token of both `Yawgmoth, Thran Physician` and
    `Urborg, Tomb of Yawgmoth`, so every artifact naming the commander appeared
    to reference Urborg; moving Urborg between zones then MISSed all thirteen
    verified stacks, a decision spread and the strategic frame at once. `cleric`
    did the same for `Starscape Cleric` against Mikaeus and Yawgmoth's type lines.

    Dropping them is safe in the direction that matters: full-name and per-face
    matching still apply, so a card whose every token is ambiguous still matches
    on its own name. The stopword list could never fix this — the collisions are
    deck-specific, and `yawgmoth` is the most distinctive word on this deck.

    `deck_names` must be a tuple so the cache can key on it.
    """
    counts = Counter()
    for name in deck_names:
        counts.update(_distinctive_tokens(name))
    return frozenset(token for token, n in counts.items() if n > 1)


def name_probes(name, ambiguous=frozenset()):
    """The lowercase substrings whose presence marks a reference to `name`."""
    probes = {name.lower()}
    faces = [f.strip() for f in name.split(" // ")] if " // " in name else [name]
    for face in faces:
        probes.add(face.lower())
    probes |= {t for t in _distinctive_tokens(name) if t not in ambiguous}
    return probes


def text_refs(text, deck_names):
    """Deck card names referenced anywhere in `text` (case-insensitive)."""
    haystack = text.lower()
    ambiguous = ambiguous_tokens(tuple(sorted(deck_names)))
    hits = set()
    for name in deck_names:
        if any(probe in haystack for probe in name_probes(name, ambiguous)):
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
