"""Pilot: the cards a deck may not lose — authored by the pilot, refused everywhere.

THE CUT LIST NEVER KNEW WHAT THE PILOT LOVES. edgar-vampires/draw-v1 staged Vish
Kal, Blood Arbiter out for Morbid Opportunist on "0 activations on record" — Forge
AI behaviour, not a judgement about the card — and nothing read the OUT side of
that branch before eight hours of games. Vish Kal is the pilot's favourite card in
the deck; the run then measured life gained 31.5 -> 21.3 a game (2026-10-04).

`protected.json` is the pilot's claim, written by hand and by nothing else:

    {"cards": [{"name": "Vish Kal, Blood Arbiter",
                "why": "the pilot's favourite: lifelink, counters, removal",
                "at": "2026-10-04"}]}

Absent means nothing is protected. Every path that can take a card out of a deck
asks `refusals` — `deck-branch stage / new / propose / merge`, `try`, the build's
must-include set, the `candidates` auto-cut and the diagnosis/prescription gates —
so the refusal has one home and one wording.
"""
from manamap.pilot.common import deck_dir, load_json

ARTIFACT = "protected.json"


def _key(name):
    # The resolver's vocabulary is the full DFC name; a pilot types the front face.
    return str(name or "").split(" // ")[0].strip().lower()


def load(slug):
    """[{name, why, at}], or [] when the deck has no protected.json."""
    try:
        path = deck_dir(slug) / ARTIFACT
    except FileNotFoundError:        # no deck directory at all: nothing protected
        return []
    if not path.is_file():
        return []
    doc = load_json(path) or {}
    return [c for c in (doc.get("cards") or []) if isinstance(c, dict) and c.get("name")]


def names(slug):
    return [c["name"] for c in load(slug)]


def refusals(slug, outs):
    """One line per protected card in `outs` (names, any DFC spelling). [] when clear."""
    by_key = {_key(c["name"]): c for c in load(slug)}
    lines, seen = [], set()
    for name in outs or []:
        hit = by_key.get(_key(name))
        if hit and _key(name) not in seen:
            seen.add(_key(name))
            why = str(hit.get("why") or "").split(". ")[0].rstrip(".")
            lines.append(f"{hit['name']} is PROTECTED on {slug} — the pilot keeps it"
                         f"{' (' + why + ')' if why else ''}. "
                         f"Release it only by editing data/decks/{slug}/{ARTIFACT}.")
    return lines


def refuse(slug, outs, what="that change"):
    """SystemExit with every protected cut named, or return quietly."""
    lines = refusals(slug, outs)
    if lines:
        raise SystemExit(f"Refusing {what}:\n  - " + "\n  - ".join(lines))
