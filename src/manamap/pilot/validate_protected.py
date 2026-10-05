"""Pilot: form-check `protected.json` — the cards the pilot keeps.

Every entry names a card that is in the deck's current 99 (a protection on a card
the deck does not run protects nothing and reads as a promise), is not the commander
(a commander cannot be swapped out at all), and says why. `at` is a date when given.
"""
import sys
from datetime import date

from manamap.pilot.common import deck_dir, load_deck_cards, load_json, report_errors
from manamap.pilot.protected import ARTIFACT, _key


def validate(slug, doc):
    errors = []
    cards = doc.get("cards")
    if not isinstance(cards, list) or not cards:
        return ["cards must be a non-empty list — delete the file to protect nothing"]
    try:
        deck = load_deck_cards(slug)
    except FileNotFoundError:
        deck = None
    rows = (deck.get("cards") if isinstance(deck, dict) else deck) or []
    in_deck = {_key(c.get("name")): c for c in rows}
    seen = set()
    for i, c in enumerate(cards):
        if not isinstance(c, dict) or not c.get("name"):
            errors.append(f"cards[{i}]: needs a name")
            continue
        k = _key(c["name"])
        if k in seen:
            errors.append(f"cards[{i}]: {c['name']!r} is listed twice")
        seen.add(k)
        if not str(c.get("why") or "").strip():
            errors.append(f"cards[{i}] ({c['name']}): no why — say what the card does for the pilot")
        if c.get("at"):
            try:
                date.fromisoformat(str(c["at"]))
            except ValueError:
                errors.append(f"cards[{i}] ({c['name']}): at {c['at']!r} is not an ISO date")
        if deck is not None:
            row = in_deck.get(k)
            if row is None:
                errors.append(f"cards[{i}]: {c['name']!r} is not in {slug}'s 99")
            elif row.get("is_commander"):
                errors.append(f"cards[{i}]: {c['name']!r} is the commander — it cannot be cut anyway")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / ARTIFACT
    if not path.is_file():
        print(f"{slug}: no {ARTIFACT} — nothing protected (absent means absent)")
        return
    doc = load_json(path) or {}
    errors = validate(slug, doc)
    report_errors(f"{slug} — {ARTIFACT}", errors)
    print(f"OK   {slug} — {ARTIFACT}: {len(doc['cards'])} protected "
          f"({', '.join(c['name'] for c in doc['cards'])})")
    sys.exit(0)
