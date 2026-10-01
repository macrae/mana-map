"""Pilot: form-check `edhrec_cards.json` — EDHREC's commander-page figures, dated.

★-tier evidence like `deck_recon.json`, so the gate is about FORM and REALITY: every card
named resolves in the corpus (DFC front face allowed), `as_of` is a date, every URL is on
EDHREC's JSON host, `synergy` is inside [-1, 1] and the counts are non-negative. It says
nothing about whether a figure is current — that is `as_of`'s job and the reader's.
"""
import sys
from datetime import date

from manamap.pilot.card_pool import corpus_names
from manamap.pilot.common import deck_dir, load_json, report_errors

ARTIFACT = "edhrec_cards.json"
HOST = "https://json.edhrec.com/"
REQUIRED = ("slug", "commander", "as_of", "themes", "cards")


def validate(slug, doc):
    errors = []
    for k in REQUIRED:
        if k not in doc:
            errors.append(f"missing required key {k!r}")
    if errors:
        return errors
    if doc["slug"] != slug:
        errors.append(f"slug is {doc['slug']!r} but the artifact lives in {slug}/")
    try:
        date.fromisoformat(str(doc["as_of"]))
    except (TypeError, ValueError):
        errors.append(f"as_of {doc['as_of']!r} is not an ISO date")
    themes = doc.get("themes") or {}
    if not isinstance(themes, dict) or not themes:
        errors.append("themes is empty — a page was fetched or the file should not exist")
    for th, meta in (themes.items() if isinstance(themes, dict) else []):
        if not str((meta or {}).get("url") or "").startswith(HOST):
            errors.append(f"themes.{th}: url {meta.get('url')!r} is not on {HOST}")
    names = corpus_names()
    cards = doc.get("cards") or {}
    if not isinstance(cards, dict) or not cards:
        errors.append("cards is empty")
    if names:
        for name, row in cards.items():
            if name not in names and name.split(" // ")[0] not in names:
                errors.append(f"cards[{name!r}]: not in the corpus")
            syn = (row or {}).get("synergy")
            if syn is not None and not (-1.0 <= float(syn) <= 1.0):
                errors.append(f"cards[{name!r}]: synergy {syn} outside [-1, 1]")
            for k in ("num_decks", "potential_decks"):
                v = (row or {}).get(k)
                if v is not None and (not isinstance(v, (int, float)) or v < 0):
                    errors.append(f"cards[{name!r}]: {k} {v!r} is not a non-negative count")
            if not (row or {}).get("by_theme"):
                errors.append(f"cards[{name!r}]: no by_theme block — which page listed it?")
    return errors


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / ARTIFACT
    if not path.is_file():
        print(f"{slug}: no {ARTIFACT} — EDHREC not fetched for this deck (absent means absent)")
        return
    doc = load_json(path) or {}
    errors = validate(slug, doc)
    report_errors(f"{slug} — {ARTIFACT}", errors)
    if not errors:
        print(f"OK   {slug} — {ARTIFACT} as_of {doc.get('as_of')}: {len(doc.get('cards') or {})} card(s) "
              f"across {', '.join(doc.get('themes') or {})}")
    sys.exit(1 if errors else 0)
