"""Simulation S3: the pod — opponent seats under data/opponents/, authored from EDHREC.

An opponent is a `decklist.txt` in the repo's own format under `data/opponents/<slug>/`,
and the harness already resolves it first (`sim.forge.seat_dir`). This module fills one
from EDHREC's *average deck* for a commander — the cheapest honest stand-in for "what a
Giada deck at the table looks like" — through its JSON endpoint, and writes a
`source.json` beside it (commander, URL, fetch date, card count) so the list's provenance
is on disk. "Don't work too hard": an average list is a representative opponent, not a
specific one; when you know the actual list at your table, paste it over the generated
one — the harness reads the file, not the source.

Basics arrive as `Plains` ×N and are written as one line with the quantity, which the
repo's parser and Forge's .dck both take. The commander is marked `*CMDR*`.
"""

import json
import re
import urllib.request
from datetime import date

from manamap.config import DECKS_DIR

EDHREC_AVERAGE = "https://json.edhrec.com/pages/average-decks/{slug}.json"
OPPONENTS_DIR = DECKS_DIR.parent / "opponents"


def edhrec_slug(commander):
    s = commander.lower().replace("'", "").replace(",", "")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def fetch_average(commander_or_slug):
    slug = commander_or_slug if re.fullmatch(r"[a-z0-9-]+", commander_or_slug) else edhrec_slug(commander_or_slug)
    url = EDHREC_AVERAGE.format(slug=slug)
    with urllib.request.urlopen(url, timeout=30) as r:
        doc = json.loads(r.read().decode("utf-8"))
    deck = doc.get("deck") or {}
    commanders = deck.get("commander") or []
    cards = []
    for _type, rows in (deck.get("cards") or {}).items():
        for name, qty in rows:
            cards.append((name, int(qty)))
    if not commanders or not cards:
        raise SystemExit(f"EDHREC returned no average deck for {slug!r} ({url})")
    return {"url": url, "slug": slug, "commanders": commanders, "cards": cards}


def decklist_text(avg):
    lines = [f"1 {c} *CMDR*" for c in avg["commanders"]]
    lines += [f"{q} {n}" for n, q in sorted(avg["cards"], key=lambda x: (x[0] in {"Plains", "Island", "Swamp", "Mountain", "Forest"}, x[0]))]
    return "\n".join(lines) + "\n"


def write_opponent(slug, avg, note=None):
    base = OPPONENTS_DIR / slug
    base.mkdir(parents=True, exist_ok=True)
    (base / "decklist.txt").write_text(decklist_text(avg), encoding="utf-8")
    total = sum(q for _, q in avg["cards"]) + len(avg["commanders"])
    (base / "source.json").write_text(json.dumps({
        "slug": slug, "commander": avg["commanders"], "source": "edhrec average deck",
        "url": avg["url"], "fetched": date.today().isoformat(), "cards": total,
        "note": note or "a representative list, not a specific one — paste the real list over "
                        "decklist.txt when you know it"}, indent=2) + "\n")
    return base, total


def main(args):
    if getattr(args, "list", False) or not getattr(args, "commander", None):
        if not OPPONENTS_DIR.is_dir() or not any(OPPONENTS_DIR.iterdir()):
            print("no opponents under data/opponents/ — "
                  "`manamap pilot fetch-opponent \"Giada, Font of Hope\" --as giada-angels`")
            return
        for d in sorted(OPPONENTS_DIR.iterdir()):
            src = d / "source.json"
            meta = json.loads(src.read_text()) if src.exists() else {}
            print(f"{d.name:<18} {', '.join(meta.get('commander') or ['?']):<34} "
                  f"{meta.get('cards', '?')} cards  {meta.get('source', 'authored')}  {meta.get('fetched', '')}")
        return
    avg = fetch_average(args.commander)
    slug = getattr(args, "as_slug", None) or edhrec_slug(avg["commanders"][0])
    base, total = write_opponent(slug, avg, note=getattr(args, "note", None))
    print(f"opponent {slug}: {', '.join(avg['commanders'])} — {total} cards from {avg['url']}")
    print(f"  → {base.relative_to(DECKS_DIR.parent.parent)}/decklist.txt (+ source.json)")
    print(f"  next: `manamap pilot simulate <your-deck> --vs {slug} …`")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot fetch-opponent \"<commander>\" [--as slug]`.")
