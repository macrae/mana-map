"""Pilot: candidate watch lists (2026-10-07) — cards Sean is reviewing for a deck.

`data/decks/<slug>/watchlist.json` holds named SETS of candidates, each from one
search or queue result (`source: "Q002"`), with a one-line `why`, which commander's
trigger the card's own discards and draws feed (`pays`), and Sean's verdict on it.
A watch list is not a branch: nothing here touches the 99. It is the step BEFORE
staging, where Sean reads the cards and decides which deserve a closer look.

Written by `add_set` (Jarvis, from a queue result Sean chose to watch) and `mark`
(Sean, from the CLI or the Atlas's review grid through `serve`'s `watch/mark`).
Mana value and oracle text are NOT stored: the page reads them from the corpus
record, so they cannot drift from the card.
"""
import datetime
import json

from manamap.pilot.common import deck_dir, load_json

ARTIFACT = "watchlist.json"
PAYS = ("both", "brallin", "shabraz", "none", "n/a")
AXES = ("interaction", "momentum", "ramp", "protection", "other")
VERDICTS = ("unreviewed", "watching", "pass")


def path(slug):
    return deck_dir(slug) / ARTIFACT


def load(slug):
    return load_json(path(slug)) or {"slug": slug, "sets": []}


def _save(slug, doc):
    from manamap.pilot import validate_watchlist

    errors = validate_watchlist.validate(slug, doc)
    if errors:
        raise SystemExit(f"FAIL {ARTIFACT}: " + "; ".join(errors[:5]) + " — nothing was written")
    path(slug).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def _set(doc, set_id):
    s = next((s for s in doc["sets"] if s["id"] == set_id), None)
    if s is None:
        raise SystemExit(f"no set {set_id!r} — sets: {', '.join(x['id'] for x in doc['sets']) or 'none'}")
    return s


def add_set(slug, set_id, title, cards, source=None, query=None, created=None):
    doc = load(slug)
    if any(s["id"] == set_id for s in doc["sets"]):
        raise SystemExit(f"set {set_id!r} exists — `mark` changes its cards")
    doc["sets"].append({
        "id": set_id, "title": title, "source": source, "query": query,
        "created": created or datetime.date.today().isoformat(),
        "cards": [{"name": c["name"], "pays": c.get("pays", "n/a"), "axis": c.get("axis", "other"),
                   "why": c["why"], "verdict": "unreviewed", "note": None, "at": None} for c in cards],
    })
    return _save(slug, doc)


def mark(slug, set_id, card, verdict=None, note=None):
    """Sean's verdict and/or note on one card. `at` is stamped on every change."""
    doc = load(slug)
    s = _set(doc, set_id)
    row = next((c for c in s["cards"] if c["name"].lower() == str(card).lower()), None)
    if row is None:
        raise SystemExit(f"{card!r} is not in set {set_id!r}")
    if verdict is not None:
        if verdict not in VERDICTS:
            raise SystemExit(f"verdict is one of {', '.join(VERDICTS)}")
        row["verdict"] = verdict
    if note is not None:
        row["note"] = note.strip() or None
    row["at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    _save(slug, doc)
    return row


def counts(s):
    out = {v: 0 for v in VERDICTS}
    for c in s["cards"]:
        out[c["verdict"]] += 1
    return out


def summary(slug, source):
    """`'watching 4 of 25'` for the set made from a queue item, or None."""
    p = path(slug)
    if not p.exists():
        return None
    for s in load(slug)["sets"]:
        if s.get("source") == source:
            n = counts(s)
            return f"watching {n['watching']} of {len(s['cards'])}, {n['pass']} passed"
    return None


def main(args):
    slug = args.slug
    verb = getattr(args, "verb", None) or "list"
    rest = list(getattr(args, "rest", None) or [])
    if verb == "list":
        doc = load(slug)
        if not doc["sets"]:
            print(f"{slug}: nothing on watch")
            return
        for s in doc["sets"]:
            n = counts(s)
            print(f"{s['id']}  {s['title']}  ({len(s['cards'])} cards · {n['watching']} watching · "
                  f"{n['pass']} passed · {n['unreviewed']} unreviewed)"
                  + (f"  from {s['source']}" if s.get("source") else ""))
            for c in s["cards"]:
                mark_ = {"watching": "★", "pass": "✗", "unreviewed": "·"}[c["verdict"]]
                print(f"   {mark_} {c['name']:<34} pays {c['pays']:<8} {c['why'][:70]}"
                      + (f"\n       note: {c['note']}" if c.get("note") else ""))
        return
    if verb == "mark":
        if len(rest) < 3:
            raise SystemExit('mark SET "Card" watching|pass|unreviewed [--note "…"]')
        row = mark(slug, rest[0], rest[1], verdict=rest[2], note=getattr(args, "note", None))
        print(f"{row['name']}: {row['verdict']}" + (f" — {row['note']}" if row.get("note") else ""))
        return
    if verb == "note":
        if len(rest) < 3:
            raise SystemExit('note SET "Card" "the note"')
        row = mark(slug, rest[0], rest[1], note=rest[2])
        print(f"{row['name']}: note saved")
        return
    raise SystemExit("verbs: list, mark, note")
