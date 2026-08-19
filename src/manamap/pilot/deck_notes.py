"""Pilot: the captain's log — what happened at the table, in the pilot's words.

WHY THIS EXISTS. Every other artifact in `data/decks/<slug>/` describes the deck
as an object: what it holds, what it can do, what a checker proved. Nothing
recorded what happened when it was PLAYED — which seat turned on it, which card
sat dead in hand for six turns, how the pilot felt about the keep. That is the
data the workbench exists to collect, and it was being kept nowhere (or in
`pilot_feedback.md`, read by one agent and written by hand).

THE LOG IS AUTHORED. `log.jsonl` is one JSON object per line, appended by
`manamap pilot deck-notes <slug> add`, and NEVER rewritten by any agent or
command — the same rule `issue.json` lives under. A note is what the pilot said,
stamped with when they said it and which decklist they were holding
(`decklist_sha256` of `decklist.txt` at that moment), so a note written before a
swap can be told from one written after without anyone remembering the date.

THE ANNOTATION IS DERIVED. `log_annotations.json` is the `debrief` agent's
structured reading of each entry — opponents named, cards that over- or
under-performed, takeaways, and `open_questions` routed to the loops that can
settle them. It is merged by id (`merge-debrief`), validated against the log it
annotates (`validate-debrief`), and can be regenerated; the log cannot.

Light structure on the entry itself: `--result win|loss|draw`, `--opponents N`
and repeatable `--tag`. Everything else is free text — the point is that writing
a note costs one sentence, not a form.
"""

import hashlib
import json
import sys
from datetime import datetime

from manamap.pilot.common import deck_dir, load_json

LOG_FILE = "log.jsonl"
ANNOTATIONS_FILE = "log_annotations.json"
RESULTS = ("win", "loss", "draw")


def log_path(slug):
    return deck_dir(slug) / LOG_FILE


def read_log(slug):
    """Every entry, in file order. A malformed line is an error, not a skip —
    a log that silently drops a game is a log you stop trusting."""
    path = log_path(slug)
    if not path.exists():
        return []
    entries = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n}: not JSON ({e}) — the log is append-only "
                             f"and hand edits must keep one object per line")
    return entries


def decklist_sha256(slug):
    """The decklist as it stands on disk, hashed the way fetch-deck hashes it.

    `decklist.txt` rather than `cards.json`'s stamp: the note is about the deck
    the pilot was HOLDING, and the text file is the thing that changes when a
    card is swapped, whether or not fetch-deck has run since.
    """
    base = deck_dir(slug)
    text_path = base / "decklist.txt"
    if text_path.exists():
        return hashlib.sha256(text_path.read_bytes()).hexdigest()
    return (load_json(base / "cards.json") or {}).get("decklist_sha256")


def next_id(entries):
    """Zero-padded sequential, like stacks and decisions — a stable key the
    annotation file can point at even if an entry is later hand-edited."""
    return f"{max((int(e['id']) for e in entries), default=0) + 1:03d}"


def append_entry(slug, text, result=None, opponents=None, tags=(), at=None):
    """Append one entry and return it. Appends; never rewrites."""
    text = (text or "").strip()
    if not text:
        raise SystemExit("a note needs text — pass it as an argument, or --file PATH "
                         "(`-` for stdin)")
    if result is not None and result not in RESULTS:
        raise SystemExit(f"--result must be one of {', '.join(RESULTS)}")
    if opponents is not None and not (1 <= int(opponents) <= 7):
        raise SystemExit("--opponents is the number of OTHER players, 1-7")
    entries = read_log(slug)
    entry = {
        "id": next_id(entries),
        "at": at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "decklist_sha256": decklist_sha256(slug),
        "result": result,
        "opponents": int(opponents) if opponents is not None else None,
        "tags": sorted(set(tags or ())),
        "text": text,
    }
    with open(log_path(slug), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def annotations(slug):
    doc = load_json(deck_dir(slug) / ANNOTATIONS_FILE) or {}
    return doc.get("entries") or {}


def _fmt(entry, annotated, width=72):
    head = (f"{entry['id']}  {entry['at'][:16]}  "
            f"{(entry.get('result') or '—'):<4}  "
            f"{('vs ' + str(entry['opponents'])) if entry.get('opponents') else '':<5}  "
            f"{'✓ debriefed' if annotated else '  —        '}")
    tags = f"  [{', '.join(entry.get('tags') or [])}]" if entry.get("tags") else ""
    body = entry["text"].replace("\n", " ")
    if len(body) > width:
        body = body[: width - 1] + "…"
    return f"{head}{tags}\n      {body}"


def main(args):
    slug = args.slug
    action = args.action
    if action == "add":
        text = args.text
        if getattr(args, "file", None):
            text = (sys.stdin.read() if args.file == "-"
                    else open(args.file, encoding="utf-8").read())
        entry = append_entry(slug, text, result=args.result, opponents=args.opponents,
                             tags=args.tag, at=args.at)
        print(f"{slug}: logged entry {entry['id']} at {entry['at']} "
              f"(deck {str(entry['decklist_sha256'])[:12]}…)")
        print("  next: `manamap pilot deck-notes "
              f"{slug} list`, or spawn `debrief` to annotate it")
        return

    entries = read_log(slug)
    if action == "show":
        want = [e for e in entries if e["id"] == args.text]
        if not want:
            raise SystemExit(f"{slug}: no log entry {args.text!r} "
                             f"({len(entries)} entries)")
        entry = want[0]
        note = annotations(slug).get(entry["id"])
        if getattr(args, "as_json", False):
            print(json.dumps({"entry": entry, "annotation": note}, indent=2,
                             ensure_ascii=False))
            return
        for k in ("id", "at", "decklist_sha256", "result", "opponents", "tags"):
            print(f"{k:<16} {entry.get(k)}")
        print()
        print(entry["text"])
        if note:
            print()
            print("— debrief —")
            print(json.dumps(note, indent=2, ensure_ascii=False))
        return

    # list (the default)
    since = getattr(args, "since", None)
    if since:
        entries = [e for e in entries if e["at"][: len(since)] >= since]
    done = annotations(slug)
    if getattr(args, "as_json", False):
        print(json.dumps({"slug": slug, "entries": entries,
                          "debriefed": sorted(done)}, indent=2, ensure_ascii=False))
        return
    if not entries:
        print(f"{slug}: no log entries — `manamap pilot deck-notes {slug} add \"…\"`")
        return
    print(f"CAPTAIN'S LOG — {slug} ({len(entries)} entries, "
          f"{sum(1 for e in entries if e['id'] in done)} debriefed)\n")
    for e in entries:
        print(_fmt(e, e["id"] in done))
    wins = sum(1 for e in entries if e.get("result") == "win")
    losses = sum(1 for e in entries if e.get("result") == "loss")
    if wins or losses:
        print(f"\n  {wins}W {losses}L over {len(entries)} logged")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-notes <slug> add|list|show`.")
