"""Pilot: the captain's log — what happened at the table, in the pilot's words.

WHY THIS EXISTS. Every other artifact in `data/decks/<slug>/` describes the deck
as an object: what it holds, what it can do, what a checker proved. Nothing
recorded what happened when it was PLAYED — which seat turned on it, which card
sat dead in hand for six turns, how the pilot felt about the keep. That is the
data the workbench exists to collect, and it was being kept nowhere (or in
`pilot_feedback.md`, read by one agent and written by hand).

THE LOG IS AUTHORED. `log.jsonl` is one JSON object per line, appended by
`manamap pilot deck-notes <slug> add`, and NEVER rewritten by any agent or
command — the same rule the deck's authored identity file lives under. A note is
what the pilot said,
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
CAUSES_FILE = "log_causes.json"
RESULTS = ("win", "loss", "draw")

#: HOW THE GAME ENDED, from a closed vocabulary.
#:
#: `--result` says whether you won. This says WHY, and the difference is the
#: whole reason the dossier's game table can show counts instead of paragraphs:
#: three losses to `removal` and three to `mana-drought` are two different decks
#: with the same record, and prose cannot be counted.
#:
#: CLOSED, not free text, and not a `--tag`. A tag is a label the pilot invents
#: per game; the moment "comboed" and "combo'd" both exist the count silently
#: splits in two and the table understates by half while looking fine. The list
#: is short on purpose — a vocabulary nobody can hold in their head gets used
#: wrong, and a cause that fits three games is worth more than one that fits one.
CAUSES = {
    "mana-drought": "colour or land screw — the deck could not cast what it drew",
    "removal": "the engine was picked apart one card at a time",
    "wipe": "a board wipe, and no rebuild",
    "combo": "an opponent assembled a combo and closed",
    "politics": "the table converged, correctly or otherwise",
    "raced": "someone was simply faster",
    "stalled": "the engine never assembled — the pieces did not arrive",
    "won": "the deck closed the game",
}


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


# ── The cause, and why it is not on the entry ────────────────────────────
#
# `log.jsonl` is APPEND-ONLY and never rewritten. Nine games were logged before
# this field existed, so putting the cause on the entry would mean either
# rewriting those lines — breaking the one contract the log has — or leaving the
# field permanently absent on the games that already happened, which is most of
# the evidence there is.
#
# So it is a sidecar, keyed by entry id: the same join `log_annotations.json`
# already uses, under the same rule. The difference from that file is WHO SPEAKS.
# An annotation is the debrief agent's structured reading and can be regenerated;
# a cause is the PILOT'S OWN CLAIM about their own game, authored and never
# derived. No agent writes this file.


def causes(slug):
    """`{entry_id: {"cause": key, "note": str, "at": iso}}`. Authored."""
    doc = load_json(deck_dir(slug) / CAUSES_FILE) or {}
    return doc.get("entries") or {}


def set_cause(slug, entry_id, cause, note=None):
    """Record how one logged game ended. One writer, so `add --cause` and the
    standalone verb cannot disagree about the shape."""
    if cause not in CAUSES:
        raise SystemExit(
            f"{cause!r} is not a cause. One of:\n  "
            + "\n  ".join(f"{k:14s} {v}" for k, v in CAUSES.items()))
    ids = {e["id"] for e in read_log(slug)}
    if entry_id not in ids:
        raise SystemExit(
            f"{slug}: no log entry {entry_id!r} — "
            f"`manamap pilot deck-notes {slug} list` shows the ids")
    path = deck_dir(slug) / CAUSES_FILE
    doc = load_json(path) or {"slug": slug, "entries": {}}
    doc.setdefault("slug", slug)
    doc.setdefault("entries", {})
    doc["entries"][entry_id] = {
        "cause": cause,
        "note": note or "",
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    doc["entries"] = dict(sorted(doc["entries"].items()))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc["entries"][entry_id]


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
        # ONE WRITER. `--cause` goes through `set_cause`, exactly as the
        # standalone verb does, so the two can never disagree about the shape of
        # the sidecar or skip its vocabulary check.
        if getattr(args, "cause", None):
            row = set_cause(slug, entry["id"], args.cause)
            print(f"  cause: {row['cause']} — {CAUSES[row['cause']]}")
        print("  next: `manamap pilot deck-notes "
              f"{slug} list`, or spawn `debrief` to annotate it")
        return

    if action == "cause":
        # `deck-notes <slug> cause <id> <code>`. `text` is the id (the positional
        # `show` already uses) and `--cause` carries the code, so the parser
        # needs no third positional.
        if not getattr(args, "cause", None):
            print(f"How a game ended. `--cause <code>` with one of:\n  "
                  + "\n  ".join(f"{k:14s} {v}" for k, v in CAUSES.items()))
            raise SystemExit(2)
        row = set_cause(slug, args.text, args.cause, note=getattr(args, "note", None))
        print(f"{slug}: entry {args.text} — {row['cause']} ({CAUSES[row['cause']]})")
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
