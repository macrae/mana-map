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

Light structure on the entry itself: `--result win|loss|draw`, `--opponents N`,
repeatable `--tag`, and since 2026-10-18 an OPTIONAL survey — `--feel power=4
--feel fun=5`, five 1-5 scores from `config.PILOT_FEEL_KEYS`, written on the entry
as `feel: {key: int}` only when given. Everything else is free text — the point is
that writing a note costs one sentence, not a form. There is no verb to add a feel
to an older entry: the log is append-only, and a score given at the table is a
different thing from one reconstructed a week later. An entry without `feel`
simply lacks the key — absent, never zero — and every reader treats it so.
"""

import hashlib
import json
import sys
from datetime import datetime

from manamap import config
from manamap.pilot.common import deck_dir, load_json

LOG_FILE = "log.jsonl"
ANNOTATIONS_FILE = "log_annotations.json"
CAUSES_FILE = "log_causes.json"
RESULTS = ("win", "loss", "draw")
#: THE SURVEY — `config.PILOT_FEEL_KEYS`, re-exported so the registry derives its
#: choices from one place (the `CAUSES` discipline).
FEEL = config.PILOT_FEEL_KEYS
FEEL_RANGE = config.PILOT_FEEL_RANGE
#: THE MARKS (2026-10-21) — `config.PILOT_MARK_KEYS`: per-game counts the pilot
#: writes, kept in a sidecar like the causes (the log is append-only and the games
#: were logged before the field existed). `{key: (what, lo, hi)}`.
MARKS = config.PILOT_MARK_KEYS
MARKS_FILE = "log_marks.json"

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
    from manamap.pilot import common

    # ONE DEFINITION (`common.decklist_sha256`). This hashed `read_bytes` while
    # `deck_versions` hashed `read_text().encode()`, so a CRLF decklist would
    # have joined a log entry to no version at all.
    return common.decklist_sha256(slug) or common.measured_sha(slug)


def next_id(entries):
    """Zero-padded sequential, like stacks and decisions — a stable key the
    annotation file can point at even if an entry is later hand-edited."""
    return f"{max((int(e['id']) for e in entries), default=0) + 1:03d}"


def parse_feel(items):
    """`["power=4", "fun=5"]` -> `{"power": 4, "fun": 5}`, or a SystemExit naming
    the first thing wrong: an unknown key, a non-integer, a value outside
    `FEEL_RANGE`, or a key given twice. `None` or `[]` -> `None` (no survey)."""
    if not items:
        return None
    lo, hi = FEEL_RANGE
    feel = {}
    for item in items:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or key not in FEEL:
            raise SystemExit(
                f"--feel takes KEY=N with KEY one of: "
                + ", ".join(sorted(FEEL)) + f" (got {item!r})\n  "
                + "\n  ".join(f"{k:12s} {v}" for k, v in FEEL.items()))
        try:
            val = int(raw.strip())
        except ValueError:
            raise SystemExit(f"--feel {key}: N must be a whole number {lo}-{hi} (got {raw!r})")
        if not lo <= val <= hi:
            raise SystemExit(f"--feel {key}: N must be {lo}-{hi} (got {val})")
        if key in feel:
            raise SystemExit(f"--feel {key} given twice")
        feel[key] = val
    return feel


def check_feel(feel):
    """The shape `append_entry` writes and `validate-log` re-checks: a dict of
    known keys to ints in range. Returns the first problem as a string, or None."""
    if not isinstance(feel, dict) or not feel:
        return "feel must be a non-empty object"
    lo, hi = FEEL_RANGE
    for k, v in feel.items():
        if k not in FEEL:
            return f"feel key {k!r} is not one of {', '.join(sorted(FEEL))}"
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            return f"feel {k} must be a whole number {lo}-{hi} (got {v!r})"
    return None


def feel_summary(entries):
    """Per key with at least one score: `{mean, n, min, max}`. A mean is not a
    result, so n, min and max ride beside it; keys nobody scored are absent."""
    out = {}
    for key in FEEL:
        vals = [e["feel"][key] for e in entries
                if isinstance(e.get("feel"), dict) and isinstance(e["feel"].get(key), int)]
        if vals:
            out[key] = {"mean": round(sum(vals) / len(vals), 2), "n": len(vals),
                        "min": min(vals), "max": max(vals)}
    return out


def append_entry(slug, text, result=None, opponents=None, tags=(), at=None, feel=None):
    """Append one entry and return it. Appends; never rewrites."""
    text = (text or "").strip()
    if not text:
        raise SystemExit("a note needs text — pass it as an argument, or --file PATH "
                         "(`-` for stdin)")
    if result is not None and result not in RESULTS:
        raise SystemExit(f"--result must be one of {', '.join(RESULTS)}")
    if opponents is not None and not (1 <= int(opponents) <= 7):
        raise SystemExit("--opponents is the number of OTHER players, 1-7")
    if feel is not None:
        why = check_feel(feel)
        if why:
            raise SystemExit(why)
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
    # THE SURVEY RIDES ON THE ENTRY, and only when given: an absent key is how
    # every reader learns there was no survey, so `None` is never written.
    if feel:
        entry["feel"] = {k: feel[k] for k in FEEL if k in feel}
    with open(log_path(slug), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def unsaved_warning(slug):
    """One line when the deck has edits no saved version holds, else None.

    The entry is stamped with the WORKING list's sha (`decklist_sha256`), and a
    list that moved since the last `save-version` is not a version yet — the
    game joins to no version until one is saved. A warning, never a block: the
    note is the pilot's and is written either way."""
    from manamap.pilot import deck_edit
    try:
        u = deck_edit.unsaved(slug)
    except (Exception, SystemExit):                        # noqa: BLE001 - a hint, never fatal
        return None
    if not u:
        return None
    ins, outs = sum(u["in"].values()), sum(u["out"].values())
    return (f"  warning: {slug} has unsaved edits (+{ins} −{outs} since {u['since']}) — "
            f"this game attaches to the WORKING list, which is not a saved version yet; "
            f"`manamap pilot save-version {slug} --note \"…\"` first to tie it to one")


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


# ── The marks: the pilot's counts, a sidecar like the cause ──────────────
#
# WHY COUNTS AND NOT A SURVEY. The feel is a mood on a scale; a mark is a FACT about
# one game the goldfish cannot see — the turn Shabraz and Brallin were both out, the
# turn the first wheel fired and whether a commander was there to be paid, what the
# seven he kept held, how often a commander was removed. Authored by the pilot, never
# inferred from his prose (the debrief may quote a mark, never write one), and the
# instrument he trusts when he does not trust the goldfish (sharknado, 2026-10-21).


def parse_marks(items):
    """`["first_wheel_turn=6", "kept_wheels=4"]` -> `{key: int}`, or a SystemExit
    naming the first thing wrong. `None` or `[]` -> `None` (no marks)."""
    if not items:
        return None
    marks = {}
    for item in items:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or key not in MARKS:
            raise SystemExit(
                f"--mark takes KEY=N with KEY one of: " + ", ".join(sorted(MARKS))
                + f" (got {item!r})\n  "
                + "\n  ".join(f"{k:22s} {v[0]} [{v[1]}-{v[2]}]" for k, v in MARKS.items()))
        what, lo, hi = MARKS[key]
        try:
            val = int(raw.strip())
        except ValueError:
            raise SystemExit(f"--mark {key}: N must be a whole number {lo}-{hi} (got {raw!r})")
        if not lo <= val <= hi:
            raise SystemExit(f"--mark {key}: N must be {lo}-{hi} (got {val}) — {what}")
        if key in marks:
            raise SystemExit(f"--mark {key} given twice")
        marks[key] = val
    return marks


def check_marks(marks):
    """The shape the sidecar holds and `validate-log` re-checks. First problem
    as a string, or None."""
    if not isinstance(marks, dict) or not marks:
        return "marks must be a non-empty object"
    for k, v in marks.items():
        if k not in MARKS:
            return f"mark key {k!r} is not one of {', '.join(sorted(MARKS))}"
        _what, lo, hi = MARKS[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            return f"mark {k} must be a whole number {lo}-{hi} (got {v!r})"
    return None


def marks(slug):
    """`{entry_id: {"marks": {key: int}, "at": iso}}`. Authored."""
    doc = load_json(deck_dir(slug) / MARKS_FILE) or {}
    return doc.get("entries") or {}


def set_marks(slug, entry_id, new):
    """Record the pilot's counts for one logged game — ONE WRITER for `add --mark`
    and the `mark` verb. Marks given later MERGE over earlier ones for the same
    game (a count he adds the morning after is not a retraction of the ones he
    gave at the table)."""
    why = check_marks(new)
    if why:
        raise SystemExit(why)
    ids = {e["id"] for e in read_log(slug)}
    if entry_id not in ids:
        raise SystemExit(
            f"{slug}: no log entry {entry_id!r} — "
            f"`manamap pilot deck-notes {slug} list` shows the ids")
    path = deck_dir(slug) / MARKS_FILE
    doc = load_json(path) or {"slug": slug, "entries": {}}
    doc.setdefault("slug", slug)
    doc.setdefault("entries", {})
    prev = (doc["entries"].get(entry_id) or {}).get("marks") or {}
    doc["entries"][entry_id] = {
        "marks": {k: v for k, v in sorted({**prev, **new}.items())},
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    doc["entries"] = dict(sorted(doc["entries"].items()))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc["entries"][entry_id]


def marks_summary(entries, sidecar):
    """Per key with at least one mark: `{mean, n, min, max}` over the entries
    given, plus two derived rates the thesis is about. A mean is not a result, so
    n, min and max ride beside it; keys nobody marked are absent. `0 = never` keys
    (`commander_turn`, `both_turn`, `first_wheel_turn`) report their mean over the
    games where it happened and carry `never` as a count beside it."""
    rows = [(e["id"], (sidecar.get(e["id"]) or {}).get("marks") or {}) for e in entries]
    out = {}
    for key, (_what, lo, hi) in MARKS.items():
        vals = [m[key] for _id, m in rows if isinstance(m.get(key), int)]
        if not vals:
            continue
        cell = {"n": len(vals), "min": min(vals), "max": max(vals)}
        if key in ("commander_turn", "both_turn", "first_wheel_turn"):
            hit = [v for v in vals if v > 0]
            cell["never"] = len(vals) - len(hit)
            cell["mean"] = round(sum(hit) / len(hit), 2) if hit else None
        else:
            cell["mean"] = round(sum(vals) / len(vals), 2)
        out[key] = cell
    fired = [m for _id, m in rows if isinstance(m.get("wheel_with_commander"), int)]
    if fired:
        out["wheel_with_commander_rate"] = round(
            sum(1 for m in fired if m["wheel_with_commander"] == 1) / len(fired), 3)
    kept = [m for _id, m in rows
            if isinstance(m.get("kept_wheels"), int) and isinstance(m.get("kept_bodies"), int)]
    if kept:
        # The pilot's own keep rule as a count: at most one wheel and a body or ramp.
        out["kept_hand_passes_rule_rate"] = round(
            sum(1 for m in kept if m["kept_wheels"] <= 1 and m["kept_bodies"] >= 1) / len(kept), 3)
    return out


def _fmt(entry, annotated, width=72):
    head = (f"{entry['id']}  {entry['at'][:16]}  "
            f"{(entry.get('result') or '—'):<4}  "
            f"{('vs ' + str(entry['opponents'])) if entry.get('opponents') else '':<5}  "
            f"{'✓ debriefed' if annotated else '  —        '}")
    tags = f"  [{', '.join(entry.get('tags') or [])}]" if entry.get("tags") else ""
    feel = entry.get("feel") or {}
    if feel:
        tags += "  " + " ".join(f"{k[:3]}{feel[k]}" for k in FEEL if k in feel)
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
                             tags=args.tag, at=args.at,
                             feel=parse_feel(getattr(args, "feel", None)))
        print(f"{slug}: logged entry {entry['id']} at {entry['at']} "
              f"(deck {str(entry['decklist_sha256'])[:12]}…)")
        if entry.get("feel"):
            print("  feel: " + ", ".join(f"{k} {v}" for k, v in entry["feel"].items()))
        warn = unsaved_warning(slug)
        if warn:
            print(warn, file=sys.stderr)
        # ONE WRITER. `--cause` goes through `set_cause`, exactly as the
        # standalone verb does, so the two can never disagree about the shape of
        # the sidecar or skip its vocabulary check.
        if getattr(args, "cause", None):
            row = set_cause(slug, entry["id"], args.cause)
            print(f"  cause: {row['cause']} — {CAUSES[row['cause']]}")
        # THE MARKS, through the one writer the `mark` verb uses.
        _m = parse_marks(getattr(args, "mark", None))
        if _m:
            row = set_marks(slug, entry["id"], _m)
            print("  marks: " + ", ".join(f"{k} {v}" for k, v in row["marks"].items()))
        print("  next: `manamap pilot deck-notes "
              f"{slug} list`, or spawn `debrief` to annotate it")
        return

    if action == "mark":
        # `deck-notes <slug> mark <id> --mark KEY=N …`: the pilot's counts for a
        # game already logged. `text` is the id, as `show` and `cause` read it.
        _m = parse_marks(getattr(args, "mark", None))
        if not _m:
            print("The pilot's counts for one game. `--mark KEY=N`, repeatable, KEY one of:\n  "
                  + "\n  ".join(f"{k:22s} {v[0]} [{v[1]}-{v[2]}]" for k, v in MARKS.items()))
            raise SystemExit(2)
        row = set_marks(slug, args.text, _m)
        print(f"{slug}: entry {args.text} — "
              + ", ".join(f"{k} {v}" for k, v in row["marks"].items()))
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
    feel = feel_summary(entries)
    if feel:
        print("  feel: " + " · ".join(f"{k} {v['mean']:.1f} (n={v['n']})"
                                      for k, v in feel.items()))


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-notes <slug> add|list|show`.")
