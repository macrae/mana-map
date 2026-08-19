"""Pilot: form-check a debrief — the annotation held to the log it annotates.

The `debrief` agent reads the pilot's own words and writes a structured reading
of them. The whole risk of that job is invention: an opponent archetype the note
never described, a card the pilot never mentioned, a line stated as fact because
it sounded like one. So the checks here are all of one shape — **the annotation
may name nothing the note and the deck do not**:

- every annotated id is a real log entry (`log.jsonl` is the authority);
- every `opponents[].evidence` is a verbatim phrase of the note;
- every `cards[].card` is in the 99 or written verbatim in the note;
- every `lines[]` is `needs a stack scenario` unless a checker-passed stack is
  cited, and a cited stack is re-read (`check_verified_line`), never trusted;
- every `open_questions[].settled_by` names a loop that exists — the doctor's
  set plus `diagnose`, because a logged game is exactly what should send a deck
  back to the doctor;
- every `engine_stages` entry is a stage the engine model actually declares.

No superlative check, no mood check, no judgment of whether a takeaway is wise:
none of that is mechanically decidable, and a validator that fires on correct
data is worse than none. What this catches is the agent writing past its source.
"""

import json
import re

from manamap.pilot.common import (
    check_verified_line, deck_dir, expand_faces, load_deck_cards, load_json,
    report_errors)
from manamap.pilot.deck_notes import ANNOTATIONS_FILE, read_log
from manamap.pilot.validate_diagnosis import REQUIRED_QUESTION_KEYS, SETTLED_BY

DEBRIEF_SETTLED_BY = SETTLED_BY | {"diagnose"}
CARD_READS = {"over", "under", "as-expected", "missed"}
LINE_STATUSES = {"verified", "needs a stack scenario"}
REQUIRED_ENTRY_KEYS = {"summary", "takeaways"}
_WS = re.compile(r"\s+")


def _norm(s):
    return _WS.sub(" ", str(s or "")).strip().lower()


def validate(slug, doc, log_entries, deck_names, stage_names, deck_path):
    errors = []
    if doc.get("slug") != slug:
        errors.append(f"slug: expected {slug!r}, got {doc.get('slug')!r}")
    entries = doc.get("entries")
    if not isinstance(entries, dict):
        return errors + ["entries: must be an object keyed by log entry id"]

    by_id = {e["id"]: e for e in log_entries}
    for eid, note in sorted(entries.items()):
        where = f"entries[{eid}]"
        if eid not in by_id:
            errors.append(f"{where}: no such log entry — the log is the authority "
                          f"and the annotation cannot add games to it")
            continue
        if not isinstance(note, dict):
            errors.append(f"{where}: must be an object"); continue
        text = _norm(by_id[eid]["text"])
        for k in sorted(REQUIRED_ENTRY_KEYS - set(note)):
            errors.append(f"{where}: missing {k!r}")
        if not str(note.get("summary") or "").strip():
            errors.append(f"{where}.summary: empty")
        if not isinstance(note.get("takeaways", []), list):
            errors.append(f"{where}.takeaways: must be a list")

        for i, opp in enumerate(note.get("opponents") or []):
            ev = opp.get("evidence")
            if not ev:
                errors.append(f"{where}.opponents[{i}]: needs `evidence` — the phrase "
                              f"of the note this reading rests on")
            elif _norm(ev) not in text:
                errors.append(f"{where}.opponents[{i}].evidence: {ev!r} is not a "
                              f"phrase of the note — the debrief may not describe "
                              f"a seat the pilot did not")

        for i, c in enumerate(note.get("cards") or []):
            name = c.get("card")
            if not name:
                errors.append(f"{where}.cards[{i}]: missing `card`"); continue
            in_deck = any(f in deck_names for f in expand_faces(name))
            in_note = any(_norm(f) in text for f in expand_faces(name))
            if not (in_deck or in_note):
                errors.append(f"{where}.cards[{i}]: {name!r} is neither in the 99 nor "
                              f"named in the note — the debrief may not name a card "
                              f"the pilot did not")
            if c.get("read") not in CARD_READS:
                errors.append(f"{where}.cards[{i}].read: {c.get('read')!r} not in "
                              f"{sorted(CARD_READS)}")

        for i, line in enumerate(note.get("lines") or []):
            st = line.get("status")
            if st not in LINE_STATUSES:
                errors.append(f"{where}.lines[{i}].status: {st!r} not in "
                              f"{sorted(LINE_STATUSES)}")
            elif st == "verified":
                errors += check_verified_line(i, line, deck_path,
                                              label=f"{where}.lines")

        for i, q in enumerate(note.get("open_questions") or []):
            for k in sorted(REQUIRED_QUESTION_KEYS - set(q)):
                errors.append(f"{where}.open_questions[{i}]: missing {k!r}")
            if q.get("settled_by") not in DEBRIEF_SETTLED_BY:
                errors.append(f"{where}.open_questions[{i}].settled_by: "
                              f"{q.get('settled_by')!r} not in "
                              f"{sorted(DEBRIEF_SETTLED_BY)}")

        if stage_names is not None:
            for s in note.get("engine_stages") or []:
                if s not in stage_names:
                    errors.append(f"{where}.engine_stages: {s!r} is not a stage in "
                                  f"engine.json ({sorted(stage_names)})")
    return errors


def main(args):
    slug = args.slug
    base = deck_dir(slug)
    path = base / ANNOTATIONS_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found — spawn `debrief` for {slug}, then "
                         f"`merge-debrief`; or nothing has been logged yet "
                         f"(`deck-notes {slug} list`).")
    doc = load_json(path)
    log_entries = read_log(slug)
    deck_names = set()
    for card in load_deck_cards(slug).get("cards", []):
        deck_names |= expand_faces(card.get("name"))
    engine = load_json(base / "engine.json")
    stage_names = ({s.get("stage") for s in engine.get("stages", [])}
                   if engine else None)
    errors = validate(slug, doc, log_entries, deck_names, stage_names, base)
    report_errors(f"{slug} debrief",
                  errors,
                  ok_line=f"OK   {slug} debrief — {len(doc.get('entries') or {})} "
                          f"annotated of {len(log_entries)} logged")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-debrief <slug>`.")
