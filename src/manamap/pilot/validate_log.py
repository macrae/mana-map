"""Pilot: form-check `log.jsonl` — the captain's log, the pilot's own record.

The log had no gate for as long as it existed, because every field on it was a
word the pilot typed and `deck_notes.append_entry` refused the bad ones at the
door. The survey (`feel`, 2026-10-18) is the first field a downstream figure is
COMPUTED from — a mean in `info.json`, a block in the fleet artifact — and a
figure computed from an unchecked file is a figure nobody can trust, so the file
gets the same form gate every other tracked artifact has.

FORM ONLY: one JSON object per line; ids `NNN`, unique and increasing; `result`
from `deck_notes.RESULTS` or null; `opponents` 1-7 or null; `tags` a list of
strings; `feel`, when present, a non-empty object of keys from
`config.PILOT_FEEL_KEYS` to whole numbers in `PILOT_FEEL_RANGE`; the stamp a
64-hex sha or null. Nothing here reads the clock, the deck or the annotations.
Absent file: "absent means absent", exit 0 — a deck nobody has played is not a
defect.
"""

import json
import re

from manamap.pilot import deck_notes
from manamap.pilot.common import deck_dir, report_errors

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^\d{3,}$")
REQUIRED = ("id", "at", "decklist_sha256", "result", "opponents", "tags", "text")


def validate_lines(raw_lines):
    """`(errors, entries)`: every problem named by line number; the parsed
    entries for a caller that wants them."""
    errors, entries, last = [], [], 0
    for n, line in enumerate(raw_lines, 1):
        if not line.strip():
            errors.append(f"line {n}: blank line (one object per line, no blanks)")
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: not JSON ({exc.msg})")
            continue
        if not isinstance(e, dict):
            errors.append(f"line {n}: not an object")
            continue
        at = f"line {n} ({e.get('id')})"
        for k in REQUIRED:
            if k not in e:
                errors.append(f"{at}: missing {k!r}")
        eid = str(e.get("id", ""))
        if not ID_RE.match(eid):
            errors.append(f"{at}: id must be zero-padded digits like 007")
        else:
            if int(eid) <= last:
                errors.append(f"{at}: id {eid} is not after {last:03d} (ids increase)")
            last = max(last, int(eid))
        if e.get("result") is not None and e.get("result") not in deck_notes.RESULTS:
            errors.append(f"{at}: result must be one of {', '.join(deck_notes.RESULTS)} or null")
        opp = e.get("opponents")
        if opp is not None and (isinstance(opp, bool) or not isinstance(opp, int)
                                or not 1 <= opp <= 7):
            errors.append(f"{at}: opponents must be 1-7 or null")
        if not isinstance(e.get("tags"), list) or not all(isinstance(t, str) for t in e.get("tags", [])):
            errors.append(f"{at}: tags must be a list of strings")
        if not isinstance(e.get("text"), str) or not e.get("text", "").strip():
            errors.append(f"{at}: text must be a non-empty string")
        sha = e.get("decklist_sha256")
        if sha is not None and not (isinstance(sha, str) and SHA_RE.match(sha)):
            errors.append(f"{at}: decklist_sha256 must be a 64-hex sha or null")
        if "feel" in e:
            why = deck_notes.check_feel(e["feel"])
            if why:
                errors.append(f"{at}: {why}")
        entries.append(e)
    return errors, entries


def main(args=None):
    slug = getattr(args, "slug", None)
    p = deck_dir(slug) / deck_notes.LOG_FILE
    if not p.is_file():
        print(f"{slug}: no {deck_notes.LOG_FILE} — nothing logged yet (absent means absent)")
        return
    errors, entries = validate_lines(p.read_text(encoding="utf-8").splitlines())
    report_errors(f"{slug} — {deck_notes.LOG_FILE}", errors)
    feel = deck_notes.feel_summary(entries)
    surveyed = sum(1 for e in entries if e.get("feel"))
    print(f"OK   {slug} — {deck_notes.LOG_FILE}: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, "
          f"{surveyed} with a survey"
          + (" (" + ", ".join(f"{k} {v['mean']:.1f}·{v['n']}" for k, v in feel.items()) + ")" if feel else "")
          + "; authored ★")
