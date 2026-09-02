"""`log_causes.json` — how each logged game ended, and whether the file still
describes the log it annotates.

WHY IT IS A SEPARATE FILE FROM THE LOG. `log.jsonl` is append-only and never
rewritten (`deck_notes` module docstring), and nine games were logged before a
cause existed. Putting the field on the entry would mean rewriting those lines —
breaking the one contract the log has — or leaving it permanently absent on most
of the evidence. So it is a sidecar keyed by entry id, the same join
`log_annotations.json` uses.

WHY IT NEEDS A GATE. It is authored, tracked, and joined by id to a file that
only grows — which is exactly the shape that goes quietly wrong. Three ways:

  1. **AN ID THAT NAMES NOTHING.** A cause filed against `004` when the log
     stops at `003` counts toward nothing and appears in no table. It does not
     error anywhere; the roll-up is simply short and looks fine.

  2. **A CAUSE OUTSIDE THE VOCABULARY.** The whole reason `CAUSES` is closed is
     that "comboed" and "combo'd" silently split one count into two while the
     table still renders. A free-text field here would undo the point of the
     field.

  3. **A CAUSE ON A GAME THAT WAS WON.** `won` is in the vocabulary because a
     win has a cause too, but filing `wipe` against a `--result win` entry is a
     contradiction between two authored claims about one game, and only a gate
     that reads both can see it.

WHAT IT DOES NOT CHECK. Whether the cause is TRUE. That is the pilot's claim
about their own game and nothing in this repo can second-guess it — the file
exists precisely so the claim is recorded rather than inferred.
"""

import json

from manamap.pilot.common import deck_dir, report_errors
from manamap.pilot.deck_notes import CAUSES, CAUSES_FILE, read_log

ARTIFACT = CAUSES_FILE

#: A win whose cause is not `won` is two authored claims that disagree. Losses
#: are deliberately unconstrained in the other direction: `won` on a loss is
#: caught below, but every other cause is legal on a loss and on a draw.
_WIN_CAUSE = "won"


def validate(doc, entries=None):
    """Check a `log_causes.json` against the log it annotates. Returns errors."""
    errors = []
    if not isinstance(doc, dict):
        return [f"{ARTIFACT} must be an object, got {type(doc).__name__}"]
    rows = doc.get("entries")
    if rows is None:
        errors.append("no `entries` key — `set_cause` always writes one, so its "
                      "absence means this file was hand-edited")
        rows = {}
    elif not isinstance(rows, dict):
        return errors + [f"entries must be an object keyed by log id, got "
                         f"{type(rows).__name__}"]

    by_id = {e["id"]: e for e in (entries or [])}
    for entry_id, row in sorted(rows.items()):
        where = f"entries[{entry_id!r}]"
        if not isinstance(row, dict):
            errors.append(f"{where} must be an object")
            continue
        cause = row.get("cause")
        if cause not in CAUSES:
            errors.append(
                f"{where}.cause {cause!r} is not one of {sorted(CAUSES)} — a "
                f"cause outside the vocabulary counts toward nothing while the "
                f"table still renders")
        if entries is not None and entry_id not in by_id:
            errors.append(
                f"{where} names no log entry — the log has "
                f"{sorted(by_id) or 'no entries'}. A cause filed against a "
                f"missing id is invisible in every roll-up")
            continue
        if entries is not None:
            result = by_id[entry_id].get("result")
            if result == "win" and cause not in (None, _WIN_CAUSE):
                errors.append(
                    f"{where}.cause is {cause!r} but the entry is a WIN — two "
                    f"authored claims about one game that disagree")
            if result in ("loss", "draw") and cause == _WIN_CAUSE:
                errors.append(
                    f"{where}.cause is {_WIN_CAUSE!r} but the entry is a "
                    f"{result} — same contradiction, other direction")
    return errors


def main(args):
    path = deck_dir(args.slug) / ARTIFACT
    if not path.exists():
        # ABSENT IS LEGAL, and on most decks it is the honest state: a deck with
        # no games has nothing to file a cause against. `deck-status` reports the
        # artifact as missing; this is not the place to turn that into a defect.
        print(f"OK   {ARTIFACT} for {args.slug} — absent (no causes filed) ◆")
        return
    doc = json.loads(path.read_text())
    errors = validate(doc, entries=read_log(args.slug))
    n = len(doc.get("entries") or {})
    logged = len(read_log(args.slug))
    report_errors(
        f"{ARTIFACT} for {args.slug}", errors,
        f"OK   {ARTIFACT} for {args.slug} — {n} of {logged} game(s) have a "
        f"stated cause ★")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-log-causes <slug>`.")
