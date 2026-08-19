"""Pilot: the queue of changes DECIDED but not yet applied (pending.json).

Nine decks and decisions made in conversation. Before this file the repo had
exactly two homes for a swap, and neither could hold an intention:

  * **applied** — derived from `git log` over `decklist.txt` (`deck_history.py`),
    exact and silent about anything not yet done;
  * **proposed** — `considering.json`, the retired Short List (frozen on the
    published decks since 2026-08-19; prescriptions carry its rule now): exactly
    ten entries, forbade a pick already in the deck, and was regenerated wholesale
    on any decklist edit, so an applied ten left no trace.

So a three-land swap decided on a Tuesday had nowhere to live, and was lost.

`CLAUDE.md`'s "swap history is DERIVED from git, never hand-kept" is narrower
than it reads: its target is a second record of what ALREADY HAPPENED, which git
owns exactly. A queue of un-applied intent cannot disagree with git, because git
has nothing to say about it yet. `deck_history.pending()` exists for that half
already; this only gives it a home that is not somebody else's artifact.

**Closure is DERIVED, never declared.** There is deliberately no `applied: true`
field. An entry is closed when the deck itself says so — every `in` card present
and every `out` card gone — checked against `cards.json`. A hand-set flag is
precisely how `HISTORY.md` became "append-only and append-forgotten", and this
file refuses to repeat it: applying a change makes its entry report APPLIED on
its own, and the entry is then deleted rather than ticked.

Report-only, computed on demand, like `deck-history` and `impact`. It is never
read by the legacy renderer (its "every issue is the reader's first" rule banned a
queue of unmade changes from print); on the bench, history is an input.
"""

import json
import re

from manamap.pilot.common import deck_dir, load_deck_cards, report_errors

REQUIRED_TOP_KEYS = {"slug", "pending"}
REQUIRED_ENTRY_KEYS = {"id", "decided", "why"}

OPEN = "OPEN"
APPLIED = "APPLIED"
PARTIAL = "PARTIAL"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _names(entry, key):
    """`in`/`out` are LISTS — a three-for-three swap is one decision, not three.

    That is the other thing `considering.json` cannot express: its `card` /
    `natural_cut` pair is strictly one-to-one, so a land fix that moves three
    cards would have to be filed as three unrelated picks that each look wrong
    on their own.
    """
    value = entry.get(key) or []
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)]


def state_of(entry, main_names):
    """OPEN / APPLIED / PARTIAL, derived from the deck as it stands right now.

    **The cuts decide it, not the additions.** The first version counted an `in`
    card as arrived whenever the deck contained it — which reports a swap that
    adds three Swamps as half-done before anything happens, because the deck
    already runs twenty-one. Presence cannot prove arrival for a card you
    already play, and the queue does not store the baseline count to compare
    against (storing one would be a second hand-kept number that goes stale the
    moment anything else moves).

    An `out` card is unambiguous: it is in the deck or it is not. So progress is
    measured on the cuts, and the additions only have to be present for the
    entry to count as fully applied. A pure-addition entry with no cuts falls
    back to its `in` cards, which is the one case where presence IS the signal.
    """
    wanted = _names(entry, "in")
    cut = _names(entry, "out")
    if not cut:
        if not wanted:
            return OPEN
        arrived = sum(1 for n in wanted if n in main_names)
        return APPLIED if arrived == len(wanted) else (OPEN if arrived == 0 else PARTIAL)

    gone = sum(1 for n in cut if n not in main_names)
    if gone == 0:
        return OPEN
    if gone == len(cut) and all(n in main_names for n in wanted):
        return APPLIED
    return PARTIAL


def validate(doc, deck_doc, known_settlers=()):
    errors = []
    missing = REQUIRED_TOP_KEYS - set(doc)
    if missing:
        errors.append(f"pending.json missing keys: {sorted(missing)}")
        return errors

    entries = doc.get("pending")
    if not isinstance(entries, list):
        errors.append("pending must be a list (an empty one is a fine state)")
        return errors

    main_names = {c.get("name") for c in (deck_doc or {}).get("cards", [])}
    seen_ids = set()
    for i, entry in enumerate(entries):
        where = f"pending[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        gaps = REQUIRED_ENTRY_KEYS - set(entry)
        if gaps:
            errors.append(f"{where} missing keys: {sorted(gaps)}")
            continue

        eid = entry["id"]
        if not _ID_RE.match(str(eid)):
            errors.append(f"{where} id {eid!r} must be lower-kebab-case")
        if eid in seen_ids:
            errors.append(f"{where} duplicate id {eid!r}")
        seen_ids.add(eid)

        if not _DATE_RE.match(str(entry["decided"])):
            errors.append(f"{where} decided {entry['decided']!r} must be YYYY-MM-DD")
        if not str(entry.get("why") or "").strip():
            errors.append(f"{where} why is empty — a queued change without a "
                          f"reason is a diff nobody can review later")

        if not _names(entry, "in") and not _names(entry, "out"):
            errors.append(f"{where} names neither an `in` nor an `out` card — "
                          f"there is nothing here to apply")

        # NO stranded-cut check here, deliberately. The obvious one — "an `out`
        # card not in the deck is a stranded name" — is UNREACHABLE: an entry
        # with any cut already gone is PARTIAL or APPLIED by construction, never
        # OPEN, so a check gated on OPEN can never fire. Ungated it is worse: a
        # missing cut means either "this entry was applied" or "that card left
        # for an unrelated reason", and nothing in the data distinguishes them.
        # PARTIAL is the honest signal and it is already reported. A validator
        # that fires on correct data is worse than none; one that can never fire
        # at all is just a comment pretending to be a check.

        settler = entry.get("settled_by")
        if settler and known_settlers and settler not in known_settlers:
            errors.append(
                f"{where} settled_by {settler!r} is not a pilot subcommand — "
                f"it must name the routine that closes this, so the queue "
                f"routes work instead of only listing it")
    return errors


def summarise(slug):
    """`{open, applied, partial, entries}` for `deck-status` and `deck-history`."""
    from manamap.pilot.common import load_json
    doc = load_json(deck_dir(slug) / "pending.json", default=None)
    if not doc:
        return {"open": 0, "applied": 0, "partial": 0, "entries": []}
    main_names = {c.get("name") for c in load_deck_cards(slug).get("cards", [])}
    out = []
    for entry in doc.get("pending", []):
        if not isinstance(entry, dict):
            continue
        out.append({**entry, "state": state_of(entry, main_names)})
    return {
        "open": sum(1 for e in out if e["state"] == OPEN),
        "applied": sum(1 for e in out if e["state"] == APPLIED),
        "partial": sum(1 for e in out if e["state"] == PARTIAL),
        "entries": out,
    }


def main(args):
    base = deck_dir(args.slug)
    path = base / "pending.json"
    if not path.exists():
        print(f"OK   no pending.json for {args.slug} — nothing queued, which is a "
              f"state rather than a gap")
        return 0
    with open(path) as f:
        doc = json.load(f)

    from manamap.pilot.registry import PILOT_STEPS
    settlers = {name for name, _, _ in PILOT_STEPS}

    errors = validate(doc, load_deck_cards(args.slug), known_settlers=settlers)
    summary = summarise(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps({"errors": errors, **summary}, indent=2))
        return 1 if errors else 0

    for e in summary["entries"]:
        mark = {OPEN: "OPEN   ", APPLIED: "APPLIED", PARTIAL: "PARTIAL"}[e["state"]]
        ins, outs = _names(e, "in"), _names(e, "out")
        print(f"  {mark} {e['id']:34} {e['decided']}  "
              f"+{len(ins)} / -{len(outs)}")
        if e["state"] == APPLIED:
            print(f"          the deck already reflects this — git owns it now, "
                  f"so delete the entry rather than ticking it")
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {summary['open']} open, {summary['applied']} applied, "
        f"{summary['partial']} partial")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-pending <slug>`.")
