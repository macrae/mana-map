"""The gate on `captains_log.json`.

THE DOCTRINE, restated because it is the whole reason this file is short: a
validator that fires on correct data is worse than no validator, and six proposed
checks in this repo have been prototyped and rejected on that ground. There is no
existing Picard prose to measure against, so the rule imposed here is harsher
than usual —

    A CHECK FAILS ONLY IF IT CANNOT FIRE ON CORRECT DATA BY CONSTRUCTION.

Everything else prints under NOTE, never moves the exit code, and is promoted to
a failure in a later commit once it has been measured over a full fleet run. That
is the treatment `merge_prose`'s content check got, and lost.

Almost every check below is possible only because the skeleton is DETERMINISTIC:
`captains_log.nights()` recomputes the grouping, the stardates and the evening
positions, and this compares. None of it reads the prose for meaning.
"""

import re
import sys

from manamap.config import DECKS_DIR
from manamap.pilot import captains_log as cl
from manamap.pilot.deck_notes import read_log

#: Reporting-only, pending measurement. Each is a real failure mode of the
#: abstraction layer and none can be proved harmless in advance.
_SHOUTY = re.compile(r"\b[A-Z]{4,}\b")
_ISSUED = re.compile(
    r"\b(I have (ordered|instructed|directed|asked|told)|"
    r"(has|have) been (ordered|instructed|directed))\b", re.I)
_SUPERLATIVE = re.compile(
    r"\b(best|worst|incredible|amazing|terrible|disaster|brutal|insane|massive|"
    r"huge|catastrophic|perfect)\b", re.I)
#: The layer's whole job is that these do not survive into the captain's mouth.
_JARGON = re.compile(
    r"\b(mulligan|wipe|sac|sacced|ETB|pod|cEDH|tutor|tutored|ramp|ramped|"
    r"goldfish|curve out|value engine)\b", re.I)


def _prose(block):
    """Every prose string in one night's account.

    Driven by `SECTION_KEYS` rather than a hand-written tuple. The old version
    listed the six sections by name, so renaming one silently dropped it from
    every style check while the checks kept reporting green — the same class of
    defect as a loop over an empty collection with no `assert checked >= N`.
    """
    out = []
    for key in cl.SECTION_KEYS:
        val = block.get(key)
        if isinstance(val, str):
            out.append((key, val))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, str):
                    out.append((f"{key}[{i}]", item))
    return out


def _check_block(where, block, night, errors, notes):
    """One night's account.

    Six sections became one. The header-quotes-the-stardate check, the
    attribution ordering and the closed station set all went with the register
    they policed: there is no ship, no officer to take an order, and a stardate
    the pilot could not read.

    What survives is the property that is not about voice — a rendered night
    must actually say something, because a stub recorded as a cache HIT renders
    empty forever with every check still green.
    """
    for key in cl.SECTION_KEYS:
        val = block.get(key)
        if val is None or (isinstance(val, str) and not val.strip()) \
                or (isinstance(val, list) and not val):
            errors.append(f"{where}.{key} is missing or empty — a rendered night "
                          f"that says nothing is worse than an unrendered one, "
                          f"which at least prints a prompt to write it")

    # ---- reporting only ----
    #
    # THE JARGON LIST IS GONE. It banned mulligan, wipe, ramp, ETB, pod and
    # tutor because the old register could not admit them, and the paraphrases
    # it forced ("a hand I chose to keep") were the strangest thing on the page.
    # Those are the pilot's own words and the log is his account.
    #
    # THE EXCLAMATION MARK IS NO LONGER A FAILURE. It was correct for a register
    # that forbade emotion; it is not a correctness property, and a check that
    # fails on prose a human would accept is the failure mode this file's own
    # doctrine refuses.
    for key, text in _prose(block):
        for label, rx in (("shouty caps carried from the source", _SHOUTY),
                          ("superlative", _SUPERLATIVE)):
            hits = sorted(set(rx.findall(text)))
            if hits:
                notes.append(f"{where}.{key}: {label} — {', '.join(map(str, hits))}")
        if "!" in text:
            notes.append(f"{where}.{key}: exclamation mark")


def _check_read(doc, slug, errors, notes):
    """The deck-level roll-up, and the one thing it may not do.

    IT MAY NOT CITE A GAME THAT DOES NOT EXIST. Everything else about a read is
    judgment, and judgment is not this file's business — but a citation is
    mechanically checkable, and a read whose evidence cannot be found is an
    opinion about a decklist. Four other artifacts already have those.

    Modelled on `validate_debrief`'s rule that the debrief may not name a card
    the pilot did not.
    """
    import re as _re

    read = doc.get("read")
    log_ids = {e["id"] for e in read_log(slug)}
    if not read:
        if log_ids:
            notes.append(f"no `read` — {len(log_ids)} logged game(s) and nothing "
                         f"synthesised across them yet")
        return
    for key in cl.READ_KEYS:
        val = read.get(key)
        if val is None or (isinstance(val, str) and not val.strip()) \
                or (isinstance(val, list) and not val):
            errors.append(f"read.{key} is missing or empty — the read is the "
                          f"reason this artifact exists and a partial one "
                          f"recorded as a HIT stays partial")
    cited = set()
    for key in cl.READ_KEYS:
        val = read.get(key)
        for text in (val if isinstance(val, list) else [val or ""]):
            cited |= set(_re.findall(r"\b(\d{3})\b", str(text)))
    unknown = sorted(cited - log_ids)
    if unknown:
        errors.append(f"read cites game(s) {', '.join(unknown)} which are not in "
                      f"the log — the log is the authority and a read cannot "
                      f"add games to it")


def validate(doc, slug):
    errors, notes = [], []
    entries = read_log(slug)
    known = {e["id"] for e in entries}
    truth = cl.nights(slug)

    nights = doc.get("nights")
    if not isinstance(nights, dict):
        return ["`nights` is missing or not an object"], notes

    # 3. THE SKELETON IS RECOMPUTED AND COMPARED. Checkable at all only because
    # the merge writes these rather than the agent. If it fires, either the file
    # was hand-edited or `stardate()` moved under a file nobody regenerated —
    # both things worth being told about.
    for key, night in sorted(nights.items()):
        if key not in truth:
            errors.append(f"nights[{key}]: {slug} logged no game that night — "
                          f"the log is the authority and a rendering cannot add "
                          f"nights to it")
            continue
        want = truth[key]
        for field in ("source_ids", "position_in_evening", "version"):
            if night.get(field) != want[field]:
                errors.append(
                    f"nights[{key}].{field} is {night.get(field)!r}, recomputed "
                    f"as {want[field]!r} — regenerate with `merge-captains-log`")

        # 1. EVERY SOURCE ID NAMES A REAL ENTRY.
        for eid in night.get("source_ids") or []:
            if eid not in known:
                errors.append(f"nights[{key}].source_ids: no log entry {eid!r}")

        logs = night.get("logs")
        if not isinstance(logs, dict):
            errors.append(f"nights[{key}].logs is missing or not an object")
            continue
        for kind, block in sorted(logs.items()):
            if kind not in cl.LOG_KINDS:
                errors.append(f"nights[{key}].logs[{kind!r}] is not a log kind — "
                              f"one of {list(cl.LOG_KINDS)}")
                continue
            if not isinstance(block, dict):
                errors.append(f"nights[{key}].logs[{kind}] is not an object")
                continue
            _check_block(f"nights[{key}].logs[{kind}]", block, want, errors, notes)
            for j, sup in enumerate(block.get("supplementals") or []):
                _check_block(f"nights[{key}].logs[{kind}].supplementals[{j}]",
                             sup, want, errors, notes)

    # 2. NO GAME FILED TWICE. This is "the raw notes stay reachable" made
    # mechanical: a game rendered under two stardates is a game the reader meets
    # twice and can trust neither copy of.
    seen = {}
    for key, night in sorted(nights.items()):
        for eid in night.get("source_ids") or []:
            if eid in seen:
                errors.append(f"log entry {eid!r} is filed under both "
                              f"{seen[eid]} and {key}")
            seen[eid] = key

    # 3. THE READ — the deck-level roll-up, and its one mechanical property.
    _check_read(doc, slug, errors, notes)
    return errors, notes


def main(args):
    slug = args.slug
    path = DECKS_DIR / slug / cl.ARTIFACT
    if not path.exists():
        # ABSENT IS LEGAL. A deck whose nights have not been rendered yet is a
        # normal state, and the cache is what says so — not a red gate.
        print(f"OK   {slug} — no {cl.ARTIFACT} (nothing rendered yet)")
        return 0

    doc = cl.read(slug)
    errors, notes = validate(doc, slug)

    # COVERAGE IS REPORTED, NEVER FAILED. A partly-rendered artifact is a normal
    # intermediate state; incompleteness belongs to the cache, not to a gate that
    # reddens history.
    truth = cl.nights(slug)
    rendered = [k for k, n in (doc.get("nights") or {}).items()
                if "pilot" in (n.get("logs") or {})]
    games = sum(len(n["source_ids"]) for n in truth.values())
    reachable = sum(len(n.get("source_ids") or [])
                    for k, n in (doc.get("nights") or {}).items() if k in truth)

    for n in notes:
        print(f"NOTE {n}")
    if errors:
        print(f"FAIL {slug} captain's log ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK   {slug} — {len(rendered)} of {len(truth)} night(s) rendered, "
          f"{reachable} of {games} game(s) reachable"
          + (f"; {len(notes)} note(s)" if notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(type("Args", (), {"slug": sys.argv[1]})()))
