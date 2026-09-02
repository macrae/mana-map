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
    """Every prose string in one log, section by section, for the text checks."""
    out = []
    for key in ("header", "situation", "narrative", "coda"):
        if isinstance(block.get(key), str):
            out.append((key, block[key]))
    for i, a in enumerate(block.get("assessment") or []):
        if isinstance(a, dict) and isinstance(a.get("text"), str):
            out.append((f"assessment[{i}]", a["text"]))
    for i, o in enumerate(block.get("orders") or []):
        if isinstance(o, dict) and isinstance(o.get("text"), str):
            out.append((f"orders[{i}]", o["text"]))
    return out


def _check_block(where, block, night, errors, notes):
    """One log — the main entry or a supplemental."""
    # 7. ALL SIX SECTIONS, PRESENT AND NON-EMPTY. A five-section log recorded as
    # a cache HIT renders short forever with every check green — the same
    # reasoning `agent_cache.record` uses to refuse a partial keyed artifact.
    for key in cl.SECTION_KEYS:
        val = block.get(key)
        if val is None or (isinstance(val, str) and not val.strip()) \
                or (isinstance(val, list) and not val):
            errors.append(f"{where}.{key} is missing or empty — a log is not a "
                          f"log with five of its six sections")

    # 4. THE HEADER QUOTES THE FACTS VERBATIM. `validate_debrief`'s substring
    # trick, turned around: it proves the prose consistent with the number the
    # renderer sorts by, while judging no word of it. The agent is handed both
    # strings, so it cannot fire on correct data.
    header = block.get("header") or ""
    if isinstance(header, str) and header.strip():
        if night["stardate"] not in header:
            errors.append(f"{where}.header does not quote the stardate "
                          f"{night['stardate']} — the header and the field the "
                          f"page sorts by must agree")
        if night.get("version") and night["version"] not in header:
            errors.append(f"{where}.header does not quote the version "
                          f"{night['version']!r}")

    # 5. RESPONSIBILITY IN ORDER: self, then ship, then circumstance, never
    # reversed. The pilot's hardest style rule, made structural.
    seen = []
    for i, a in enumerate(block.get("assessment") or []):
        if not isinstance(a, dict):
            errors.append(f"{where}.assessment[{i}] is not an object")
            continue
        attr = a.get("attribution")
        if attr not in cl.ATTRIBUTION_ORDER:
            errors.append(f"{where}.assessment[{i}].attribution {attr!r} is not "
                          f"one of {list(cl.ATTRIBUTION_ORDER)}")
            continue
        seen.append(cl.ATTRIBUTION_ORDER.index(attr))
    if seen:
        if seen[0] != 0:
            errors.append(f"{where}.assessment does not begin with `self` — the "
                          f"captain assigns responsibility to himself first")
        if any(b < a for a, b in zip(seen, seen[1:])):
            errors.append(f"{where}.assessment attributes out of order "
                          f"({[cl.ATTRIBUTION_ORDER[i] for i in seen]}) — the "
                          f"order is self, ship, circumstance and never reversed")

    # 6. THE STATIONS ARE A CLOSED SET.
    for i, o in enumerate(block.get("orders") or []):
        if not isinstance(o, dict):
            errors.append(f"{where}.orders[{i}] is not an object")
            continue
        if o.get("station") not in cl.STATIONS:
            errors.append(f"{where}.orders[{i}].station {o.get('station')!r} is "
                          f"not a station — one of {sorted(cl.STATIONS)}")

    # 8. NO EXCLAMATION MARKS. The one style rule in the spec that is binary, and
    # correct Picard prose contains none by construction.
    for key, text in _prose(block):
        if "!" in text:
            errors.append(f"{where}.{key} contains an exclamation mark")

    # ---- reporting only, pending measurement over a full fleet run ----
    for key, text in _prose(block):
        for label, rx in (("shouty caps carried from the source", _SHOUTY),
                          ("superlative", _SUPERLATIVE),
                          ("jargon", _JARGON)):
            hits = sorted(set(rx.findall(text)))
            if hits:
                notes.append(f"{where}.{key}: {label} — {', '.join(map(str, hits))}")
    for i, o in enumerate(block.get("orders") or []):
        text = (o or {}).get("text") or ""
        if text and not _ISSUED.search(text):
            notes.append(f"{where}.orders[{i}] is not phrased as already issued "
                         f"— \"I have ordered …\"")


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
        for field in ("stardate", "source_ids", "position_in_evening", "version"):
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
                if "ship" in (n.get("logs") or {})]
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
