"""`diagnostic.json` — the form check the artifact shipped without.

IT WAS TRACKED, WRITTEN BY `diagnose --write`, AND GATED BY NOTHING: no
validator, no freshness test, no `deck_status` row. It is also composed from the
goldfish, so it goes stale on every model change — the one artifact whose
staleness would be least visible.

WHAT IS WORTH CHECKING HERE IS THE EVIDENCE CONTRACT, NOT THE NUMBERS. The
figures are deterministic under a fixed seed and the freshness test re-derives
them; a validator that also re-derived them would be the same check twice. What
no other gate asserts is the discipline this document exists to keep:

  1. EVERY RATE CARRIES AN INTERVAL. A rate published bare is the thing this
     repo refuses everywhere else, and `output` was one careless commit from
     being the exception — the magnitude block originally emitted raw floats.
  2. ABSENT MEANS ABSENT, NEVER ZERO. A deck that declares no `required`
     component has no engine figure and a deck that opts into neither model has
     no magnitude block. Both must be MISSING with a stated reason, because 0.0
     is a measurement nobody made.
  3. AN INTERVAL MUST BRACKET ITS OWN RATE. Cheap, and it catches a whole class
     of transcription error that reads as plausible.
"""

import json

from manamap.pilot.common import deck_dir, report_errors

ARTIFACT = "diagnostic.json"

#: Blocks that carry `{rate, ci95, n}` cells, and how deep they sit.
RATE_BLOCKS = (("stall", "by_turn"), ("stall", "two_in_a_row"),
               ("engine", "online_by_turn"), ("engine", "any_route_by_turn"),
               ("mana", "missed_land_drop_by_five"), ("mana", "mulliganed"),
               ("output", "hoard_by_turn"), ("output", "damage_by_turn"),
               ("output", "board_power_by_turn"), ("output", "kill_by_turn"))


def _cells(value):
    """Yield every `{rate, ...}` cell, whether the block is one or by-turn."""
    if not isinstance(value, dict):
        return
    if "rate" in value:
        yield "", value
        return
    for key, cell in value.items():
        if isinstance(cell, dict) and "rate" in cell:
            yield key, cell


def validate(doc):
    errors = []
    for key in ("slug", "harness", "limits"):
        if key not in doc:
            errors.append(f"missing top-level key {key!r}")
    for block, name in RATE_BLOCKS:
        section = doc.get(block)
        if not isinstance(section, dict) or name not in section:
            continue
        seen = False
        for turn, cell in _cells(section[name]):
            seen = True
            where = f"{block}.{name}" + (f"[{turn}]" if turn else "")
            if "ci95" not in cell:
                errors.append(f"{where}: a rate with no interval")
                continue
            lo, hi = cell["ci95"]
            if not (lo <= cell["rate"] <= hi):
                errors.append(
                    f"{where}: ci95 {cell['ci95']} does not bracket its own "
                    f"rate {cell['rate']}")
            if "n" not in cell:
                errors.append(f"{where}: a rate with no sample size")
        if not seen:
            errors.append(f"{block}.{name}: present but holds no rate cell")

    # ABSENT ⇒ ABSENT. Both blocks report `available` and owe a reason when it
    # is false; a zeroed figure standing in for a missing one is the failure.
    #
    # WHAT THE RULE IS ACTUALLY ABOUT, and what it banned by accident. This
    # allowed only `available`/`why`/`basis` and rejected everything else — but
    # the producer writes `declared_targets` and `declaration_mismatch` on an
    # unavailable block, and those are not measurements. They are facts about
    # the AUTHORED DECLARATION (`goldfish_targets.json`): how many targets it
    # names, and which of them name cards this list does not run. They are the
    # evidence FOR the unavailability, and the rule against zeroed figures was
    # never meant to strip a reason of its detail.
    #
    # Found by creating the two `diagnostic.json` files the pinned decks were
    # missing: both were rejected by the gate the instant they were written,
    # which is a producer and its validator disagreeing about their own format
    # (GitHub #37). Still a CLOSED set — a rate, a count of games, anything
    # measured still fails here, which is the check that matters.
    explanatory = {"available", "why", "basis",
                   "declared_targets", "declaration_mismatch"}
    for block in ("engine", "output"):
        section = doc.get(block) or {}
        if "available" not in section:
            errors.append(f"{block}: no `available` flag")
        elif not section["available"]:
            if not section.get("why"):
                errors.append(f"{block}: unavailable with no reason given")
            for stray in set(section) - explanatory:
                errors.append(
                    f"{block}.{stray}: unavailable blocks must be ABSENT, not "
                    f"zeroed — a figure here is one nobody measured")
    return errors


def main(args):
    branch = getattr(args, "branch", None)
    where = args.slug + (f"@{branch}" if branch else "")
    path = deck_dir(args.slug, branch) / ARTIFACT
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `manamap pilot diagnose {args.slug}"
            + (f" --branch {branch}" if branch else "") + " --write` first.")
    doc = json.loads(path.read_text())
    errors = validate(doc)
    blocks = sum(1 for b, n in RATE_BLOCKS if n in (doc.get(b) or {}))
    report_errors(
        f"{ARTIFACT} for {where}", errors,
        f"OK   {ARTIFACT} for {where} — {blocks} measured block(s), "
        f"every rate carries its interval ◆")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-diagnostic <slug>`.")
