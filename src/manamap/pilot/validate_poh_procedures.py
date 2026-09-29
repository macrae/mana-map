"""Pilot: form-check `poh_procedures.json`, the authored half of the handbook.

THIS ARTIFACT HAD NO GATE AT ALL. It is tracked on six decks, `install_agent` stamps a
`decklist_sha256_prefix` into it, and nothing has ever checked its shape: no entry in
`deck_status.VALIDATED`, no `deck_status.STAGES` row, no freshness test.
`validate-poh` exists and checks the RENDERED HTML — dangling cross-references, callouts
per page, no `<script>` — which is a different question and cannot see a condition outside
the closed vocabulary or a `grounded_in` pointing at a game nobody logged.

That is a standing violation of CLAUDE.md's own rule: "A new tracked artifact needs a gate
in the same commit — a validator, a freshness test, or both — and a `deck_status.VALIDATED`
entry so the status command sees what the tests see."

WHAT IT CHECKS AND WHY EACH ONE CANNOT FIRE ON CORRECT DATA. Measured against all six
tracked files before shipping, which is this repo's bar — six proposed validators have been
rejected for firing on correct data, and one written earlier today would have:

  * the three owned keys are present. All six have them; the agent charter says it owns
    exactly `emergency`, `normal` and `handling`.
  * EXTRA keys are allowed. heliod carries `slug` and `decklist_sha256` that the other five
    lack, and a validator that erred on them would fire on a correct file the day it
    shipped. They are reported as a NOTE so the drift is visible without being fatal.
  * conditions come from `poh_spec.EMERGENCY_CONDITIONS`, which is `deck_notes.CAUSES`
    mirrored — so a page names a way games actually end and the dossier can count them.
  * phases are exactly `poh_spec.NORMAL_PHASES`' five keys, in any order.
  * `immediate` and `subsequent` are LISTS, because the renderer emits `<ol>` from them and
    a string would render as one character per step.
  * a NON-EMPTY `grounded_in` must resolve against `log.jsonl`. An EMPTY one is fine and
    deliberately so: the charter says "a page with an empty `grounded_in` is honest", and
    zur-enchantress has zero across all seven pages. Erroring on that would demand the
    pilot invent games.

WHAT IT DOES NOT CHECK. Whether a step is good advice, whether the prose hedges, whether a
condition is the right one for this deck. `validate_captains_log` states the doctrine this
follows: a check fails only if it cannot fire on correct data by construction; everything
else prints as a NOTE and is promoted later, once measured over a full fleet run.
"""

import json
import sys

from manamap.pilot import poh_spec
from manamap.pilot.common import deck_dir, load_json, report_errors

#: The keys the `poh-procedures` agent owns, and the only ones required.
OWNED = ("emergency", "normal", "handling")

#: Every phase `poh.render_normal` knows how to draw.
PHASES = tuple(p[0] for p in poh_spec.NORMAL_PHASES)

#: `poh.render_handling`'s four subsections.
HANDLING = ("optics", "reveal", "alliances", "targets")

#: Ordered fields the renderer emits as `<ol>`; a string here renders one character
#: per step, which is the shape of failure a form check exists to stop.
ORDERED = ("immediate", "subsequent")


def validate(doc, log_ids=None):
    """Return `(errors, notes)`. Errors are form; notes are drift worth seeing."""
    errors, notes = [], []
    if not isinstance(doc, dict):
        return [f"poh_procedures.json is {type(doc).__name__}, not an object"], notes

    missing = [k for k in OWNED if k not in doc]
    if missing:
        errors.append(f"missing the key(s) the agent owns: {missing}")

    extra = sorted(set(doc) - set(OWNED) - {"decklist_sha256_prefix", "decklist_sha256",
                                            "slug"})
    if extra:
        notes.append(f"keys nothing reads: {extra}")
    drift = sorted({"slug", "decklist_sha256"} & set(doc))
    if drift:
        notes.append(f"carries {drift}, which five of the six tracked files do not — "
                     f"harmless, and reported so the drift stays visible")

    # ------------------------------------------------------------- emergency
    pages = doc.get("emergency")
    if pages is not None:
        if not isinstance(pages, list):
            errors.append(f"emergency is {type(pages).__name__}, not a list of pages")
        else:
            seen = []
            for i, page in enumerate(pages):
                where = f"emergency[{i}]"
                if not isinstance(page, dict):
                    errors.append(f"{where} is {type(page).__name__}, not an object")
                    continue
                cond = page.get("condition")
                if cond not in poh_spec.EMERGENCY_CONDITIONS:
                    errors.append(
                        f"{where}: condition {cond!r} is not one the log can record; "
                        f"pick from {sorted(poh_spec.EMERGENCY_CONDITIONS)}")
                else:
                    seen.append(cond)
                for field in poh_spec.EMERGENCY_FIELDS:
                    if field not in page:
                        errors.append(f"{where} ({cond}): no {field!r}")
                for field in ORDERED:
                    val = page.get(field)
                    if val is not None and not isinstance(val, list):
                        errors.append(
                            f"{where} ({cond}): {field} is {type(val).__name__}, not a "
                            f"list — the renderer emits <ol> from it and a string would "
                            f"draw one character per step")
                ind = page.get("indications")
                if ind is not None and not isinstance(ind, list):
                    errors.append(f"{where} ({cond}): indications must be a list")
                # A CITED GAME MUST EXIST. An UNCITED page is honest — the charter says
                # so, and zur-enchantress has seven of them.
                for gid in (page.get("grounded_in") or []):
                    ref = str(gid).split(":", 1)[-1]
                    if log_ids is not None and ref not in log_ids:
                        errors.append(
                            f"{where} ({cond}): grounded_in {gid!r} is not a log entry — "
                            f"the log is the authority and a page cannot cite a game "
                            f"nobody played")
            dupes = sorted({c for c in seen if seen.count(c) > 1})
            if dupes:
                errors.append(f"emergency has two pages for the same condition: {dupes}")

    # ---------------------------------------------------------------- normal
    normal = doc.get("normal")
    if normal is not None:
        if not isinstance(normal, dict):
            errors.append(f"normal is {type(normal).__name__}, not an object of phases")
        else:
            for phase in PHASES:
                if phase not in normal:
                    errors.append(f"normal has no {phase!r} phase")
            unknown = sorted(set(normal) - set(PHASES))
            if unknown:
                errors.append(f"normal has phase(s) the renderer cannot draw: {unknown}")
            for phase, body in normal.items():
                if phase not in PHASES:
                    continue
                # `render_normal` tolerates a bare list OR a dict; both are fine.
                if isinstance(body, dict):
                    steps = body.get("steps") or body.get("keep") or body.get("ship")
                    if steps is None:
                        errors.append(f"normal.{phase}: no steps, keep or ship")
                elif not isinstance(body, list):
                    errors.append(f"normal.{phase} is {type(body).__name__}, not steps")

    # -------------------------------------------------------------- handling
    handling = doc.get("handling")
    if handling is not None:
        if not isinstance(handling, dict):
            errors.append(f"handling is {type(handling).__name__}, not an object")
        else:
            for key in HANDLING:
                if key not in handling:
                    errors.append(f"handling has no {key!r}")
    return errors, notes


def main(args=None):
    from manamap.pilot import deck_notes

    slug = getattr(args, "slug", None)
    base = deck_dir(slug)
    path = base / poh_spec.PROCEDURES_ARTIFACT
    if not path.is_file():
        print(f"{slug}: no {poh_spec.PROCEDURES_ARTIFACT} — nothing to check "
              f"(run /poh-procedures to author one)")
        return
    doc = load_json(path) or {}
    try:
        log_ids = {e["id"] for e in deck_notes.read_log(slug)}
    except (SystemExit, FileNotFoundError, KeyError):
        # NO LOG IS NOT A FAILURE. A deck can have a handbook before it has games, and a
        # gate that reddened on that would demand the pilot play before writing.
        log_ids = None
    errors, notes = validate(doc, log_ids=log_ids)
    for note in notes:
        print(f"  ! {note}")
    if log_ids is None:
        print("  ! no captain's log, so grounded_in references are not checked")
    # LABEL FIRST: `report_errors(fail_label, errors)`. Reversed, it iterates the
    # label STRING as the error list and prints one character per line — which is
    # exactly what it did, reporting 28 "errors" that were the letters of the slug.
    report_errors(f"{slug} — {poh_spec.PROCEDURES_ARTIFACT}", errors)
    if not errors:
        pages = len(doc.get("emergency") or [])
        cited = sum(1 for p in (doc.get("emergency") or []) if p.get("grounded_in"))
        print(f"OK   {slug} — {pages} emergency page(s), {cited} grounded in real games")
    sys.exit(1 if errors else 0)
