"""Pilot: mechanically enforce the form of a strategic frame.

strategic_frame.json is the highest-leverage unvalidated artifact in the
system: it feeds the notes writer, the engineer, the doctor and the decision
spreads, and
nothing checked its shape — a frame with a missing key or an
unflagged candidate line propagated silently into prose. Same division of
labour as every other validator here: code enforces **form**, the consuming
agents judge substance.

The one substantive rule this file does own is the candidate-line flag. A
missing line the researcher merely suspects must carry
`status: "needs a stack scenario"` — the exact wording the architect and
sideboard analyst use — because that string is what routes it into the
resolve-stack queue instead of into print.
"""

import json

from manamap.pilot.common import deck_dir, report_errors

REQUIRED_KEYS = {
    "slug", "archetype", "schools", "overall_assessment", "role_assignment",
    "engines", "candidate_missing_lines", "matchup_frames",
    "distribution_notes", "gaps",
}
REQUIRED_LINE_KEYS = {"cards", "title", "why_plausible", "status"}
REQUIRED_ENGINE_KEYS = {"piece", "engine", "evidence", "strategy_ref"}
CANDIDATE_STATUS = "needs a stack scenario"


def validate(doc, slug=None):
    """Return a list of human-readable error strings (empty = the form holds)."""
    errors = []

    missing = REQUIRED_KEYS - set(doc)
    if missing:
        errors.append(f"missing required key(s): {', '.join(sorted(missing))}")
        return errors

    if slug and doc.get("slug") != slug:
        errors.append(f"slug is {doc.get('slug')!r}, expected {slug!r}")

    for key in ("archetype", "overall_assessment", "distribution_notes"):
        if not (isinstance(doc[key], str) and doc[key].strip()):
            errors.append(f"{key} must be a non-empty string")
    for key in ("schools", "gaps", "engines", "candidate_missing_lines"):
        if not isinstance(doc[key], list):
            errors.append(f"{key} must be a list")
    for key in ("role_assignment", "matchup_frames"):
        if not isinstance(doc[key], dict):
            errors.append(f"{key} must be an object")

    for i, engine in enumerate(doc.get("engines") or []):
        if not isinstance(engine, dict):
            errors.append(f"engines[{i}] must be an object")
            continue
        missing = REQUIRED_ENGINE_KEYS - set(engine)
        if missing:
            errors.append(f"engines[{i}] missing {', '.join(sorted(missing))}")

    for i, line in enumerate(doc.get("candidate_missing_lines") or []):
        if not isinstance(line, dict):
            errors.append(f"candidate_missing_lines[{i}] must be an object")
            continue
        missing = REQUIRED_LINE_KEYS - set(line)
        if missing:
            errors.append(f"candidate_missing_lines[{i}] missing {', '.join(sorted(missing))}")
        if "status" in line and line["status"] != CANDIDATE_STATUS:
            errors.append(
                f"candidate_missing_lines[{i}].status must be {CANDIDATE_STATUS!r} "
                f"(got {line['status']!r}) — a candidate line is never a fact"
            )
        if "cards" in line and not (isinstance(line["cards"], list) and line["cards"]):
            errors.append(f"candidate_missing_lines[{i}].cards must be a non-empty list")

    return errors


def main(args):
    path = deck_dir(args.slug) / "strategic_frame.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — the strategy-researcher (consult mode) writes it; "
            f"see the write-manual skill."
        )
    with open(path) as f:
        doc = json.load(f)
    errors = validate(doc, slug=args.slug)
    from manamap.pilot.model_staleness import note as _stale_note
    _stale = _stale_note(args.slug, doc if isinstance(doc, dict) else {},
                         getattr(args, "branch", None))
    report_errors(
        path.name, errors,
        f"OK   {path.name} — {len(doc.get('engines', []))} engine(s), "
        f"{len(doc.get('candidate_missing_lines', []))} candidate line(s), "
        f"form holds" + _stale)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-strategic-frame <slug>`.")
