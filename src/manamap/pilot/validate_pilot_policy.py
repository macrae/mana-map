"""Pilot: form-check `pilot_policy.json`, the declared piloting rules.

THE GATE ARRIVES BEFORE THE FIRST FILE DOES, which is the point. No deck has a policy
today — the one rule ever written was CR-proven correct, measured monotonically harmful at
20,000 games/arm and withdrawn the same hour — so this checks nothing yet. It exists now
because CLAUDE.md's rule is "a gate in the same commit", and a policy is about to grow a
`forge` section over 121 `AiProps` keys. A vocabulary that expands with no gate in front of
it is how a flag nobody reads gets shipped.

`pilot_policy.validate` already raises `PolicyError` at LOAD time inside the goldfish, which
is a gate on the simulator's path and not on the artifact: it fires only when somebody runs
a measurement, reports through a traceback, and is invisible to `deck-status` and to the
fleet sweep in `tests/test_pilot_tracked_artifacts_validate.py`. This is the same predicate
reached the way every other tracked artifact's is.
"""

import sys

from manamap.pilot import pilot_policy
from manamap.pilot.common import deck_dir, load_json, report_errors


def validate(doc):
    """Return a list of error strings — the load-time predicate, as a report."""
    if not doc:
        return []
    try:
        pilot_policy.validate(doc)
    except pilot_policy.PolicyError as exc:
        return [str(exc)]
    return []


def main(args=None):
    slug = getattr(args, "slug", None)
    path = deck_dir(slug) / "pilot_policy.json"
    if not path.is_file():
        print(f"{slug}: no pilot_policy.json — the simulator uses its own heuristics "
              f"(absent means absent; this is the normal state)")
        return
    doc = load_json(path) or {}
    errors = validate(doc)
    report_errors(f"{slug} — pilot_policy.json", errors)
    if not errors:
        rules = doc.get("rules") or []
        print(f"OK   {slug} — {len(rules)} rule(s): "
              + ", ".join(r.get("id", "?") for r in rules))
        for line in pilot_policy.render(doc):
            print(f"  {line}")
    sys.exit(1 if errors else 0)
