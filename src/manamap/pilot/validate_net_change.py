"""`net_change.json` — the form check on the document a purchase rests on.

GATED IN THE COMMIT THAT CREATED IT. This repo has now shipped two tracked
artifacts with no gate — `diagnostic.json` and `pool.txt` — and found both in an
audit rather than in use. A net-change report is the worst one to leave ungated:
it is read once, acted on, and the action costs money.

WHAT IT CHECKS is the discipline, not the numbers. The figures are deterministic
under a fixed seed and the freshness test re-derives them; re-checking them here
would be the same check twice. What no other gate asserts:

  1. EVERY MEASURED ROW CARRIES ITS MDE, and a row whose delta is under it is
     marked `noise` rather than ranked. A report that ranks noise is how a
     spending decision gets made on a coin flip.
  2. AN ENGINE LIFT STATES WHETHER ITS INTERVAL EXCLUDES ZERO. The lift is the
     measurement that decided the Ur-Dragon question; published without that it
     is a number with no claim attached.
  3. A FORGE BLOCK STATES ITS MDE, so an underpowered run says so instead of
     reading as no difference.
  4. AN OBJECTIVE, IF PRESENT, IS GRADED — and the grade is one of the four
     states, never a bare boolean.
  5. A RECOMMENDATION NAMES ROWS THAT EXIST, and is one of the five states.

WHAT IT DELIBERATELY DOES NOT CHECK is whether the recommendation follows the
rule. `net_change.recommend` derives it from this same document, so re-deriving
it here would be a test that re-derives the rule it is testing — the failure
this repo has shipped four times, once guarding the flagship metric. The rule is
held to fixtures in `tests/test_pilot_net_change.py`; what a GATE can add is that
the summary does not name a measure the table never carried.
"""

import json

from manamap.pilot.common import deck_dir, report_errors


def net_change_states():
    from manamap.pilot.net_change import STATES as S
    return S

ARTIFACT = "net_change.json"
STATES = {"met", "not met", "not resolvable", "not measured"}
#: Read from the module that writes them, so the two cannot drift.
RECOMMENDATION_STATES = set(net_change_states())


def validate(doc):
    errors = []
    for key in ("slug", "branch", "harness", "table", "limits"):
        if key not in doc:
            errors.append(f"missing top-level key {key!r}")

    table = doc.get("table") or []
    if not table:
        errors.append("table: no measured rows — the report claims nothing")
    for row in table:
        where = f"table[{row.get('measure', '?')}]"
        for key in ("champion", "branch", "delta", "mde", "verdict"):
            if key not in row:
                errors.append(f"{where}: no {key!r}")
        if "mde" not in row or "delta" not in row or "verdict" not in row:
            continue
        under = abs(row["delta"]) <= row["mde"]
        if under and row["verdict"] != "noise":
            errors.append(
                f"{where}: delta {row['delta']} is under the MDE {row['mde']} and "
                f"is reported as {row['verdict']!r} — a difference this run could "
                f"not resolve must be marked noise, not ranked")
        if not under and row["verdict"] == "noise":
            errors.append(
                f"{where}: delta {row['delta']} clears the MDE {row['mde']} and is "
                f"reported as noise")

    # ABSENT MEANS ABSENT, AND IT OWES A REASON. This used to be checked on the
    # engine-lift block alone, so deleting that block took the whole rule with
    # it and left `mana` and `forge` free to report `available: false` with no
    # explanation — a blank section a reader cannot tell from a measured
    # nothing. Stated once, over every block that has the key.
    for name in ("mana", "forge"):
        block = doc.get(name) or {}
        if "available" not in block:
            continue
        if not block["available"] and not str(block.get("why") or "").strip():
            errors.append(f"{name}: unavailable with no reason given")

    f = doc.get("forge") or {}
    if f.get("available"):
        if f.get("mde") is None:
            errors.append(
                "forge: no MDE — an underpowered run that does not say so reads "
                "as no difference")
        if "ci95" not in f:
            errors.append("forge: a delta with no interval on the difference")

    grade = doc.get("objective_grade")
    if doc.get("objective") and not grade:
        errors.append("an objective is stated and never graded")
    if grade and grade.get("state") not in STATES:
        errors.append(f"objective_grade.state {grade.get('state')!r} is not one of "
                      f"{sorted(STATES)}")

    rec = doc.get("recommendation")
    if rec is not None:
        if rec.get("state") not in RECOMMENDATION_STATES:
            errors.append(
                f"recommendation.state {rec.get('state')!r} is not one of "
                f"{sorted(RECOMMENDATION_STATES)}")
        if not (rec.get("because") or "").strip():
            errors.append(
                "recommendation: no `because` — a verdict with no sentence "
                "behind it is the thing this report exists not to be")
        # A SUMMARY THAT NAMES A MEASURE THE TABLE NEVER CARRIED is the one
        # inconsistency a gate can see without re-deriving the rule.
        measured = {r.get("measure") for r in table}
        for key in ("rose", "fell", "no_call"):
            for name in rec.get(key) or []:
                if name not in measured:
                    errors.append(
                        f"recommendation.{key} names {name!r}, which is not a "
                        f"row in the table")
    return errors


def main(args):
    branch = getattr(args, "branch", None)
    if not branch:
        raise SystemExit(
            f"{ARTIFACT} lives on a branch — `--branch <name>`.")
    path = deck_dir(args.slug, branch) / ARTIFACT
    if not path.exists():
        raise SystemExit(
            f"{path} not found — `manamap pilot net-change {args.slug} "
            f"--branch {branch} --write` first.")
    doc = json.loads(path.read_text())
    errors = validate(doc)
    graded = (doc.get("objective_grade") or {}).get("state", "ungraded")
    report_errors(
        f"{ARTIFACT} for {args.slug}@{branch}", errors,
        f"OK   {ARTIFACT} for {args.slug}@{branch} — {len(doc.get('table') or [])} "
        f"measured row(s), objective {graded} ◆")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-net-change <slug> --branch <name>`.")
