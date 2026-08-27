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
"""

import json

from manamap.pilot.common import deck_dir, report_errors

ARTIFACT = "net_change.json"
STATES = {"met", "not met", "not resolvable", "not measured"}


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

    for who, lift in (doc.get("engine_lift") or {}).items():
        if not lift.get("available"):
            if not lift.get("why"):
                errors.append(f"engine_lift.{who}: unavailable with no reason given")
            continue
        for key in ("lift", "ci95", "excludes_zero", "reading"):
            if key not in lift:
                errors.append(f"engine_lift.{who}: no {key!r}")

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
