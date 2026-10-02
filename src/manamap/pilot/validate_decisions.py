"""The gate on `decisions.jsonl` — form only, so it runs anywhere.

One object per line, ids sequential from 001, a closed `kind` vocabulary, a
withdraw or reject that says why, a merge that names the list it made, and an
outcome that names an existing merge. A backfilled line may lack a prediction;
a live one written by `propose` or `merge` carries one. Never fires on a
correct file: swept over every deck's ledger before it shipped.
"""

import json

from manamap.pilot import decisions
from manamap.pilot.common import report_errors


def validate(entries):
    errors = []
    ids = [e.get("id") for e in entries]
    want = [f"{i:03d}" for i in range(1, len(entries) + 1)]
    if ids != want:
        errors.append(f"ids are {ids[:6]}… and must be sequential from 001 — the ledger is "
                      f"append-only and a gap is a deleted line")
    merges = {e["id"] for e in entries if e.get("kind") == "merge"}
    closed = set()
    for e in entries:
        where = f"{e.get('id', '?')}"
        kind = e.get("kind")
        if kind not in decisions.KINDS:
            errors.append(f"{where}: kind {kind!r} is not one of {decisions.KINDS}")
            continue
        if not e.get("at"):
            errors.append(f"{where}: no `at`")
        if kind in decisions.NEEDS_REASON and not str(e.get("reason") or "").strip():
            errors.append(f"{where}: a {kind} with no reason")
        if kind == "merge" and not e.get("decklist_sha256"):
            errors.append(f"{where}: a merge that does not name the list it made")
        if (kind in ("propose", "merge") and not e.get("backfilled")
                and not e.get("prediction")
                and not str(e.get("prediction_note") or "").strip()):
            # A NAMED ABSENCE IS ALLOWED; A SILENT ONE IS NOT. Nine of the
            # fleet's ten merges freeze a report and should. The tenth —
            # sharknado 005, swords-v1 — merged four swaps the pilot had already
            # made in cardboard, off a branch that was never measured, so there
            # was no report to freeze and a `prediction` would have been
            # invented. This fires on a merge that forgot its prediction and
            # passes one that says why it has none, which is the difference
            # between a missing measurement and a hidden one.
            errors.append(f"{where}: a live {kind} carries neither a prediction nor a "
                          f"`prediction_note` saying why — the report's figures at the "
                          f"moment of the decision are the point, and an absent figure "
                          f"must name its reason")
        if kind == "outcome":
            if e.get("of") not in merges:
                errors.append(f"{where}: outcome of {e.get('of')!r}, which is not a merge here")
            elif e["of"] in closed:
                errors.append(f"{where}: merge {e['of']} already has an outcome")
            else:
                closed.add(e["of"])
            r = e.get("realised") or {}
            if not isinstance(r.get("rate"), (int, float)) or not r.get("run_ids"):
                errors.append(f"{where}: an outcome with no realised rate or no run ids")
    return errors


def main(args):
    slug = args.slug
    p = decisions.path(slug)
    if not p.exists():
        print(f"{slug}: no {decisions.LEDGER}")
        return
    entries = decisions.read(slug)
    errors = validate(entries)
    report_errors(f"{slug}: {decisions.LEDGER}", errors)
    print(f"{slug}: {decisions.LEDGER} — {len(entries)} line(s), OK")
