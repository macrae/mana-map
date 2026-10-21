"""Form-check a pod's seat audit (`data/pods/<name>.audit.json`, 2026-10-21): the
harness stamp, the definition, one row per seat with a verdict in the vocabulary and a
reason wherever it fails, and a table verdict that agrees with its seats."""

from manamap.pilot.common import report_errors


def main(args):
    from manamap.sim import seat_audit
    pod = args.pod
    doc = seat_audit.load(pod)
    if doc is None:
        print(f"{pod}: no seat audit on disk (absent means absent) — "
              f"`manamap pilot forge-seat-audit {pod} --write`")
        return
    errors = seat_audit.validate(doc)
    n_bad = sum(1 for r in doc.get("seats", []) if r.get("verdict") != "PILOTABLE")
    report_errors(f"{pod} — seat audit", errors,
                  f"OK   {pod} — seat audit {doc.get('verdict')} ({len(doc.get('seats', []))} seats, "
                  f"{n_bad} failing) under overrides {(doc.get('harness') or {}).get('overrides')}")
