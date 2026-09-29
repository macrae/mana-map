"""Merge the sim-debrief agent's prose into `sim_findings.json`.

`captains_log`'s split, which is the strongest synthesis pattern in the repo:
the skeleton is RECOMPUTED, always — it is a pure function of the tracked run
records — and only `prose` is taken from the handoff, per run, whitelisted to
`PROSE_KEYS`. So an agent that invents a finding, a run or a figure lands
nothing: the validator refuses it before the write, and the skeleton it would
have needed to lie about is rebuilt from the records underneath it.
"""

import json

from manamap import config
from manamap.pilot import pilot_findings as pf
from manamap.pilot import validate_sim_findings as vsf
from manamap.pilot.common import load_json

AGENT_FILE = "sim-debrief.json"


def merge(slug):
    base = config.DECKS_DIR / slug
    handoff = load_json(base / ".agent-out" / AGENT_FILE)
    if handoff is None:
        raise SystemExit(f"no {base / '.agent-out' / AGENT_FILE} — spawn the `sim-debrief` agent "
                         f"first; it writes there and returns the path")
    incoming = handoff.get("runs") or {}
    if not incoming:
        raise SystemExit(f"{AGENT_FILE} carries no `runs` — nothing to merge")
    doc = pf.with_prose(pf.skeleton(slug), pf.read(slug))
    merged, rejected = [], []
    for rid, block in incoming.items():
        if rid not in doc["runs"]:
            rejected.append(rid)
            continue
        prose = (block or {}).get("prose") or block or {}
        doc["runs"][rid]["prose"] = {k: prose[k] for k in pf.PROSE_KEYS if k in prose}
        merged.append(rid)
    if not merged:
        raise SystemExit(f"none of the {len(incoming)} run(s) in {AGENT_FILE} is a run of {slug} — "
                         f"the records are the authority. Rejected: {', '.join(sorted(rejected))}")
    errors, notes = vsf.validate(doc, slug)
    if errors:
        raise SystemExit(f"refusing to merge — the prose does not hold to the record:\n  - "
                         + "\n  - ".join(errors))
    path = pf.write(slug, doc)
    return merged, rejected, notes, path


def main(args):
    merged, rejected, notes, path = merge(args.slug)
    print(f"{args.slug}: merged prose for {len(merged)} run(s) -> {path}")
    for n in notes:
        print(f"  ! {n}")
    if rejected:
        print(f"  rejected (not runs of this deck): {', '.join(rejected)}")
