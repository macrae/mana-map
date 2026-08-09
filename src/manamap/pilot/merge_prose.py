"""Pilot: merge an agent's prose into manual_prose.json — only the keys it owns.

Two agents write into one file. `manual-writer` owns six keys and `pilot-coach`
owns two, and neither may touch the other's: a whole-file copy from either
`.agent-out/` artifact silently deletes the other agent's work, and the loss is
invisible until the manual renders a section short.

That made the merge a ~40-line script every orchestrator re-derived from scratch
at the end of a prose refresh, which is exactly the shape of thing that gets it
subtly wrong once. The ownership map is not a convention to be remembered — it is
`AGENT_ROUTINES[routine]["artifact_keys"]`, the same declaration the cache
fingerprints per key, so this command and the staleness check cannot disagree.

Deterministic, no LLM, no network. It reads what an agent already wrote.
"""

import json

from manamap.config import AGENT_ROUTINES
from manamap.pilot.common import deck_dir

# Which `.agent-out/` file each routine's agent writes. The agent name is the
# filename by the handoff convention (agents return a path, never inline JSON).
AGENT_FILE = {
    "coach-prose": "pilot-coach.json",
    "writer-prose": "manual-writer.json",
}

PROSE_ARTIFACT = "manual_prose.json"


def unwrap(doc, owned):
    """The agent's payload, whether it nested its keys or returned them flat.

    Charters ask for the keys at the top level and mostly comply, but a wrapper
    ({"prose": {...}}) is a recurring and harmless variation. Guessing wrong
    here means merging zero keys and reporting success, so the test is whether
    an OWNED key is actually present rather than what the wrapper is called.
    """
    if any(k in doc for k in owned):
        return doc
    for value in doc.values():
        if isinstance(value, dict) and any(k in value for k in owned):
            return value
    return doc


def merge(slug, routine):
    """Merge `routine`'s owned keys into the deck's manual_prose.json.

    Returns (merged, missing, total_keys). Raises SystemExit if the agent
    produced nothing usable — a silent no-op here reads downstream as "the
    agent wrote nothing worth keeping", which is a different and wrong
    conclusion from "the merge did not find its keys".
    """
    owned = AGENT_ROUTINES[routine]["artifact_keys"]
    base = deck_dir(slug)
    source = base / ".agent-out" / AGENT_FILE[routine]
    if not source.exists():
        raise SystemExit(
            f"{source} not found — spawn {AGENT_ROUTINES[routine]['agent']} for "
            f"{slug} first. Agents hand off by path, so the artifact is the "
            f"contract; there is nothing to merge until it exists.")

    payload = unwrap(json.loads(source.read_text()), owned)
    target = base / PROSE_ARTIFACT
    doc = json.loads(target.read_text()) if target.exists() else {}

    merged = [k for k in owned if k in payload]
    missing = [k for k in owned if k not in payload]
    if not merged:
        raise SystemExit(
            f"{source} holds none of {routine}'s keys ({', '.join(owned)}) — "
            f"refusing to write. Merging nothing and reporting success is how a "
            f"section renders empty with every check still green.")

    for key in merged:
        doc[key] = payload[key]
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return merged, missing, sorted(doc)


def main(args):
    routine = args.routine
    if routine not in AGENT_FILE:
        raise SystemExit(
            f"unknown prose routine {routine!r} — expected one of "
            f"{', '.join(sorted(AGENT_FILE))}")
    merged, missing, keys = merge(args.slug, routine)
    print(f"{args.slug}: merged {len(merged)} key(s) from {routine} — "
          f"{', '.join(merged)}")
    if missing:
        print(f"  MISSING (the agent did not write these): {', '.join(missing)}")
    print(f"  {PROSE_ARTIFACT} now holds {len(keys)} key(s): {', '.join(keys)}")
    # Missing keys are a real finding, not a failure: a partial revision is a
    # supported mode (charters have one), so this must not break a scripted loop.


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot merge-prose <slug> <routine>`.")
