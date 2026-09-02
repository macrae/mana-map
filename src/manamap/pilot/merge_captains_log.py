"""Merge the captains-log agent's prose into `captains_log.json`.

The agent writes `.agent-out/captains-log.json` and this puts it beside the log
it renders. Same shape as `merge_debrief`: additive by key, refuses to write on
zero matches, and the MERGE STAMPS RATHER THAN THE AGENT.

THE WHITELIST IS THE POINT. The skeleton — stardate, grouping, version, the
games — is recomputed here from `log.jsonl` on every merge, and only the six
prose sections are taken from the handoff. An agent that invents a stardate, or
groups two nights into one because the story flowed better, is not caught later
by a reader; its invention never lands at all.
"""

import json

from manamap.config import DECKS_DIR
from manamap.pilot import captains_log as cl
from manamap.pilot.common import load_json

AGENT_FILE = "captains-log.json"


def _sections(incoming):
    """Only the prose, and only prose the agent actually wrote."""
    return {k: incoming[k] for k in cl.SECTION_KEYS if k in incoming}


def merge(slug, kind="ship"):
    base = DECKS_DIR / slug
    handoff = load_json(base / ".agent-out" / AGENT_FILE)
    if handoff is None:
        raise SystemExit(
            f"no {base / '.agent-out' / AGENT_FILE} — spawn the `captains-log` "
            f"agent first; it writes there and returns the path")
    if kind not in cl.LOG_KINDS:
        raise SystemExit(f"unknown log kind {kind!r} — one of {list(cl.LOG_KINDS)}")

    incoming = handoff.get("nights") or {}
    if not incoming:
        # MERGING NOTHING AND REPORTING SUCCESS is how a log reads as rendered
        # with every check still green. Same refusal as `merge_debrief`.
        raise SystemExit(f"{AGENT_FILE} carries no `nights` — nothing to merge")

    # THE SKELETON IS RECOMPUTED, ALWAYS. It is a pure function of tracked
    # inputs, so recomputing costs nothing and propagates a corrected stardate or
    # a newly-tagged release to every night without re-spawning the prose. The
    # prose is the expensive half; the facts are free.
    doc = cl.skeleton(slug)
    previous = cl.read(slug).get("nights") or {}

    merged, rejected = [], []
    for key, night in doc["nights"].items():
        # Carry forward prose already on disk for a night the agent did not write.
        for k, block in ((previous.get(key) or {}).get("logs") or {}).items():
            if k in cl.LOG_KINDS:
                night["logs"][k] = block
        if key in incoming:
            night["logs"][kind] = _sections(incoming[key])
            merged.append(key)
    for key in incoming:
        if key not in doc["nights"]:
            rejected.append(key)

    if not merged:
        raise SystemExit(
            f"none of the {len(incoming)} night(s) in {AGENT_FILE} are nights "
            f"{slug} logged a game on — the log is the authority and a rendering "
            f"cannot add nights to it. Rejected: {', '.join(sorted(rejected))}")

    path = base / cl.ARTIFACT
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return merged, rejected, path


def main(args):
    merged, rejected, path = merge(args.slug, getattr(args, "kind", None) or "ship")
    print(f"merged {len(merged)} night(s) into {path}: {', '.join(merged)}")
    if rejected:
        print(f"  REJECTED (no such night in the log): {', '.join(sorted(rejected))}")
    print(f"  next: manamap pilot validate-captains-log {args.slug}")
    return 0
