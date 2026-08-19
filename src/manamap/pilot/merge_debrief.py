"""Pilot: merge the debrief agent's annotations into log_annotations.json, by id.

Same argument `merge-prose` and `merge-deck-map` make, one layer over: the
tracked file holds a DERIVED reading keyed by log entry id, and a whole-file
copy from `.agent-out/` would let a scoped spawn (one new game) silently drop
every earlier annotation. The merge writes entry by entry, accepts only ids the
log actually has, and never touches `log.jsonl` — the log is authored and no
command writes it but `deck-notes add`.
"""

import json

from manamap.pilot.common import deck_dir
from manamap.pilot.deck_notes import ANNOTATIONS_FILE, read_log

AGENT_FILE = "debrief.json"


def merge(slug):
    """Returns (merged_ids, rejected_ids, total_annotated)."""
    base = deck_dir(slug)
    source = base / ".agent-out" / AGENT_FILE
    if not source.exists():
        raise SystemExit(
            f"{source} not found — spawn `debrief` for {slug} first. Agents hand "
            f"off by path, so the artifact is the contract; there is nothing to "
            f"merge until it exists.")
    payload = json.loads(source.read_text())
    incoming = payload.get("entries")
    if not isinstance(incoming, dict) or not incoming:
        raise SystemExit(f"{source} holds no `entries` — refusing to write. Merging "
                         f"nothing and reporting success is how a debrief reads as "
                         f"done with every check still green.")

    known = {e["id"] for e in read_log(slug)}
    target = base / ANNOTATIONS_FILE
    doc = json.loads(target.read_text()) if target.exists() else {}
    doc.setdefault("slug", slug)
    doc.setdefault("entries", {})

    merged, rejected = [], []
    for eid in sorted(incoming):
        if eid in known:
            doc["entries"][eid] = incoming[eid]
            merged.append(eid)
        else:
            rejected.append(eid)
    if not merged:
        raise SystemExit(f"{source}: none of its ids {sorted(incoming)} are in the log "
                         f"— refusing to write")
    doc["entries"] = dict(sorted(doc["entries"].items()))
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return merged, rejected, len(doc["entries"])


def main(args):
    merged, rejected, total = merge(args.slug)
    print(f"{args.slug}: merged {len(merged)} debrief entr{'y' if len(merged) == 1 else 'ies'} "
          f"— {', '.join(merged)}; {ANNOTATIONS_FILE} now annotates {total}")
    if rejected:
        print(f"  REJECTED (not in the log): {', '.join(rejected)} — the annotation "
              f"cannot add games the pilot did not log")
    print(f"  next: `manamap pilot validate-debrief {args.slug}`")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot merge-debrief <slug>`.")
