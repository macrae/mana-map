"""An agent's output becomes a TRACKED artifact — one home for it, and a stamp.

THE HOLE THIS FILLS. `merge_prose` and `merge_debrief` exist because their
artifacts are merged by key or by id. Everything else — `engine.json`,
`tutor_guide.json`, `strategic_frame.json`, `deck_map.json`, `diagnosis.json` —
is a WHOLE-FILE handoff, and the skills said "copy that to the tracked path".
So the copy was a `cp`, and a `cp` cannot stamp.

WHAT THAT COST, measured on the fleet the day this was written:

    engine.json          stamped 0, unstamped 9
    tutor_guide.json     stamped 0, unstamped 9
    manual_prose.json    stamped 1, unstamped 8
    strategic_frame.json stamped 3, unstamped 6

`deck_status.STAGES` declares `engine.json`'s staleness key as
`decklist_sha256` and has done for months. NOT ONE ENGINE MODEL IN THE FLEET
CARRIES ANY SHA AT ALL, so the check was wired and could never fire — and
`deck_status.py` itself records the previous instance of the same failure:
edgar-vampires' engine named twelve cards that were not in the 99, with the row
reading OK.

It happened again. Ur-Dragon's `engine.json` names SEVENTEEN cards that are not
in the 99 — Counterspell, Smothering Tithe, Swan Song, Moltensteel Dragon — and
it was found by accident, because a tutor-guide agent mentioned it in passing
while doing something else.

THE MERGE STAMPS, NOT THE AGENT. Same reasoning `merge_prose` gives: the merge
is the moment the artifact BECOMES current, so it is the only place that can
honestly assert what it is current against. An agent's stamp is a claim about a
file it cannot see the final state of.

AND A STAMP IS NEVER BACKFILLED. Stamping the nine stale engine models would
assert they are current, which is the exact lie the missing stamp allowed —
worse, because it would be written deliberately. A stale artifact is re-run
through this command or it stays unstamped and reads as unknown.
"""

import json
import shutil

from manamap.config import AGENT_ROUTINES, DECKS_DIR
from manamap.pilot.common import load_json

#: Where each routine's agent leaves its handoff, when the filename is not just
#: `<agent>.json`. Read from the registry first; this covers the ones whose
#: agent writes under a task-specific name.
AGENT_FILE = {
    "deck-engine": "deck-engineer.json",
    "tutor-guide": "pilot-notes-tutor-guide.json",
    "strategic-frame": "strategy-researcher.json",
    "deck-map-names": "deck-cartographer.json",
    "deck-diagnosis": "deck-doctor.json",
    "deck-recon": "deck-doctor-recon.json",
    "candidate-pool": "deck-analyst.json",
    "poh-procedures": "poh-procedures.json",
}

#: Routines with their own merge module, because their artifact is merged rather
#: than replaced. This command refuses them by name rather than silently
#: clobbering an accumulating file.
MERGED_ELSEWHERE = {
    "pilot-notes": "manamap pilot merge-prose <slug> pilot-notes",
    "debrief": "manamap pilot merge-debrief <slug>",
    "captains-log": "manamap pilot merge-captains-log <slug>",
}

STAMP_KEY = "decklist_sha256_prefix"


def agent_file(routine):
    spec = AGENT_ROUTINES.get(routine)
    if not spec:
        raise SystemExit(f"unknown routine {routine!r} — see `cache-status --help`")
    return AGENT_FILE.get(routine, f"{spec['agent']}.json")


def decklist_sha(slug):
    """The sha `cards.json` was built from — the same one `merge_prose` uses."""
    return (load_json(DECKS_DIR / slug / "cards.json") or {}).get("decklist_sha256")


def install(slug, routine, force=False):
    if routine in MERGED_ELSEWHERE:
        raise SystemExit(
            f"{routine} is MERGED, not replaced — its artifact accumulates and a "
            f"whole-file copy would drop what is already there. Use:\n"
            f"    {MERGED_ELSEWHERE[routine]}")
    spec = AGENT_ROUTINES[routine]
    base = DECKS_DIR / slug
    src = base / ".agent-out" / agent_file(routine)
    dst = base / spec["artifact"]
    if not src.exists():
        raise SystemExit(
            f"no {src} — spawn the `{spec['agent']}` agent first; it writes there "
            f"and returns the path")
    doc = load_json(src)
    if doc is None:
        raise SystemExit(f"{src} is not readable JSON")

    # THE STAMP, written here and nowhere else. Absent when `cards.json` has no
    # sha (a deck that has never been fetched), because an absent stamp reads as
    # UNKNOWN and a wrong one reads as current.
    sha = decklist_sha(slug)
    if sha:
        doc[STAMP_KEY] = sha[:12]
    elif not force:
        raise SystemExit(
            f"{slug} has no decklist_sha256 in cards.json — run `fetch-deck "
            f"{slug}` first, or pass --force to install without a stamp (the "
            f"artifact will read as staleness-unknown forever)")

    if dst.exists():
        shutil.copy2(dst, base / ".agent-out" / f"{spec['artifact']}.prev")
    dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return dst, sha


def main(args):
    routine = args.routine
    dst, sha = install(args.slug, routine, force=getattr(args, "force", False))
    spec = AGENT_ROUTINES[routine]
    print(f"installed {spec['agent']} -> {dst}"
          + (f"  (stamped {sha[:12]})" if sha else "  (UNSTAMPED — no decklist sha)"))
    from manamap.pilot.deck_status import VALIDATED
    gate = VALIDATED.get(spec["artifact"])
    if gate:
        print(f"  next: manamap pilot {gate.rsplit('.', 1)[-1].replace('_', '-')} "
              f"{args.slug}")
    print(f"  then: manamap pilot cache-record {args.slug} --routine {routine}")
    return 0
