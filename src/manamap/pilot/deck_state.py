"""Pilot: a deck's LIFECYCLE — is this thing still a deck, or is it a pile of cards.

WHY THIS EXISTS. The status was authored by hand-editing `issue.json` and nothing
wrote it, which is how `yawgmoth-swarm` came to be rendering under **SLEEVED**,
with a live paper lock at V5, weeks after it was broken down for parts to fund
another build. The front door's whole job is answering *what can I play tonight*
and it was confidently naming a box of nothing — the same failure
`common.DECK_STATUSES` was moved out of the frozen magazine renderer to prevent,
one surface later.

WHAT IT WRITES. `deck_versions.json`'s `lifecycle` block, through
`deck_versions.set_lifecycle` — which is the one writer of that file, so `paper`
and `lifecycle` cannot end up contradicting each other. See
`common.deck_lifecycle` for why the fact lives there and not on `issue.json`.

IT REGENERATES WHAT IT INVALIDATES, and that is not a convenience. The moment a
deck is archived, `regen.is_retired` starts SKIPPING it (`regen.py:63-80`), so
`manamap pilot regen` will never refresh its `info.json` again — and the front
door would read a file that still says the deck is live, forever. The command
that creates that state is the only place that can still fix it.
"""

import argparse

from manamap.pilot import deck_versions
from manamap.pilot.common import DECK_STATUSES, deck_dir, deck_lifecycle

#: What the pilot types, and the status it sets. `archive` is the pilot's own
#: word for the rack these decks land in; `broken-down` is the vocabulary the
#: rest of the repo reads. Both, rather than making them type the hyphenated one.
ACTIONS = {
    "archive": "broken-down",
    "supersede": "superseded",
    "retire": "retired",
}


def show(slug):
    life = deck_lifecycle(slug)
    block = deck_versions.lifecycle(slug) or {}
    paper = deck_versions.paper(slug)
    if not life:
        print(f"{slug}: LIVE")
    else:
        print(f"{slug}: {life[1]}")
        if block.get("at"):
            print(f"  since   {block['at']}")
        if block.get("reason"):
            print(f"  reason  {block['reason']}")
        print(f"  {life[2]}")
    if paper:
        name = paper.get("release") or f"V{paper.get('version')}"
        print(f"  sleeved {name}, built {paper.get('built_at')}")
    return 0


def main(args):
    slug = args.slug
    if not deck_dir(slug).is_dir():
        raise SystemExit(f"{slug}: no such deck under data/decks/")
    action = getattr(args, "action", None)
    if not action:
        return show(slug)

    if action == "revive":
        deck_versions.set_lifecycle(slug, clear=True)
        print(f"{slug}: LIVE again — the lifecycle mark is withdrawn.")
        print("  Its paper lock was withdrawn when it was archived and is NOT "
              "restored: whether the cards are back in sleeves is a fact about "
              "cardboard, so say so yourself with `deck-version "
              f"{slug} paper`.")
    else:
        status = ACTIONS[action]
        block, withdrew = deck_versions.set_lifecycle(
            slug, status=status, reason=getattr(args, "reason", None))
        print(f"{slug}: {DECK_STATUSES[status][0]} (as of {block['at']})")
        if block["reason"]:
            print(f"  reason  {block['reason']}")
        if withdrew:
            # NEVER SILENTLY. Archiving withdraws a claim about cardboard the
            # pilot made deliberately, so the version it named is printed here
            # and stays in `tags` — the lock is gone, the record of it is not.
            name = withdrew.get("release") or f"V{withdrew.get('version')}"
            print(f"  withdrew the paper lock on {name} "
                  f"(built {withdrew.get('built_at')}) — a deck whose cards "
                  f"are in a pile cannot also have an exact 99 in sleeves.")

    _refresh(slug)
    return 0


def _refresh(slug):
    """`info.json` then the manifest, in that order — the manifest reads neither
    but the deck page reads both, and a half-refreshed pair is the staleness this
    command exists to end."""
    from manamap.pilot import build_index, deck_info

    if (deck_dir(slug) / "cards.json").exists():
        deck_info.main(argparse.Namespace(slug=slug, write=True, as_json=False,
                                          verify=False, branch=None))
        print(f"  rewrote data/decks/{slug}/info.json")
    build_index.main(argparse.Namespace())
