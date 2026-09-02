"""`deck_versions.json` — the form check on the two facts about this deck's cardboard.

**THE LAST TRACKED PILOT ARTIFACT WITH NO GATE** (GitHub #24), and it earned one
the moment it grew a third key. It already held `paper` — the pilot's assertion
that one exact 99 is sleeved, the single most load-bearing authored claim in the
repo, since the front door filters on it — and `tags`, the release names every
other surface quotes. Nothing checked either.

WHAT IT CHECKS is form and contradiction, never figures. Nothing here re-derives
a version: that is a git walk and `deck_versions.versions` owns it.

  1. **`lifecycle.status` IS IN THE VOCABULARY.** `common.deck_status_of`
     deliberately tolerates an unknown value by returning None rather than
     raising — a typo must not take the workbench offline — so a misspelled
     status reads as LIVE and the deck stays in the playable rack. Tolerated
     there, reported here.

  2. **A DEAD DECK IS NOT ALSO SLEEVED.** `lifecycle.status` in
     `UNPLAYABLE_STATUSES` together with a `paper` block is a contradiction:
     "these cards are in a pile" and "this exact 99 is in sleeves" cannot both be
     true. It is an ERROR rather than a warning because the workbench used to
     resolve it by filtering `locked` first, which put a deck that had been
     broken down for parts under **SLEEVED — you can play these tonight**.

     `superseded` is deliberately NOT part of this check, and it reuses
     `UNPLAYABLE_STATUSES` rather than naming its own set so it cannot drift: a
     superseded list is still sleeved and still playable, it is just no longer
     the best version of itself.

  3. **EVERY SHA IS A SHA.** A `decklist_sha256` is what joins a log entry, a
     release tag and a paper lock to one list. A truncated or mistyped one does
     not error anywhere — it simply matches nothing, so a lock silently reports
     `unresolved` and a tag silently stops resolving to a release.

  4. **A TAG NAME IS A RELEASE OR A NICKNAME, NEVER NEARLY A RELEASE.**
     `deck_versions._NEARLY_RELEASE_RE` exists because `v1.2` sorts as a
     nickname — alphabetically, in the wrong place, forever, while looking
     exactly like a version.

WHAT IT DOES NOT FAIL ON. An `unresolved` paper lock — one naming a version git
no longer carries — is reported by `paper_state` as a fact worth seeing, not as
a defect, so it is not re-litigated here. And `slug` is checked against the
directory rather than assumed: it is the only field that can be copied wrong by
duplicating a file.
"""

import json
import re

from manamap.pilot.common import (
    DECK_STATUSES, UNPLAYABLE_STATUSES, deck_dir, report_errors)
from manamap.pilot.deck_versions import (
    _NEARLY_RELEASE_RE, _RELEASE_RE, BASELINE_KEY, LIFECYCLE_KEY, PAPER_KEY,
    TAGS_FILE)

ARTIFACT = TAGS_FILE
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha_errors(where, block):
    sha = (block or {}).get("decklist_sha256")
    if sha is None:
        return [f"{where}: no decklist_sha256 — nothing can join a list to it"]
    if not _SHA256.match(str(sha)):
        return [f"{where}: decklist_sha256 {str(sha)[:16]!r} is not 64 hex "
                f"characters — it will match no list and fail silently"]
    return []


def validate(doc, slug=None):
    """Check the form of a `deck_versions.json`. Returns error strings."""
    errors = []
    if slug and doc.get("slug") != slug:
        errors.append(f"slug is {doc.get('slug')!r} but the file sits in "
                      f"data/decks/{slug}/")

    life = doc.get(LIFECYCLE_KEY)
    if life is not None:
        if not isinstance(life, dict):
            errors.append(f"{LIFECYCLE_KEY} must be an object, got "
                          f"{type(life).__name__}")
            life = {}
        status = life.get("status")
        if status not in DECK_STATUSES:
            errors.append(
                f"{LIFECYCLE_KEY}.status {status!r} is not one of "
                f"{sorted(DECK_STATUSES)} — `deck_status_of` returns None for an "
                f"unknown value, so this deck reads as LIVE and stays in the "
                f"playable rack")
        if not life.get("at"):
            errors.append(f"{LIFECYCLE_KEY}.at is absent — a lifecycle mark with "
                          f"no date cannot be read against anything")
        if status in UNPLAYABLE_STATUSES and doc.get(PAPER_KEY):
            paper = doc[PAPER_KEY]
            errors.append(
                f"{LIFECYCLE_KEY}.status is {status!r} AND a paper lock is set "
                f"(V{paper.get('version')}, built {paper.get('built_at')}). "
                f"A deck whose cards are in a pile cannot also have an exact 99 "
                f"in sleeves — withdraw the lock "
                f"(`deck-version {doc.get('slug', '<slug>')} paper --clear`) or "
                f"revive the deck")

    if doc.get(PAPER_KEY) is not None:
        paper = doc[PAPER_KEY]
        if not isinstance(paper, dict):
            errors.append(f"{PAPER_KEY} must be an object")
        else:
            errors += _sha_errors(PAPER_KEY, paper)
            if not isinstance(paper.get("version"), int):
                errors.append(f"{PAPER_KEY}.version must be the integer ordinal, "
                              f"got {paper.get('version')!r}")
            if not paper.get("built_at"):
                errors.append(f"{PAPER_KEY}.built_at is absent — a lock with no "
                              f"date cannot report drift honestly")

    if doc.get(BASELINE_KEY) is not None:
        errors += _sha_errors(BASELINE_KEY, doc[BASELINE_KEY])

    tags = doc.get("tags")
    if tags is None:
        errors.append("no `tags` key — `_write_tags` always writes one, so its "
                      "absence means this file was hand-edited")
    elif not isinstance(tags, dict):
        errors.append(f"tags must be an object, got {type(tags).__name__}")
    else:
        for name, tag in tags.items():
            errors += _sha_errors(f"tags[{name!r}]", tag)
            if not isinstance((tag or {}).get("version"), int):
                errors.append(f"tags[{name!r}].version must be the integer "
                              f"ordinal, got {(tag or {}).get('version')!r}")
            if not _RELEASE_RE.match(name) and _NEARLY_RELEASE_RE.match(name):
                errors.append(
                    f"tags[{name!r}] is nearly a release and is therefore a "
                    f"NICKNAME — it will sort alphabetically among the nicknames "
                    f"while looking like a version. Use vMAJOR.MINOR.PATCH")
    return errors


def main(args):
    path = deck_dir(args.slug) / ARTIFACT
    if not path.exists():
        # ABSENT IS LEGAL. A deck with no tags, no baseline, no lock and no
        # lifecycle mark has nothing to author — most of the bench is in that
        # state. `deck-status` reports the artifact as missing; this is not the
        # place to turn an absence into a defect.
        print(f"OK   {ARTIFACT} for {args.slug} — absent (nothing authored yet) ◆")
        return
    doc = json.loads(path.read_text())
    errors = validate(doc, slug=args.slug)
    life = (doc.get(LIFECYCLE_KEY) or {}).get("status")
    paper = doc.get(PAPER_KEY) or {}
    state = (DECK_STATUSES[life][0] if life in DECK_STATUSES
             else (f"sleeved V{paper.get('version')}" if paper else "live"))
    report_errors(f"{ARTIFACT} for {args.slug}", errors,
                  f"OK   {ARTIFACT} for {args.slug} — {state}, "
                  f"{len(doc.get('tags') or {})} tag(s) ◆")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot validate-deck-versions <slug>`.")
