"""Pilot: remove a deck that was never anything — the only destructive fleet verb.

THE DOCTRINE IT WORKS AGAINST. `issue_spec.py:288-292` says a published deck is a
record and the honest move is to MARK it, never to delete it. That rule is right
and this does not overturn it — it draws the line the rule always implied. A deck
that was **never sleeved, never played and never published** is not a record of
anything; it is a build plan that did not become a deck, and three of them
(`kianne`, `kinnan`, `blar`) were deterministic baselines built from the whole
format to test the builder.

AND THEY WERE NOT INERT. `deck_branch._deck_holders` walks every `cards.json` on
disk and treats a build plan as a holder of the cards it names, so Edgar's
`bloodline-v4` branch was `mergeable: false` on *"unsleeve The Ozolith from
kianne"* — a deck that has never existed in paper. `collection.py` documents the
same hazard from the other side: counting kinnan's whole-format baseline made 99
unowned cards read as owned. A phantom deck is not neutral; it is a confident,
wrong instruction about cardboard.

WHY THE REFUSAL LIST IS NOT KEYED ON "PUBLISHED". `build_index` computes
`published` as *the frozen magazine renderer ran on this deck*, and
`docs/manual-v5-spec.md` retires that renderer and gives every deck a
`manuals/p/` page. A destructive gate keyed on a predicate whose meaning is
scheduled to change is a gate that silently inverts. `blockers()` asks the three
questions that will still mean the same thing afterwards: was it ever sleeved,
was it ever played, did it ever go to press.

IT DOES NOT COMMIT. `git rm -r` stages the removal so it shows up in
`git status` as a deliberate act rather than as a pile of missing files, and
stops there. The pilot reviews and commits — the same opt-in shape
`serve._build_finish` uses for its own commit, and for the same reason.
"""

import shutil
import subprocess

from manamap.config import DATA_DIR
from manamap.pilot.common import deck_dir, load_json

MANUALS = DATA_DIR.parent / "manuals"


def blockers(slug, base=None):
    """Why this deck may not be deleted. Empty list = it may.

    ONE HOME for the verdict, because the browser needs it too: if `workbench.js`
    re-derived "never sleeved and never played" it would be a second
    implementation of this refusal, free to disagree with the command's. The
    manifest carries the rows; nothing re-computes them.

    `base` IS AN ARGUMENT because `build_index` calls this while walking the
    deck directories and must not raise on one. `common.deck_dir` VALIDATES —
    it raises `FileNotFoundError` naming the decklist to create, which is right
    for a per-deck command and wrong for a manifest pass that has the path in
    its hand already. It also made this unusable under a monkeypatched
    `DECKS_DIR`, which four `build_index` tests noticed at once.
    """
    base = base if base is not None else deck_dir(slug)
    out = []
    if (MANUALS / f"{slug}.html").exists():
        out.append("it went to press — `manuals/{}.html` is a published record; "
                   "archive it instead (`deck-state {} retire`)".format(slug, slug))
    # The log is read from `base`, not by slug, for the same reason. An
    # unreadable or absent log is not a game we played.
    log = base / "log.jsonl"
    try:
        games = sum(1 for line in log.read_text().splitlines() if line.strip()) \
            if log.exists() else 0
    except OSError:
        games = 0
    if games:
        out.append(f"{games} logged game(s) — the captain's log is the one thing "
                   f"here that cannot be re-derived; archive it instead")
    versions = load_json(base / "deck_versions.json", {})
    if versions.get("paper"):
        out.append("it is SLEEVED — withdraw the lock and archive it, so "
                   "`deck_is_apart` can tell the branches its cards are free")
    return out


def holders(slug):
    """Other decks' branch pull lists that currently name this deck as a holder.

    NOT A REFUSAL — a report. When the cards physically exist the right verb was
    archive, and `blockers` already catches that (a deck whose cards exist was
    sleeved). When they never existed, every one of these rows was wrong the
    whole time and deleting the deck is what corrects them. Either way the pilot
    should see the number before they act, and re-run `net-change` after.
    """
    out = []
    for deck in sorted((DATA_DIR / "decks").iterdir()):
        if not deck.is_dir() or deck.name == slug:
            continue
        for branch in sorted((deck / "branches").glob("*")):
            # `recommendation.cost`, NOT `cost`. The first cut read a top-level
            # `cost` key that does not exist, so this returned nothing on every
            # deck and printed nothing — a report that looks correct because it
            # is silent. Caught by deleting kianne, whose holder claim I had read
            # out of the same file by hand ten minutes earlier.
            doc = load_json(branch / "net_change.json", {})
            cost = (doc.get("recommendation") or {}).get("cost") or {}
            named = [r["name"] for r in (cost.get("must_unsleeve") or [])
                     if slug in (r.get("decks") or [])]
            if named:
                out.append((deck.name, branch.name, named))
    return out


def delete(slug, force=False):
    base = deck_dir(slug)
    if not base.is_dir():
        raise SystemExit(f"{slug}: no such deck under data/decks/")
    why = blockers(slug)
    if why and not force:
        lines = "\n".join(f"  - {w}" for w in why)
        raise SystemExit(
            f"{slug}: refusing to delete.\n{lines}\n"
            f"  A deck that was sleeved, played or published is a record. "
            f"`manamap pilot deck-state {slug} retire` keeps it and moves it to "
            f"the archive rack.")

    claims = holders(slug)
    page = MANUALS / "p" / f"{slug}.html"
    paths = [base] + ([page] if page.exists() else [])

    # `git rm -r --cached` then remove, rather than `git rm -rf`: the latter
    # refuses when the tree has uncommitted changes under the path, which is
    # exactly the state a deck being deleted is usually in. Untracked files are
    # not in the index at all, so a failure here is not fatal — the removal is.
    rel = [str(p.relative_to(DATA_DIR.parent)) for p in paths]
    subprocess.run(["git", "rm", "-r", "-q", "--cached", "--ignore-unmatch", *rel],
                   cwd=DATA_DIR.parent, check=False)
    for p in paths:
        shutil.rmtree(p) if p.is_dir() else p.unlink()

    print(f"{slug}: DELETED " + ", ".join(rel))
    if why:
        print("  --force overrode: " + "; ".join(w.split(" — ")[0] for w in why))
    for deck, branch, names in claims:
        print(f"  {deck}/{branch} named this deck as the holder of "
              f"{', '.join(names)} — re-run `manamap pilot regen --only net-change`")
    print(f"  staged, NOT committed. Undo:  git checkout HEAD -- {' '.join(rel)}")
    print(f"  after committing:             git show HEAD~1:data/decks/{slug}/decklist.txt")
    return claims


def main(args):
    delete(args.slug, force=getattr(args, "force", False))
    from manamap.pilot import build_index
    build_index.main()
