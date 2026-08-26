"""Pilot: a candidate 99 you cannot yet sleeve (`branches/<name>/`).

    manamap pilot deck-branch <slug> list|new|show|diff|source|merge

THE GAP THIS FILLS. A deck had exactly two states: the list in `decklist.txt`,
and nothing. `decklist.txt` is tracked, so writing it MINTS A VERSION, and the
captain's log stamps games against versions — which makes a version you cannot
physically play a version that lies. So a refactor that needs cards you have not
bought had nowhere to live. The Ur-Dragon treasure rebuild is 34 out and 35 in
with 23 cards to source; it was designed, measured, briefed, and unappliable.

A BRANCH IS A WHOLE LIST, NOT A QUEUE OF SWAPS. `pending.json` already holds
decided-but-unapplied in/out pairs and is right for a three-land swap decided on
a Tuesday. It is the wrong shape here for two reasons: 35 swaps written as pairs
is unreadable, and — the load-bearing one — **you can goldfish a list and you
cannot goldfish a swap queue**. Being measurable is the whole point: it is what
lets the pilot find out whether the refactor is better BEFORE spending money on
it.

WHY NOT A GIT BRANCH. Versions derive from `git log` over `decklist.txt`, so a
git branch would mint version numbers that never shipped, and checking one out
would move every OTHER deck's artifacts with it.

WHAT `source` ANSWERS, AND THE CATEGORY NOTHING ELSE COMPUTES. For every card a
branch adds there are four states, not two. `deck_history.pending()` already
derives owned-or-purchase from `COLLECTION_DIR`; what it cannot say is that a
card is SLEEVED IN ANOTHER DECK. That is neither owned nor a purchase — it is a
trade-off, and whether it is available at all depends on the other deck: three
of this prototype's cards sit in `goblin-storm`, which is finished and locked.

Ownership itself is still `pilot/collection.py` and still means A BOX. This adds
no second answer: deck membership is reported as INFORMATION beside ownership,
never folded into it. That distinction is why `collection.include_decks=True`
exists and why no gate uses it.

MERGE IS THE MOMENT THE DECK CHANGES, so it refuses on unsourced cards. It
reuses `check_in.analyze`'s blocking checks verbatim rather than restating them,
writes `decklist.txt`, and prints the commit command WITHOUT committing — the
commit is what `deck-version` numbers and what the log stamps games against, so
it stays the pilot's act. It never touches the `paper` block: three placeholder
locks were once withdrawn for being written to demonstrate machinery, and a
merge that silently claimed cardboard would be that mistake with a command
attached.
"""

import datetime
import json
import re

from manamap.pilot import check_in, collection
from manamap.pilot.common import (
    BRANCHES_DIR,
    DECKS_DIR,
    deck_dir,
    deck_lifecycle,
    load_json,
)
from manamap.pilot.deck_history import _entries

BRANCH_FILE = "branch.json"
#: A branch name is a path segment and a git-visible directory, so keep it to
#: what a shell and a filesystem both treat as one word.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,48}$")

#: The four answers to "can I put this card in the deck". `elsewhere` is the one
#: nothing in the repo computed before, and it is the one that changes a
#: decision: a card sleeved in a LOCKED deck is not available at all.
IN_DECK, BOX, ELSEWHERE, BUY = "in_deck", "box", "elsewhere", "buy"
SOURCED = (IN_DECK, BOX)


def branch_root(slug):
    return deck_dir(slug) / BRANCHES_DIR


def names(slug):
    root = branch_root(slug)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "decklist.txt").exists())


def _list_text(slug, branch=None):
    return (deck_dir(slug, branch) / "decklist.txt").read_text(encoding="utf-8")


def meta(slug, branch):
    return load_json(deck_dir(slug, branch) / BRANCH_FILE) or {}


def diff(slug, branch):
    """What the branch adds and drops against the deck's current list.

    Named for the hands rather than the diff, the same way `paper_state` is:
    `add` is what goes in, `out` is what comes out.
    """
    now = _entries(_list_text(slug))
    cand = _entries(_list_text(slug, branch))
    # COPIES and NAMES are different numbers and the repo has been bitten by
    # conflating them: `size` is what the shuffler sees (36 basics are 36 cards),
    # `names` is what you have to source (36 Forests are one line on a buy list).
    return {"add": sorted(n for n in cand if n not in now),
            "out": sorted(n for n in now if n not in cand),
            "size": sum(cand.values()), "base_size": sum(now.values()),
            "names": len(cand), "base_names": len(now)}


def _deck_holders(name, skip):
    """Which OTHER tracked decks hold this card, with whether they are locked.

    A card in a deck is not a card you own — `collection.py` is the only
    ownership answer and it means a box. This is reported alongside so the pilot
    can see the trade-off, and it carries the holder's lifecycle because a
    finished deck is not a donor.
    """
    from manamap.pilot import deck_versions
    out = []
    for d in sorted(DECKS_DIR.iterdir()):
        if not d.is_dir() or d.name == skip:
            continue
        doc = load_json(d / "cards.json")
        if not doc:
            continue
        if any(c.get("name") == name for c in (doc.get("cards") or [])):
            life = deck_lifecycle(d.name)
            locked = bool(deck_versions.paper(d.name))
            out.append({"slug": d.name, "locked": locked,
                        "status": life[0] if life else None})
    return out


def source(slug, branch):
    """Where every card in the branch would come from. The reason this exists.

    OVER THE WHOLE LIST, not over the diff. Walking only the added cards makes
    `in_deck` structurally zero — a card the branch ADDS is by definition not in
    the deck already — and that count is the useful one: it is how much of the
    build is already sleeved in front of you. The unsourced set is identical
    either way, because a card already in the deck is sourced by being there.
    """
    d = diff(slug, branch)
    box = collection.owned_index()
    in_deck = set(_entries(_list_text(slug)))
    rows = []
    for name in sorted(_entries(_list_text(slug, branch))):
        if name in in_deck:
            rows.append({"name": name, "state": IN_DECK, "where": None})
            continue
        if name in box:
            rows.append({"name": name, "state": BOX,
                         "where": ", ".join(sorted(collection.sources_for(name)))})
            continue
        holders = _deck_holders(name, slug)
        if holders:
            rows.append({"name": name, "state": ELSEWHERE, "where": holders})
            continue
        rows.append({"name": name, "state": BUY, "where": None})
    counts = {s: sum(1 for r in rows if r["state"] == s)
              for s in (IN_DECK, BOX, ELSEWHERE, BUY)}
    unsourced = [r["name"] for r in rows if r["state"] not in SOURCED]
    # `counts` are DISTINCT NAMES, not copies — you source Sol Ring once, and a
    # basic land is not a purchase at all.
    return {"slug": slug, "branch": branch, "cards": rows, "counts": counts,
            "unsourced": unsourced, "diff": d, "counts_are": "distinct names",
            "mergeable": not unsourced}


def report(slug, branch=None):
    if branch:
        return {"slug": slug, "branches": [_one(slug, branch)]}
    return {"slug": slug, "branches": [_one(slug, b) for b in names(slug)]}


def _one(slug, branch):
    s = source(slug, branch)
    m = meta(slug, branch)
    return {"name": branch, "opened": m.get("opened"), "why": m.get("why"),
            "base_version": m.get("base_version"),
            "size": s["diff"]["size"], "add": len(s["diff"]["add"]),
            "out": len(s["diff"]["out"]), "counts": s["counts"],
            "unsourced": s["unsourced"], "mergeable": s["mergeable"],
            "has_cards": (deck_dir(slug, branch) / "cards.json").exists()}


def new(slug, branch, text, why=None, at=None):
    if not NAME_RE.match(branch or ""):
        raise SystemExit(
            f"'{branch}' is not a usable branch name — lowercase letters, digits, "
            f"dot, dash and underscore, starting with a letter or digit.")
    root = branch_root(slug)
    path = root / branch
    if path.exists():
        raise SystemExit(f"{path} already exists — pick another name, or edit it in place.")
    # The SAME refusals a check-in gets. A branch that cannot become a deck is
    # not worth carrying, and finding out at merge time wastes the work in
    # between.
    checked = check_in.analyze(slug, text)
    if checked["blocking"]:
        raise SystemExit("Refusing to open the branch:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    path.mkdir(parents=True)
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")
    from manamap.pilot import deck_versions
    doc = {"slug": slug, "branch": branch,
           "opened": (at or datetime.date.today().isoformat()),
           "why": why or "",
           # What it was branched FROM, so a stale branch is visible as one.
           "base_version": deck_versions.report(slug).get("current_version")}
    (path / BRANCH_FILE).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return {"path": str(path), "warnings": checked["warnings"],
            "size": sum(e.get("quantity") or 1 for e in checked["entries"])}


def merge(slug, branch, write=False, force=False, reason=None):
    """Make the branch the deck's list. Refuses what it cannot honestly apply."""
    s = source(slug, branch)
    text = _list_text(slug, branch)
    checked = check_in.analyze(slug, text)
    blocking = list(checked["blocking"])
    if s["unsourced"] and not force:
        held = {r["name"]: r for r in s["cards"]}
        detail = []
        for n in s["unsourced"][:40]:
            r = held[n]
            if r["state"] == ELSEWHERE:
                where = ", ".join(
                    h["slug"] + (" (LOCKED)" if h["locked"] else "") for h in r["where"])
                detail.append(f"{n} — sleeved in {where}")
            else:
                detail.append(n)
        blocking.append(
            f"{len(s['unsourced'])} card(s) not sourced:\n      "
            + "\n      ".join(detail)
            + "\n    Source them, or `--force --reason \"…\"` to say why anyway.")
    if force and not reason:
        blocking.append("--force needs --reason: a merge that skips the sourcing "
                        "gate should say what it is assuming.")
    out = {"slug": slug, "branch": branch, "blocking": blocking,
           "warnings": checked["warnings"], "source": s, "written": False}
    if blocking or not write:
        return out
    (deck_dir(slug) / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")
    out["written"] = True
    return out


# ── CLI ──────────────────────────────────────────────────────────────────

def _print_source(s):
    c = s["counts"]
    print(f"SOURCING — {s['slug']}/{s['branch']}  "
          f"+{len(s['diff']['add'])} -{len(s['diff']['out'])} vs the current list  "
          f"({s['diff']['size']} cards, {s['diff']['names']} distinct)")
    print(f"  in the deck {c[IN_DECK]} · in a box {c[BOX]} · "
          f"sleeved elsewhere {c[ELSEWHERE]} · to buy {c[BUY]}\n")
    for state, label in ((BOX, "IN A BOX — free"),
                         (ELSEWHERE, "SLEEVED ELSEWHERE — a trade-off, not a purchase"),
                         (BUY, "TO BUY")):
        rows = [r for r in s["cards"] if r["state"] == state]
        if not rows:
            continue
        print(f"  {label} ({len(rows)}):")
        for r in rows:
            if state == ELSEWHERE:
                where = ", ".join(h["slug"] + (" — LOCKED" if h["locked"] else "")
                                  for h in r["where"])
                print(f"    {r['name']:38} {where}")
            elif state == BOX:
                print(f"    {r['name']:38} {r['where']}")
            else:
                print(f"    {r['name']}")
        print()
    print("  MERGEABLE" if s["mergeable"]
          else f"  NOT MERGEABLE — {len(s['unsourced'])} card(s) unsourced")


def main(args):
    slug, action = args.slug, args.action
    branch = getattr(args, "name", None)
    if action == "list":
        doc = report(slug)
        if getattr(args, "json", False):
            print(json.dumps(doc, indent=1)); return
        if not doc["branches"]:
            print(f"No branches on {slug}. "
                  f"`manamap pilot deck-branch {slug} new <name> --from <file>`")
            return
        print(f"BRANCHES — {slug} ({len(doc['branches'])})")
        for b in doc["branches"]:
            mark = "mergeable" if b["mergeable"] else f"{len(b['unsourced'])} to source"
            print(f"  {b['name']:24} opened {b['opened']}  +{b['add']:<3} -{b['out']:<3} "
                  f"[{b['size']:>3}]  {mark}")
            if b["why"]:
                print(f"      {b['why'][:96]}")
        return
    if not branch:
        raise SystemExit(f"`deck-branch {slug} {action}` needs a branch name.")
    if action == "new":
        got = new(slug, branch, check_in.read_list(args.source), why=getattr(args, "why", None))
        print(f"Opened {got['path']}  ({got['size']} cards)")
        for w in got["warnings"]:
            print(f"  warning: {w}")
        print(f"  next: `manamap pilot deck-branch {slug} source {branch}`")
        return
    if action == "show":
        print(_list_text(slug, branch), end="")
        return
    if action == "diff":
        d = diff(slug, branch)
        if getattr(args, "json", False):
            print(json.dumps(d, indent=1)); return
        print(f"DIFF — {slug}/{branch} vs the current list "
              f"({d['base_size']} -> {d['size']} cards)")
        print(f"\n  OUT ({len(d['out'])}):")
        for n in d["out"]:
            print(f"    - {n}")
        print(f"\n  IN ({len(d['add'])}):")
        for n in d["add"]:
            print(f"    + {n}")
        return
    if action == "source":
        s = source(slug, branch)
        if getattr(args, "json", False):
            print(json.dumps(s, indent=1)); return
        _print_source(s)
        return
    if action == "merge":
        got = merge(slug, branch, write=getattr(args, "write", False),
                    force=getattr(args, "force", False),
                    reason=getattr(args, "reason", None))
        if got["blocking"]:
            print(f"REFUSED — {slug}/{branch} cannot be merged:")
            for b in got["blocking"]:
                print(f"  - {b}")
            raise SystemExit(1)
        for w in got["warnings"]:
            print(f"  warning: {w}")
        if not got["written"]:
            print(f"{slug}/{branch} is mergeable. `--write` applies it.")
            return
        print(f"Merged {branch} into {slug}/decklist.txt")
        print("  NOT COMMITTED — the commit is what `deck-version` numbers and what the")
        print("  captain's log stamps games against, so it stays yours:")
        print(f"      git add data/decks/{slug}/decklist.txt && \\")
        print(f"        git commit -m \"{slug}: merge branch {branch}\"")
        print(f"  then: `manamap pilot fetch-deck {slug}`")
