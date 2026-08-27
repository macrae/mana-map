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
import hashlib
import re
import shutil

from manamap.pilot import check_in, collection
from manamap.pilot.common import (
    BRANCHES_DIR,
    DECKS_DIR,
    deck_dir,
    deck_lifecycle,
    expand_faces,
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

#: What counts as "you can put this in the deck tonight".
#:
#: `elsewhere` IS A LOGISTICS PROBLEM, NOT AN OWNERSHIP ONE, and the first cut
#: got that wrong. A card sleeved in another deck is a card you OWN — the only
#: question is whether you are willing to proxy it or unsleeve the other deck,
#: and that is a fact about the pilot rather than about the card. Reported as
#: unsourced it reads as "buy a second copy", which is advice to spend money on
#: something already in the house.
#:
#: So it stays its own state — the trade-off is real and worth seeing — and
#: `--proxy` says the pilot is happy to proxy across their own decks, which makes
#: it sourced. `buy` is never proxiable here: that would be a claim about a card
#: nobody owns, which is a different decision and not one this should make
#: quietly.
SOURCED = (IN_DECK, BOX)
SOURCED_WITH_PROXY = (IN_DECK, BOX, ELSEWHERE)


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


def _canonical(slug, branch=None):
    """decklist name -> the name every other artifact uses.

    A DOUBLE-FACED CARD IS NAMED TWO WAYS AND BOTH ARE CORRECT. `cards.json` and
    everything derived from it (deck_map, the graph) key the joined `A // B`
    form; a decklist names whichever face the writer had in front of them, and
    for transform cards it HAS to — Scryfall answers the joined name with a 404
    and resolves the front face alone, so a decklist carrying `A // B` cannot be
    fetched at all.

    Left unreconciled the seam is silent and small: the roster joined
    `deck_map`'s names to this module's and lost exactly two cards, under-marking
    the additions 30 against 32 and the buy list 19 against 21. Nothing errors —
    two cards simply render unmarked.
    """
    doc = load_json(deck_dir(slug, branch) / "cards.json")
    if not doc:
        return {}
    out = {}
    for c in (doc.get("cards") or []):
        for face in expand_faces(c.get("name")):
            out[face] = c["name"]
    return out


def meta(slug, branch):
    return load_json(deck_dir(slug, branch) / BRANCH_FILE) or {}


def _named(slug, branch, text_slug, text_branch):
    """Both lists in one vocabulary, preferring what the resolver settled on."""
    canon = dict(_canonical(slug))
    canon.update(_canonical(slug, branch))
    def fix(entries):
        out = {}
        for n, q in entries.items():
            out[canon.get(n, n)] = out.get(canon.get(n, n), 0) + q
        return out
    return fix(text_slug), fix(text_branch)


def diff(slug, branch):
    """What the branch adds and drops against the deck's current list.

    Named for the hands rather than the diff, the same way `paper_state` is:
    `add` is what goes in, `out` is what comes out.
    """
    now, cand = _named(slug, branch,
                       _entries(_list_text(slug)), _entries(_list_text(slug, branch)))
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
        # Either face, for the same reason `_canonical` exists: the holder's
        # cards.json keys the joined form and a decklist may not.
        faces = expand_faces(name)
        if any(expand_faces(c.get("name")) & faces for c in (doc.get("cards") or [])):
            life = deck_lifecycle(d.name)
            locked = bool(deck_versions.paper(d.name))
            out.append({"slug": d.name, "locked": locked,
                        "status": life[0] if life else None})
    return out


def source(slug, branch, proxy=False):
    """Where every card in the branch would come from. The reason this exists.

    OVER THE WHOLE LIST, not over the diff. Walking only the added cards makes
    `in_deck` structurally zero — a card the branch ADDS is by definition not in
    the deck already — and that count is the useful one: it is how much of the
    build is already sleeved in front of you. The unsourced set is identical
    either way, because a card already in the deck is sourced by being there.
    """
    d = diff(slug, branch)
    box = collection.owned_index()
    now, cand = _named(slug, branch,
                       _entries(_list_text(slug)), _entries(_list_text(slug, branch)))
    in_deck = set(now)
    rows = []
    for name in sorted(cand):
        if name in in_deck:
            rows.append({"name": name, "state": IN_DECK, "where": None})
            continue
        if any(f in box for f in expand_faces(name)):
            rows.append({"name": name, "state": BOX,
                         "where": ", ".join(sorted(
                             f for face in expand_faces(name)
                             for f in collection.sources_for(face)))})
            continue
        holders = _deck_holders(name, slug)
        if holders:
            rows.append({"name": name, "state": ELSEWHERE, "where": holders})
            continue
        rows.append({"name": name, "state": BUY, "where": None})
    counts = {s: sum(1 for r in rows if r["state"] == s)
              for s in (IN_DECK, BOX, ELSEWHERE, BUY)}
    ok = SOURCED_WITH_PROXY if proxy else SOURCED
    unsourced = [r["name"] for r in rows if r["state"] not in ok]
    # `counts` are DISTINCT NAMES, not copies — you source Sol Ring once, and a
    # basic land is not a purchase at all.
    return {"slug": slug, "branch": branch, "cards": rows, "counts": counts,
            "unsourced": unsourced, "diff": d, "counts_are": "distinct names",
            "proxy": proxy, "owned_but_elsewhere": counts[ELSEWHERE],
            "mergeable": not unsourced}


def report(slug, branch=None):
    if branch:
        return {"slug": slug, "branches": [_one(slug, branch)]}
    return {"slug": slug, "branches": [_one(slug, b) for b in names(slug)]}


def _one(slug, branch):
    s = source(slug, branch)
    m = meta(slug, branch)
    add = set(s["diff"]["add"])
    return {"name": branch, "opened": m.get("opened"), "why": m.get("why"),
            "base_version": m.get("base_version"),
            "size": s["diff"]["size"], "add": len(s["diff"]["add"]),
            "out": len(s["diff"]["out"]), "counts": s["counts"],
            "unsourced": s["unsourced"], "mergeable": s["mergeable"],
            # PER-CARD PROVENANCE, so a roster can mark a card without asking a
            # second time. `is_new` is the diff's answer and `state` is the
            # collection's; they are different questions and a card can be new
            # and already in a box.
            "cards": [{"name": r["name"], "state": r["state"],
                       "where": r["where"], "is_new": r["name"] in add}
                      for r in s["cards"]],
            "has_cards": (deck_dir(slug, branch) / "cards.json").exists()}


#: A BRANCH THAT CANNOT BE FALSIFIED GETS GRADED ON WHETHER IT DID WHAT IT DOES.
#: The Ur-Dragon treasure branch said "treasure is the engine" and achieved that —
#: 4.4x the hoard — while missing the purpose nobody wrote down, which was winning.
#: It was measured for a week before anyone could say it had failed, because there
#: was nothing it could fail against.
#:
#: So an objective names a MEASURE, a DIRECTION and a NUMBER, and the measure has to
#: be one the bench already computes: the vocabulary is `candidates.AXES`, which is a
#: registry, is independence-checked fleet-wide by `tests/test_metric_hygiene.py`, and
#: cannot silently grow a second name for the same thing.
_OBJECTIVE_RE = re.compile(
    r"^\s*([a-z_0-9]+)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")

_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def parse_objective(text):
    """`"kill_by_8 >= 0.30"` -> {axis, op, value}. Raises with the vocabulary."""
    from manamap.pilot import candidates
    m = _OBJECTIVE_RE.match(text or "")
    if not m:
        raise SystemExit(
            f"--objective must read `<measure> <op> <number>`, e.g. "
            f'"kill_by_8 >= 0.30". Got: {text!r}')
    axis, op, value = m.group(1), m.group(2), float(m.group(3))
    if axis not in candidates.OBJECTIVE_AXES:
        raise SystemExit(
            f"'{axis}' is not something the bench measures. Pick one of:\n  "
            + "\n  ".join(sorted(candidates.OBJECTIVE_AXES))
            + "\n(`candidates.OBJECTIVE_AXES` — wider than what a SWEEP may rank "
              "on, because a correlated measure is a fine thing to aim at and a "
              "useless thing to sort by.)")
    return {"axis": axis, "op": op, "value": value}


def grade_objective(objective, reading, mde=None):
    """met / not met / not resolvable, given a measured value.

    NOT RESOLVABLE IS A THIRD STATE AND IT IS THE HONEST ONE. A reading that
    misses by less than the run could detect has not failed — the run could not
    see it. Reporting that as "not met" is the same error as reporting a null as
    a finding, which this repo refuses in three other places.
    """
    if reading is None:
        return {"state": "not measured",
                "why": "the axis has no reading on this list"}
    hit = _OPS[objective["op"]](reading, objective["value"])
    miss = abs(reading - objective["value"])
    if not hit and mde is not None and miss < mde:
        return {"state": "not resolvable", "reading": reading, "shortfall": round(miss, 4),
                "why": (f"missed by {miss:.4f}, which is under the {mde:.4f} this run "
                        f"could detect — evidence of nothing, not evidence of failure")}
    return {"state": "met" if hit else "not met", "reading": reading,
            "shortfall": None if hit else round(miss, 4)}


def new(slug, branch, text, why=None, at=None, objective=None):
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
           # v2: an objective and a commit trail. v1 files (objective absent)
           # still load — the grading section is simply absent for them, which
           # is the honest report for a branch that never stated one.
           "v": 2,
           "opened": (at or datetime.date.today().isoformat()),
           "why": why or "",
           "objective": objective,
           "commits": [],
           # What it was branched FROM, so a stale branch is visible as one.
           "base_version": deck_versions.report(slug).get("current_version")}
    (path / BRANCH_FILE).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return {"path": str(path), "warnings": checked["warnings"],
            "size": sum(e.get("quantity") or 1 for e in checked["entries"])}


def _write_meta(slug, branch, doc):
    (branch_root(slug) / branch / BRANCH_FILE).write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def _parsed(slug, branch=None):
    """The decklist as ENTRIES, not as a name->copies map.

    `deck_history._entries` is the right reader for a diff and the wrong one
    here: it collapses to `{name: copies}` and throws away `is_commander`, the
    set code, the collector number and the foil marker. Rendering from that would
    drop the `Commander:` header — the list would still be 100 cards and would no
    longer be a Commander deck — and would silently re-resolve a Secret Lair to
    its cheapest reprint on the next `fetch-deck`.
    """
    from manamap.pilot.fetch_deck import parse_decklist
    return parse_decklist(_list_text(slug, branch))


def _resolve_in_list(entries, name):
    """Which entry is this card, allowing for either face of a DFC.

    Same seam `_canonical` closes one function up: the library holds `A // B`
    and a pilot types whichever face they are looking at.
    """
    for e in entries:
        if e["name"] == name:
            return e
    for e in entries:
        if name in expand_faces(e["name"]) or e["name"] in expand_faces(name):
            return e
    return None


def stage(slug, branch, out_name, in_name, strength=None, why=None):
    """One card out, one card in — the staging area, and its provenance.

    THE SWAP IS THE UNIT, not the card. A card added and a card cut are two edits
    a reader has to pair up by hand; a swap is one edit that already says what it
    displaced, and `net-change` can then name WHICH swaps bought the delta rather
    than reporting a list that changed somehow.

    ONE FOR ONE, ALWAYS. The list stays at its legal size through every
    intermediate state, so a branch is never briefly a 98 that some command
    measures. It also means the sweep in `candidates` prices exactly this — a
    substitution, which needs no placebo because the library never shrinks.

    It writes through `check_in.analyze`, so a staged swap gets the same refusals
    a paper list gets: singleton, size, commander, and a name the corpus does not
    know. Editing `decklist.txt` directly would skip all four.
    """
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    entries = _parsed(slug, branch)
    out_e = _resolve_in_list(entries, out_name)
    if out_e is None:
        raise SystemExit(
            f"{out_name!r} is not in {slug}/{branch} — nothing to swap out. "
            f"`deck-branch {slug} diff {branch}` shows what is.")
    if out_e.get("is_commander"):
        raise SystemExit(
            f"{out_e['name']} is the COMMANDER. Changing it is a different deck, "
            f"not a swap — open a new branch from a new list.")
    if _resolve_in_list(entries, in_name) is not None:
        raise SystemExit(f"{in_name!r} is already in {slug}/{branch}.")

    staged_entries = []
    for e in entries:
        if e is out_e:
            # Basics carry a quantity; a singleton does not. Decrement rather
            # than delete, or swapping one Mountain would cut all of them.
            left = int(e.get("quantity") or 1) - 1
            if left > 0:
                staged_entries.append(dict(e, quantity=left))
            continue
        staged_entries.append(e)
    staged_entries.append({"name": in_name, "quantity": 1})

    text = check_in.render_decklist(staged_entries)
    checked = check_in.analyze(slug, text)
    if checked["blocking"]:
        raise SystemExit("Refusing to stage that swap:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")

    doc = meta(slug, branch) or {"slug": slug, "branch": branch, "v": 2}
    doc.setdefault("staged", []).append({
        "at": datetime.date.today().isoformat(),
        "out": out_e["name"], "in": in_name,
        "strength": strength, "why": why or ""})
    _write_meta(slug, branch, doc)
    return {"slug": slug, "branch": branch, "out": out_e["name"], "in": in_name,
            "staged": len(doc["staged"]), "warnings": checked["warnings"]}


def unstage(slug, branch, out_name=None, in_name=None):
    """Put a staged swap back. Reverses the most recent match.

    A staging area you cannot back out of is a decision, not a draft.
    """
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc = meta(slug, branch) or {}
    staged = doc.get("staged") or []
    if not staged:
        raise SystemExit(f"Nothing staged on {slug}/{branch}.")
    hit = None
    for i in range(len(staged) - 1, -1, -1):
        row = staged[i]
        if ((out_name is None or row["out"] == out_name)
                and (in_name is None or row["in"] == in_name)):
            hit = i
            break
    if hit is None:
        raise SystemExit(
            f"No staged swap matches that on {slug}/{branch} — "
            f"`deck-branch {slug} log {branch}` lists them.")
    row = staged[hit]
    entries = _parsed(slug, branch)
    in_e = _resolve_in_list(entries, row["in"])
    if in_e is None:
        raise SystemExit(
            f"{row['in']} is no longer in the list, so this swap cannot be "
            f"reversed cleanly. Edit decklist.txt and drop the record by hand.")
    rebuilt = []
    for e in entries:
        if e is in_e:
            left = int(e.get("quantity") or 1) - 1
            if left > 0:
                rebuilt.append(dict(e, quantity=left))
            continue
        rebuilt.append(e)
    back = _resolve_in_list(rebuilt, row["out"])
    if back is not None:
        rebuilt = [dict(e, quantity=int(e.get("quantity") or 1) + 1)
                   if e is back else e for e in rebuilt]
    else:
        rebuilt.append({"name": row["out"], "quantity": 1})

    checked = check_in.analyze(slug, check_in.render_decklist(rebuilt))
    if checked["blocking"]:
        raise SystemExit("Refusing to unstage that swap:\n  - "
                         + "\n  - ".join(checked["blocking"]))
    (path / "decklist.txt").write_text(
        check_in.render_decklist(checked["entries"]), encoding="utf-8")
    staged.pop(hit)
    doc["staged"] = staged
    _write_meta(slug, branch, doc)
    return {"slug": slug, "branch": branch, "out": row["out"], "in": row["in"],
            "staged": len(staged)}


def commit(slug, branch, message):
    """Freeze this candidate list with a message. NOT a merge.

    THE TWO-STEP IS THE PILOT'S OWN DISTINCTION: a commit says "this is the deck
    I am committed to running"; a merge says "this is the deck". The gap between
    them is CARDBOARD — you can decide on a list months before you own it.
    So a commit is allowed with unsourced cards and a merge is not, which is
    exactly the gate `source()` already enforces.

    It records the decklist sha, so a commit names one exact 99 and a later edit
    is visibly a different one.
    """
    import hashlib
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    if not (message or "").strip():
        raise SystemExit(
            "A commit needs a message. It is the only record of WHY this list "
            "and not the last one.")
    text = _list_text(slug, branch)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc = meta(slug, branch) or {"slug": slug, "branch": branch}
    commits = doc.setdefault("commits", [])
    if commits and commits[-1].get("decklist_sha256") == sha:
        raise SystemExit(
            f"The list has not changed since the last commit "
            f"({commits[-1]['at']}: {commits[-1]['message'][:48]}). "
            f"Edit the list, or say something new about the same one by "
            f"deleting that entry.")
    commits.append({"at": datetime.date.today().isoformat(),
                    "decklist_sha256": sha, "message": message.strip()})
    doc.setdefault("v", 2)
    _write_meta(slug, branch, doc)
    s = source(slug, branch)
    return {"slug": slug, "branch": branch, "n": len(commits),
            "decklist_sha256": sha, "message": message.strip(),
            "unsourced": len(s["unsourced"]),
            "mergeable": not s["unsourced"]}


def log(slug, branch):
    doc = meta(slug, branch) or {}
    return {"slug": slug, "branch": branch,
            "objective": doc.get("objective"), "why": doc.get("why"),
            "opened": doc.get("opened"), "base_version": doc.get("base_version"),
            "commits": doc.get("commits") or [], "merged": doc.get("merged")}


def delete(slug, branch, force=False):
    """Remove a branch. Refuses an unmerged one without `--force`.

    A branch holds measurements that cost real time — a 100-game Forge run is
    45 minutes — so deleting one that was never merged is throwing away the
    evidence for a decision nobody recorded.
    """
    import shutil
    path = branch_root(slug) / branch
    if not path.is_dir():
        raise SystemExit(f"No branch '{branch}' on {slug}.")
    doc = meta(slug, branch) or {}
    if not doc.get("merged") and not force:
        raise SystemExit(
            f"'{branch}' was never merged. Deleting it discards its "
            f"measurements and whatever decision they supported — "
            f"`--force` if that is what you mean.")
    shutil.rmtree(path)
    return {"slug": slug, "branch": branch, "deleted": str(path)}


def merge(slug, branch, write=False, force=False, reason=None, proxy=False,
          run_chain=True):
    """Make the branch the deck's list. Refuses what it cannot honestly apply."""
    s = source(slug, branch, proxy=proxy)
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
    # A BACKUP FIRST, the way `check_in.apply` does. Merge overwrites the deck's
    # tracked list; check-in has made a `.txt.bak` since it shipped and merge
    # never did, which is the more destructive of the two.
    target = deck_dir(slug) / "decklist.txt"
    if target.exists():
        shutil.copy(target, target.with_suffix(".txt.bak"))
    target.write_text(check_in.render_decklist(checked["entries"]), encoding="utf-8")
    out["written"] = True

    # THE CHAIN, because a merge that leaves the figures behind makes the deck
    # read stale forever — `goldfish_metrics.json` and `mana_analysis.json` stamp
    # the decklist sha. `check_in.apply` has run it since it shipped; merge
    # printed "next: fetch-deck" and hoped.
    if run_chain:
        from types import SimpleNamespace
        from manamap.pilot import fetch_deck, goldfish, mana_analysis
        ran = []
        for name, mod in (("fetch-deck", fetch_deck), ("goldfish", goldfish),
                          ("mana-analysis", mana_analysis)):
            try:
                mod.main(SimpleNamespace(slug=slug, branch=None))
                ran.append(name)
            except Exception as exc:                    # pragma: no cover - env
                out.setdefault("chain_failed", []).append(f"{name}: {exc}")
                break
        out["chain"] = ran

    # THE BRANCH RECORDS THAT IT LANDED. Without this the branch survives
    # untouched, `diff` reads +0 -0 forever, and nothing links the resulting
    # version back to the work that produced it.
    from manamap.pilot import deck_versions
    doc = meta(slug, branch) or {"slug": slug, "branch": branch}
    doc["merged"] = {
        "at": datetime.date.today().isoformat(),
        "decklist_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        # The version this became. It is minted from git by the COMMIT the pilot
        # makes next, so at merge time we can only name what it was merged INTO.
        "into_version_before": deck_versions.report(slug).get("current_version"),
    }
    if reason:
        doc["merged"]["forced_reason"] = reason
    _write_meta(slug, branch, doc)

    # WHAT IS NOW STALE. The registry already carries the command for each stage,
    # so this reads `deck_status` rather than keeping a second list that can
    # disagree with it.
    try:
        from manamap.pilot import deck_status
        rep = deck_status.report(slug)
        out["stale"] = [{"stage": r["key"], "how": r.get("how", "")}
                        for r in rep.get("stages", [])
                        if r.get("state") == "STALE"]
    except Exception:                                   # pragma: no cover - env
        out["stale"] = []
    return out


# ── CLI ──────────────────────────────────────────────────────────────────

def _print_source(s):
    c = s["counts"]
    print(f"SOURCING — {s['slug']}/{s['branch']}  "
          f"+{len(s['diff']['add'])} -{len(s['diff']['out'])} vs the current list  "
          f"({s['diff']['size']} cards, {s['diff']['names']} distinct)")
    owned = c[IN_DECK] + c[BOX] + c[ELSEWHERE]
    print(f"  in the deck {c[IN_DECK]} · in a box {c[BOX]} · "
          f"sleeved elsewhere {c[ELSEWHERE]} · to buy {c[BUY]}")
    print(f"  you already own {owned} of {sum(c.values())}"
          + ("  (--proxy counts the elsewhere ones as sourced)"
             if c[ELSEWHERE] and not s.get("proxy") else "") + "\n")
    for state, label in ((BOX, "IN A BOX — free"),
                         (ELSEWHERE, "SLEEVED ELSEWHERE — you own these; proxy or unsleeve"),
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
        raw = getattr(args, "objective", None)
        if not raw:
            raise SystemExit(
                "A branch needs an --objective: `<measure> <op> <number>`, e.g.\n"
                '  --objective "kill_by_8 >= 0.30"\n'
                "A branch that cannot be falsified gets graded on whether it did "
                "what it does. The Ur-Dragon treasure branch said 'treasure is the "
                "engine', achieved that 4.4x over, and missed the purpose nobody "
                "wrote down.")
        objective = parse_objective(raw)
        objective["why"] = getattr(args, "why", None) or ""
        got = new(slug, branch, check_in.read_list(args.source),
                  why=getattr(args, "why", None), objective=objective)
        print(f"Opened {got['path']}  ({got['size']} cards)")
        print(f"  objective: {objective['axis']} {objective['op']} {objective['value']}")
        for w in got["warnings"]:
            print(f"  warning: {w}")
        print(f"  next: `manamap pilot deck-branch {slug} source {branch}`")
        return
    if action in ("stage", "unstage"):
        out_name = getattr(args, "swap_out", None)
        in_name = getattr(args, "swap_in", None)
        if action == "stage" and not (out_name and in_name):
            raise SystemExit(
                f"A swap is ONE CARD OUT AND ONE CARD IN:\n"
                f'  manamap pilot deck-branch {slug} stage {branch} '
                f'--out "<card>" --in "<card>"\n'
                f"`manamap pilot upgrades {slug} --branch {branch}` proposes them.")
        if action == "stage":
            got = stage(slug, branch, out_name, in_name,
                        strength=getattr(args, "strength", None),
                        why=getattr(args, "why", None))
            print(f"Staged on {slug}/{branch}:  - {got['out']}  + {got['in']}")
        else:
            got = unstage(slug, branch, out_name, in_name)
            print(f"Unstaged on {slug}/{branch}:  + {got['out']}  - {got['in']}")
        print(f"  {got['staged']} swap(s) staged. Nothing is measured yet — "
              f"`manamap pilot net-change {slug} --branch {branch}`")
        for w in got.get("warnings") or []:
            print(f"  warning: {w}")
        return
    if action == "commit":
        got = commit(slug, branch, getattr(args, "message", None))
        print(f"Committed #{got['n']} on {slug}/{branch}  "
              f"[{got['decklist_sha256'][:12]}]")
        print(f"  {got['message']}")
        print("  " + ("mergeable" if got["mergeable"] else
                      f"{got['unsourced']} card(s) still to source — a commit is a "
                      f"decision, a merge needs the cardboard"))
        return
    if action == "log":
        got = log(slug, branch)
        if getattr(args, "json", False):
            print(json.dumps(got, indent=1)); return
        o = got.get("objective") or {}
        print(f"BRANCH — {slug}/{branch}   opened {got['opened']}"
              f"   from V{got.get('base_version')}")
        if o:
            print(f"  objective: {o['axis']} {o['op']} {o['value']}"
                  + (f"   — {o['why']}" if o.get("why") else ""))
        else:
            print("  objective: NONE — this branch predates the requirement and "
                  "cannot be graded")
        if not got["commits"]:
            print("\n  no commits yet — `deck-branch "
                  f"{slug} commit {branch} -m \"…\"`")
        for i, c in enumerate(got["commits"], 1):
            print(f"\n  #{i}  {c['at']}  [{c['decklist_sha256'][:12]}]")
            print(f"      {c['message']}")
        m = got.get("merged")
        if m:
            print(f"\n  MERGED {m['at']} into the list after V"
                  f"{m.get('into_version_before')}")
        return
    if action == "delete":
        got = delete(slug, branch, force=getattr(args, "force", False))
        print(f"Deleted {got['deleted']}")
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
        s = source(slug, branch, proxy=getattr(args, "proxy", False))
        if getattr(args, "json", False):
            print(json.dumps(s, indent=1)); return
        _print_source(s)
        return
    if action == "merge":
        got = merge(slug, branch, write=getattr(args, "write", False),
                    force=getattr(args, "force", False),
                    reason=getattr(args, "reason", None),
                    proxy=getattr(args, "proxy", False))
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
        print(f"Merged {branch} into {slug}/decklist.txt  "
              f"(previous list kept as decklist.txt.bak)")
        if got.get("chain"):
            print(f"  recomputed: {', '.join(got['chain'])}")
        for failed in got.get("chain_failed") or []:
            print(f"  CHAIN FAILED — {failed}")
        stale = got.get("stale") or []
        if stale:
            # WHAT THE MERGE COULD NOT REDO. The chain covers the deterministic
            # figures; every narrative artifact was written about the OLD list
            # and no command can regenerate it — that is an agent's work, and
            # naming it here is the difference between a merge that finishes and
            # one that quietly leaves the deck describing a deck you deleted.
            print(f"\n  STALE — written against the previous list ({len(stale)}):")
            for row in stale:
                print(f"    {row['stage']:10} {row['how']}")
        print("\n  NOT COMMITTED — the commit is what `deck-version` numbers and what the")
        print("  captain's log stamps games against, so it stays yours:")
        print(f"      git add data/decks/{slug} && \\")
        print(f"        git commit -m \"{slug}: merge branch {branch}\"")
