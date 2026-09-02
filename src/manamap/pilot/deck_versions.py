"""Pilot: deck versions — every list this deck has been, numbered, tagged, and joined
to the games played on it.

WHY THIS EXISTS. `deck-history` answers "what moved and when"; nothing answered
"which list was I holding in game four", "what did I call the build I took to the
Orinda weekly", or "put the deck back to the list that went 4–1". The captain's log
stamps every entry with the sha of `decklist.txt` as it stood, and every change to
that file is a commit, so the join is already on disk — this module performs it.

VERSIONS ARE DERIVED; TAGS ARE AUTHORED. A version is a commit whose parsed
decklist differs from the one before — `V1` is the first tracked list, `V2` the
first swap, and so on. Derived from git, like `deck-history`, so it cannot drift
and needs no maintenance. A comment-only edit to `decklist.txt` changes the file's
bytes but not the 99; it gets no new version, but its byte-sha still maps to the
version it belongs to, which is how a log entry stamped on it finds its list.

A tag is a name the pilot gives a version ("the-lock", "orinda-4-1") and lives in
the tracked, authored `deck_versions.json` — the same rule the log and `issue.json`
(the deck's authored identity) live under. Tags are the one piece of version data a browser can read without
git, which is why they are a file and the version list is not.

WHY THE VERSION LIST IS NOT A TRACKED FILE. The commit that changes `decklist.txt`
receives its sha AFTER anything written in the same commit, so a generated
`versions.json` would be one version behind forever. Computed on demand here; the
history viewer in the viz gets its copy from a deploy-time step with git available.

`restore` is the only writer of `decklist.txt` in this module and it is a DRY RUN
unless `--write` is passed; git can always undo it, and `fetch-deck` must follow.
"""

import hashlib
import json
import re
from datetime import date

from manamap.pilot import deck_history as dh
from manamap.pilot import common
from manamap.pilot.common import deck_dir, load_json
from manamap.pilot.deck_notes import read_log

TAGS_FILE = "deck_versions.json"
PAPER_KEY = "paper"
BASELINE_KEY = "baseline"
#: The deck's LIFECYCLE — broken down, superseded, retired, or absent for a live
#: deck. It moved here from `issue.json` (see `common.deck_lifecycle`); this
#: module owns the write because `paper` is the fact it contradicts.
LIFECYCLE_KEY = common.LIFECYCLE_KEY


def _sha(blob):
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def versions(slug):
    """Every content-distinct decklist, oldest first, each with the byte-shas that
    map to it (a comment-only edit adds a sha, not a version)."""
    path = f"data/decks/{slug}/decklist.txt"
    revs = dh.revisions(slug)
    # One git process for every blob rather than one each — see `dh._blobs`.
    blobs = dh._blobs(revs, path)
    b = baseline(slug)
    if b and b.get("decklist_sha256"):
        # Skip forward to the commit that first carries the baseline LIST. Matching
        # on content rather than on a commit sha is what lets the baseline be
        # written before the commit that would name it exists.
        want = b["decklist_sha256"]
        for i, rev in enumerate(revs):
            blob = blobs.get(rev["sha"])
            if blob is not None and _sha(blob) == want:
                revs = revs[i:]
                break
    out, previous = [], None
    for rev in revs:
        blob = blobs.get(rev["sha"])
        if blob is None:
            continue
        entries = dh._entries(blob)
        byte_sha = _sha(blob)
        if entries == previous:
            out[-1]["decklist_sha256s"].append(byte_sha)
            out[-1]["sha"] = rev["sha"][:12]        # latest commit carrying this list
            out[-1]["date"] = rev["date"]
            continue
        added = sorted(n for n in entries if n not in (previous or {}))
        removed = sorted(n for n in (previous or {}) if n not in entries)
        out.append({
            "version": len(out) + 1,
            "sha": rev["sha"][:12], "first_sha": rev["sha"][:12],
            "date": rev["date"], "first_date": rev["date"],
            "subject": rev["subject"],
            "decklist_sha256": byte_sha,             # the first blob of this version
            "decklist_sha256s": [byte_sha],
            "size": sum(entries.values()),
            "in": added, "out": removed,
        })
        previous = entries
    return out


def tags(slug):
    doc = load_json(deck_dir(slug) / TAGS_FILE) or {}
    return doc.get("tags") or {}


def baseline(slug):
    """The list version numbering RESTARTS from, or None.

    WHY AN AUTHORED INPUT TO A DERIVED QUANTITY. This module's doctrine is that
    versions are derived from git so they cannot drift and need no maintenance,
    and a baseline is the one deliberate exception: a deck gets rebuilt in paper
    and the pilot wants its history to start there, not to carry the development
    scaffolding that preceded it. Nothing is destroyed — the earlier commits stay
    in git and remain reachable by sha — they simply stop being numbered.

    IT NAMES A LIST, NOT A COMMIT, and that is the load-bearing part. A commit's
    sha is not knowable inside the commit that creates it, which is exactly why
    `versions.json` cannot be tracked. A decklist's content hash is known before
    anything is committed, so the baseline can be written in the same breath as
    the list it names.
    """
    doc = load_json(deck_dir(slug) / TAGS_FILE) or {}
    return doc.get(BASELINE_KEY) or None


def set_baseline(slug, sha256=None, note=None, clear=False):
    """Restart version numbering at a list. Authored, tracked, never derived."""
    path = deck_dir(slug) / TAGS_FILE
    doc = load_json(path) or {"slug": slug, "tags": {}}
    if clear:
        if BASELINE_KEY not in doc:
            raise SystemExit(f"{slug}: no baseline to clear")
        doc.pop(BASELINE_KEY)
        _write_tags(path, doc)
        return None
    sha256 = sha256 or working_sha(slug)
    if not sha256:
        raise SystemExit(f"{slug}: no decklist.txt to baseline")
    doc[BASELINE_KEY] = {"decklist_sha256": sha256,
                         "at": date.today().isoformat(), "note": note or ""}
    _write_tags(path, doc)
    return doc[BASELINE_KEY]


def paper(slug):
    """The version that is SLEEVED, or None. Authored, tracked, never derived.

    LOCKED IS AN ASSERTION AND NOTHING ELSE COULD MAKE IT. `DECK_STATUSES` marks
    only the DEAD decks — `broken-down`, `superseded`, `retired` — so "live" has
    always meant "not explicitly killed": an absence, not a claim. Nothing in the
    repo said *this deck is sleeved and I can play it tonight*, which is the one
    question the workbench's front door needs to filter on, and it is not
    derivable from any artifact because it is a fact about cardboard.

    It hangs off a VERSION rather than a deck, because what is sleeved is one
    exact list. That is also what makes drift computable: the repo moves on every
    swap and the cardboard does not.
    """
    doc = load_json(deck_dir(slug) / TAGS_FILE) or {}
    sleeved = doc.get(PAPER_KEY) or None
    if not sleeved:
        return None

    # THE RELEASE TAG, RESOLVED. The paper lock stores an ORDINAL — "V3" — which
    # is the number git can derive and means nothing to a reader. `v1.0.2` is
    # what the pilot ships and asks for, and it lives one key over in `tags`.
    # Matched on the decklist SHA rather than on the ordinal alone: a re-baseline
    # renumbers the ordinals, and a tag pointing at a list that no longer has
    # that number would otherwise be silently wrong rather than absent.
    sleeved = dict(sleeved)
    for name, tag in (doc.get("tags") or {}).items():
        if (tag.get("decklist_sha256")
                and tag.get("decklist_sha256") == sleeved.get("decklist_sha256")):
            sleeved["release"] = name
            break
    return sleeved


def paper_state(slug, vers=None, current_version=None):
    """Locked, in sync, and if not — exactly which cards differ.

    The two sides of the drift are the physical instruction, which is why they
    are named for the hands rather than for the diff: `pull` is in the sleeved
    list and no longer in the repo's, `add` is the reverse.
    """
    p = paper(slug)
    if not p:
        return None
    vers = vers if vers is not None else versions(slug)
    if current_version is None:
        by_sha = {s: v["version"] for v in vers for s in v["decklist_sha256s"]}
        current_version = by_sha.get(working_sha(slug))
    n = p.get("version")
    # `release` rides along so a surface can name what the pilot SHIPS rather
    # than the ordinal git derives. It is the same resolution `paper()` does; not
    # propagating it meant the workbench had the tag available nowhere and fell
    # back to "V3", which is the number and not the name.
    out = {"version": n, "release": p.get("release"),
           "built_at": p.get("built_at"), "note": p.get("note") or "",
           "locked": True, "in_sync": None, "versions_behind": None, "drift": None}
    target = next((v for v in vers if v["version"] == n), None)
    if target is None:
        # A lock naming a version git no longer carries — a rewritten history, or
        # a hand-edited file. Report it rather than crashing or silently
        # unlocking: an unresolvable lock is a fact worth seeing.
        out["unresolved"] = True
        return out
    out["in_sync"] = current_version == n
    if current_version is not None:
        out["versions_behind"] = max(0, current_version - n)
    if not out["in_sync"]:
        d = diff_vs_working(slug, target)
        out["drift"] = {"pull": d["in_then_not_now"], "add": d["in_now_not_then"]}
    return out


def set_paper(slug, ref=None, built_at=None, note=None, clear=False):
    """Assert (or withdraw) that a version is the one sleeved."""
    path = deck_dir(slug) / TAGS_FILE
    doc = load_json(path) or {"slug": slug, "tags": {}}
    if clear:
        if PAPER_KEY not in doc:
            raise SystemExit(f"{slug}: not locked — nothing to clear")
        doc.pop(PAPER_KEY)
        _write_tags(path, doc)
        return None
    # BOTH DIRECTIONS, OR THE INVARIANT IS HALF AN INVARIANT. `set_lifecycle`
    # withdraws the lock when a deck is archived; without this, locking an
    # archived deck rebuilt the same contradiction from the other side — and it
    # is easy to do by accident, because `paper` with no ref is a WRITE that
    # reads like a report. Caught in exactly that way, minutes after archiving
    # yawgmoth-swarm: `deck-version yawgmoth-swarm paper`, meant as a check,
    # silently re-locked a deck that is in a pile.
    life = common.deck_lifecycle(slug)
    if life and life[0] in common.UNPLAYABLE_STATUSES:
        raise SystemExit(
            f"{slug}: {life[1]} \u2014 its cards are in a pile, so no list of "
            f"them is sleeved. `manamap pilot deck-state {slug} revive` first "
            f"if you have rebuilt it.")
    vers = versions(slug)
    if not vers:
        raise SystemExit(f"{slug}: no committed versions to lock")
    target = resolve(slug, ref, vers) if ref else None
    if ref and target is None:
        raise SystemExit(f"{slug}: no version {ref!r}")
    if target is None:
        cur = working_sha(slug)
        target = next((v for v in vers if cur in v["decklist_sha256s"]), None)
        if target is None:
            raise SystemExit(f"{slug}: decklist.txt is uncommitted — commit it, or "
                             f"name the version you sleeved with --at")
    doc[PAPER_KEY] = {"version": target["version"], "sha": target["sha"],
                      "decklist_sha256": target["decklist_sha256"],
                      "built_at": built_at or date.today().isoformat(),
                      "note": note or ""}
    _write_tags(path, doc)
    return target


def lifecycle(slug):
    """The raw lifecycle block, or None. `common.deck_lifecycle` is the PREDICATE."""
    doc = load_json(deck_dir(slug) / TAGS_FILE) or {}
    return doc.get(LIFECYCLE_KEY) or None


def set_lifecycle(slug, status=None, reason=None, clear=False):
    """Mark a deck broken down / superseded / retired, or bring it back.

    RETURNS `(block, withdrew)`, where `withdrew` is the paper lock this call
    removed or None.

    ARCHIVING WITHDRAWS THE SLEEVED LOCK, and that is the whole reason this
    writer lives beside `set_paper` rather than in its own module. A deck whose
    cards are in a pile cannot also have an exact 99 in sleeves; leaving both
    set leaves every reader to guess which one is true, and the workbench guessed
    WRONG — it filtered on `locked` before `status`, so a deck that had been
    broken down for parts rendered under SLEEVED, on the one screen whose whole
    job is answering what you can play tonight.

    `superseded` is deliberately NOT unlocked: a superseded list can still be
    sleeved and played, it is just no longer the best version of itself. That is
    the same distinction `UNPLAYABLE_STATUSES` draws, so it is read from there
    rather than re-listed here.

    The reason is a note about a DECISION, never a claim about cardboard — it
    records why the pilot did this, and nothing derives anything from it.
    """
    path = deck_dir(slug) / TAGS_FILE
    doc = load_json(path) or {"slug": slug, "tags": {}}
    doc.setdefault("slug", slug)
    if clear:
        if LIFECYCLE_KEY not in doc:
            raise SystemExit(f"{slug}: already live \u2014 nothing to revive")
        doc.pop(LIFECYCLE_KEY)
        _write_tags(path, doc)
        return None, None
    if status not in common.DECK_STATUSES:
        raise SystemExit(
            f"{slug}: {status!r} is not a lifecycle status \u2014 "
            f"one of {sorted(common.DECK_STATUSES)}")
    withdrew = None
    if status in common.UNPLAYABLE_STATUSES and doc.get(PAPER_KEY):
        withdrew = doc.pop(PAPER_KEY)
    doc[LIFECYCLE_KEY] = {"status": status, "at": date.today().isoformat(),
                          "reason": reason or ""}
    _write_tags(path, doc)
    return doc[LIFECYCLE_KEY], withdrew


# A RELEASE tag, in the pilot's semantics (docs/pilot.md "What a version bump
# means"). `v` is optional so a hand-typed `1.2.0` is still a release.
_RELEASE_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
# ...and this is what a near-miss looks like: nothing but an optional `v`, digits
# and dots, yet not a valid release. `v1.2`, `v1`, `1.2.3.4`. Caught rather than
# silently filed as a nickname, because a nickname sorts alphabetically and would
# then sit in the wrong place forever while looking exactly like a version.
#
# It requires the WHOLE name to be version-shaped, not merely to start with a
# digit: `3rd-rebuild` and `2026-rebuild` are perfectly good nicknames, and a
# first cut at this rule refused both.
_NEARLY_RELEASE_RE = re.compile(r"^v?[\d.]+$")


def _tag_key(name):
    """Sort releases numerically, nicknames alphabetically, releases first.

    Plain `sorted()` is lexical, so **`v1.10.0` sorts before `v1.9.0`** — the
    tenth minor bump of a deck files itself between the first and the second and
    stays there. A deck reaches v1.10.0 by shipping ten changes that alter what
    it can do, which is a perfectly ordinary year of a deck's life, so this is a
    bug with a date on it rather than a hypothetical.
    """
    m = _RELEASE_RE.match(name)
    if m:
        return (0, tuple(int(g) for g in m.groups()), "")
    return (1, (), name)


def _write_tags(path, doc):
    """One writer, so `paper` and `tags` cannot disagree about key order."""
    doc.setdefault("tags", {})
    doc["tags"] = dict(sorted(doc["tags"].items(), key=lambda kv: _tag_key(kv[0])))
    ordered = {"slug": doc.get("slug")}
    # LIFECYCLE FIRST, because it is the fact that overrides the rest: a deck in
    # a pile has no sleeved list and no working version, and a reader scanning
    # this file should meet that before the numbers it invalidates.
    if LIFECYCLE_KEY in doc:
        ordered[LIFECYCLE_KEY] = doc[LIFECYCLE_KEY]
    if BASELINE_KEY in doc:
        ordered[BASELINE_KEY] = doc[BASELINE_KEY]
    if PAPER_KEY in doc:
        ordered[PAPER_KEY] = doc[PAPER_KEY]
    ordered["tags"] = doc["tags"]
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")


def working_sha(slug):
    path = deck_dir(slug) / "decklist.txt"
    return _sha(path.read_text(encoding="utf-8")) if path.exists() else None


def resolve(slug, ref, vers=None):
    """A version by number ("V4" / "4"), by tag name, or by git sha prefix."""
    vers = vers if vers is not None else versions(slug)
    ref = str(ref or "").strip()
    if not ref:
        return None
    # A `V` prefix is an UNAMBIGUOUS request for a version number, so a miss is a
    # miss. A BARE digit string is ambiguous — it could be a sha prefix — so that
    # one falls through. The first version of this fall-through did not draw the
    # distinction and let `V9` reach the sha matcher, where a hex sha beginning
    # with 9 matches about one time in seven: an intermittent wrong answer, and a
    # flake in `test_an_arm_resolves_from_a_version_or_the_working_copy`.
    explicit_version = ref.upper().startswith("V") and ref[1:].isdigit()
    if explicit_version:
        ref = ref[1:]
    if ref.isdigit():
        n = int(ref)
        hit = next((v for v in vers if v["version"] == n), None)
        if hit is not None or explicit_version:
            return hit
        # FALL THROUGH for a bare number only. A git sha is hex, so an all-digit
        # short prefix is not exotic: (10/16)**7 is about one 7-char prefix in
        # 27. Returning None here meant a perfectly good sha silently resolved
        # to nothing roughly 3.7% of the time — which is exactly how often
        # `test_tags_are_authored_and_resolve` failed under the full suite while
        # passing every time in isolation, and it would have done the same to a
        # pilot typing `deck-version <slug> show 1234567`.
    t = tags(slug).get(ref)
    if t:
        return next((v for v in vers if v["version"] == t.get("version")), None)
    return next((v for v in vers if v["first_sha"].startswith(ref) or v["sha"].startswith(ref)
                 or v["decklist_sha256"].startswith(ref)), None)


def report(slug):
    """Versions, tags and the log joined: which games were played on which list."""
    vers = versions(slug)
    by_sha = {s: v["version"] for v in vers for s in v["decklist_sha256s"]}
    current = working_sha(slug)
    current_version = by_sha.get(current)
    games = {v["version"]: [] for v in vers}
    unmatched = []
    for e in read_log(slug):
        n = by_sha.get(e.get("decklist_sha256"))
        if n is None and e.get("decklist_sha256") == current:
            n = current_version          # the working copy, possibly uncommitted
        if n is None:
            unmatched.append(e["id"])
        else:
            games[n].append(e)
    for v in vers:
        played = games[v["version"]]
        v["games"] = len(played)
        v["record"] = {"win": sum(1 for e in played if e.get("result") == "win"),
                       "loss": sum(1 for e in played if e.get("result") == "loss"),
                       "draw": sum(1 for e in played if e.get("result") == "draw")}
        v["log_ids"] = [e["id"] for e in played]
        v["tags"] = sorted(name for name, t in tags(slug).items()
                           if t.get("version") == v["version"])
    notes = []
    if current is not None and current_version is None:
        notes.append("decklist.txt on disk differs from every committed version — "
                     "the working copy is uncommitted; games logged on it show as "
                     "unmatched until it is committed")
    if unmatched:
        notes.append(f"{len(unmatched)} log entr{'y' if len(unmatched) == 1 else 'ies'} "
                     f"({', '.join(unmatched)}) stamped with a decklist no commit carries")
    base = baseline(slug)
    if base:
        notes.append(f"version numbering restarts at V1 from the list baselined "
                     f"{base.get('at','')} — earlier commits are still in git and "
                     f"reachable by sha, they are simply no longer numbered")
    state = paper_state(slug, vers, current_version)
    if state and state.get("in_sync") is False and state.get("drift"):
        d = state["drift"]
        notes.append(f"the sleeved list is V{state['version']}; the repo is at "
                     f"V{current_version} — pull {len(d['pull'])}, add {len(d['add'])} "
                     f"to bring the cardboard level")
    return {"slug": slug, "current_version": current_version,
            "working_decklist_sha256": current, "versions": vers,
            "tags": tags(slug), "paper": state, "baseline": base,
            "unmatched_log_entries": unmatched, "notes": notes}


def release_suggestion(slug, version):
    """What to call this version, now that it exists in paper. Proposes only.

    **v0.x is a list; v1.0.0 is a deck you can hold.** A deck lives on the bench
    at v0.1.0, v0.4.2, whatever it takes — digital, unproven, freely rewritten.
    Sleeving it is the act that makes it real, so that is where 1.0.0 belongs,
    and the major version then means something physical rather than something a
    maintainer felt.

    THE TOOL PROPOSES AND THE PILOT CONFIRMS, which is why this prints a command
    instead of writing a tag. `deck_history` is explicit that *why* a card moved
    is not knowable from a commit; the same modesty applies to what a version
    should be CALLED. Auto-tagging would also make `paper` non-idempotent — a
    re-run would either fail on the duplicate or silently move the name.

    Silent when the version already carries a release, and silent when the deck
    has released before: proposing v1.0.0 to a deck already at v2.1.0 is worse
    than proposing nothing.
    """
    have = [t for t in (version.get("tags") or []) if _RELEASE_RE.match(t)]
    if have:
        return []
    doc = load_json(deck_dir(slug) / TAGS_FILE) or {}
    if any(_RELEASE_RE.match(t) for t in (doc.get("tags") or {})):
        return []
    return [f"  no release tag yet — sleeving it is what makes it 1.0.0:",
            f"      manamap pilot deck-version {slug} tag v1.0.0 "
            f"--at V{version['version']} --note \"…\""]


def tag(slug, name, ref=None, note=None, force=False):
    """Name a version. Authored, tracked, never derived."""
    name = str(name or "").strip()
    if not name or "/" in name or " " in name:
        raise SystemExit("a tag is one word: letters, digits, dashes (e.g. the-lock)")
    # A near-miss is refused rather than accepted as a nickname. `v1.2` files
    # itself alphabetically among the release tags and looks correct in the
    # listing; the mistake only surfaces once there are enough versions for the
    # order to matter, which is long after the tag was written.
    if _NEARLY_RELEASE_RE.match(name) and not _RELEASE_RE.match(name):
        raise SystemExit(
            f"{name!r} looks like a release tag but is not one — use "
            f"vMAJOR.MINOR.PATCH (e.g. v1.2.0), or a name that does not start "
            f"with a digit. See docs/pilot.md, 'What a version bump means'.")
    vers = versions(slug)
    if not vers:
        raise SystemExit(f"{slug}: no committed versions to tag")
    target = resolve(slug, ref, vers) if ref else None
    if ref and target is None:
        raise SystemExit(f"{slug}: no version {ref!r}")
    if target is None:
        cur = working_sha(slug)
        target = next((v for v in vers if cur in v["decklist_sha256s"]), None)
        if target is None:
            raise SystemExit(f"{slug}: decklist.txt is uncommitted — commit it, or name "
                             f"a version with --at")
    path = deck_dir(slug) / TAGS_FILE
    doc = load_json(path) or {"slug": slug, "tags": {}}
    doc.setdefault("tags", {})
    # Re-tagging is a silent overwrite: `tag v1.0.0` at a later version moves the
    # name and the old version keeps its games while losing its label, so every
    # artifact that quoted "v1.0.0" now names a different 99. A tag is a claim
    # about one exact list.
    prev = doc["tags"].get(name)
    if prev and prev.get("version") != target["version"] and not force:
        raise SystemExit(
            f"{slug}: {name!r} already names {prev.get('version')} "
            f"({prev.get('decklist_sha256', '')[:12]}) — a tag is a claim about one "
            f"exact list. Pick another name, or pass --force to move it.")
    doc["tags"][name] = {"version": target["version"], "sha": target["sha"],
                         "decklist_sha256": target["decklist_sha256"],
                         "at": date.today().isoformat(), "note": note or ""}
    _write_tags(path, doc)
    return target


def blob_at(slug, version):
    path = f"data/decks/{slug}/decklist.txt"
    return dh._git("show", f"{version['first_sha']}:{path}")


def diff_vs_working(slug, version):
    then = dh._entries(blob_at(slug, version) or "")
    now = dh._entries((deck_dir(slug) / "decklist.txt").read_text(encoding="utf-8"))
    return {"in_then_not_now": sorted(n for n in then if n not in now),
            "in_now_not_then": sorted(n for n in now if n not in then)}


def restore(slug, version, write=False):
    blob = blob_at(slug, version)
    if blob is None:
        raise SystemExit(f"{slug}: cannot read V{version['version']} from git")
    d = diff_vs_working(slug, version)
    if write:
        (deck_dir(slug) / "decklist.txt").write_text(blob, encoding="utf-8")
    return d


def _print_list(doc):
    p = doc.get("paper") or {}
    lock = ""
    if p:
        lock = (f", SLEEVED V{p['version']}"
                + ("" if p.get("in_sync") else f" — {p.get('versions_behind') or '?'} behind"))
    print(f"DECK VERSIONS — {doc['slug']} ({len(doc['versions'])}, "
          f"current: {'V' + str(doc['current_version']) if doc['current_version'] else 'uncommitted'}"
          f"{lock})")
    b = doc.get("baseline")
    if b:
        print(f"  re-baselined {b.get('at','')} — earlier history is pre-baseline "
              f"and still in git" + (f": {b['note']}" if b.get("note") else ""))
    print()
    for v in doc["versions"]:
        tag_s = f"  [{', '.join(v['tags'])}]" if v["tags"] else ""
        cur = " ◀ current" if v["version"] == doc["current_version"] else ""
        if p and v["version"] == p.get("version"):
            cur += " ◆ SLEEVED"
        r = v["record"]
        rec = f"{r['win']}W {r['loss']}L" if v["games"] else "—"
        print(f"V{v['version']:<3} {v['first_date']}  {v['first_sha']}  [{v['size']:>3}]  "
              f"+{len(v['in']):<2} -{len(v['out']):<2}  games {v['games']:<2} {rec:<8}"
              f"{tag_s}{cur}")
        print(f"      {v['subject'][:80]}")
    for n in doc["notes"]:
        print(f"\n  note: {n}")


def main(args):
    slug = args.slug
    action = getattr(args, "action", "list") or "list"
    if action == "list":
        doc = report(slug)
        if getattr(args, "write", False):
            # THE DEPLOY-TIME ARTIFACT. Versions are a git walk, so this cannot be
            # committed: the commit that changes `decklist.txt` receives its sha
            # AFTER anything written in the same commit, which would leave a
            # tracked copy one version behind forever. Gitignored, and regenerated
            # by `make demo` immediately before it is read.
            out = deck_dir(slug) / "versions.json"
            out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
            print(f"{slug}: wrote {out.name} "
                  f"({len(doc['versions'])} version(s), current "
                  f"V{doc['current_version']})")
            return
        if getattr(args, "as_json", False):
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        else:
            _print_list(doc)
        return
    if action == "baseline":
        if getattr(args, "clear", False):
            set_baseline(slug, clear=True)
            print(f"{slug}: baseline cleared — the full git history is numbered again")
            return
        b = set_baseline(slug, note=getattr(args, "note", None))
        doc = report(slug)
        print(f"{slug}: BASELINED at the working list ({b['decklist_sha256'][:12]}) — "
              f"version numbering restarts at V1 from here")
        print(f"  {len(doc['versions'])} version(s) now numbered; earlier commits stay "
              f"in git and are still reachable by sha")
        return
    if action == "paper":
        if getattr(args, "clear", False):
            set_paper(slug, clear=True)
            print(f"{slug}: lock withdrawn — no longer marked as built in paper")
            return
        v = set_paper(slug, ref=getattr(args, "at", None) or getattr(args, "ref", None),
                      built_at=getattr(args, "built_at", None),
                      note=getattr(args, "note", None))
        state = paper_state(slug)
        print(f"{slug}: SLEEVED at V{v['version']} ({v['first_sha']}, {v['first_date']}) "
              f"— built in paper, playable at a table → {TAGS_FILE}")
        for line in release_suggestion(slug, v):
            print(line)
        if state and state.get("in_sync") is False:
            d = state["drift"]
            print(f"  the repo has moved on: pull {len(d['pull'])}, add {len(d['add'])} "
                  f"to bring the cardboard level")
        return
    if action == "tag":
        if not args.ref:
            raise SystemExit("tag needs a name: `deck-version <slug> tag <name> [--at V4]`")
        v = tag(slug, args.ref, ref=getattr(args, "at", None),
                note=getattr(args, "note", None), force=getattr(args, "force", False))
        print(f"{slug}: tagged V{v['version']} ({v['first_sha']}, {v['first_date']}) as "
              f"{args.ref!r} → {TAGS_FILE}")
        return
    v = resolve(slug, getattr(args, "ref", None))
    if v is None:
        raise SystemExit(f"{slug}: no version {getattr(args, 'ref', None)!r} — "
                         f"`deck-version {slug} list` shows them (V4, a tag, or a sha prefix)")
    if action == "show":
        d = diff_vs_working(slug, v)
        print(f"V{v['version']}  {v['first_date']}  {v['first_sha']}  [{v['size']} cards]  "
              f"{v['subject']}")
        print(f"  vs the working list: -{len(d['in_then_not_now'])} +{len(d['in_now_not_then'])}")
        for n in d["in_then_not_now"]:
            print(f"    then, not now: {n}")
        for n in d["in_now_not_then"]:
            print(f"    now, not then: {n}")
        if getattr(args, "full", False):
            print()
            print(blob_at(slug, v))
        return
    if action == "restore":
        write = getattr(args, "write", False)
        d = restore(slug, v, write=write)
        verb = "RESTORED" if write else "would restore (dry run; add --write)"
        print(f"{slug}: {verb} decklist.txt to V{v['version']} ({v['first_sha']})")
        for n in d["in_now_not_then"]:
            print(f"    - {n}")
        for n in d["in_then_not_now"]:
            print(f"    + {n}")
        if write:
            print(f"  next: `manamap pilot fetch-deck {slug}`, then goldfish → mana-analysis; "
                  f"commit the decklist so the log can stamp it as a version")
        return
    raise SystemExit(f"unknown action {action!r}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-version <slug> list|show|tag|restore|paper`.")
