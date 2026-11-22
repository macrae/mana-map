"""Pilot: what has actually changed in this deck, derived from git.

Three places recorded swaps before this and none of them could be trusted:
comment blocks in `decklist.txt` (prose, hand-written, and the deck it describes
moves out from under it), `HISTORY.md` on four of eight decks (append-only and
append-forgotten), and `considering.json`, which is replaced wholesale on every
regeneration so a ten that gets applied leaves no trace of having existed.

The ledger is already on disk and it is exact: `decklist.txt` is tracked, so
every change to the 99 is a commit. Deriving from `git log` cannot drift,
cannot be forgotten, and needs no maintenance — the same argument
`build-index` makes for the deck manifest, and the same one the manual-freshness
guard makes for the rendered deck page.

**Report-only and computed on demand**, like `deck-facts` and `impact`. It reads
git; it never writes a tracked path.

What it cannot know, stated rather than guessed: *why* a card moved. The commit
subject is the closest honest proxy and is reported verbatim as `reason`. A swap
whose commit says "Ten bought cards" tells you less than one that says what the
cards were for, which is an argument for good commit subjects rather than for a
second hand-kept file that would disagree with this one.
"""

import json
import subprocess

from manamap import config
from manamap.pilot.common import deck_dir, resolve_out_path

# The git root is a sibling of `data/`, not of `data/decks/`. Named once because
# `config.DECKS_DIR.parent.parent` read as an accident at both call sites, and a wrong
# number of `.parent`s fails silently as "no history".
_REPO_ROOT = config.DECKS_DIR.parent.parent


def _git(*args):
    """Run a git command from the repo root; return stdout, or None on failure."""
    root = _REPO_ROOT
    try:
        out = subprocess.run(["git", "-C", str(root), *args],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


#: ONE SUBPROCESS FOR EVERY BLOB, INSTEAD OF ONE EACH. `history` and
#: `deck_versions.versions` both walk every revision of a decklist and ran
#: `git show <sha>:<path>` per revision — an N+1 against a process boundary.
#: Profiled on `deck-info ur-dragon`: 14 git invocations, 1.33s of a 3.9s
#: command, and `deck_versions.report` is called twice per compose so the deck
#: paid it twice. `git cat-file --batch` answers the whole list down one pipe.
def _blobs(revs, path):
    """`{sha: text}` for `<sha>:<path>` at every revision, in one git process.

    A revision whose blob is missing is simply absent from the map — the same
    contract the per-call `_git("show", …)` had when it returned None, so every
    caller's `if blob is None: continue` still reads correctly.
    """
    if not revs:
        return {}
    spec = "".join(f"{rev['sha']}:{path}\n" for rev in revs)
    try:
        out = subprocess.run(["git", "-C", str(_REPO_ROOT), "cat-file", "--batch"],
                             input=spec.encode(), capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    blobs, buf, i = {}, out.stdout, 0
    for rev in revs:
        nl = buf.find(b"\n", i)
        if nl < 0:
            break
        header = buf[i:nl].decode("utf-8", "replace").split()
        # "<oid> missing" for anything git could not resolve.
        if len(header) < 3:
            i = nl + 1
            continue
        size = int(header[2])
        body = buf[nl + 1:nl + 1 + size]
        blobs[rev["sha"]] = body.decode("utf-8", "replace")
        i = nl + 1 + size + 1          # +1 for the trailing newline git adds
    return blobs


def _entries(text):
    """Card names -> copies from a decklist blob, mainboard only.

    Reuses the one parser both the CLI and the browser are fixture-locked to,
    so a history entry can never disagree with what `fetch-deck` read.
    """
    from manamap.pilot.fetch_deck import parse_mainboard
    out = {}
    # `parse_mainboard`, not `parse_decklist`: a sideboard edit is not a swap
    # and must never read as a version.
    for e in parse_mainboard(text):
        out[e["name"]] = out.get(e["name"], 0) + int(e.get("quantity") or 1)
    return out


def revisions(slug):
    """Every commit that touched this deck's decklist, oldest first."""
    path = f"data/decks/{slug}/decklist.txt"
    log = _git("log", "--follow", "--format=%H%x1f%ad%x1f%s", "--date=short",
               "--", path)
    if not log:
        return []
    rows = []
    for line in log.strip().splitlines():
        sha, date, subject = line.split("\x1f", 2)
        rows.append({"sha": sha, "date": date, "subject": subject})
    return list(reversed(rows))


def history(slug):
    """In/out per revision, derived by diffing the parsed decklist at each commit."""
    path = f"data/decks/{slug}/decklist.txt"
    revs = revisions(slug)
    blobs = _blobs(revs, path)
    changes, previous = [], {}
    for rev in revs:
        blob = blobs.get(rev["sha"])
        if blob is None:
            continue
        current = _entries(blob)
        added = sorted(n for n in current if n not in previous)
        removed = sorted(n for n in previous if n not in current)
        requantified = sorted(
            f"{n} {previous[n]}->{current[n]}"
            for n in current if n in previous and current[n] != previous[n])
        if previous and (added or removed or requantified):
            changes.append({
                "sha": rev["sha"][:12], "date": rev["date"],
                "reason": rev["subject"],
                "in": added, "out": removed, "quantity_changes": requantified,
                "size": sum(current.values()),
            })
        elif not previous:
            changes.append({
                "sha": rev["sha"][:12], "date": rev["date"],
                "reason": rev["subject"], "in": [], "out": [],
                "quantity_changes": [], "size": sum(current.values()),
                "note": "first tracked revision — the baseline, not a swap",
            })
        previous = current
    return changes


def pending(slug):
    """The swaps the deck is CONSIDERING but has not made.

    `considering.json` is exactly this and always was; it simply had no name for
    it. Reporting it beside the applied history is what makes the pair a plan
    rather than two unrelated files.

    A pending swap is a card in and a card out, and nothing about where the card
    would come from. Until 2026-11-22 this derived an `acquisition` (owned or
    purchase) from the pilot's boxes; the bench no longer has a notion of what
    anyone holds, so an `acquisition` block an older `considering.json` still
    carries is read past. The Short List was ownership-free by design all along:
    it names ten cards worth knowing about, and whether one is already in a box
    was never what the list was for.
    """
    from manamap.pilot.common import load_json
    doc = load_json(deck_dir(slug) / "considering.json", default=None)

    out = []
    # Two sources, tagged. `considering.json` is the retired Short List (frozen on
    # published decks; prescriptions carry its rule now) — ten cards
    # worth knowing about, authored by an agent and regenerated wholesale.
    # `pending.json` is a decision somebody actually made and has not applied
    # yet, which the ten cannot hold: it is fixed at ten, forbids a pick already
    # in the deck, and cannot express a three-for-three swap as one change.
    try:
        from manamap.pilot.validate_pending import summarise
        for e in summarise(slug)["entries"]:
            ins = e.get("in") or []
            outs = e.get("out") or []
            ins = [ins] if isinstance(ins, str) else ins
            outs = [outs] if isinstance(outs, str) else outs
            out.append({
                "in": ", ".join(ins) or None,
                "out": ", ".join(outs) or None,
                "closes": e.get("closes"),
                "source": "pending.json",
                "id": e.get("id"),
                "state": e.get("state"),
            })
    except Exception:
        pass

    if not doc:
        return out
    for e in doc.get("ten", []):
        out.append({
            "in": e.get("card"),
            "out": e.get("natural_cut"),
            "closes": e.get("closes"),
            "source": "considering.json",
            "id": None,
            "state": None,
        })
    return out


def analyze(slug):
    deck_dir(slug)  # fail early with the standard message if the slug is wrong
    applied = history(slug)
    prop = pending(slug)
    notes = []
    if not applied:
        notes.append(
            "No git history for this decklist — either the repo has no commits "
            "touching it, or git is unavailable. The applied ledger is empty "
            "for that reason, not because the deck has never changed.")
    notes.append(
        "`reason` is the commit subject verbatim. It is the closest honest "
        "proxy for why a card moved; nothing in the repo records intent per "
        "card, and a second hand-kept file would disagree with this one.")
    if prop:
        unnamed = sum(1 for p in prop if not p["out"])
        if unnamed:
            notes.append(
                f"{unnamed} of {len(prop)} pending swaps name no card to cut. "
                f"They are additions looking for a slot, and applying them "
                f"needs a decision this file cannot make.")
    return {"slug": slug, "applied": applied, "pending": prop, "notes": notes}


def format_report(doc):
    lines = [f"{doc['slug']}: deck history — {len(doc['applied'])} tracked revision(s)"]
    for rev in doc["applied"]:
        head = f"  {rev['date']}  {rev['sha']}  [{rev['size']} cards]  {rev['reason'][:58]}"
        lines.append(head)
        if rev.get("note"):
            lines.append(f"      {rev['note']}")
        for name in rev["out"]:
            lines.append(f"      - {name}")
        for name in rev["in"]:
            lines.append(f"      + {name}")
        for change in rev["quantity_changes"]:
            lines.append(f"      ~ {change}")
    if doc["pending"]:
        lines.append(f"  PENDING — {len(doc['pending'])} considered, not applied:")
        for p in doc["pending"]:
            lines.append(f"      + {str(p['in']):<34}"
                         + (f"- {p['out']}" if p["out"] else "(no cut named)"))
    for note in doc["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def main(args):
    doc = analyze(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_report(doc))
    out = getattr(args, "out", None)
    if out:
        path = resolve_out_path(out, args.slug, "deck-history")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-history <slug>`.")
