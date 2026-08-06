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
guard makes for the rendered issue.

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

from manamap.pilot.common import deck_dir


def _git(*args):
    """Run a git command from the repo root; return stdout, or None on failure."""
    from manamap.config import DECKS_DIR
    root = DECKS_DIR.parent.parent
    try:
        out = subprocess.run(["git", "-C", str(root), *args],
                             capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


def _entries(text):
    """Card names -> copies from a decklist blob, mainboard only.

    Reuses the one parser both the CLI and the browser are fixture-locked to,
    so a history entry can never disagree with what `fetch-deck` read.
    """
    from manamap.pilot.fetch_deck import parse_decklist
    out = {}
    for e in parse_decklist(text):
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
    changes, previous = [], {}
    for rev in revs:
        blob = _git("show", f"{rev['sha']}:{path}")
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


def _owned_index():
    """Every card in share/*.txt -> the files holding it, front faces included.

    The pilot's box, read the same way `pool-facts` reads a collection.
    """
    from manamap.config import DECKS_DIR
    from manamap.pilot.fetch_deck import parse_decklist
    share = DECKS_DIR.parent.parent / "share"
    index = {}
    if not share.is_dir():
        return index
    for path in sorted(share.glob("*.txt")):
        try:
            entries = parse_decklist(path.read_text())
        except (OSError, ValueError):
            continue
        for entry in entries:
            name = entry["name"]
            faces = {name} | ({f.strip() for f in name.split(" // ")}
                              if " // " in name else set())
            for face in faces:
                index.setdefault(face, set()).add(path.stem)
    return index


def pending(slug):
    """The swaps the deck is CONSIDERING but has not made.

    `considering.json` is exactly this and always was; it simply had no name for
    it. Reporting it beside the applied history is what makes the pair a plan
    rather than two unrelated files.

    Ownership is DERIVED when the artifact does not carry it. `acquisition` is
    optional and one regeneration dropped it from a deck that previously had it,
    which made every pending swap read as "buy". An absent field is not evidence
    that a card is unowned, so `share/` is checked before anything is called a
    purchase.

    Note this is the ONLY ownership question left in the repo, and it is about a
    physical collection (`share/*.txt`), not about a deck. The Short List itself
    is ownership-free by design: it names ten cards worth knowing about, and
    whether one is already in a box is not what the list is for.
    """
    from manamap.pilot.common import load_json
    doc = load_json(deck_dir(slug) / "considering.json", default=None)
    if not doc:
        return []
    box = _owned_index()

    out = []
    for e in doc.get("ten", []):
        name = e.get("card")
        acq = e.get("acquisition") or {}
        status, source = acq.get("status"), acq.get("file")
        if not status:
            files = box.get(name) or set()
            if files:
                status, source = "owned", ", ".join(sorted(files))
            else:
                status = "purchase"
            source = source or None
        out.append({
            "in": name,
            "out": e.get("natural_cut"),
            "closes": e.get("closes"),
            "acquisition": status,
            "derived": not (e.get("acquisition") or {}).get("status"),
            "source_file": source,
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
            mark = "owned" if p["acquisition"] == "owned" else "buy"
            src = f"  <- {p['source_file']}" if p.get("source_file") else ""
            lines.append(f"      [{mark:<5}] + {str(p['in']):<34}"
                         + (f"- {p['out']}" if p["out"] else "(no cut named)") + src)
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
        path = deck_dir(args.slug) / out if "/" not in str(out) else out
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-history <slug>`.")
