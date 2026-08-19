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
from datetime import date

from manamap.pilot import deck_history as dh
from manamap.pilot.common import deck_dir, load_json
from manamap.pilot.deck_notes import read_log

TAGS_FILE = "deck_versions.json"


def _sha(blob):
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def versions(slug):
    """Every content-distinct decklist, oldest first, each with the byte-shas that
    map to it (a comment-only edit adds a sha, not a version)."""
    path = f"data/decks/{slug}/decklist.txt"
    out, previous = [], None
    for rev in dh.revisions(slug):
        blob = dh._git("show", f"{rev['sha']}:{path}")
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


def working_sha(slug):
    path = deck_dir(slug) / "decklist.txt"
    return _sha(path.read_text(encoding="utf-8")) if path.exists() else None


def resolve(slug, ref, vers=None):
    """A version by number ("V4" / "4"), by tag name, or by git sha prefix."""
    vers = vers if vers is not None else versions(slug)
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.upper().startswith("V") and ref[1:].isdigit():
        ref = ref[1:]
    if ref.isdigit():
        n = int(ref)
        return next((v for v in vers if v["version"] == n), None)
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
    return {"slug": slug, "current_version": current_version,
            "working_decklist_sha256": current, "versions": vers,
            "tags": tags(slug), "unmatched_log_entries": unmatched, "notes": notes}


def tag(slug, name, ref=None, note=None):
    """Name a version. Authored, tracked, never derived."""
    name = str(name or "").strip()
    if not name or "/" in name or " " in name:
        raise SystemExit("a tag is one word: letters, digits, dashes (e.g. the-lock)")
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
    doc["tags"][name] = {"version": target["version"], "sha": target["sha"],
                         "decklist_sha256": target["decklist_sha256"],
                         "at": date.today().isoformat(), "note": note or ""}
    doc["tags"] = dict(sorted(doc["tags"].items()))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
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
    print(f"DECK VERSIONS — {doc['slug']} ({len(doc['versions'])}, "
          f"current: {'V' + str(doc['current_version']) if doc['current_version'] else 'uncommitted'})\n")
    for v in doc["versions"]:
        tag_s = f"  [{', '.join(v['tags'])}]" if v["tags"] else ""
        cur = " ◀ current" if v["version"] == doc["current_version"] else ""
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
        if getattr(args, "as_json", False):
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        else:
            _print_list(doc)
        return
    if action == "tag":
        if not args.ref:
            raise SystemExit("tag needs a name: `deck-version <slug> tag <name> [--at V4]`")
        v = tag(slug, args.ref, ref=getattr(args, "at", None), note=getattr(args, "note", None))
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
    raise SystemExit("Run via `manamap pilot deck-version <slug> list|show|tag|restore`.")
