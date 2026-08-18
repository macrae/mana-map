"""Pilot: is this deck finished, and is any of it stale?

WHY THIS EXISTS. The deck lifecycle is eighteen skills and forty-four subcommands,
and until now nothing said what a COMPLETE deck looks like. Each phase knew its
own inputs; none knew the sequence. So a capability added in one development cycle
— `deck-map`, its naming pass, `analyze-engine` — was reachable only by somebody
who remembered it existed, and a deck built the following month silently inherited
the old pipeline instead.

A runbook alone does not fix that: prose drifts and nobody diffs it. This does,
because it is checked. `STAGES` below is the single machine-readable statement of
what a deck can have and in what order, and the skill that sequences the work
reads the same list rather than restating it.

STALENESS IS THE OTHER HALF, and it is the failure that actually ships. Most of
these artifacts stamp the `decklist_sha256` they were derived from. A deck whose
`engine.json` was written against a decklist two swaps ago is not incomplete — it
is CONFIDENT AND WRONG, which is worse, and it looks finished from every angle
except this one.

Read-only. Computed on demand, never committed.
"""

import json

from manamap.pilot.common import deck_dir, load_json, presentable, report_errors

# The lifecycle, in dependency order. `sha` names the key an artifact stamps the
# decklist hash into, when it stamps one — an artifact with no `sha` cannot be
# checked for staleness and says so rather than being assumed current.
STAGES = [
    ("brief",      "brief.json",             None,  False, "the written intent a build starts from"),
    ("decklist",   "decklist.txt",           None,  True,  "the 99, tracked, and the identity everything else hangs on"),
    ("cards",      "cards.json",             None,  True,  "Scryfall resolution with printings — `fetch-deck`"),
    ("bracket",    "bracket_report.json",    None,  False, "computed power floor — `bracket-check`"),
    ("targets",    "goldfish_targets.json",  None,  False, "the engine DECLARATION: any_of groups, size = redundancy"),
    ("goldfish",   "goldfish_metrics.json",  "meta.decklist_sha256", False, "seeded Monte Carlo — `goldfish`"),
    ("mana",       "mana_analysis.json",     "decklist_sha256",      False, "hypergeometric colour sources — run AFTER goldfish"),
    ("frame",      "strategic_frame.json",   None,  False, "the strategist's read — `research-strategy` consult"),
    ("map",        "deck_map.json",          "decklist_sha256",      False, "the constellation — `deck-map`"),
    ("map-names",  "deck_map.json",          None,  False, "cities named by `deck-cartographer` + `merge-deck-map`"),
    ("engine",     "engine.json",            "decklist_sha256",      False, "how it RUNS — `analyze-engine` loop"),
    ("stacks",     "stacks/",                None,  False, "checker-passed lines: the only fact tier"),
    ("shortlist",  "considering.json",       None,  False, "The Ten — `short-list`"),
    ("shortlist-art", "considering_art.json", None,  False, "Scryfall art for The Ten — `short-list-art`"),
    ("tutors",     "tutor_guide.json",       None,  False, "the tutor guide — `tutor-guide`"),
    ("prose",      "manual_prose.json",      None,  False, "writer + coach prose — `write-manual`"),
    ("panel",      "manual_prose.json",      None,  False, "the front of book — `pilot-panel`"),
    ("issue",      "issue.json",             None,  False, "authored identity: volume, date, price"),
    ("plan",       "issue_plan.json",        None,  False, "the magazine editor's packaging — `design-issue`"),
]

# Stages whose artifact exists for another reason, so file presence proves nothing:
# the panel writes `editors_letter` and `pilots_log` INTO the writer's file, which
# `prose` already created. Checked by key instead, or a deck with no front of book
# reports complete.
KEYED_STAGES = {"panel": ("editors_letter", "pilots_log")}

# Stages this development cycle added. Named explicitly so a deck built before
# them reports them as MISSING rather than as complete-by-omission — which is the
# exact way a new capability fails to propagate.
ADDED_2026_08 = {"map", "map-names", "engine", "panel", "shortlist-art"}


def _dig(doc, path):
    for part in path.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(part)
    return doc


def status(slug):
    base = deck_dir(slug)
    cards = load_json(base / "cards.json") or {}
    truth = cards.get("decklist_sha256")

    rows = []
    for key, name, sha_path, required, what in STAGES:
        path = base / name
        if name.endswith("/"):
            files = sorted(path.glob("*.json")) if path.is_dir() else []
            passing = [f for f in files
                       if presentable(json.loads(f.read_text()))]
            rows.append({"stage": key, "artifact": name, "what": what,
                         "state": "present" if passing else "missing",
                         "detail": f"{len(passing)} passing of {len(files)}",
                         "required": required, "new": key in ADDED_2026_08})
            continue

        if not path.exists():
            rows.append({"stage": key, "artifact": name, "what": what,
                         "state": "missing", "detail": "", "required": required,
                         "new": key in ADDED_2026_08})
            continue

        doc = load_json(path) if name.endswith(".json") else {}
        detail, state = "", "present"

        if key == "map-names":
            regions = (doc or {}).get("regions") or []
            cities = [r for r in regions if r.get("level") == 0]
            named = [r for r in cities if r.get("label")]
            state = "present" if cities and len(named) == len(cities) else "missing"
            detail = f"{len(named)}/{len(cities)} cities named"
        elif key == "engine":
            verdict = ((doc or {}).get("critic") or {}).get("verdict")
            detail = f"critic: {verdict or 'not run'}"
            if verdict == "fail":
                state = "unverified"
        elif key in KEYED_STAGES:
            wanted = KEYED_STAGES[key]
            have = [k for k in wanted if (doc or {}).get(k)]
            state = "present" if len(have) == len(wanted) else "missing"
            detail = f"{len(have)}/{len(wanted)} keys"

        elif sha_path and truth:
            stamped = _dig(doc, sha_path)
            if stamped and stamped != truth:
                state = "STALE"
                detail = f"built against {str(stamped)[:12]}…, deck is {truth[:12]}…"

        rows.append({"stage": key, "artifact": name, "what": what, "state": state,
                     "detail": detail, "required": required,
                     "new": key in ADDED_2026_08})
    return rows


def fleet():
    """One row per deck. Nine decks and no way to ask "what is outstanding
    everywhere" is the other half of the problem `pending.json` exists for —
    `deck-status` answered it one slug at a time, so nobody ever asked it nine
    times and the fleet picture lived only in a hand-kept PLAN.md table."""
    from manamap.config import DECKS_DIR
    from manamap.pilot.validate_pending import summarise
    out = []
    for path in sorted(DECKS_DIR.glob("*/cards.json")):
        slug = path.parent.name
        rows = status(slug)
        stale = [r["stage"] for r in rows if r["state"] == "STALE"]
        try:
            pend = summarise(slug)
        except Exception:
            pend = {"open": 0, "applied": 0, "partial": 0}
        out.append({
            "slug": slug,
            "done": sum(1 for r in rows if r["state"] == "present"),
            "total": len(rows),
            "stale": stale,
            "pending_open": pend["open"],
            "pending_applied": pend["applied"],
            "pending_partial": pend["partial"],
        })
    return out


def _fleet_main(args):
    rows = fleet()
    if getattr(args, "as_json", False):
        print(json.dumps({"decks": rows}, indent=2))
        return
    print(f"FLEET STATUS — {len(rows)} decks\n")
    print(f"  {'deck':18}{'stages':>9}{'stale':>7}{'queued':>8}   notes")
    for r in rows:
        q = r["pending_open"] + r["pending_partial"]
        note = []
        if r["stale"]:
            note.append("STALE: " + ", ".join(r["stale"]))
        if r["pending_applied"]:
            note.append(f"{r['pending_applied']} queued entry now applied — delete it")
        print(f"  {r['slug']:18}{r['done']:>4}/{r['total']:<4}{len(r['stale']):>7}"
              f"{q:>8}   {'; '.join(note)}")
    tot_stale = sum(len(r["stale"]) for r in rows)
    tot_q = sum(r["pending_open"] + r["pending_partial"] for r in rows)
    print(f"\n  {tot_stale} stale artifact(s), {tot_q} queued change(s) across the fleet")
    errors = [f"{r['slug']}: {s} is STALE" for r in rows for s in r["stale"]]
    report_errors("fleet status", errors,
                  f"OK   {len(rows)} decks, nothing stale, {tot_q} change(s) queued")


def main(args):
    if getattr(args, "all_decks", False):
        return _fleet_main(args)
    if not args.slug:
        raise SystemExit("deck-status needs a slug, or --all for the fleet view.")
    rows = status(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps({"slug": args.slug, "stages": rows}, indent=2))
        return

    mark = {"present": "OK  ", "missing": "  --", "STALE": "STALE", "unverified": " ?  "}
    print(f"DECK STATUS — {args.slug}\n")
    for row in rows:
        flag = " NEW" if row["new"] and row["state"] == "missing" else ""
        print(f"  {mark[row['state']]:6s} {row['stage']:11s} {row['artifact']:24s}"
              f" {row['detail']}{flag}")
        if row["state"] == "missing" and row["new"]:
            print(f"         ^ added 2026-08 — a deck built before it does not have it")

    done = sum(1 for r in rows if r["state"] == "present")
    print(f"\n  {done}/{len(rows)} stages complete")

    # Deliberately NOT a STAGES row: a queued change is intent, not a lifecycle
    # stage, and must not move the completeness count. An APPLIED entry is the
    # queue erasing itself — git owns the change now, so the entry goes.
    from manamap.pilot.validate_pending import summarise
    pend = summarise(args.slug)
    if pend["entries"]:
        bits = [f"{pend['open']} open"]
        if pend["partial"]:
            bits.append(f"{pend['partial']} partial")
        if pend["applied"]:
            bits.append(f"{pend['applied']} APPLIED (delete the entry)")
        print(f"  pending.json: {', '.join(bits)}")

    # Staleness is an ERROR; incompleteness is a state. A half-built deck is a
    # deck in progress; a deck whose artifacts disagree about which decklist they
    # describe is one that will publish a wrong number.
    errors = [f"{r['stage']} ({r['artifact']}) is STALE — {r['detail']}"
              for r in rows if r["state"] == "STALE"]
    missing_required = [r["stage"] for r in rows
                        if r["required"] and r["state"] == "missing"]
    errors += [f"{s} is required and absent" for s in missing_required]
    report_errors(f"deck status for {args.slug}", errors,
                  f"OK   {args.slug} — nothing stale, "
                  f"{done}/{len(rows)} lifecycle stages present")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-status <slug>`.")
