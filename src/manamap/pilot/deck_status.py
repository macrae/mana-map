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
#
# `how` IS PART OF THE REGISTRY, not a second table somewhere else. The dossier
# used to make an absent stage VANISH — `if (!b) return ''` in every panel — so a
# freshly-built deck rendered as a thin one with nothing saying what was missing
# or how to get it. Any mapping from stage to command written in JavaScript would
# be a second copy of this sequence, which is the thing this file exists to be
# the single statement of. So the command travels with the stage, and every
# surface renders what it is given.
#
# A stage advanced by an agent names its SKILL (`/analyze-engine`), because that
# is the honest instruction — there is no subcommand that writes an engine model.
# A stage that is authored says so instead of naming a command that cannot exist.
STAGES = [
    ("brief",      "brief.json",             None,  False, "the written intent a build starts from",
     "manamap pilot brew {slug} --commander \"<name>\" --theme <style>"),
    ("decklist",   "decklist.txt",           None,  True,  "the 99, tracked, and the identity everything else hangs on",
     "manamap pilot check-in {slug} --from <paper-list.txt>"),
    ("cards",      "cards.json",             None,  True,  "Scryfall resolution with printings — `fetch-deck`",
     "manamap pilot fetch-deck {slug}"),
    ("bracket",    "bracket_report.json",    None,  False, "computed power floor — `bracket-check`",
     "manamap pilot bracket-check {slug}"),
    ("targets",    "goldfish_targets.json",  None,  False, "the engine DECLARATION: any_of groups, size = redundancy",
     "manamap pilot scaffold-targets {slug}   # a DRAFT to edit — the groups are "
     "yours to name; `validate-goldfish-targets {slug}` checks it"),
    ("goldfish",   "goldfish_metrics.json",  "meta.decklist_sha256", False, "seeded Monte Carlo — `goldfish`",
     "manamap pilot goldfish {slug}"),
    ("mana",       "mana_analysis.json",     "decklist_sha256",      False, "hypergeometric colour sources — run AFTER goldfish",
     "manamap pilot mana-analysis {slug}"),
    ("frame",      "strategic_frame.json",
     "decklist_sha256_prefix|decklist_sha256_12|as_of_decklist_sha256|decklist_sha256",
     False, "the strategist's read — `research-strategy` consult",
     "/research-strategy — MODE consult, on {slug}"),
    ("map",        "deck_map.json",          "decklist_sha256",      False, "the constellation — `deck-map`",
     "manamap pilot deck-map {slug}"),
    ("engine",     "engine.json",            "decklist_sha256",      False, "how it RUNS — `analyze-engine` loop",
     "/analyze-engine {slug}"),
    ("stacks",     "stacks/",                None,  False, "checker-passed lines: the only fact tier",
     "/resolve-stack — one board, one rules domain, cited"),
    # `decklist_sha256` and its two spellings, same as the frame: the guide is
    # agent-written and the producers have not agreed on a key. `None` here
    # meant no staleness check AT ALL on an artifact whose entire content is
    # card names — the thing most certain to rot when a list moves.
    ("tutors",     "tutor_guide.json",
     "decklist_sha256|decklist_sha256_prefix|decklist_sha256_12",
     False, "the tutor guide — `tutor-guide`",
     "/write-manual — the tutor guide is one of pilot-notes' artifacts"),
    ("prose",      "manual_prose.json",
     "decklist_sha256_prefix|decklist_sha256_12|as_of_decklist_sha256|decklist_sha256",
     False, "the pilot's notes — `write-manual`",
     "/write-manual {slug}"),
    ("log",        "log.jsonl",              None,  False, "the captain's log — `deck-notes add`; debriefed by the `debrief` agent",
     "manamap pilot deck-notes {slug} add \"<what happened>\" --result win|loss --opponents 3"),
    ("issue",      "issue.json",             None,  False, "authored identity: name, commander, status (volume/price are legacy fields)",
     'AUTHORED: data/decks/{slug}/issue.json — three keys matter now: '
     '{{"deck_name": "…", "commander": "…"}} plus "status" once the deck stops '
     'being sleeved (broken-down | superseded | retired). The cover/volume '
     'fields are the frozen magazine renderer\'s and no live surface reads them.'),
]

# Stages this development cycle added. Named explicitly so a deck built before
# them reports them as MISSING rather than as complete-by-omission — which is the
# exact way a new capability fails to propagate.
ADDED_2026_08 = {"map", "engine", "log"}

# RETIRED 2026-08-19 with the workbench pivot (docs/agent-audit-2026-08-19.md):
# `map-names` (the cartographer is optional now — the deterministic fallback
# names are honest, and a gate on wit is a gate nobody should have to pass),
# `panel` (the Editor's Letter and Pilot's Log — pilot-panel is deleted) and
# `plan` (issue_plan.json — magazine-editor is deleted; build-manual renders
# with department defaults when no plan exists, and the tracked plans on the
# published decks are frozen legacy inputs until the manual is simplified).
# `shortlist` and `shortlist-art` followed on the same day: The Short List's rule
# (ten ranked cards worth knowing about, ownership not a criterion) lives in the
# doctor's prescriptions now; the surviving considering.json files are frozen
# legacy the renderer still reads and `validate-considering` still gates.
# A stage whose artifact exists for another reason cannot be checked by file
# presence — `panel` was checked by KEY for that reason, and the mechanism went
# with the stage; bring it back with the first such stage, not before.


def _dig(doc, path):
    """Read a dotted path. `path` may name SEVERAL, separated by `|`.

    Several because the same stamp is written under different keys by
    different producers: `strategic_frame.json` carries
    `decklist_sha256_prefix` on two decks and `decklist_sha256_12` on a
    third. One dotted path could not reach both, which is half of why the
    frame's staleness went unchecked for as long as it did. The first path
    that resolves wins.
    """
    for candidate in path.split("|"):
        cur = doc
        for part in candidate.split("."):
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        if cur:
            return cur
    return None


# WHY manual_prose IS STAMP-CHECKED AND NOT CONTENT-CHECKED.
#
# The defect that prompted this was found by a human reading the rendered page,
# not by any gate: ur-dragon's `manual_prose.json` survived a re-baseline and
# went on coaching the pilot about Aggravated Assault — 18 mentions of a card
# that had left the 99, plus goldfish figures from before `model_treasures` was
# turned on. `deck-status` said `OK prose` throughout, because these five keys
# have no validator and carried no stamp.
#
# The obvious fix is a CONTENT check: flag prose naming a card this deck used to
# run and no longer does. It was built and measured before being rejected.
# Fleet-wide it finds 45 names; DFC-normalising the faces (a front face is not a
# departed card) drops that to 35 across 7 decks. But reading the singletons in
# context kills it — heliod's "Lightning Greaves is the cheap version of the same
# worry" is an UPGRADE being considered, gishath's "Ravenous Sailback, at 3 power,
# does not" is a RULES COMPARISON, and yawgmoth's "Culling the Weak takes the
# smallest engine class" is a SHORT-LIST entry. Prose legitimately discusses cards
# the deck does not hold.
#
# The discriminator against ur-dragon's real staleness — "play it the turn it
# wins", an instruction about a card you are holding — is SEMANTIC, and no regex
# separates an instruction from a discussion. A validator that fires on correct
# data trains its reader to ignore it, so the stamp ships and the content check
# does not. Same verdict, same reasoning, as the four checks rejected on
# 2026-08-24 at 46-of-57, 217-of-220, 2-of-4 and one-false-positive-fleet-wide.


def _stamp_is_stale(stamped, truth):
    """Compare a stamp to the truth, tolerating a PREFIX.

    The other half of why the frame went unchecked: the comparison was
    `stamped != truth`, and three decks store twelve characters rather than
    the full sha. An exact test would have called gishath and heliod stale
    while they were current — a validator that fires on correct data, which
    this repo rejects checks for. Compare over the shorter of the two.
    """
    a, b = str(stamped), str(truth)
    n = min(len(a), len(b))
    return n == 0 or a[:n] != b[:n]


def status(slug, validate=True):
    """Presence, staleness AND validity.

    Validity was the missing third. `deck-status` compared shas and counted
    files, which is a claim about bookkeeping rather than about correctness —
    PLAN.md recorded it reading nine decks green while two were failing their
    own validators, and it did it again live on ur-dragon mid-swap. The gates
    existed; nothing in the command ran them.
    """
    base = deck_dir(slug)
    cards = load_json(base / "cards.json") or {}
    truth = cards.get("decklist_sha256")

    rows = []
    for key, name, sha_path, required, what, how in STAGES:
        path = base / name
        if name.endswith("/"):
            files = sorted(path.glob("*.json")) if path.is_dir() else []
            passing = [f for f in files
                       if presentable(json.loads(f.read_text()))]
            rows.append({"stage": key, "artifact": name, "what": what,
                         "how": how.format(slug=slug),
                         "state": "present" if passing else "missing",
                         "detail": f"{len(passing)} passing of {len(files)}",
                         "required": required, "new": key in ADDED_2026_08})
            continue

        if not path.exists():
            rows.append({"stage": key, "artifact": name, "what": what,
                         "how": how.format(slug=slug),
                         "state": "missing", "detail": "", "required": required,
                         "new": key in ADDED_2026_08})
            continue

        doc = load_json(path) if name.endswith(".json") else {}
        detail, state = "", "present"

        if key == "log":
            # One line per game; `detail` says how many the debrief has read.
            # The stage's artifact is the AUTHORED log, which has no gate; the
            # DERIVED annotation beside it does, and the row runs it — a green
            # row over a broken debrief is the exact failure the last commit
            # before this stage existed to stop.
            from manamap.pilot.deck_notes import ANNOTATIONS_FILE, annotations, read_log
            entries = read_log(slug)
            state = "present" if entries else "missing"
            done = sum(1 for e in entries if e["id"] in annotations(slug))
            detail = f"{len(entries)} logged, {done} debriefed"
            if state == "present" and validate and (base / ANNOTATIONS_FILE).exists():
                ok, why = _validity(slug, ANNOTATIONS_FILE)
                if ok is False:
                    state, detail = "INVALID", why
        elif sha_path and truth:
            stamped = _dig(doc, sha_path)
            if stamped and _stamp_is_stale(stamped, truth):
                state = "STALE"
                detail = f"built against {str(stamped)[:12]}…, deck is {truth[:12]}…"
            elif not stamped:
                # UNSTAMPED is not STALE and must not be reported as it: the
                # artifact may be perfectly current and simply not say so.
                # Reported anyway, because "nothing can check this" is a fact
                # the lifecycle view owes its reader — six of nine frames are
                # in this state, and one of the three that WAS stamped had
                # gone stale under a re-baseline with the row reading OK.
                detail = detail or "unstamped — staleness cannot be checked"

        # THE ENGINE'S CRITIC VERDICT, ADDED TO the staleness read rather than
        # replacing it.
        #
        # This used to be an `elif key == "engine"` branch ABOVE the staleness
        # check, so it short-circuited the chain: `STAGES` declares the engine's
        # stamp path as `decklist_sha256` and that check could never run, on any
        # deck, ever. The wiring was there and the current flowed nowhere.
        #
        # What it cost: edgar-vampires' `engine.json` named twelve cards the
        # deck stopped running the day the bloodline branch merged, and the row
        # read `OK  engine  critic: pass` for a week. Only a hand audit found it.
        # A model that describes a different list is stale whatever its critic
        # thought of it — the critic signed off on the OLD deck.
        if key == "engine":
            verdict = ((doc or {}).get("critic") or {}).get("verdict")
            critic = f"critic: {verdict or 'not run'}"
            detail = f"{detail} · {critic}" if detail else critic
            # STALE outranks it, same rule the validity check below follows: an
            # engine built against another decklist is wrong for a reason the
            # critic never looked at.
            if verdict == "fail" and state == "present":
                state = "unverified"

        # Validity LAST, and only for an artifact that is otherwise fine.
        # STALE already outranks it: an artifact built against another decklist
        # is wrong for a reason the gate cannot see, and reporting both would
        # bury the one that explains the other.
        if state == "present" and validate:
            ok, why = _validity(slug, name)
            if ok is False:
                state, detail = "INVALID", why
            elif ok is None and why:
                detail = detail or why

        rows.append({"stage": key, "artifact": name, "what": what,
                         "how": how.format(slug=slug), "state": state,
                     "detail": detail, "required": required,
                     "new": key in ADDED_2026_08})

    # GATED ARTIFACTS THAT ARE NOT LIFECYCLE STAGES.
    #
    # `VALIDATED` and `STAGES` are different lists and always were: `diagnosis.json`,
    # `build_plan.json` and `deck_recon.json` have gates but no stage row, so the loop
    # above never reached them and `deck-status` could not report them. It said
    # "0 failing a gate" for the whole fleet while `validate-diagnosis heliod` failed
    # in the same second — the precise divergence the `VALIDATED` map was extracted to
    # end, reappearing through the other door. A dashboard that is green while a gate
    # is red is worse than no dashboard, because people stop checking the gate.
    #
    # Reported as `state: "gate"` rather than as a stage: these are not steps in
    # building a deck and must not move the "N/15 stages" count.
    staged = {r["artifact"] for r in rows}
    for artifact in sorted(set(VALIDATED) - staged):
        if not (base / artifact).exists():
            continue
        row = {"stage": "—", "artifact": artifact, "what": "gated, not a lifecycle stage",
               "state": "gate", "detail": "", "required": False, "new": False}
        if validate:
            ok, why = _validity(slug, artifact)
            if ok is False:
                row["state"], row["detail"] = "INVALID", why
            elif ok is None and why:
                row["detail"] = why
        rows.append(row)
    return rows


# artifact filename -> the module whose `main(args)` gates it. Modules import
# LAZILY inside `_validity` so `manamap --help` stays fast, the same reason
# `registry.PILOT_STEPS` holds dotted strings rather than modules.
#
# This map used to live only in `tests/test_pilot_tracked_artifacts_validate.py`,
# which meant the TEST knew which artifacts had gates and `deck-status` — the
# command the runbook says to start with, every time — did not. It reported
# presence and staleness and called that health. Measured on ur-dragon mid-swap:
# `deck-status` FAIL=0 while `validate-issue` (legacy plan gate) FAIL=1 on the same deck, in the
# same second. A dashboard that is green while the gate is red is worse than no
# dashboard, because people stop checking the gate.
VALIDATED = {
    "cards.json": "manamap.pilot.validate_deck",
    "considering.json": "manamap.pilot.validate_considering",
    "deck_map.json": "manamap.pilot.validate_deck_map",
    "deck_recon.json": "manamap.pilot.validate_recon",
    "diagnosis.json": "manamap.pilot.validate_diagnosis",
    # `diagnostic.json` — the vitals — was tracked, written by `diagnose
    # --write`, and reported by nothing: no validator, no freshness test, no row
    # here. It is composed from the goldfish, so it goes stale on every model
    # change, which made it the artifact whose staleness was least visible.
    "diagnostic.json": "manamap.pilot.validate_diagnostic",
    # A branch artifact, and the first one in this map. It is reached because
    # the validator test now recurses into `branches/*` — before that the whole
    # branch tree was gated by nothing.
    "net_change.json": "manamap.pilot.validate_net_change",
    # THE LAST TRACKED PILOT ARTIFACT THAT HAD NO GATE. It holds the objective a
    # branch is graded against and the pilot's acceptance of it, and nothing
    # checked either until `propose` shipped.
    "branch.json": "manamap.pilot.validate_branch",
    # THE LAST TRACKED PILOT ARTIFACT THAT HAD NO GATE (#24) — and it held the
    # most load-bearing authored claim in the repo the whole time: `paper`, the
    # assertion that one exact 99 is sleeved, which is the predicate the front
    # door filters on. It earned a gate when it grew `lifecycle`, whose new
    # invariant is that the two cannot both be set.
    "deck_versions.json": "manamap.pilot.validate_deck_versions",
    "engine.json": "manamap.pilot.validate_engine",
    "goldfish_targets.json": "manamap.pilot.validate_goldfish_targets",
    # NOT a lifecycle stage, and that is exactly why it was missing: the test
    # hand-added `GATED["build_plan.json"] = validate_build` because there was
    # no `STAGES` row to hang it on. That left the TEST knowing about a gate the
    # status command did not — the two-maps divergence this map was extracted to
    # end, still open for one artifact. `deck_status` renders a validator with
    # no stage as a GATE row, so it belongs here.
    "build_plan.json": "manamap.pilot.validate_build",
    "log_annotations.json": "manamap.pilot.validate_debrief",
    # A GATE ROW, NOT A STAGE — deliberately. The stage list is the sequence a
    # deck advances through, and a captain's log is not a step toward a finished
    # deck: it is a record of a night, and it does not go stale when the deck
    # changes. Putting it in STAGES would put "N/16" on every deck and mark it
    # incomplete for nights that have not happened.
    "captains_log.json": "manamap.pilot.validate_captains_log",
    # The pilot's own claim about how each game ended, joined to the log by id.
    # Authored, tracked, and joined to a file that only grows — the shape that
    # goes quietly wrong, which is why it earned a gate on the day it shipped.
    "log_causes.json": "manamap.pilot.validate_log_causes",
    "pending.json": "manamap.pilot.validate_pending",
    "strategic_frame.json": "manamap.pilot.validate_strategic_frame",
    "tutor_guide.json": "manamap.pilot.validate_tutor_guide",
}

# Two validators reach for the gitignored strategy DB and report every
# `strategy:<id>` citation as an error when it is absent; one reads the corpus
# through `card_pool`. On a fresh clone those are MISSING INPUTS, not defects, so
# they report `unverified` rather than failing — the same distinction
# `tests/conftest.py`'s markers make.
_NEEDS_STRATEGY = {"tutor_guide.json", "diagnosis.json"}
_NEEDS_CORPUS = {"build_plan.json", "deck_recon.json"}


def _validity(slug, artifact):
    """`(ok, detail)` — None when nothing gates it or the gate cannot run here."""
    import contextlib
    import importlib
    import io

    dotted = VALIDATED.get(artifact)
    if not dotted:
        return None, ""
    from manamap.config import OUTPUT_CSV_PATH, STRATEGY_INDEX_PATH
    if artifact in _NEEDS_STRATEGY and not STRATEGY_INDEX_PATH.exists():
        return None, "gate needs the strategy DB"
    if artifact in _NEEDS_CORPUS and not OUTPUT_CSV_PATH.exists():
        return None, "gate needs the card corpus"

    module = importlib.import_module(dotted)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            module.main(type("Args", (), {"slug": slug})())
    except SystemExit as exit_:
        if exit_.code:
            first = next((ln.strip() for ln in buf.getvalue().splitlines()
                          if ln.strip().startswith("-")), "")
            n = sum(1 for ln in buf.getvalue().splitlines()
                    if ln.strip().startswith("-"))
            return False, f"{n} error(s){': ' + first.lstrip('- ')[:60] if first else ''}"
    except Exception as exc:                      # a broken gate is not a green deck
        return False, f"validator raised {type(exc).__name__}: {exc}"[:90]
    return True, ""


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
        # Name the ARTIFACT, not the stage: a gate row has no stage and reported as
        # "FAILS ITS GATE: —", which tells a reader nothing about what to fix.
        def _name(r):
            return r["stage"] if r["stage"] != "—" else r["artifact"]
        stale = [_name(r) for r in rows if r["state"] == "STALE"]
        invalid = [_name(r) for r in rows if r["state"] == "INVALID"]
        # Gates are not stages — counting them makes a deck with MORE evidence look
        # less finished. Same rule the single-deck view follows.
        stage_rows = [r for r in rows if r["stage"] != "—"]
        try:
            pend = summarise(slug)
        except Exception:
            pend = {"open": 0, "applied": 0, "partial": 0}
        out.append({
            "slug": slug,
            "done": sum(1 for r in stage_rows if r["state"] == "present"),
            "total": len(stage_rows),
            "stale": stale,
            "invalid": invalid,
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
    print(f"  {'deck':18}{'stages':>9}{'stale':>7}{'FAIL':>6}{'queued':>8}   notes")
    for r in rows:
        q = r["pending_open"] + r["pending_partial"]
        note = []
        if r["stale"]:
            note.append("STALE: " + ", ".join(r["stale"]))
        if r["invalid"]:
            note.append("FAILS ITS GATE: " + ", ".join(r["invalid"]))
        if r["pending_applied"]:
            note.append(f"{r['pending_applied']} queued entry now applied — delete it")
        print(f"  {r['slug']:18}{r['done']:>4}/{r['total']:<4}{len(r['stale']):>7}"
              f"{len(r['invalid']):>6}{q:>8}   {'; '.join(note)}")
    tot_stale = sum(len(r["stale"]) for r in rows)
    tot_bad = sum(len(r["invalid"]) for r in rows)
    tot_q = sum(r["pending_open"] + r["pending_partial"] for r in rows)
    print(f"\n  {tot_stale} stale, {tot_bad} failing their own gate, "
          f"{tot_q} queued change(s) across the fleet")
    errors = [f"{r['slug']}: {st} is STALE" for r in rows for st in r["stale"]]
    errors += [f"{r['slug']}: {bad} FAILS its validator" for r in rows for bad in r["invalid"]]
    report_errors("fleet status", errors,
                  f"OK   {len(rows)} decks, nothing stale, nothing failing a gate, "
                  f"{tot_q} change(s) queued")


def main(args):
    if getattr(args, "all_decks", False):
        return _fleet_main(args)
    if not args.slug:
        raise SystemExit("deck-status needs a slug, or --all for the fleet view.")
    rows = status(args.slug)
    if getattr(args, "as_json", False):
        print(json.dumps({"slug": args.slug, "stages": rows}, indent=2))
        return

    mark = {"present": "OK  ", "missing": "  --", "STALE": "STALE",
            "INVALID": "FAIL", "unverified": " ?  ",
            # A gated artifact that is not a lifecycle stage — it passed its gate
            # but is not a step in building a deck, so it must not read as one.
            "gate": "GATE"}
    print(f"DECK STATUS — {args.slug}")
    # Which list this is. Derived from git on demand; a deck outside a repo
    # reports nothing rather than guessing.
    try:
        from manamap.pilot.deck_versions import report as _versions
        vdoc = _versions(args.slug)
        if vdoc["versions"]:
            cur = vdoc["current_version"]
            v = next((x for x in vdoc["versions"] if x["version"] == cur), None)
            if v:
                tag_s = f" [{', '.join(v['tags'])}]" if v["tags"] else ""
                print(f"  version V{cur} of {len(vdoc['versions'])} · {v['first_date']} · "
                      f"{v['games']} game(s) logged{tag_s}")
            else:
                print(f"  version: uncommitted working list "
                      f"({len(vdoc['versions'])} committed)")
    except Exception:                    # noqa: BLE001 — a header, never a gate
        pass
    print()
    for row in rows:
        flag = " NEW" if row["new"] and row["state"] == "missing" else ""
        print(f"  {mark[row['state']]:6s} {row['stage']:11s} {row['artifact']:24s}"
              f" {row['detail']}{flag}")
        if row["state"] == "missing" and row["new"]:
            print(f"         ^ added 2026-08 — a deck built before it does not have it")

    # Gate rows are NOT stages — they are artifacts that have a validator but no
    # step in building a deck. Counting them would make "13/15" become "13/17" and
    # a deck look less finished for having MORE evidence, which is backwards.
    stages = [r for r in rows if r["stage"] != "—"]
    done = sum(1 for r in stages if r["state"] == "present")
    print(f"\n  {done}/{len(stages)} stages complete")

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
    # An artifact that fails its own gate is a published error, not a warning.
    errors += [f"{r['stage']} ({r['artifact']}) FAILS its validator — {r['detail']}"
               for r in rows if r["state"] == "INVALID"]
    missing_required = [r["stage"] for r in rows
                        if r["required"] and r["state"] == "missing"]
    errors += [f"{s} is required and absent" for s in missing_required]
    report_errors(f"deck status for {args.slug}", errors,
                  f"OK   {args.slug} — nothing stale, "
                  f"{done}/{len(rows)} lifecycle stages present")


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-status <slug>`.")
