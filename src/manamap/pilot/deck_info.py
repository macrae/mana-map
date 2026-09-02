"""Pilot: the workbench view — one deck, one screen, and what to do next.

WHY THIS EXISTS. Seven commands each answer a piece of "where is this deck":
`deck-status` (is it finished, is it stale), `deck-version` (which list is this),
`deck-notes` (what happened at the table), `prescribe --list` (what was asked and
answered), `deck-audit` (is it any good), `goldfish` (how fast), `bracket-check`
(how strong). A pilot sitting down to tinker ran four of them and held the join in
their head. This is the join, composed from those modules and the tracked
artifacts — it computes nothing new, so it can never disagree with the command
that owns each figure.

THE `next` BLOCK IS THE POINT. A dashboard tells you the state; a workbench tells
you the move. Every suggestion here is DERIVED from a condition that is true right
now (un-debriefed games → `/debrief`; an uncommitted working list → commit it so
the log can stamp a version; a stale stage → the command that regenerates it; no
games logged → play it), and each names the command. None of it is judgment about
the deck — that is the doctor's, behind `/prescribe`.

Read-only and computed on demand. `--json` is the shape a future UI reads; the human
print is the same dict laid out.

`--write` puts that shape on disk as `info.json` so the deck page can fetch it, and
that artifact IS committed — unlike `deck-facts`, which stays uncommitted because
nothing reads it but an agent standing right there. Two consequences, both handled:
it is staleness-gated like `mana_analysis.json` (regenerate after a decklist change or
the test fails), and it OMITS the version block. Versions are derived from a git walk,
so a committed copy is one commit behind forever — the deck page gets them from a
deploy-time `versions.json` instead, which CI can build because `deck_versions` needs
only git while `deck_audit` needs the gitignored corpus.
"""

import json

from manamap.pilot import deck_facts as facts_mod
from manamap.pilot import deck_model as dm
from manamap.pilot import deck_status as status_mod
from manamap.pilot import deck_versions as versions_mod
from manamap.pilot.common import UNPLAYABLE_STATUSES, deck_dir, deck_lifecycle, load_json
from manamap.pilot.deck_notes import annotations, causes, read_log
from manamap.pilot.prescribe import list_all as prescriptions_of
from manamap.sim.experiment import list_all as experiments_of
from manamap.sim.forge import list_runs as sim_runs


def _pct(x):
    return None if x is None else round(100 * float(x))


def _goldfish_block(base):
    """The goldfish, as a MODEL BLOCK — weighted, defined, intervals attached.

    THE PANEL THIS FEEDS WAS 3,833 PIXELS TALL AND CLAIMED NO HEADLINE. It
    rendered every target, every turn and every assumption at full length,
    because nothing upstream had an opinion about which two figures a pilot came
    for. `deck_model.block` now refuses more than four headlines, and everything
    else is `body` or `detail` — so the same data cannot render as a wall
    whatever the renderer does.

    Runs beside the legacy `_goldfish()` roll-up, which `info.json` keeps until
    every panel reads the model.
    """
    doc = load_json(base / "goldfish_metrics.json")
    if not doc:
        return dm.block("Goldfish", {"run": dm.absent(
            "no goldfish_metrics.json — run `manamap pilot goldfish <slug>`")},
            tier="data", source="goldfish_metrics.json")
    m = doc.get("metrics") or {}
    meta = doc.get("meta") or {}
    cmd = m.get("commander") or {}
    combat = m.get("combat") or {}
    facts = {}

    if cmd.get("cast_by_turn_6_rate") is not None:
        facts["commander_by_t6"] = dm.figure(
            _pct(cmd["cast_by_turn_6_rate"]), tier="data", weight="headline",
            unit="%", n=m.get("iterations"), source="goldfish_metrics.json",
            definition="share of seeded games where the commander was CAST by "
                       "turn six — cast, not drawn")
    else:
        facts["commander_by_t6"] = dm.absent("the model did not measure the commander")

    if combat.get("median_kill_turn") is not None:
        facts["kill_turn"] = dm.figure(
            combat["median_kill_turn"], tier="data", weight="headline",
            unit="turn", n=m.get("iterations"), source="goldfish_metrics.json",
            definition="MEDIAN turn the goldfish kills on, not the mean — a mean "
                       "over a skewed sample is a true number describing no game")
    else:
        facts["kill_turn"] = dm.absent(
            "combat is opt-in and this deck has not been re-baselined for it "
            "(`model_combat` in goldfish.py)")

    if cmd.get("mean_cast_turn") is not None:
        facts["commander_mean_turn"] = dm.figure(
            cmd["mean_cast_turn"], tier="data", weight="body", unit="turn",
            source="goldfish_metrics.json",
            definition="mean turn the commander is first cast")
    for t in (m.get("targets") or []):
        if t.get("by_turn_6_rate") is None:
            continue
        facts["target:" + str(t.get("label"))] = dm.figure(
            _pct(t["by_turn_6_rate"]), tier="data", weight="body", unit="%",
            source="goldfish_metrics.json",
            definition=f"share of games assembling {t.get('label')!r} by turn six")

    # THE ASSUMPTIONS ARE DETAIL, NOT BODY — 28 strings on ur-dragon. They must
    # travel with the figures (a model's stated assumptions are the model) and
    # they must never be what a reader scrolls through to reach the numbers.
    if meta.get("model_assumptions"):
        facts["assumptions"] = dm.figure(
            list(meta["model_assumptions"]), tier="data", weight="detail",
            source="goldfish_metrics.json",
            definition="what the seeded model does NOT do; an assumption stated "
                       "is a model, one hidden is a guess")
    # THE PER-TURN CURVES. `detail` on the dossier — a three-row, ten-column
    # table is not what a pilot opens the page for — and the chart material for
    # the handbook's Performance section, which is the whole reason one model
    # feeds both.
    oh = m.get("opening_hand") or {}
    if oh.get("keep_first_seven_rate") is not None:
        facts["keepable_sevens"] = dm.figure(
            _pct(oh["keep_first_seven_rate"]), tier="data", weight="body",
            unit="%", n=m.get("iterations"), source="goldfish_metrics.json",
            definition="share of opening sevens the model would keep — two to "
                       "five lands, up to two redraws")
    for key, label, unit in (
            ("mean_available_mana_by_turn", "mana_by_turn", "mana"),
            ("land_drop_hit_rate_by_turn", "land_drops_by_turn", "rate"),
            ("mean_bodies_by_turn", "bodies_by_turn", "creatures")):
        if m.get(key):
            facts[label] = dm.figure(
                m[key], tier="data", weight="detail", unit=unit,
                source="goldfish_metrics.json",
                definition=f"{label.replace('_', ' ')}, turns one to ten")
    if oh.get("first_seven_land_histogram"):
        facts["opening_land_distribution"] = dm.figure(
            oh["first_seven_land_histogram"], tier="data", weight="detail",
            source="goldfish_metrics.json",
            definition="lands in the opening seven, over every seeded hand")

    if combat.get("kill_turn_histogram"):
        facts["kill_distribution"] = dm.figure(
            combat["kill_turn_histogram"], tier="data", weight="detail",
            source="goldfish_metrics.json",
            definition="how many of the seeded games killed on each turn")

    return dm.block("Goldfish", facts, tier="data",
                    source="goldfish_metrics.json",
                    definition=f"{m.get('iterations') or '?'} seeded games, "
                               f"seed {meta.get('seed')}; no blockers and no "
                               f"opponent — a floor on speed, never a verdict "
                               f"on board quality")


def _goldfish(base):
    doc = load_json(base / "goldfish_metrics.json")
    if not doc:
        return None
    m = doc.get("metrics") or {}
    cmd = m.get("commander") or {}
    out = {"seed": (doc.get("meta") or {}).get("seed"),
           "iterations": m.get("iterations"),
           "commander_mean_cast_turn": cmd.get("mean_cast_turn"),
           "commander_cast_by_turn_6_pct": _pct(cmd.get("cast_by_turn_6_rate")),
           "targets": [{"label": t.get("label"), "by_turn_6_pct": _pct(t.get("by_turn_6_rate"))}
                       for t in (m.get("targets") or [])]}
    # The clock, when the deck opted into `model_combat`. Absent otherwise, which
    # is every deck that has not been deliberately re-baselined — see the opt-in
    # contract in `goldfish.py`.
    #
    # THIS BLOCK HAD NEVER EXECUTED. It asked for `kill_by_turn_8_pct`,
    # `kill_turn_distribution` and `never_by_turn_10_pct`; the producer emits
    # `kill_by_turn_rate`, `kill_turn_histogram` and `no_kill_by_max_turn_rate`.
    # Not one key matched, so the first deck to opt in would have written
    # `"combat": {}` into a tracked `info.json` and rendered nothing — dead code
    # that no test could catch, because no artifact has ever carried a combat
    # block for it to run against.
    combat = m.get("combat") or {}
    if combat:
        max_turn = (doc.get("meta") or {}).get("max_turn")
        by_turn = combat.get("kill_by_turn_rate") or {}
        out["combat"] = {
            "mean_kill_turn": combat.get("mean_kill_turn"),
            "median_kill_turn": combat.get("median_kill_turn"),
            "kill_by_turn_6_pct": _pct(by_turn.get("6")),
            "kill_by_turn_8_pct": _pct(by_turn.get("8")),
            # Named from the artifact rather than hardcoded: the producer's rate is
            # over GOLDFISH_MAX_TURN, and a key saying "by_turn_10" would quietly
            # lie the moment that constant moved.
            "max_turn": max_turn,
            "no_kill_by_max_turn_pct": _pct(combat.get("no_kill_by_max_turn_rate")),
            "kill_turn_histogram": combat.get("kill_turn_histogram"),
        }
    return out


def _binding_axes(slug):
    """The audit's under/over axes — what the measurement says binds. Never a score."""
    try:
        from manamap.pilot import deck_audit
        doc = deck_audit.analyze(slug)
    except Exception:                      # noqa: BLE001 — an optional panel, never a gate
        return None
    axes = doc.get("axes") or []
    arch = doc.get("archetype")
    if isinstance(arch, dict):              # detect_archetype returns its evidence too
        arch = arch.get("archetype")
    return {"archetype": arch,
            "under": [a["axis"] for a in axes if a.get("verdict") == "under"],
            "over": [a["axis"] for a in axes if a.get("verdict") == "over"],
            "stale": [k for k, v in (doc.get("freshness") or {}).items()
                      if isinstance(v, dict) and v.get("state") not in (None, "current")]}


def _open_questions(base):
    out = []
    for name, key in (("engine.json", "engine"), ("diagnosis.json", "diagnosis")):
        doc = load_json(base / name) or {}
        for q in doc.get("open_questions") or []:
            out.append({"from": key, "question": q.get("question"),
                        "settled_by": q.get("settled_by")})
    for eid, note in sorted((load_json(base / "log_annotations.json") or {}).get("entries", {}).items()):
        for q in note.get("open_questions") or []:
            out.append({"from": f"log:{eid}", "question": q.get("question"),
                        "settled_by": q.get("settled_by")})
    return out


def _branches(slug):
    """Open branches: their state, what each would cost, and the pull list.

    Composes; computes nothing of its own. `deck_branch.source` is the one place
    that answers where a card comes from, and it is the only reader of the
    collection through `pilot/collection.py`; `deck_branch.branch_state` is the
    one place that decides whether a branch is an experiment, a decision waiting
    on cardboard, or ready to merge.

    This is what puts a PROPOSAL on a static page. The deck dossier and the
    workbench both read `info.json` and neither can run a command, so the state
    and the pull list have to travel in the artifact — and both are derived on
    write, so the blocker a page shows is the blocker at the moment it was
    written rather than a stored claim that can go quietly wrong.
    """
    try:
        from manamap.pilot import deck_branch
        rows = []
        for name in deck_branch.names(slug):
            try:
                rows.append(deck_branch.one(slug, name))
            except Exception:
                # A branch whose list will not parse is a fact worth showing,
                # not a reason the whole dossier fails to load.
                rows.append({"name": name, "unreadable": True})
        return rows
    except Exception:
        return []


# ── The engine, said in one word ─────────────────────────────────────────
#
# A VERDICT, WHICH THIS REPO NORMALLY REFUSES TO PUBLISH. The obsolescence index
# is the standing lesson: it shipped a judgement ("Obsoleted By") over a measure
# and **36.5% of 22,753 pairs failed a purely mechanical check** — the retrieval
# half was fine, the judgement half was not. The rule that came out of it is that
# a measure ships and THE PILOT SETS THE LINE.
#
# The cover sheet needs a word anyway: the whole point of a cover sheet is that
# you absorb it in thirty seconds, and `0.4025 [0.3929, 0.4121]` is not a
# thirty-second fact. So the word ships under three conditions, all of which the
# obsolescence index lacked:
#
#   1. THE THRESHOLDS ARE A NAMED CONSTANT the pilot can move, right here.
#   2. THE MEASURE TRAVELS WITH THE WORD — `rate`, `n` and `by_turn` are in the
#      same block, so the reader never has to look it up and never has to guess
#      what "HEALTHY" was computed from.
#   3. ABSENT MEANS ABSENT. A deck with no diagnostic, or one whose engine block
#      says `available: false`, gets NO WORD and a stated reason. It does not get
#      WEAK — that would be a measurement of a deck nobody measured.
#
# The axis is "is the engine online by turn five", which is the one figure that
# is about the machine rather than about any single card. Turn five because it is
# where the diagnostic's own bottleneck analysis sits.
ENGINE_HEALTH_TURN = "5"
#: `(floor, word)` — highest floor that the rate clears wins. The pilot's line.
ENGINE_HEALTH_BANDS = (
    (0.60, "EXCEPTIONAL"),
    (0.40, "HEALTHY"),
    (0.20, "BRITTLE"),
    (0.00, "WEAK"),
)


def engine_health(vitals):
    """One word for the cover sheet, or None with a reason. Never a bare word."""
    block = (vitals or {}).get("engine") or {}
    if not vitals:
        return None
    if not block.get("available"):
        return {"word": None, "why": block.get("why")
                or "the engine was not modelled on this deck",
                "basis": "diagnostic.json"}
    row = (block.get("online_by_turn") or {}).get(ENGINE_HEALTH_TURN) or {}
    rate = row.get("rate")
    if rate is None:
        return {"word": None,
                "why": f"no engine-online rate at turn {ENGINE_HEALTH_TURN}",
                "basis": "diagnostic.json"}
    word = next(w for floor, w in ENGINE_HEALTH_BANDS if rate >= floor)
    return {"word": word, "rate": rate, "ci95": row.get("ci95"), "n": row.get("n"),
            "turn": int(ENGINE_HEALTH_TURN),
            "bands": [[f, w] for f, w in ENGINE_HEALTH_BANDS],
            "why": (f"the engine is online by turn {ENGINE_HEALTH_TURN} in "
                    f"{rate:.0%} of {row.get('n', 0):,} seeded games"),
            "basis": "diagnostic.json"}


def compose(slug, verify=False):
    """The workbench view. `verify` RUNS THE GATES, and costs about two seconds.

    `deck_status.status(validate=True)` imports and executes all fourteen
    validator modules in-process, and `validate_diagnosis` alone is ~1s because
    it re-derives the audit. That is the right thing for `deck-status`, whose
    JOB is to say whether the artifacts hold up, and the wrong thing for the
    command the pilot runs to remember where a deck stands — measured at 7.8s
    end to end, of which ~2.3s was gates nobody asked for.

    OFF MEANS UNKNOWN, NEVER CLEAN. With `verify=False` the `invalid` key is
    None rather than `[]`, because a reader cannot tell "nothing failed" from
    "nothing was checked" and this repo has paid for that confusion elsewhere
    (`absent means ABSENT, never zero`). Every consumer must handle None.
    """
    base = deck_dir(slug)
    facts = facts_mod.analyze(slug)
    counts = facts.get("counts") or {}
    identity = sorted({c for v in (facts.get("colours") or {}).values()
                       for c in (v.get("card") or [])}, key="WUBRG".index)
    rows = status_mod.status(slug, validate=verify)
    vdoc = versions_mod.report(slug)
    log = read_log(slug)
    done = annotations(slug)
    rx = prescriptions_of(slug)
    brief = load_json(base / "brief.json") or {}
    branches = _branches(slug)
    vitals = load_json(base / "diagnostic.json") or {}
    bracket = load_json(base / "bracket_report.json") or {}
    engine = load_json(base / "engine.json") or {}
    diag = load_json(base / "diagnosis.json") or {}

    # THE MODEL. One composed structure both surfaces read; no renderer touches a
    # raw artifact. Built beside the legacy roll-ups until every panel is
    # converted, so the page never renders half-shaped.
    model = {"goldfish": _goldfish_block(base)}

    cur = next((v for v in vdoc["versions"] if v["version"] == vdoc["current_version"]), None)
    record = {"games": len(log),
              "win": sum(1 for e in log if e.get("result") == "win"),
              "loss": sum(1 for e in log if e.get("result") == "loss"),
              "draw": sum(1 for e in log if e.get("result") == "draw"),
              "last_played": max((e["at"][:10] for e in log), default=None),
              # DATE FIRST BOOKED — the cover sheet's one biographical fact, and
              # it is not derivable from anything else: a deck's directory is
              # created when the BUILD starts, not when it first hit a table.
              "first_played": min((e["at"][:10] for e in log), default=None),
              "undebriefed": [e["id"] for e in log if e["id"] not in done]}
    # HOW EACH GAME ENDED, folded in by id — authored, see `deck_notes.CAUSES`.
    # Counted HERE rather than in the browser so the per-game rows and the
    # roll-up beneath them cannot disagree about the vocabulary they count.
    _causes = causes(slug)
    record["causes"] = {e["id"]: _causes[e["id"]]["cause"]
                        for e in log if e["id"] in _causes}
    record["cause_counts"] = {
        c: sum(1 for v in record["causes"].values() if v == c)
        for c in sorted(set(record["causes"].values()))}
    lines = engine.get("lines") or []
    # A GATE ROW IS NOT A STAGE, and its `stage` is the literal "—".
    stage_rows = [r for r in rows if r["stage"] != "—"]

    def _name(r):
        return r["stage"] if r["stage"] != "—" else r["artifact"]

    life = deck_lifecycle(slug)
    info = {
        "slug": slug,
        "commander": facts.get("commander"),
        "lifecycle": ({"status": life[0], "headline": life[1], "body": life[2]}
                      if life else None),
        "colour_identity": identity,
        "size": counts.get("copies"), "lands": counts.get("land_copies"),
        "version": {"current": vdoc["current_version"], "of": len(vdoc["versions"]),
                    "date": cur["first_date"] if cur else None,
                    "tags": cur["tags"] if cur else [],
                    "uncommitted": vdoc["current_version"] is None and bool(vdoc["versions"])},
        # STAGES ONLY IN THE DENOMINATOR. `deck_status.status()` returns two
        # kinds of row: the lifecycle STAGES, and GATE rows for artifacts that
        # have a validator but no step in building a deck. `deck_status` itself
        # excludes the gates from its count — "counting them would make 13/15
        # become 13/17 and a deck look less finished for having MORE evidence,
        # which is backwards" (deck_status.py:483-485) — and this counted all of
        # them, so the two commands printed different fractions for one deck and
        # decks were being compared against different totals (14/20, 14/19,
        # 14/17, 8/16).
        #
        # It went unnoticed while the gate set was stable and bit the moment one
        # was added: registering `deck_versions.json` in `VALIDATED` moved every
        # deck's denominator by one with no new stage in sight.
        #
        # `_name(r)` for the same reason: a gate row's `stage` is the literal
        # "—", so `invalid` read `["—"]` on seven decks and `next` printed
        # "1 artifact(s) fail their own gate (—)", naming nothing. The fleet
        # view already had this fix (deck_status.py:386) and this did not.
        "status": {"complete": sum(1 for r in stage_rows if r["state"] == "present"),
                   "of": len(stage_rows),
                   "stale": [_name(r) for r in rows if r["state"] == "STALE"],
                   # None, not [] — see `compose`'s docstring.
                   "invalid": ([_name(r) for r in rows if r["state"] == "INVALID"]
                               if verify else None),
                   "verified": verify,
                   "missing": [r["stage"] for r in stage_rows
                               if r["state"] == "missing"],
                   # WHAT EACH ABSENT STAGE IS, AND HOW TO GET IT — carried from
                   # `deck_status.STAGES`, which is the one machine-readable
                   # statement of the lifecycle. The dossier made an absent
                   # section VANISH, so a freshly-built deck rendered as a thin
                   # one and said nothing about the difference; the fix is not a
                   # lookup table in JavaScript, which would be this sequence
                   # written down twice.
                   "todo": [{"stage": r["stage"], "what": r["what"], "how": r["how"]}
                            for r in stage_rows if r["state"] == "missing"]},
        # THE BRIEF IS WHAT THE DECK IS TRYING TO BE, and it was a staged,
        # tracked artifact with nowhere to read it: `deck_status` reports it,
        # `build_deck` consumes it, and neither `info.json` nor the dossier had
        # ever surfaced one. A deck whose brief and whose 99 disagree is the
        # normal state DURING a refactor, and the page could not show the
        # difference. Free-text fields are carried verbatim — this composes and
        # computes nothing, same as every other key here.
        "brief": ({"playstyle": brief.get("playstyle"),
                   "commander_rationale": brief.get("commander_rationale"),
                   "mana": brief.get("mana"),
                   "design_rules": brief.get("design_rules") or [],
                   "win_conditions": brief.get("win_conditions"),
                   "targets": brief.get("targets") or {},
                   "notes": brief.get("notes"),
                   "must_include": brief.get("must_include") or [],
                   "must_exclude": brief.get("must_exclude") or []}
                  if brief else None),
        # CANDIDATE LISTS the pilot cannot yet sleeve. A branch is not a stage
        # and deliberately not in `deck_status.STAGES`: adding one would change
        # the denominator for every deck at once and mark twelve newly
        # incomplete for something optional — the same argument that keeps `sim`
        # a todo rather than a stage.
        "branches": branches,
        # THE VITALS. Composed like everything else here; `diagnose --write` owns
        # the figures. It is not a lifecycle stage for the same reason `sim` is
        # not — optional, and adding one would mark twelve decks newly incomplete.
        # `vitals`, NOT `diag` — that name was already the DIAGNOSIS, and
        # `diagnosis.json` and `diagnostic.json` are one letter apart. The
        # second assignment silently won and the dossier rendered "not
        # measured" over a file full of data.
        "diagnostic": ({"harness": vitals.get("harness"),
                        "engine": vitals.get("engine"),
                        "stall": vitals.get("stall"),
                        "mana": vitals.get("mana"),
                        "decklist_sha256": vitals.get("decklist_sha256")}
                       if vitals else None),
        "bracket": {"floor": bracket.get("floor"), "floor_name": bracket.get("floor_name"),
                    "target": bracket.get("target"),
                    "within_target": bracket.get("within_target")} if bracket else None,
        # WHETHER THIS DECK EXISTS IN PAPER, as the pilot asserted it.
        #
        # The AUTHORED half only — version, when, and the note. `paper_state`'s
        # drift needs a git walk, and `info.json` is committed and omits
        # everything git-derived because the commit that changes `decklist.txt`
        # gets its sha after anything written alongside it. `paper()` is a plain
        # file read of `deck_versions.json`, so this side of the split is free
        # and CI can compute it.
        #
        # `_next` needs it for one reason: an UNLOCKED deck is not a dead deck,
        # it is an UNKNOWN one, and telling the pilot to go and play something
        # that may never have been sleeved is a quieter version of the defect
        # that had this command recommending a deck whose cards were in another
        # deck's sleeves.
        "paper": versions_mod.paper(slug),
        "record": record,
        # THE COMPOSED MODEL — see `deck_model`. Both the dossier and the
        # handbook read this; neither reads a raw artifact. `goldfish` below is
        # the legacy roll-up and is deleted when the last panel converts.
        "model": model,
        "goldfish": _goldfish(base),
        # A VERDICT, travelling with its measure and its bands — see
        # `engine_health`. Absent, never WEAK, when nothing was measured.
        "engine_health": engine_health(vitals),
        "engine": {"thesis": engine.get("thesis"),
                   "critic": (engine.get("critic") or {}).get("verdict"),
                   "lines": len(lines),
                   "verified_lines": sum(1 for l in lines if l.get("verified_by"))} if engine else None,
        "diagnosis": {"verdict": diag.get("verdict"),
                      "skeptic": (diag.get("skeptic") or {}).get("verdict"),
                      "stale": diag.get("as_of_decklist_sha256") not in
                      (None, (load_json(base / "cards.json") or {}).get("decklist_sha256"))}
        if diag else None,
        "audit": _binding_axes(slug),
        "prescriptions": {"count": len(rx),
                          "answered": sum(1 for p in rx if "add_candidates" in p),
                          "latest": ({"id": rx[-1]["id"], "prompt": rx[-1]["prompt"],
                                      "adds": len(rx[-1].get("add_candidates") or []),
                                      "cuts": len(rx[-1].get("cut_candidates") or []),
                                      "skeptic": (rx[-1].get("skeptic") or {}).get("verdict")}
                                     if rx else None)},
        "open_questions": _open_questions(base),
        "simulation": _simulation(slug),
        "experiments": _experiments(slug),
    }
    info["next"] = _next(info)
    return info


def _current_sha(slug):
    return ((load_json(deck_dir(slug) / "cards.json") or {}).get("decklist_sha256"))


def _simulation(slug):
    """The latest run, and WHETHER IT MEASURED THIS LIST.

    A run record stamps every seat's decklist sha, so a measurement made against
    a list the deck no longer holds is mechanically detectable — and nothing was
    detecting it. Edgar shipped a 0.25 win rate on the workbench for a deck that
    had been checked in and re-baselined under it. A stale figure presented as
    current is worse than an absent one: the reader has no way to know, and the
    number is exactly as precise-looking as a true one.
    """
    runs = sim_runs(slug)
    if not runs:
        return None
    r = runs[-1]
    me = (r.get("analysis") or {}).get("seats", {}).get(slug, {})
    tok = me.get("tokens") or {}
    ran_on = next((s.get("decklist_sha256") for s in r.get("seats", [])
                   if s.get("slug") == slug), None)
    cur = _current_sha(slug)
    return {"runs": len(runs), "latest": r["run_id"], "at": r.get("at"),
            "stale": bool(ran_on and cur and ran_on != cur),
            "ran_on_decklist_sha256": ran_on,
            "games": r.get("games_completed"),
            "vs": [s["slug"] for s in r.get("seats", [])[1:]],
            "win_rate": me.get("win_rate"), "win_rate_ci95": me.get("win_rate_ci95"),
            # A WIN RATE NEVER TRAVELS WITHOUT THE PILOTING READING. Forge's AI
            # is documented as untrained and measured here at 0.67 land drops per
            # own turn — so the number is only readable beside whether OUR seat
            # was handled like the rest of the table. Every surface that shows one
            # shows this.
            "piloting": _piloting(r),
            "eliminated_by": me.get("eliminated_by"),
            "mean_round": (r.get("summary") or {}).get("mean_round"),
            "token_damage_share": (tok.get("token_damage_share") or {}).get("mean"),
            "tokens_observed": (tok.get("tokens_observed") or {}).get("mean")}


def _piloting(rec):
    """Was the AI playing this deck, or holding it? Derived, never stored."""
    try:
        from manamap.sim import pilot_quality
        q = pilot_quality.from_record(rec)
        if not q:
            return None
        return {"comparable": q["comparable"],
                "lands_ratio": q[pilot_quality.LANDS]["ratio"],
                "casts_ratio": q[pilot_quality.CASTS]["ratio"],
                "reading": q["reading"]}
    except Exception:
        return None


def _experiments(slug):
    docs = experiments_of(slug)
    if not docs:
        return None
    d = docs[-1]
    w = d["delta"]["win_rate"]
    power = d["delta"].get("power") or {}
    # An A/B is stale when NEITHER arm is the list the deck now holds — it
    # compared two lists, and if the deck has moved past both, the delta is a
    # fact about history rather than about this deck.
    cur = _current_sha(slug)
    arms = {d["arms"]["a"]["decklist_sha256"], d["arms"]["b"]["decklist_sha256"]}
    stale = bool(cur and cur not in arms)
    # `overlap` is gone rather than deprecated. It named the overlap fallacy —
    # two marginal intervals overlapping says nothing about their difference —
    # and leaving the key in the artifact re-invites the error it was removed
    # for. What replaces it is the interval on the DIFFERENCE, and the effect
    # size this experiment could have detected at all.
    return {"count": len(docs), "latest": {
        "question": d["question"], "at": d["at"], "stale": stale,
        "games_per_arm": d["games_per_arm"], "win_a": w["a"], "win_b": w["b"],
        "win_diff_ci95": w.get("ci95_diff"),
        "differs": w.get("excludes_zero"),
        "minimum_detectable_difference": power.get("minimum_detectable_difference"),
        "reading": d["delta"]["reading"]}}


def _next(info):
    """Each suggestion is derived from a condition true right now and names the
    command. No judgment about the deck lives here."""
    nxt = []
    slug = info["slug"]
    # A deck that has been pulled apart cannot be shuffled, so every suggestion
    # that ends in "play it" or "measure it before you play it" is an
    # instruction the pilot cannot follow. Say the status instead, and say what
    # is being withheld — a silently shorter list reads as "nothing to do here",
    # which is a different claim. What survives is everything that still works
    # on a published record: a failing gate, a stale artifact, an open rules
    # question. `superseded` is NOT in this set — that list is still sleeved.
    closed = (info["lifecycle"] or {}).get("status") in UNPLAYABLE_STATUSES
    if closed:
        nxt.append(f"{info['lifecycle']['headline'].lower()} — the play/measure loop is "
                   f"closed for this deck ({info['lifecycle']['status']}); its artifacts "
                   f"stay as published. Suggestions to log a game, simulate or run an "
                   f"experiment are withheld")
    if info["status"]["invalid"]:
        nxt.append(f"{len(info['status']['invalid'])} artifact(s) fail their own gate "
                   f"({', '.join(info['status']['invalid'])}) — `deck-status {slug}` names them; "
                   f"fix before anything else reads them")
    if info["status"]["stale"]:
        nxt.append(f"stale: {', '.join(info['status']['stale'])} — regenerate against the "
                   f"current list (`/publish-deck` sequences it)")
    if info["version"]["uncommitted"]:
        nxt.append("decklist.txt differs from every committed version — commit it so the "
                   "log can stamp games against a version (`deck-version` will then show it)")
    # Three states, not two. LOCKED means the pilot said this exact list is
    # sleeved; a dead `status` means it demonstrably is not; and ABSENT means
    # nobody has said either way — which is most decks, and is the state every
    # build plan sits in. Only the first earns "go and play it".
    unknown = not closed and not info.get("paper")
    if unknown:
        nxt.append(f"not marked as built in paper — if it is sleeved, "
                   f"`deck-version {slug} paper --note \"…\"` locks the list you hold "
                   f"(then drift is computed on every swap); if it is not, it is a build "
                   f"plan and the play/measure suggestions below need cardboard first")
    if info["record"]["games"] == 0 and not closed:
        nxt.append(f"nothing in the captain's log — play it, then "
                   f"`deck-notes {slug} add \"…\" --result win|loss --opponents N`")
    elif info["record"]["undebriefed"]:
        ids = info["record"]["undebriefed"]
        nxt.append(f"{len(ids)} logged game(s) not yet debriefed ({', '.join(ids)}) — `/debrief {slug}`")
    from manamap.pilot import deck_branch

    # A branch that is fully sourced is a decision waiting to be taken; one that
    # is not is a shopping list. Both are worth saying, and neither is a judgement
    # about the deck.
    #
    # A PROPOSED BRANCH IS A DIFFERENT SENTENCE FROM AN OPEN ONE, and until
    # `propose` existed this loop could not tell them apart: "needs 12 card(s)
    # sourced" is what it said about a branch nobody had an opinion on and about
    # one the pilot had accepted as the next version. The state does the talking
    # now; `deck_branch.branch_state` is the only thing that decides it.
    for b in (info.get("branches") or []):
        prop = b.get("proposal") or {}
        as_v = prop.get("as_version")
        if b.get("unreadable"):
            nxt.append(f"branch `{b['name']}` will not parse — fix "
                       f"data/decks/{slug}/branches/{b['name']}/decklist.txt")
        elif b.get("state") == deck_branch.PROPOSED_BLOCKED:
            pl = b.get("pull_list") or {}
            bits = [f"{len(pl.get(k) or [])} to {v}"
                    for k, v in (("buy", "buy"), ("unsleeve", "unsleeve"))
                    if pl.get(k)]
            nxt.append(f"`{b['name']}` is PROPOSED as {as_v} and waiting on "
                       f"cardboard — {pl.get('blocking', 0)} card(s)"
                       + (f" ({', '.join(bits)})" if bits else "")
                       + f" — `manamap pilot deck-branch {slug} show {b['name']}` "
                         f"has the pull list")
        elif b.get("state") == deck_branch.PROPOSED_READY:
            nxt.append(f"`{b['name']}` is PROPOSED as {as_v} and every card is "
                       f"sourced — `manamap pilot deck-branch {slug} merge "
                       f"{b['name']} --write`")
        elif b.get("state") == deck_branch.PROPOSED_STALE:
            nxt.append(f"`{b['name']}` was proposed as {as_v} and the list has "
                       f"changed since — re-run `net-change` and propose again, "
                       f"or `deck-branch {slug} withdraw {b['name']}`")
        elif b.get("state") == deck_branch.PROPOSED_OUTRUN:
            nxt.append(f"`{b['name']}` was proposed as {as_v} against "
                       f"V{prop.get('base_version')} and the deck has moved on — "
                       f"{as_v} may be taken; re-measure and propose again")
        elif b.get("mergeable"):
            nxt.append(f"branch `{b['name']}` is fully sourced (+{b['add']} -{b['out']}) — "
                       f"`manamap pilot deck-branch {slug} merge {b['name']} --write`")
        elif b.get("unsourced"):
            nxt.append(f"branch `{b['name']}` needs {len(b['unsourced'])} card(s) sourced — "
                       f"`manamap pilot deck-branch {slug} source {b['name']}`")
    if info["prescriptions"]["count"] > info["prescriptions"]["answered"]:
        nxt.append(f"{info['prescriptions']['count'] - info['prescriptions']['answered']} open "
                   f"prescription(s) — `/prescribe {slug}` runs the doctor ⇄ skeptic loop")
    if info["diagnosis"] and info["diagnosis"]["stale"]:
        nxt.append(f"diagnosis.json describes an older list — `/diagnose-deck {slug}`")
    if info["engine"] and info["engine"]["critic"] != "pass":
        nxt.append(f"engine.json's critic verdict is {info['engine']['critic']!r} — `/analyze-engine {slug}`")
    routed = [q for q in info["open_questions"] if q.get("settled_by")]
    if routed:
        by = {}
        for q in routed:
            by.setdefault(q["settled_by"], 0)
            by[q["settled_by"]] += 1
        nxt.append("open questions routed: " + ", ".join(f"{k} ×{v}" for k, v in sorted(by.items())))
    if info["simulation"] is None and not closed:
        nxt.append(f"no simulation runs — `simulate {slug} --vs <opp> [--vs …] --games N` "
                   f"(Forge; ◆ seeded)")
        # SIMULATION IS NOT A LIFECYCLE STAGE and is not being made one here.
        # Adding it to `STAGES` would change the denominator for all eleven decks
        # in one commit and mark nine of them newly incomplete for a measurement
        # that is optional and costs 45 minutes. It is a todo because the dossier
        # has a panel for it and that panel used to vanish — which is the same
        # defect, on an artifact that happens not to be a stage.
        info["status"]["todo"].append({
            "stage": "sim", "what": "how it does against your actual pod",
            "how": f"manamap pilot simulate {slug} --vs <opponent> --games 12"})
    elif info["experiments"] is None and info["version"]["of"] > 1 and not closed:
        nxt.append(f"{info['version']['of']} versions and no experiment — "
                   f"`experiment {slug} --a V<n> --b working --vs <pod> --games N` measures a swap")
    if info["bracket"] and info["bracket"].get("within_target") is False:
        nxt.append(f"bracket floor {info['bracket']['floor']} exceeds target "
                   f"{info['bracket']['target']} — `bracket-check {slug}` names the drivers")
    if not nxt:
        nxt.append("nothing outstanding — ask the doctor something (`/prescribe`) or play it")
    elif closed and len(nxt) == 1:
        nxt.append("nothing else outstanding — this deck's record is complete as it stands")
    return nxt


#: The three states a deck can be in, as words a reader can act on.
#:
#: NOT the on-disk vocabulary, deliberately: `paper` / `locked` stay exactly as
#: they are in `deck_versions.json`, because renaming a tracked key migrates
#: artifacts to change nothing a reader sees. Only the human-facing words are
#: settled here, in one place, so three surfaces cannot each invent their own.
#:
#: "Pinned" is the PRD's word for the first of these (§3.1) and is NOT used,
#: because the same document uses "pin" for the immutable decklist hash (§3.2) —
#: one word for a deck's physical existence and for a content sha is the
#: collision the ManaMap/Atlas rename was made to avoid. SLEEVED says the
#: physical thing plainly and cannot be confused with a hash.
STATE_SLEEVED = "SLEEVED"        # the pilot asserted this exact list is in paper
STATE_ON_BENCH = "ON THE BENCH"  # a list, not yet sleeved — every build plan
STATE_RETIRED = "RETIRED"        # broken-down / superseded / retired, one bucket


def deck_state(info):
    """Which of the three, and a short reason. See `STATE_*` above."""
    if info.get("lifecycle"):
        return STATE_RETIRED, info["lifecycle"]["headline"].lower()
    if info.get("paper"):
        p = info["paper"]
        return STATE_SLEEVED, f"V{p.get('version')}, built {p.get('built_at')}"
    return STATE_ON_BENCH, "nobody has said whether this exists in paper"


def _print(info):
    ci = "".join(info["colour_identity"]) or "C"
    state, why = deck_state(info)
    # The deck names itself. This line used to read "WORKBENCH — <slug>", which
    # named the LANDING PAGE at the top of a per-deck command — harmless while
    # "workbench" meant the whole bench, wrong once it became one surface.
    print(f"{info['slug'].upper()} — {' / '.join(info['commander'] or [])} · {ci} · "
          f"{info['size']} cards ({info['lands']} lands)")
    print(f"  {state} · {why}")
    if info["lifecycle"]:
        print(f"  ⚑ {info['lifecycle']['headline']} — {info['lifecycle']['body']}")
    print()
    v = info["version"]
    vline = (f"V{v['current']} of {v['of']} · {v['date']}" if v["current"]
             else (f"uncommitted working list ({v['of']} committed)" if v["of"] else "no git history"))
    if v["tags"]:
        vline += f" · [{', '.join(v['tags'])}]"
    print(f"  version    {vline}")
    s = info["status"]
    sline = f"{s['complete']}/{s['of']} stages"
    if s["stale"]:
        sline += f" · STALE: {', '.join(s['stale'])}"
    if s["invalid"]:
        sline += f" · INVALID: {', '.join(s['invalid'])}"
    elif s["invalid"] is None:
        # NOT CHECKED IS NOT CLEAN. Silence here would read as "every gate
        # passed", which is the one thing this command must not imply.
        sline += " · gates not run (--verify)"
    if info["bracket"]:
        b = info["bracket"]
        sline += (f" · bracket floor {b['floor']}" + (f" ({b['floor_name']})" if b.get("floor_name") else "")
                  + (f" · target {b['target']} {'✓' if b.get('within_target') else '✗'}"
                     if b.get("target") else ""))
    print(f"  status     {sline}")
    r = info["record"]
    rline = (f"{r['games']} game(s) · {r['win']}W {r['loss']}L" + (f" {r['draw']}D" if r["draw"] else "")
             + (f" · last {r['last_played']}" if r["last_played"] else "")
             + (f" · {len(r['undebriefed'])} un-debriefed" if r["undebriefed"] else ""))
    print(f"  record     {rline}")
    g = info["goldfish"]
    if g:
        gl = f"commander mean t{g['commander_mean_cast_turn']} ({g['commander_cast_by_turn_6_pct']}% by t6)"
        if g.get("combat"):
            gl += f" · combat: {g['combat']}"
        print(f"  goldfish   {gl}")
        for t in g["targets"][:3]:
            print(f"             {t['by_turn_6_pct']}% by t6  {str(t['label'])[:70]}")
    if info["engine"]:
        e = info["engine"]
        print(f"  engine     {e['verified_lines']}/{e['lines']} lines verified · critic {e['critic']}")
        print(f"             {str(e['thesis'])[:100]}")
    if info["audit"]:
        a = info["audit"]
        print(f"  audit      {a.get('archetype')} · under: {', '.join(a['under']) or '—'} · "
              f"over: {', '.join(a['over']) or '—'}")
    if info["diagnosis"]:
        d = info["diagnosis"]
        print(f"  diagnosis  skeptic {d['skeptic']}{' · STALE' if d['stale'] else ''}")
        print(f"             {str(d['verdict'])[:100]}")
    sm = info["simulation"]
    if sm:
        print(f"  simulated  {sm['runs']} run(s)"
              + ("  ** STALE — measured on a list this deck no longer holds **" if sm.get("stale") else "")
              + f" · latest {sm['games']} games vs {', '.join(sm['vs'])} · "
              f"win {sm['win_rate']} ci95 {sm['win_rate_ci95']}"
              + ("" if (sm.get("piloting") or {}).get("comparable", True)
                 else " (AI PILOTED OUR SEAT WORSE THAN THE POD)")
              + f" · mean round {sm['mean_round']} · "
              f"eliminated by {sm['eliminated_by']} · token dmg share {sm['token_damage_share']}")
    xp = info["experiments"]
    if xp:
        l = xp["latest"]
        print(f"  tested     {xp['count']} experiment(s)"
              + ("  ** STALE — neither arm is the current list **" if l.get("stale") else "")
              + f" · latest {l['question'][:70]}")
        print(f"             win {l['win_a']} → {l['win_b']} over {l['games_per_arm']}/arm · {l['reading'][:80]}")
    p = info["prescriptions"]
    if p["count"]:
        lt = p["latest"]
        print(f"  asked      {p['count']} prescription(s), {p['answered']} answered · latest "
              f"{lt['id']}: {lt['prompt'][:60]} → {lt['adds']} add / {lt['cuts']} cut")
    if info["open_questions"]:
        print(f"  questions  {len(info['open_questions'])} open")
        for q in info["open_questions"][:5]:
            print(f"             [{q['from']} → {q['settled_by']}] {str(q['question'])[:80]}")
    print("\n  NEXT")
    for n in info["next"]:
        print(f"   · {n}")


# Everything except the version block, which cannot be committed accurately.
def fetchable(info):
    """`info` minus what a committed artifact would misreport.

    `version` is derived by walking git, and the commit that changes `decklist.txt`
    receives its sha AFTER anything written in the same commit — so a committed copy
    names the previous version forever. It is dropped rather than frozen, and the
    page reads `versions.json` (built at deploy time) instead. A wrong version number
    is worse than an absent one: the log stamps games against it.
    """
    out = {k: v for k, v in info.items() if k != "version"}
    out["_note"] = ("Written by `deck-info --write`. The version block is deliberately "
                    "absent: it is a git walk, and a committed copy is one commit "
                    "behind forever. The deck page reads versions.json instead.")
    return out


def main(args):
    # --write implies --verify: the tracked info.json is read by the deck page
    # and must carry real verdicts rather than "not checked".
    info = compose(args.slug, verify=(getattr(args, "verify", False)
                                      or getattr(args, "write", False)))
    if getattr(args, "write", False):
        path = deck_dir(args.slug) / "info.json"
        path.write_text(json.dumps(fetchable(info), indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")
        return
    if getattr(args, "as_json", False):
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        _print(info)


if __name__ == "__main__":
    raise SystemExit("Run via `manamap pilot deck-info <slug>`.")
