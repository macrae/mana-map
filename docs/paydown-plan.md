# The paydown plan — six phases, in order

**What this is.** The implementation plan for the debt named in
`docs/audit-2026-09-12.md`, in the order the audit recommended: housekeeping and
doc corrections, then the daily-loop test cost, then the branch-lifecycle
cluster while it is fresh, then the three predicates, then the magazine, then
goldfish.

**This file is the tracker.** Every task has an id, a status and a gate. Update
the status column **in the same commit as the work**, and put the commit sha in
the Landed column. A task is not done because the code is written; it is done
when its gate passes and its proof is recorded. Phases run in order because
each one removes a hazard the next one would otherwise trip over — Phase 4's
registry work, for instance, is what makes Phase 1's `viz_ladder` finding
structurally impossible rather than patched.

**Status legend.** `TODO` · `WIP` · `DONE` (gate passes, proof recorded) ·
`BLOCKED` (on what, named) · `DROPPED` (with the reason, kept for the record).

---

**A note on how this file names files that do not exist yet.**
`test_docs_section_count.test_live_docs_do_not_name_source_files_that_do_not_exist`
fired on the first draft of this plan, correctly: a backtick-quoted filename
reads as a claim that the file exists. Its author already handled the proposal
case — a path whose parent directory does not exist is read as a proposal and
passes — so proposed modules here are written as paths into their new directory
(`goldfish/profiles.py`, `pilot/artifacts.py`). A proposed file inside a
directory that already exists has no such escape, which is a real gap in that
gate; P1-13 is the place to decide whether it needs one.

## Ground rules this plan must obey

These are the repo's own, from `CLAUDE.md` and the gotchas pages. Each has been
paid for and each constrains a task below.

1. **A validator that fires on correct data is worse than no validator, and the
   only way to know is to measure it against the whole fleet first.** Every new
   gate in this plan (P1-12, P3-01, P4-01, P4-04) runs against all 12 decks and
   shows zero false positives *before* it is committed. Six proposed checks have
   already been rejected on this ground.
2. **Prove a test by re-introducing the bug it was written for.** A test that
   re-derives the rule is testing itself. P2-02, P2-03, P4-01 and P4-02 each
   name the mutation that must make them fail.
3. **A loop over a possibly-empty collection needs `assert checked >= N`.**
   34 existing loops lack it; do not add a 35th.
4. **Absent means absent, never zero.** No task here may write a `0.0` where a
   figure was not measured.
5. **A new tracked artifact needs a gate in the same commit,** plus a
   `deck_status.VALIDATED` entry.
6. **Never `cache-record` to make a board green.** Phase 6 uses
   `cache-snapshot` + `cache-rerecord`, which exist for a format change, not
   `cache-record`.
7. **A model change makes every derived artifact stale.** Phase 6 changes
   `model_version` by construction and therefore owns a fleet regen.
8. **Commit code and regenerated data together only when the regeneration is
   the proof.** 30% of history mixes them; Phases 1–5 should not add to that.
   Phase 6 must, and says why.

---

## Tracker

| id | phase | task | size | status | landed |
|---|---|---|---|---|---|
| P1-01 | 1 | Diagnose `viz_ladder` before closing #42 | S | DONE | |
| P1-02 | 1 | Fix the `serve_cli` fails-open test | S | DONE | |
| P1-03 | 1 | Decide the Edgar run the piloting gate flags | S | DONE | |
| P1-04 | 1 | Edgar `info.json` staleness on a branch write | S | DONE | |
| P1-05 | 1 | File the new findings as issues; close #27, #43 | S | DONE | |
| P1-06 | 1 | Rewrite `known-issues.md` against the measured board | M | DONE | |
| P1-07 | 1 | `docs/README.md`: index, sizes, descriptions | S | DONE | |
| P1-08 | 1 | Archive six superseded docs to `docs/history/` | S | DONE | |
| P1-09 | 1 | `CLAUDE.md` corrections | M | DONE | |
| P1-10 | 1 | The other live docs: vision, simulation, pilot, pipeline, viz, data-artifacts | M | DONE | |
| P1-11 | 1 | Prune `PLAN.md` to open work | M | DONE | |
| P1-12 | 1 | `testing.md`: one runtime figure, real counts, every marker | S | DONE | |
| P1-13 | 1 | New drift gates, each measured first | M | DONE | |
| P1-14 | 1 | The four-way pilot-subcommand count disagreement in code | S | DONE | |
| P1-15 | 1 | Align the frontend cache busts and gate them | S | DONE | |
| P2-01 | 2 | #48.2 — guard the cache accessor | S | DONE | |
| P2-02 | 2 | #28 — key the freshness cache on a derived import closure | M | DONE | |
| P2-03 | 2 | #31 — one home for the deck root; complete `clear_memo` | M | DONE | |
| P2-04 | 2 | Re-measure and state ONE runtime figure | S | DONE | |
| P3-01 | 3 | #45 — a branched measurement carries the branch's own sha | M | DONE | |
| P3-02 | 3 | #46.1 — `stage`/`commit` warn on a PROPOSED branch | S | DONE | |
| P3-03 | 3 | #46.2 — `propose` amends instead of requiring withdraw | M | DONE | |
| P3-04 | 3 | #48.1 — `propose --reason` refuses or maps to `--why` | S | DONE | |
| P3-05 | 3 | #47 — branch-scoped goldfish targets | M | DONE | |
| P3-06 | 3 | #25.1 — `elsewhere` stops counting free cardboard | S | DONE | |
| P3-07 | 3 | #25.2 — `log` shows staged swaps | S | DONE | |
| P4-01 | 4 | One decklist sha, two named meanings | M | TODO | |
| P4-02 | 4 | Copies: three questions, three names, one default | M | TODO | |
| P4-03 | 4 | Six artifact registries become one | L | TODO | |
| P4-04 | 4 | Config paths are read at call time, never `from`-imported | M | TODO | |
| P5-01 | 5 | Sever the live edge: `build_index` splits in two | M | TODO | |
| P5-02 | 5 | Strip magazine fields from the manifest and `build.js` | S | TODO | |
| P5-03 | 5 | Retarget `make manuals` and the CI determinism gate | S | TODO | |
| P5-04 | 5 | Update the skills and the charter that call magazine commands | S | TODO | |
| P5-05 | 5 | Delete the renderer, its subcommands, its output, its tests | L | TODO | |
| P5-06 | 5 | Collapse the two-registry section-count truth | S | TODO | |
| P5-07 | 5 | Retire the magazine xfails and tracked artifacts | M | TODO | |
| P5-08 | 5 | Docs sweep after the delete | S | TODO | |
| P6-01 | 6 | Snapshot the fleet and the agent cache | S | TODO | |
| P6-02 | 6 | `goldfish.py` becomes a package, imports unchanged | L | TODO | |
| P6-03 | 6 | `model_version` hashes the package | S | TODO | |
| P6-04 | 6 | The proof: a byte-identical fleet regen | M | TODO | |
| P6-05 | 6 | `run()` stops raising SystemExit at its callers | M | TODO | |
| P6-06 | 6 | Metric hygiene learns the `model_*` flags | M | TODO | |
| P6-07 | 6 | `simulate_once`'s 239 locals become a state object | L | TODO | |

---

# Phase 1 — housekeeping and doc corrections

**Goal.** The repo's own account of itself is true, the red board matches a real
run, and the drift that produced twelve wrong counts cannot recur silently.

**Why first.** Every later phase changes counts, deletes modules and moves
figures. Correcting the docs afterwards means correcting them twice, and the
new gates in P1-13 are what keep Phases 2–6 from re-introducing drift.

**Exit criteria.** `known-issues.md` lists exactly the tests that fail on a
fresh run, with a named owner or unblocker each. No doc states a count the repo
contradicts. `make test` red count is unchanged or lower, and every new gate has
been shown to fire zero times on correct data.

### P1-01 · Diagnose `viz_ladder` before closing #42

`tests/test_viz_ladder.py::test_the_gate_count_matches_what_promote_would_print`
fails `assert 2 >= 3`. It compares `info.json`'s `gates` against
`promote.stage(slug)` over `promote.LADDER`. Issue #42 was "deck_info counted
gate rows in its denominator", fixed and tested — this is the same disagreement
on a different surface, so it may be the same bug incompletely fixed or a
genuine third definition.

Do this before P1-05 closes anything. Find which slug fails and whether
`info.json` or `promote.stage` is wrong. If the cause is two registries
disagreeing, do not patch it here: record it as the motivating case for **P4-03**
and leave the test red with that pointer in `known-issues.md`. A patch that
makes two of six registries agree is the same debt with a smaller radius.

**Gate.** Either the test passes, or `known-issues.md` carries a row naming
P4-03 as the unblocker.

### P1-02 · Fix the `serve_cli` fails-open test

`tests/test_serve_cli.py:85` asserts `cli._daemon_run([...]) is None` with the
comment "nothing on :1", and never sets `MANAMAP_DAEMON`. So it hits the default
`127.0.0.1:8000` from `cli.py:34`, where a real `manamap serve` answers, and the
call correctly returns an exit code. **The test fails for anyone following the
project's own recommended warm-worker workflow**, which is why it is red on this
checkout and green in CI.

Fix: `monkeypatch.setenv("MANAMAP_DAEMON", "127.0.0.1:1")` so the test tests
what its comment says. Add a sibling that a *wrong* port fails open too, and one
that a non-manamap HTTP server on the right port fails open — `http.server`
answers POST with 501, so `cli.py:51`'s status check already covers it, and a
test pins that. This is the "a control can be blind to the class it exists for"
shape: the test proves fail-open against a closed socket and never against a
stranger answering.

**Gate.** The three tests pass with `manamap serve` running and with it stopped.

### P1-03 · Decide the Edgar run the piloting gate flags

`test_sim_pilot_quality::test_no_tracked_run_is_flagged_by_accident` fires at
0.826 on
`data/decks/edgar-vampires/sim/sythis-enchantress-vs-jarad-graveyard-vs-abaddon-n40-cdaddf26-s341395706-podExperimental-c600.json`.
The test's own message says what to do: check it is a true positive before
widening the threshold, and if it is, add it to `KNOWN_FLAGGED` with the reason.
Read the run against `sim/pilot_quality.py`'s definition of the gate. A 40-game
run is under-powered anyway (MDE at 20 games/arm is 42 points), so the honest
outcome may be that the run should not be tracked.

**Gate.** The test passes, either because the run is in `KNOWN_FLAGGED` with a
stated reason or because the run is gone.

### P1-04 · Edgar `info.json` staleness on a branch write

`artifact_freshness::test_info_json_matches_a_fresh_run[edgar-vampires]` went
red after `elenda-v1` was staged on 2026-09-12. `info.json` composes a branch
summary, so **staging a branch dirties the champion's tracked artifact** — which
means every branch iteration reds a freshness gate on a deck whose own list has
not moved.

Two candidate fixes and the choice is a decision: either `deck-branch stage`
rewrites `info.json` as part of the write (consistent with "SLEEVED is built
automatically"), or `info.json`'s branch block is excluded from the freshness
comparison because it is derived from files the deck does not own. Prefer the
first: the workbench reads `info.json` and a pilot staging a swap wants the rack
to show it. Record the decision in the commit body either way.

**Gate.** The test passes, and a fresh `deck-branch stage` on any deck leaves it
passing.

### P1-05 · File the new findings; close what is fixed

File issues for, or append evidence to:

- **#47** — append the measured red: `heliod/goldfish_targets.json` now fails
  its validator with "Greater Auramancy is declared in a target but is not in
  the deck". The issue predicted this; it has happened on a sleeved deck.
- **New** — P1-04's class: a branch write dirties the champion's `info.json`.
- **New** — P1-02's class: a test that names a port it never sets, and the
  missing fail-open control against a stranger on the right port.
- **New** — P1-01's finding, if it turns out to be a third gate registry.
- The seven follow-ups the audit found named in commit bodies and
  `known-issues.md` with no issue: the parser naming split (§4), the 50
  undispatched engine open questions (§8), `experiment` not rotating seats
  (§12, `experiment.py:416-417`), `net_change.json` carrying neither model nor
  champion stamp (O4), `list_runs` returning aborted runs (O7), `token_bodies`
  reading the whole text, and the `captains-log` printer's `KeyError 'ship'`.

Close **#27** (`forge.py:67` rewritten) and **#43** (`fetch-depth: 0` at
`test.yml:56,120`). Close **#42** only if P1-01 clears it.

**Gate.** Every row in the measured red board maps to an issue or a
`known-issues.md` entry, and no issue claims something the code already does.

### P1-06 · Rewrite `known-issues.md` against the measured board

The file was last verified 2026-09-08 at 9 failing. A fresh run gives **13
failed / 3,371 passed / 327 skipped (318 cache hits) / 6 xfailed / 319 s**.

- Update the header count, date and the "last verified" line.
- **§9 is resolved** — 15 games logged, 15 annotated. Move it to the FIXED
  section rather than deleting it.
- **§13 is resolved** — heliod v1.2.1 runs exist.
- **§4's header** says unfixed while its own body at :261 records the owner map
  as fixed in `parse.py:257-285`. Split the row: the owner map is fixed, the
  naming split is open.
- **§5** lists `zur-enchantress` as live; it was broken down 2026-09-10.
- **§1's cause has moved** — ur-dragon's paper is locked at v1.2.1, so the
  unblocker is `/analyze-engine`, not a check-in.
- Add the five rows that are on no board (§3 of the audit).

**Gate.** Every failing test named in the file fails on a fresh run, and every
failing test on a fresh run is named in the file. That is a mechanical claim and
P1-13 can gate it.

### P1-07 · `docs/README.md`: index, sizes, descriptions

- Index the four unlisted files, or archive them under P1-08:
  `ur-dragon-fork.md`, `ur-dragon-refactor.md`, `naya-treasure-shell.md`,
  `ur-dragon-treasure.decklist.txt`. The "Two files" claim about
  `docs/history/` is wrong; there are three.
- The size column is stale on nearly every row and mixes units — the gotchas
  rows quote bullet counts while every other row quotes lines. Make the column
  one unit and **derive it** (P1-13).
- ":3 About 7,500 lines" against ~12,000 actual.
- ":48 18 then; 15 now" agents against 17 actual.
- ":49 manual-v5-spec DRAFT" against the file's own SUPERSEDED header.

**Gate.** P1-13's derived-size check passes.

### P1-08 · Archive six superseded docs

Move to `docs/history/`, which `test_docs_counts.py` and
`test_docs_section_count.py` both already exempt from count checks — so
archiving is also how a dated document stops being a drift source:

| file | why |
|---|---|
| `ur-dragon-fork.md` | 2026-08-26 design memo, "nothing applied", cited by nothing |
| `ur-dragon-refactor.md` | same |
| `naya-treasure-shell.md` | same |
| `ur-dragon-treasure.decklist.txt` | the memo's list |
| `manual-v5-spec.md` | self-declares SUPERSEDED 2026-09-02 by the POH |
| `agent-audit-2026-08-19.md` | executed; `agent-inventory.md` supersedes it |

**`prd-2026-08.md` stays put** — ~27 `PRD-v1 §N` citations resolve against it,
and `test_docs_counts.DESIGN_RECORDS` already exempts it.

**MEASURED 2026-09-12, and two of the six are deferred.** Counting referrers
before moving anything changed the answer:

| file | referrers | call |
|---|---|---|
| the four 08-26 memos | 2, both written by this audit | **moved** |
| `manual-v5-spec.md` | **35**, including ten legacy modules and thirteen test files | **defer to P5-08** — most of them are deleted by Phase 5, so moving it now means editing paths twice |
| `agent-audit-2026-08-19.md` | 11, including `config.py` and `deck_status.py` comments | **defer to P5-08** — same reason, smaller |

Archiving a document that thirty-five files cite is churn, not housekeeping.

Careful: `test_docs_section_count.SURFACES` and `test_docs_counts.SURFACES`
glob `docs/*.md` and skip anything with `history` in its parts, so archiving
silently removes those files from both gates. That is correct for a dated
record and wrong if a live doc is archived by mistake — check each one is really
superseded, not merely old.

**Gate.** `make test` unchanged; no live doc links to a moved file by its old
path (grep the repo for each filename after the move).

### P1-09 · `CLAUDE.md` corrections

- **":205 all 18 subcommands"** — there are 28. Write it as "28 **top-level**
  subcommands" so `test_docs_counts`' `top-level-subcommands` truth actually
  guards it. The unqualified phrase is why this drifted: the gate requires the
  qualifier and the docstring at `test_docs_counts.py:88-94` already says an
  unqualified count "is a documentation bug in its own right" — and nothing
  fails on it. P1-13 makes it fail.
- **"FOUR pages"** — five, since `spaces.html` shipped 2026-09-01. It is in
  `shell.js:67`'s SURFACES nav, documented at `docs/viz.md:22` and tested by
  `tests/test_viz_spaces_page.py`.
- **"`design.py` and `issue_length.py` are imported by LIVE code"** — true of
  `design` (`build_index.py:22-23`), false of `issue_length` (only
  `build_page.py:575`).
- The compact page "replaces" the magazine in future tense; the POH replaced it
  on 2026-09-02.
- The `make test` runtime line. This is the one CLAUDE.md already apologises
  for: it "said ~22s/~29s for weeks while the real figure was 772s". It now says
  772 s and the measured figure is 319 s warm. **Do not put a number here at
  all** — point at `docs/testing.md`, which is already declared the only place
  test counts live. Extend that rule to runtimes in P1-12.
- **The layout comment omits ~60 modules**: 35 in `pilot/` (archetypes, assess,
  autobuild, benchmark, brew, build_corpus, calibrate, candidates, captains_log,
  card_value, close, commander_search_cmd, deck_model, diagnostic, formats,
  install_agent, mana_fit, merge_captains_log, model_coverage, model_staleness,
  net_change, page_design, page_spec, promote, regen, retrieve, scaffold_targets,
  upgrades, validate_branch, validate_brief, validate_captains_log,
  validate_diagnostic, validate_log_causes, validate_net_change,
  validate_pending) and ~25 elsewhere (all of
  `sim/{experiment,pods,power,progress,failure,forge_cards,pilot_quality,pod_behaviour}`,
  `metrics.py`, `console.py`, `spaces.py`, `serve.py`, the whole `sven/`
  package, `ingest/edhrec.py`, the CardBERT set in `training/`,
  `analysis/commander_search`).

  A hand-kept list of 167 modules will drift again. Two honest options: keep the
  comment as an **annotated map of the subsystems that matter** and say so
  explicitly (it is a reading guide, not an inventory), or generate the
  inventory into `docs/architecture.md` from the filesystem and have the comment
  point there. Prefer the first for `CLAUDE.md` — it loads into every session
  and a 167-line listing is not worth the context — plus the second for the doc.
  Either way, add the ~15 modules a reader would be actively misled by their
  absence: `net_change`, `regen`, `promote`, `assess`, `diagnostic`,
  `model_coverage`, `benchmark`, `candidates`, `mana_fit`, `autobuild`,
  `captains_log`, `metrics`, `console`, `spaces`, `sven`.

**Gate.** `make test -k docs` passes; P1-13's new checks pass.

### P1-10 · The other live docs

| file | correction |
|---|---|
| `vision.md:87` | "Two surfaces over one data layer", then lists three, and omits `branch.html`. Five pages now. |
| `vision.md:118-128` | Calls `build_page` "legacy, frozen" and, four lines later, the compact manual that replaced the magazine. `poh.py` is the live renderer. |
| `vision.md:3`, `simulation.md:4` | "Last revised 2026-08-22" over 09-03 and 09-10 content. |
| `vision.md` | "Two games. Not two hundred." and "five of eleven decks" — 15 games, 12 decks. |
| `pilot.md:1505` | Says the magazine was replaced by `manual-v5-spec.md`; it was replaced by `poh.py`. |
| `pipeline.md:26,28` | Names routines `writer-prose`, `the-ten`, `issue-plan` that no longer exist. `AGENT_ROUTINES` has 12. |
| `viz.md:72` | tokens.css "~810 lines" (1,230); says tokens are shared by three pages (four). |
| `viz.md:1063` | "nine" script busts on index.html — eleven. |
| `viz.md:1071` | nine card-map artifacts — fifteen in `MM.DATA`. |
| `viz.md:1076` | "every `window.MM` member has a live caller" — eight have none, eleven are read only by tests. |
| `viz.md:1404` | A "Known Plotly gotcha" section for a dependency that is gone. |
| `data-artifacts.md` | No row for `data/pods/`, `engine_casts`, `span_vectors.npy`, the four cardbert artifacts, or twelve per-deck files the frontend fetches (audit §5.I). |
| all | Legacy vocabulary in live docs: `sideboard` (pilot.md, data-artifacts.md), "Short List" (viz.md, data-artifacts.md), "Deck Lens", kianne/kinnan as decks (pilot.md, simulation.md, viz.md). |

`test_docs_counts.test_no_surface_names_a_deleted_module` already guards seven
deleted filenames. The kianne/kinnan deck references are the same class for
**decks** and are not guarded — add them to P1-13.

### P1-11 · Prune `PLAN.md`

1,314 lines, mostly done history, and the single most-edited file in the repo
after `CLAUDE.md` (184 commits). Its deck table lists 10 of 12 decks and is
wrong about four: ur-dragon (locked v1.2.1, not "v1.0.1 SLEEVED, v1.0.2 on the
way"), heliod and gishath (locked, not PLACEHOLDER), zur (broken down 09-10, not
"on the bench"). Items listed as next that are done: stack 007 (passing),
"zero prescription files exist" (two), "`supersedes` has no scaffolding"
(`deck-state supersede` exists), "five stale diagnoses", "six of nine
strategic_frame unstamped", "three more not marked in paper" (kianne/kinnan
deleted 09-01).

Rule to apply: **PLAN.md holds open work and the current state, nothing else.**
A completed measurement belongs in `docs/gotchas-bench.md`, where it is
findable and where the gotchas index points. Move the done sections (speed
sprint, the embedding detour, propose, granted mana) and leave a one-line
pointer each. Target under 400 lines.

Do this *after* P1-06, so the red board it references is already correct.

### P1-12 · `testing.md`: one runtime figure, real counts, every marker

- **Counts**: 180 files, 3,717 cases in the `make test` selection, 3,971 with
  `-m ""`, 249 browser, 4 fleet, 1 forge, 3 serial_only. The file currently
  states 119 files and 2,882 tests (2026-08-31), plus a 09-08 table summing to
  3,522, plus "2,183 `def test_`" against 2,959 actual.
- **Runtime**: the file contradicts itself — header ~2.5 min, the 09-08 table
  735 s, ":323" 232 s for the browser suite parallel, ":57,76" 400 s. Measured
  now: **319 s warm with 318 cache hits**. State one warm figure and one fresh
  figure, each with its date and the exact command, and declare this file the
  only home for runtimes as it already is for counts.
- **Markers**: the file lists five; there are seven skipif gates in
  `conftest.py` plus four pytest markers, and ~69 ad-hoc `skipif`s. Document
  `requires_branch` (`conftest.py:98`) and `requires_roles` (`:105`), and record
  the item counts: `requires_deck` 547, `requires_data` 231, `requires_branch`
  34, `requires_roles` 20, `requires_rules` 11, `requires_strategy` 8.
- **Record the gate asymmetry**: `requires_deck` gates on a *tracked* file
  (`decks/goblin-storm/cards.json`), so none of its 547 items ever skip on a
  clone, while `requires_data` gates on an untracked one and all 231 do. That is
  a real property of the suite and it is nowhere written down.
- **xfails**: six, all strict. The file says "three tests" in one place and
  "one" in another.

### P1-13 · New drift gates, each measured first

Extend `tests/test_docs_counts.py`. **Run each new check against the repo before
committing it and record the hit count in the commit body** — a check that fires
on correct prose is worse than none (ground rule 1), and this file's own history
is the proof: its `_NOT_A_COUNT` pattern exists because an earlier version fired
on six correct charters.

1. **Unqualified subcommand counts fail.** The docstring already argues this is
   a bug; make it one. Match `N subcommands` not preceded by `top-level` or
   `pilot`, and fail with "say which: `top-level` or `pilot`".
2. **The page count is derived** from `viz/*.html` minus any documented
   non-page, so "FOUR pages" cannot survive a sixth page.
3. **`docs/README.md`'s size column is derived** from `wc -l` of each indexed
   file, with a tolerance of zero and one unit. Also assert every `docs/*.md`
   is either indexed or under `history/` — that is what let four files go
   unlisted.
4. **Deleted decks are named like deleted modules.** Extend
   `test_no_surface_names_a_deleted_module` with a derived list: any slug
   referenced in prose that is not a directory under `data/decks/` and not
   inside `history/`. Measure first — kianne and kinnan will hit, and so may
   legitimate historical prose, in which case the check needs the same
   "sentence about its removal" exemption the general section-count check
   already has.
5. **The red board matches a run.** A test that parses the failing-test names
   out of `known-issues.md` and asserts each names a real test id (via
   `--collect-only`, not by running them). It cannot assert they fail without
   running them, but it can catch a row naming a test that no longer exists,
   which is half the drift.
**Measured 2026-09-12 and two were reshaped.** Gate 1 (a bare `N subcommands`
must be qualified) hit five places and only one was wrong — the rest were this
audit quoting the defect and the magazine's own correct count of a subset. Gate
2 (the page count) hit sixteen and eleven were correct prose about a subset. Both
were rewritten to check something mechanical: the sentence beside `manamap
--help`, and the LIST of pages rather than a number. The shipped three each
measured one true hit and zero false, and each is proved by re-introducing its
defect. The three below are not yet written.

6. **Runtime figures live in one file.** Fail if a number followed by
   `s`/`sec`/`seconds`/`min` appears next to `make test` outside
   `docs/testing.md`. Measure this one especially carefully; it will hit
   prose in the gotchas pages that is recording history, which must be
   exempted the same way design records are.

### P1-14 · The four-way pilot-subcommand count in code

`console.py:3`, `serve.py:40`, `cli.py:198` and `CLAUDE.md:208` state the pilot
subcommand count as 66, 69, 102 and 107. Only the last is right.
`test_docs_counts` guards prose surfaces but its `SURFACES` list covers
`.md` files and `src/manamap/pilot/*.py` only for the section-count test — the
three offending strings are in non-pilot Python.

Fix: derive where the string is user-facing (`len(PILOT_STEPS)` is one import
away), and delete the number where it is a module docstring that does not need
it. Then add `src/manamap/*.py` to the count test's surfaces so it stays true.

### P1-15 · Align the frontend cache busts and gate them

`branch.html` is at `?v=216`; `workbench.html`, `deck.html`, `index.html` and
`spaces.html` are at `?v=215`. No test asserts cross-page agreement:
`test_viz_drill.py:168` and `test_viz_deck_lens.py:63` check `index.html` only,
`:190` checks that `deck.html` has *some* bust, and `workbench.html` and
`branch.html` have no bust test at all.

Also: **three cache-bust constants for one data file.** `data/decks/index.json`
is fetched with `?v=3` (`workbench.js:33`), `?v=2` (`build.js:47`) and `?v=9`
via `MM.DATA_VERSION` (`discovery.js:1049`), and `deck-view.js:2085` fetches
`log.jsonl` with no bust and no `no-cache`.

Fix: one bust value per page, bumped together; one `DATA_VERSION` for
`data/` fetches, read from one place. Add a test that every `?v=` in every HTML
page is the same integer and that no `../data/` fetch in `viz/js` carries a
hand-written version. Measure it first — `spaces-view.js:17` deliberately
mirrors `DATA_VERSION` and is pinned by `test_viz_drill.py:174`, so the check
must permit that one or the mirror must go.

---

# Phase 2 — the daily-loop test cost

**Goal.** An edit anywhere under `src/manamap/` stops re-running 279 heavy
freshness cases, the suite runs without the cache plugin, and a test can patch
the deck root without poisoning later tests.

**Why second.** It is the tax on every remaining phase. Phases 3–6 are all
`src/manamap/` edits, and each one currently pays a 5-minute freshness re-run it
does not need.

**Exit criteria.** A one-line edit to `sim/forge.py` invalidates zero freshness
cases; an edit to `pilot/goldfish.py` invalidates exactly the cases that read
it. `pytest -p no:cacheprovider` runs green. The memo-leak control passes.

### P2-01 · #48.2 — guard the cache accessor

`tests/conftest.py:247` (`cache = request.config.cache`) and `:264`
(`cache = item.config.cache`) both have a `cache is not None` check **on the
next line**, which is one line too late: with `-p no:cacheprovider` the
attribute does not exist at all, so the read itself raises
`AttributeError: 'Config' object has no attribute 'cache'`. Collection succeeds;
running any of the ~497 `unchanged`-fixture cases fails.

Fix: `getattr(request.config, "cache", None)` at both sites. The existing
`is not None` branches then do the right thing — no cache means no hit and no
record, which is exactly `--no-test-cache` behaviour.

Extract the two reads into one module-level `_cache_of(config)` so there is one
definition, and unit-test it with an object that has no `cache` attribute. That
is testable from inside pytest, where `-p no:cacheprovider` is not.

Then decide the doc question the issue raises: is the plugin load-bearing? After
this fix it is not, so `docs/testing.md` should say the suite runs with or
without it and that `make test-fresh` is the supported uncached path.

**Gate.** `.venv/bin/pytest -p no:cacheprovider -n0 tests/test_pilot_imports.py`
passes. The unit test on `_cache_of` passes.

### P2-02 · #28 — key the freshness cache on a derived import closure

**The problem.** Three files key the regenerate-and-compare cache on the whole
source tree: `CODE = (SRC,)` at `test_pilot_artifact_freshness.py:50` and
`test_pilot_manual_freshness.py:38`, and `INPUTS = (SRC, …)` at
`test_pilot_tracked_artifacts_validate.py:51`. That is 105 + 165 + 9 = **279
cases** re-run on any edit under `src/manamap/` — a change in `sven/`, in
`training/train_vae.py` or in a docstring re-runs 90,000 seeded goldfish games.
The issue counted 236; it has grown.

**What must not happen.** `conftest.py`'s `_digest` docstring argues *for* the
over-invalidation, and its argument is correct as far as it goes: "naming a
producer's exact inputs means tracing its transitive imports by hand, and a
missed edge does not fail — it silently serves a stale pass, which is the one
outcome this cache must never have." So the fix is **not** a hand-written list
of inputs per test. That trades a 5-minute cost for a silent-stale-pass risk,
and this repo's standing rule is that a cache may never make a board green.

**The fix: derive the closure, then prove it covers reality.**

1. Add `module_closure(*paths)` to `tests/conftest.py`, beside the `_digest`
   machinery it feeds: AST-parse each file, collect
   every `import`/`ImportFrom` node **including those inside function bodies**
   (the codebase uses lazy imports heavily — `pipeline.py`'s STEPS registry,
   `registry.py`'s dispatch, and ~40 function-local `from manamap.config import
   …`), resolve every `manamap.*` target to a file, recurse, and return the file
   set. Always include `config.py` and the conftest itself.
2. The three files call `unchanged(*module_closure(SRC / "pilot/goldfish.py"), …)`
   instead of naming `SRC`.
3. **Control 1 — the closure covers what actually gets imported.** Snapshot
   `sys.modules` before and after a real producer run; assert every
   `manamap.*` module that appeared is in the closure. This is the test that
   catches an `importlib`, a `__import__`, or a plugin edge the AST cannot see.
   Without it this task is not safe to land.
4. **Control 2 — re-introduce the bug.** Append a comment to
   `pilot/common.py` (a transitive dependency of every producer) and assert a
   freshness case misses the cache. Then append one to `sven/llm.py` (not a
   dependency of any) and assert it hits. Both directions, or the test proves
   nothing.
5. **Control 3 — it actually narrows.** Assert the closure is a strict subset of
   the tree and record the measured invalidation counts in the commit body: how
   many of the 279 cases a `goldfish.py` edit invalidates, and how many a
   `sven/` edit does (target: zero).

**Risk.** A missed dynamic edge serves a stale pass. Controls 1 and 2 are the
mitigation and neither is optional. If Control 1 cannot be made to pass — if
some producer reaches code by a path the closure cannot model — **stop and keep
`SRC`**, and record why in the conftest docstring. A 5-minute cost is cheaper
than a false green.

Rewrite the `_digest` docstring either way: it currently defends a decision this
task reverses, and leaving a stale argument in place is how the next person
re-reverses it.

### P2-03 · #31 — one home for the deck root; complete `clear_memo`

**Reproduce first.** The issue's diagnosis (a slug-keyed resolved-path memo) no
longer matches the code: `common._JSON_MEMO:56` keys on `str(path)` (absolute,
so a tmp path is a different key), `_MTIME_MEMO:72` keys on an explicit string
plus a file signature, and `deck_dir:548` memoizes nothing. The only slug-keyed
key is `scenario_facts.py:105`'s `bodies:{slug}`, and it is signature-guarded.

The surviving mechanism is **by-value imports**: `common.py` does
`from manamap.config import DECKS_DIR` and `deck_dir:568` reads that module
global, so patching `config.DECKS_DIR` alone does nothing. **22 modules**
`from`-import `DECKS_DIR`, which is why `test_serve.py:498-507` patches three
modules by hand and why the tests that broke had to be moved to
`test_serve_cli.py` with the reason in its docstring. One module has already
learned this the hard way — `deck_branch.py:199` carries a comment insisting on
`_common.DECKS_DIR` at call time rather than a `from`-import binding.

So write the failing test first:

```
def test_patching_the_configured_root_moves_every_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DECKS_DIR", tmp_path)
    (tmp_path / "slug").mkdir()
    assert common.deck_dir("slug") == tmp_path / "slug"
```

**Fix, in two parts.**

1. `deck_dir` (and `deck_file:582`, and anything else resolving a deck path)
   reads `config.DECKS_DIR` at call time. That is one patch point for the whole
   package, and it is the same "one predicate, one home" shape as Phase 4 — so
   the general rule goes in **P4-04** and only the deck root is fixed here.
2. **Complete `clear_memo()`** at `common.py:108`. It clears `_JSON_MEMO`,
   `_MTIME_MEMO`, `_STRATEGY_SHA_MEMO`, `_RULES_DB_MEMO` and
   `collection._COLLECTION_MEMO`. It does **not** clear three that outlive a
   patch: `agent_cache._SHA_MEMO:64` (its own docstring at :109 claims to
   "mirror `_SHA_MEMO.clear()`", and `sven/cache.py:109-110` records it as a
   known leak that never evicts), `goldfish._CREATURE_TYPES_CACHE:2607`, and
   `card_refs.py:73`'s `lru_cache(32)`. Three test files clear `_SHA_MEMO` by
   hand today (`test_pilot_prescribe.py:47,132,140`), which is the tell.

Then add an autouse fixture that clears when a test moved a configured path:
capture `config.DECKS_DIR` and `config.DATA_DIR` at setup, compare at teardown,
clear on difference. Cheaper and more targeted than clearing after every test,
which would make every case re-parse the 27 MB synergy graph.

**Gate.** The reproduction test passes. A second test — patch, resolve, unpatch,
assert the next resolve returns the real path — passes. `test_serve_cli.py`'s
docstring loses its workaround paragraph, and its tests move back into
`test_serve.py` and still pass under both `-n0` and `-n auto`. That last move is
the real proof: the workaround was the symptom.

### P2-04 · Re-measure and state ONE runtime figure

After P2-01..03, run `make test` (warm) and `make test-fresh` (cold), record
both with their dates and the exact command in `docs/testing.md`, and record the
invalidation win from P2-02: the number of freshness cases a single-file edit
now re-runs, before and after. CLAUDE.md points at the doc rather than repeating
the number (P1-09).

---

# Phase 3 — the branch-lifecycle cluster

**Goal.** A branch measurement cannot describe a list it did not measure, a
proposal can be amended, and the branch surface stops overstating.

**Why third.** Four of these five issues were filed on 2026-09-12, the day
before this plan, from one session of real use — the details are in the pilot's
head now. Three sleeved decks have PROPOSED branches waiting on cardboard
(`gishath/mana-v1`, `heliod/splendor-v1`, `ur-dragon/final-v1`), so this is
live workflow, not cleanup. And #47 is red on a sleeved deck today.

**Exit criteria.** `net-change --write` refuses or re-measures rather than
stamping a stale sha. Staging on a proposed branch says what it will do.
`propose` amends. `heliod/goldfish_targets.json` validates. The bill's counts
agree with the pull list.

### P3-01 · #45 — a branched measurement carries the branch's own sha

**What happens now.** `net_change.py:716` takes `decklist_sha256` from the
branch's `goldfish_metrics.json` (`b.get("decklist_sha256")`). If the goldfish
has not been re-run since the last commit, `net_change.json` is written
describing a list it did not measure, `validate-net-change` passes it, and the
only refusal comes from `deck-branch propose` — whose message tells the pilot to
re-run `net-change`, which reproduces the same stale file. The real sequence is
`fetch-deck --branch` → `goldfish --branch` → `net-change --write`.

This is the "a branched write needs a branched read" class one layer up, and it
is the third instance of that class.

**Fix, all three parts.**

1. `net-change` compares the branch goldfish's `decklist_sha256` to
   `deck_branch._sha_of_list(slug, branch)` (`deck_branch.py:865`) and refuses
   with the **three-command sequence**, naming all three commands in order.
   Re-running the goldfish automatically is tempting — it is 10k games and
   seconds — but it hides a measurement inside a reporting command, and the
   pilot should see that the list moved.
2. `validate-net-change` fails when `decklist_sha256` differs from the list on
   disk, the way `propose` already does. **Sweep the fleet first**: every
   existing `net_change.json` on all 28 branch directories must be checked
   before this lands, and any that is already stale is either regenerated or
   the check is staged behind that regeneration. A validator that reds three
   branches on landing is the thing ground rule 1 forbids.
3. Fix `propose`'s refusal text at `deck_branch.py:970-974` to name the full
   sequence instead of the single command that reproduces the bug.

Related and worth doing here: known-issues **O4** says `net_change.json` carries
neither a model stamp nor which champion it was measured against. Both are the
same "a figure must name where it was measured" rule. Add `model_version` and
the champion's `decklist_sha256` to the doc in the same commit, since every
consumer is being touched anyway.

**Gate.** A test that stages a swap on a branch with a measured `net_change.json`
and asserts `net-change --write` refuses; a second that the full sequence
produces a doc whose sha matches the list. Both drive the production functions.

### P3-02 · #46.1 — `stage` and `commit` warn on a PROPOSED branch

`stage` (`deck_branch.py:586`) and `commit` (`:707`) succeed silently on a
branch in `PROPOSED · BLOCKED`, and the commit makes the proposal read
`PROPOSED · STALE` because `branch_state:816` compares the frozen
`proposal.decklist_sha256` against `_sha_of_list`. The pilot found out from
`deck-branch list`.

`stage` already calls `meta()`, so the state is one call away. Print one line:
*"this branch is PROPOSED as vX; committing makes it STALE — re-propose after
`net-change --write`"*. Not a refusal: iterating on a proposed list is exactly
the case the pilot said the bench should support.

**Gate.** A test asserting the line appears on a proposed branch and does not on
an open one.

### P3-03 · #46.2 — `propose` amends instead of requiring withdraw

`propose` refuses an already-proposed branch (`deck_branch.py:947-950`) and
there is no amend path, so getting back to PROPOSED took `withdraw` +
`propose --as <same> --why "<the whole original reason, plus the new one>"`.
`withdraw` (`:1029`) discards `proposal.accepted_on` — the objective, its grade
and its reading at acceptance — and it is rebuilt from scratch.

Fix: `propose` with the **same** `--as` amends. It keeps the version, appends
the new `--why` to the old, refreshes `decklist_sha256` and `accepted_on`, and
pushes the previous acceptance onto `proposal.history`. A **different** `--as`
still refuses and still points at `withdraw`, because that is a change of mind
about what the list is meant to become.

`proposal.history` is a new key on a tracked artifact, so it needs a gate in the
same commit: `validate_branch` learns it, and `branch_state` ignores it (the
current acceptance is the live one).

Check `branch-view.js` and `deck-view.js` before landing: a proposal with a
history should render the current acceptance, not the first one.

**Gate.** A test that amending twice keeps one `as_version`, appends both
reasons, and leaves two entries in `history`.

### P3-04 · #48.1 — `propose --reason` refuses or maps to `--why`

`--reason` is shared on the `deck-branch` parser at `registry.py:633`, for
`merge`'s sourcing override and `propose --anyway`'s justification; `--why` is
at `:609` for `new` and `stage`. So `propose … --reason "…"` parsed, proposed,
and wrote `proposal.why: ""` with no warning. The pilot's reason went nowhere
and was found by reading `branch.json`.

Fix: `propose` refuses `--reason` without `--anyway`, with *"did you mean
--why?"*. Mapping a bare `--reason` to `--why` is the friendlier option and the
wrong one — the two words mean different things elsewhere on the same parser,
and silently reinterpreting one as the other is how this bug class starts.

This is a symptom of `add_pilot_parser` being a 754-line function of 74
`if name ==` branches (`registry.py:222`) where options are declared per
*command group* rather than per *subcommand*, so `propose` inherits `merge`'s
flags. Note it as a candidate for a later phase; do not restructure the parser
here.

**Gate.** A test that `propose --reason` without `--anyway` exits non-zero and
names `--why`.

### P3-05 · #47 — branch-scoped goldfish targets

**What happens now, measured.** `heliod/goldfish_targets.json` has a group whose
`any_of` names Lightning Greaves. `splendor-v1` swaps Greaves for Greater
Auramancy. The branch goldfish read that group at **45%** against the
champion's 68% — not because the branch is less protected, but because the
authored group could not see the replacement. Adding Auramancy makes
`validate-goldfish-targets` warn on **both** lists, which is why the test is red
today. With both names present the branch reads 53%, the honest figure for a
five-card group.

This is the `MEMBERSHIP_AXES` shape: an authored file steering a measured
figure. The deleted engine-lift metric is the precedent — three defensible
declarations of one list gave +0.007, −0.036 and +0.014 on the same 10,000
games.

**Fix, two parts, in this order.**

1. **Un-red it first.** `validate-goldfish-targets` learns branches: warn about
   a declared name only when it is in **no** list — not the deck and not any
   open branch. The shared file can then carry both names during iteration
   without reading as wrong. Sweep the fleet before landing.
2. **Then make it structural.** A branch-scoped overlay at
   `branches/<name>/goldfish_targets.json`. The read path already exists and
   already prefers it: `common.deck_file:582` returns the branch's copy when
   present. Nothing writes one. Add the writer to `stage`: when a staged swap's
   outgoing card is named in any target group, write the branch overlay with the
   incoming card substituted, and say so on stdout.

Part 2 contradicts a deliberate decision recorded in `deck_file`'s docstring —
"a branch does not have its own AUTHORED files: nobody writes a second
`goldfish_targets.json`". That was right when nothing did. Update the docstring
with the measurement above rather than leaving the two in conflict; the
45%-vs-53% reading is the argument.

**Gate.** `test_tracked_artifact_passes_its_validator[heliod/goldfish_targets.json]`
passes. A test that a staged swap touching a declared group produces an overlay
naming the incoming card, and that the deck's own file is untouched.

### P3-06 · #25.1 — `elsewhere` stops counting free cardboard

`source()` classifies a card held only by broken-down or retired decks as
`free` — loose cardboard, nothing to unsleeve, nothing to buy — and the
`unsourced` set correctly excludes it. `counts["elsewhere"]` still includes it.
Measured on `ur-dragon/eminence-v3`: `elsewhere = 6`, `free = 1`, actually
contested 5. The sixth is in `sisay`, which is retired and in a pile. The dossier
renders "sleeved in another deck: 6".

**The fix is a decision.** The pull list already chose: BUY / UNSLEEVE / PROXY /
FREE, because each costs a different thing. Make the counts agree with it —
`elsewhere` becomes contested-only and `free` gets its own tile. The alternative
(keep the coarse count, relabel the tile) leaves two vocabularies for one fact.

Touches `deck_branch.source()`'s counts, `deck-view.js`'s `branchPanel`
(`:343`) and `branch-view.js`'s `billPanel` (`:239`), and needs a `?v=` bump per
P1-15.

**Gate.** A test that a branch wanting a card held only by a broken-down deck
reports `elsewhere: 0` and `free: 1`. Note that `common.deck_is_apart:523` is
already the one predicate for "is this deck in a pile" — use it, do not add a
seventh reader.

### P3-07 · #25.2 — `log` shows staged swaps

`log()` (`deck_branch.py:748`) reads `objective`, `why`, `opened`,
`base_version`, `commits` and `merged`. It has never read `staged`. So
`unstage`'s refusal — *"No staged swap matches that — `deck-branch <slug> log
<name>` lists them"* — points at a command that does not list them, and the 21
staged swaps on `eminence-v3` are visible only through `net-change` or by
opening `branch.json`.

`log` is the better home than changing the refusal: it is the only read command
that costs nothing, and `unstage` is the thing that needs the list.

**Gate.** A test that `log` on a branch with staged swaps prints each swap, and
that `unstage`'s refusal names a command whose output contains them.

---

# Phase 4 — the three predicates

**Goal.** Three questions this codebase answers in several places get one home
each, and the divergences become impossible rather than merely absent.

**Why fourth.** Phase 3 touches the sha and the branch predicates; doing this
first would mean doing it twice. Doing it *after* Phase 3 and *before* Phase 5
matters because the magazine delete moves `line_cards` and the manifest, and the
artifact registry (P4-03) is what tells the delete which artifacts are live.

**Exit criteria.** One function per question, every caller through it, and a
test per predicate that fails when the divergence is re-introduced.

### P4-01 · One decklist sha, two named meanings

**The finding.** "The current decklist sha" means two different things and has
six definitions.

*The hash of `decklist.txt` on disk*: `deck_notes.decklist_sha256:87`
(`read_bytes`), `deck_versions.working_sha:367` (`read_text` → `.encode`),
`deck_branch._sha_of_list:865` (`read_text`), `check_in.py:177` (`read_bytes`),
`deck_branch.py:1196` (`text.encode`), `fetch_deck.py:403` (`text.encode`).

*The stamp recording what `cards.json` was built from*:
`install_agent.decklist_sha:78`, `deck_info._current_sha:534`,
`validate_diagnosis.is_stale:569`.

**Two live consequences.** `read_bytes` and `read_text().encode("utf-8")` differ
whenever the file has CRLF endings or a BOM, so the same list hashes differently
in `deck_notes` than in `deck_versions` — a captain's-log entry would not join
to its own version. And `deck_status._stamp_is_stale:160` is prefix-tolerant
because "three decks store twelve characters" while
`validate_diagnosis.is_stale:569` compares exactly — the second is the bug the
first was written to fix.

**Fix.**

1. `common.decklist_sha256(slug, branch=None)` — one definition, `read_bytes`,
   the full 64 hex. Every list-hash caller goes through it.
2. `common.measured_sha(slug, branch=None)` — the stamp `cards.json` carries.
   **A different name for a different question**, so a reader can see which one
   a call site means.
3. `common.sha_matches(a, b)` — one comparison, prefix-tolerant, with the
   twelve-character case in its docstring. Both `_stamp_is_stale` and
   `is_stale` call it.

**Before landing, prove no tracked sha changes.** Every decklist in the repo
must be LF-only with no BOM, so that `read_bytes` and `read_text().encode()`
agree today and the change is a no-op on current data. Check it
(`file data/decks/*/decklist.txt`, and a grep for `\r`), and if any file has
CRLF, fix the file and regenerate the stamps that referenced it **in a separate
commit** so the two changes are not entangled.

**Control.** Write `decklist.txt` with CRLF into a tmp deck and assert every
caller returns the same digest. That test fails today against
`deck_versions.working_sha`, which is the re-introduction proof.

### P4-02 · Copies: three questions, three names, one default

`common.expand_copies:324` exists and is used 22 times. There are 23
`get("quantity")` sites, but — an important correction to the audit's framing —
they are **not** 23 missed calls. They are three different questions:

| question | sites | the right home |
|---|---|---|
| expand to one element per physical card | `goldfish.py:2893`, and the 22 existing callers | `expand_copies` |
| sum a count of copies | `deck_facts.py:349,353`, `mana_analysis.py:187,195`, `poh.py:114`, `pool_facts.py:134`, `check_in.py:54`, `deck_history.py:95`, `deck_branch.py:549`, `goldfish.py:5499`, `validate_deck.py:21`, `artist_credits.py:66,203,240`, `build_page.py:98` | a new `count_copies(cards)` |
| arithmetic on one entry's quantity | `card_value.py:91`, `deck_branch.py:624,682,689`, `validate_deck.py:41`, `diagnostic.py:580` | leave alone — it is genuinely per-entry |

**The real defect is the default.** `validate_deck.py:21` sums
`c.get("quantity", 0)`; every other site defaults to 1. An entry written without
a `quantity` key makes the deck read as 99 cards and fail the 100-card
invariant — latent today only because `fetch_deck` always writes the key.

**Fix.** Add `common.count_copies(cards)` with the default of 1, route the
middle column through it, and fix `validate_deck.py:21`. Leave the third column,
and add a one-line comment at each of those six sites saying it is per-entry on
purpose — otherwise a later sweep "fixes" them.

**Control.** `count_copies(cards) == len(expand_copies(cards))` for every deck's
`cards.json`, with `assert checked >= 12`. Plus a test that an entry with no
`quantity` key counts as one, which fails today against `validate_deck`.

### P4-03 · Six artifact registries become one

**The finding.** "What artifacts does a deck have, need, or get gated on" is
declared six times: `deck_status.STAGES:43` and `VALIDATED:318`,
`regen.STAGES:51` and `BOOTSTRAP:100`, `promote.GATES:63`, `assess._GATES:45`,
`deck_info._gates:650`, and `serve.MEASURES`/`SCAFFOLDS:797-819`. None disagreed
loudly enough to notice until `viz_ladder` started failing `assert 2 >= 3` —
which is P1-01, and which is why that task defers here rather than patching.

This is the exact shape of the `deck_is_apart` consolidation: four modules had
grown their own answer to "is this deck in a pile", none disagreed *yet*, and it
was already costing something.

**Fix.** One module, `pilot/artifacts.py`, declaring each artifact once:

```
ARTIFACT = namedtuple("Artifact", "name stage producer validator "
                                  "bootstrap gate authored branch_scoped")
```

- `name` — the filename.
- `stage` — where it sits in the sequence `deck_status.STAGES` defines.
- `producer` — the command that writes it.
- `validator` — the command that gates it, or None.
- `bootstrap` — may `regen` create it when absent on a pinned deck.
- `gate` — does `promote`/`assess` count it as a requirement.
- `authored` — is it written by hand or by an agent (so freshness means
  something different).
- `branch_scoped` — does a branch get its own copy (this is what P3-05 adds and
  what `common.deck_file` reads).

Then every consumer derives its view: `deck_status.STAGES` is a projection,
`regen.BOOTSTRAP` is a filter on `bootstrap`, `promote.GATES` a filter on
`gate`, `deck_info._gates` the same filter, `serve.MEASURES` a projection of
`producer`.

**Control — nothing is lost.** For each of the six existing lists, a test that
the projection from the new registry equals the old literal. Keep the old
literals in the test file as expected values for one commit, then delete them.
This is the only safe way to refactor six hand-kept lists: prove the new one
reproduces each, then remove them.

**Gate.** `viz_ladder` passes for the structural reason, not a patched
denominator. `deck_status.VALIDATED` gains any artifact that has a validator and
was missing from it — the gotcha says a new tracked artifact needs a
`VALIDATED` entry so status sees what the tests see, and a single registry makes
that automatic.

**Size.** Largest task in the plan outside Phases 5 and 6. Expect the registry
itself to be small and the consumer rewrites to be most of the work.

### P4-04 · Config paths are read at call time, never `from`-imported

The general form of P2-03. **22 modules** `from`-import `DECKS_DIR`, so patching
`config.DECKS_DIR` reaches none of them, and the same is true of `DATA_DIR`,
`COLLECTION_DIR`, `MANUALS_DIR` and `FORGE_DECKS_DIR`. `deck_branch.py:199`
already carries the lesson in a comment; nothing enforces it.

**Fix.** A rule, applied: a path constant is read as `config.NAME` at call time.
Non-path constants (`BRACKET_MAX`, `MECHANICAL_TAGS`, `AGENT_ROUTINES`) keep
their `from`-imports — they are values, not environment, and re-reading them
buys nothing.

**Control.** An AST check over `src/manamap/**.py`: no `ImportFrom` of
`manamap.config` may name a constant ending in `_DIR` or `_PATH`. **Measure it
first** — it will hit ~40 sites, and each must be converted before the check can
land, so the check is the last commit of the task, not the first. If some site
genuinely needs the module-level binding (a default argument evaluated at import
time, say), the check needs an allowlist with a reason per entry, and an
allowlist of 40 is a sign the rule is wrong.

**Why it is worth doing.** It is the root of #31, of the `test_serve.py`
three-module patch, of the workaround file `test_serve_cli.py`, and of the fact
that `MANAMAP_DATA_DIR` and `MANAMAP_COLLECTION_DIR` — both documented as
overrides — only work if set before the first import.

---

# Phase 5 — delete the magazine (#4)

**Goal.** ~6,600 lines of frozen renderer, its eight subcommands, its HTML
output and its 13 test files leave the repo, and nothing live loses a function.

**Why fifth.** It is the largest single reduction available and it is now a
small job, because the dependency map is one edge. It goes after Phase 4 because
P4-03's registry is what says which artifacts are live, and before Phase 6
because deleting 6,600 lines from `pilot/` shrinks what Phase 6's import-closure
cache (P2-02) has to consider.

**Exit criteria.** `grep -rn "build_manual\|issue_spec\|artist_credits" src/`
returns nothing. `manuals/p/` still renders byte-identically. CI's determinism
gate still runs and still guards something.

### The dependency map, measured

Eight modules self-labelled `LEGACY (2026-08-19)` plus two more:
`build_manual.py` (2008), `design.py` (2011), `issue_spec.py` (412),
`validate_issue.py` (757), `validate_considering.py` (232),
`issue_length.py` (176), `artist_credits.py` (308), `short_list_art.py` (136),
`build_page.py` (585), `page_design.py` (174).

**The only live-code edge** is `build_index.py:20-26`, which imports
`stylesheet_link`, `write_stylesheet`, `FONT_LINK`, `badge`, `barcode`, `esc`
from `design`, and `MASTHEAD`, `SERIES_SLUG`, `STANDING_TAGLINE` from
`issue_spec`. And `build_index` is load-bearing twice over:

- `build_index.line_cards:99` is imported by `engine_facts.py:33`,
  `deck_map.py:55` and `validate_engine.py:86`.
- `build_index.write_manifest:478` writes `data/decks/index.json`, the manifest
  the entire frontend fetches, and `build_index.main` is called by
  `serve.py:437,1101`, `deck_delete.py:149`, `deck_state.py:96` and
  `autobuild.py:443`.

`page_spec.py` is **live by test** — `test_viz_deck_lens.py:242` locks
`deck-view.js`'s `SECTIONS` to `page_spec.DOSSIER_SECTIONS`. It stays.
`page_design.py` is legacy by use (only `build_page.py` imports it) and goes.

Outside `src/`: `Makefile:96` (`make manuals` → `build-manual`), CI's
`git diff --exit-code -- manuals/`, three skills
(`write-manual`, `publish-deck`, `author-decision`), `pilot-notes.md:53,72`,
and 13 test files including `test_pilot_build_manual.py` (32 legacy imports) and
`test_pilot_validate_issue.py` (19).

### P5-01 · Sever the live edge

Split `build_index.py` in two:

- **`pilot/deck_manifest.py`** (live) — `line_cards`, `gather_entries`,
  `write_manifest`, and the `build-index` subcommand. No magazine imports.
- The newsstand HTML renderer (`render_index`, `EXTRA_CSS`, the `design` and
  `issue_spec` imports) stays behind in the legacy set and dies with it in
  P5-05.

Retarget the three `line_cards` importers and the five `build_index.main`
callers. Keep `build-index` as the subcommand name — it is in CLAUDE.md, in
skills and in the Makefile — and have it write only the manifest.

**Gate.** `data/decks/index.json` is byte-identical before and after, which is
the whole claim. CI's determinism job already diffs it.

### P5-02 · Strip magazine fields from the manifest and `build.js`

`write_manifest:478` publishes `volume`, `coverline` and `published` per deck.
Measured consumers: `volume` only at `build.js:1092` (rendering
`Vol. 003 — Name` in the Build-mode deck picker), `coverline` only at
`build.js:1130`, `published` only at `deck-view.js:2115`. `deck-view.js:2037`
already carries a comment rejecting volume numbers on the workbench as "the
magazine's sentinel leaking".

Drop `volume` and `coverline` from the manifest and from those two `build.js`
sites. Decide `published`: either it becomes a live predicate (does
`manuals/p/<slug>.html` exist) or it goes and `deck-view.js:2115` reads
something else. `gather_entries` also sorts decks by volume with a sentinel of
999 for a deck without an `issue.json` — replace that ordering with one that
means something now (paper-locked first, then alphabetical, matching the
workbench racks).

**Gate.** A `?v=` bump (P1-15), the browser tests for Build mode's picker, and
the manifest schema test.

### P5-03 · Retarget `make manuals` and the CI determinism gate

`Makefile:96-98` runs `build-manual` for any deck with an `issue.json`, and CI
runs `make manuals && git diff --exit-code -- manuals/ data/decks/index.json`.
That gate is valuable — it is the determinism claim asserted from outside the
code that asserts it, and it caught a real break on CI's first run — so it must
survive the delete pointed at the live renderer.

Retarget `make manuals` to `build-poh` over the decks that have a handbook, and
the diff to `manuals/p/` plus `data/decks/index.json`.

Also fix the three Makefile smells found in the audit while the file is open:
`demo` runs `build-index` three times, `serve` labels `manuals/index.html` as
"issues", and the comment claiming "exactly ONE" serial test (there are three).

**Gate.** CI green, and a deliberate one-byte edit to a POH input makes the
diff gate fail (prove the gate still guards something).

### P5-04 · Update the skills and the charter

`.claude/skills/write-manual/SKILL.md`, `publish-deck`, `author-decision` and
`.claude/agents/pilot-notes.md:53,72` invoke magazine commands. Rewrite them
against `build-poh` / the POH procedures skill. `write-manual`'s own description
already says "legacy magazine renderer until manual-v5" — the POH is that
replacement and shipped 2026-09-02.

`test_docs_counts.SURFACES` includes `.claude/**.md`, so a stale command name in
a skill is guarded for deleted *modules* but not for deleted *subcommands*.
Consider adding that to P1-13's list.

### P5-05 · Delete

In one commit, after P5-01..04:

- The ten modules named above (`page_spec.py` stays).
- The eight subcommands: `build-manual`, `build-page`, `validate-issue`,
  `validate-considering`, `artist-credits`, `short-list-art`, `issue-length`,
  and the newsstand half of `build-index`. Remove their `registry.py` entries
  and their branches in `add_pilot_parser`. `registry.py:12` currently
  describes `build-page` as "the compact deck page (the Pilot's Manual)", which
  contradicts CLAUDE.md — the delete resolves it.
- `manuals/*.html` (nine issue pages) and `manuals/magazine.css`,
  `manuals/index.html`. **Keep `manuals/p/`** and its `poh.css`/`page.css`.
- The 13 test files, or their legacy portions:
  `test_pilot_build_manual.py`, `test_pilot_validate_issue.py`,
  `test_pilot_validate_considering.py`, `test_pilot_artist_credits.py`,
  `test_pilot_issue_length.py`, `test_pilot_issue_status.py`, and the legacy
  parts of `test_pilot_deck_map.py`, `test_viz_deck_lens.py`,
  `test_docs_section_count.py`, `test_pilot_voice_lint.py`,
  `test_pilot_poh.py`, `test_pilot_tracked_artifacts_validate.py`,
  `test_pilot_manual_freshness.py`.

**Count the deletion in the commit body**: modules, lines, subcommands, tests,
and the resulting `pilot/` LOC. The claim "~6,600 lines, 15% of the pilot
package" should be re-measured, not repeated from this plan.

### P5-06 · Collapse the two-registry section count

`test_docs_section_count.test_no_surface_states_a_wrong_section_count` accepts
two truths — `issue_spec.DEPARTMENTS`' seventeen and `page_spec.SECTIONS`' nine
— and its docstring says: *"This collapses back to a single truth the moment
`build_manual.py` is deleted — if you are reading this after that, take the set
apart."* Do exactly that. Also delete
`test_no_surface_hardcodes_the_department_id_list`, whose subject is gone.

### P5-07 · Retire the magazine xfails and tracked artifacts

Six strict xfails, four of them magazine-shaped: `STALE_XFAIL` on
heliod/`considering.json` (`test_pilot_tracked_artifacts_validate.py:189`) and
`ISSUE_XFAIL` on edgar/ur-dragon/heliod (`:337-356`). A strict xfail whose
subject is deleted must be removed, not left to fail.

Decide the fate of the tracked artifacts nothing will regenerate:
`issue.json`, `considering.json`, `issue_plan.json` per deck, and the per-deck
panel keys. They are a record of the magazine era. Options: delete them (the
renderer is gone, the HTML is gone, git holds both), or keep them and add a
registry entry (P4-03) marking them `producer: None, validator: None` so nothing
gates them. **Prefer deleting** — a tracked artifact with no producer and no
validator is exactly the "half-in" state the audit flags for the embedding
experiments, and `docs/gotchas-magazine-legacy.md` already holds the lessons.

Also resolves known-issues §11 ("retired agents' artifacts") and the
`hapatra`/`yawgmoth` file gates at `test_pilot_issue_status.py:114` and
`test_pilot_pending.py:146`, which key on files that exist only on archived
decks.

### P5-08 · Docs sweep after the delete

`docs/pilot.md`'s LEGACY block at :1505 goes. `docs/gotchas-magazine-legacy.md`
stays as history and gets a header saying the code it describes was deleted on
this date. `docs/data-artifacts.md` loses the magazine rows. CLAUDE.md's layout
comment loses the legacy block (and P1-09's honesty about what the comment is
makes that easy). `docs/README.md` sizes re-derived by P1-13's check.

---

# Phase 6 — goldfish

**Goal.** `goldfish.py` becomes a package of comprehensible modules with
**byte-identical output**, its declaration errors stop killing callers, and the
metric-hygiene test learns the flags it currently cannot see.

**Why last.** It is the highest-risk change in the repo: the simulator produces
every measured figure on the bench, `model_version()` is a sha over its bytes,
and that sha keys the agent invocation cache for 12 routines. Every earlier
phase reduces the risk — P2-02 makes the test loop fast enough to iterate,
P4-03 makes the artifact dependencies explicit, P5 removes 6,600 lines of noise
from the package.

**Exit criteria.** Every deck's `goldfish_metrics.json` is byte-identical apart
from `meta.model_version`. No module in the package exceeds ~800 lines. `run()`
raises no `SystemExit`. `test_metric_hygiene` covers all 12 `model_*` flags and
`model_coverage.CHANNELS` is complete.

### The cost that must be budgeted first

`model_version()` at `goldfish.py:61` is `sha256(pathlib.Path(__file__).read_bytes())[:12]`.
Its docstring defends the coarseness: *"a comment edit bumps it, which costs a
regeneration nobody needed. The alternative is a curated list of 'model-facing'
lines, which is exactly the judgement call that goes wrong silently."* That
reasoning is right and it means **the split changes the stamp by construction**.

The stamp appears in `meta.model_version` in every deck's
`goldfish_metrics.json`, and `config.py:1391-1597` names it in the input
fingerprint of **12 `AGENT_ROUTINES`**. So the split invalidates the agent
invocation cache fleet-wide — which is LLM spend, tracked in
`docs/agent-cost.md`.

The repo already has the right tool: `cache-snapshot` records every routine's
status *before* a cache-format change, and `cache-rerecord` re-fingerprints what
the change invalidated, **gated on that snapshot**. This is a format change in
all but name. Use those two, not `cache-record`, which is the thing ground rule
6 forbids.

### P6-01 · Snapshot the fleet and the agent cache

Before touching a line:

1. `manamap pilot cache-snapshot` for every deck.
2. Copy every `data/decks/*/goldfish_metrics.json` and every branch's copy to a
   scratch directory outside the repo. These are the comparison baseline for
   P6-04.
3. Record which artifacts embed a goldfish figure — `mana_analysis.json`
   embeds them by design (CLAUDE.md: run `mana-analysis` after `goldfish`), and
   `net_change.json`, `diagnostic.json` and `info.json` all read them. P4-03's
   registry should be able to answer this; if it cannot, that is a gap worth
   noting.

### P6-02 · `goldfish.py` becomes a package

`src/manamap/pilot/goldfish/` with `goldfish/__init__.py` re-exporting **every public
name the current module exposes**, so all 4 in-process callers
(`benchmark.py:213`, `diagnostic.py:488`, `deck_branch.py:497`,
`calibrate.py:164`), the registry entry, and every test import keep working
unchanged. Verified boundaries, from the current file:

| new module | current lines | contents |
|---|---|---|
| `goldfish/assumptions.py` | 55–305 | `model_version`, `X_DRAW_MIN`, `MODEL_ASSUMPTIONS`, `TREASURE_ASSUMPTIONS`, `DISCARD_ASSUMPTIONS`, `COMBAT_ASSUMPTIONS` |
| `goldfish/profiles.py` | 507–2743 | the ~120 regexes and every classifier: `event_payoffs`, `token_doubler`, `cast_token_profile`, `sac_outlet_profile`, `death_profile`, `draw_profile`, `is_tutor`, `treasure_profile`, `team_haste_grant`, `creature_body_count`, `is_etb_engine`, `drain_profile`, `combat_profile`, `room_profile`, `produced_mana`, `body_count`, `cost_reduction`, `reduced_cost`, `devotion_gate`, … — pure functions of card text |
| `goldfish/library.py` | 2744–2939 | `classify`, `build_library`, `keepable`, `_target_met` |
| `goldfish/turn.py` | 2940–4940 | `simulate_once` (see P6-07) |
| `goldfish/aggregate.py` | 4941–5199 | `_round`, `aggregate`, `BAND_ROWS` |
| `goldfish/run.py` | 5200–5689 | `run` (see P6-05) |
| `goldfish/__init__.py` | 5690–5830 | `_band_value`, `_ability_band`, `_coverage_preflight`, `main`, `_print_band`, and the re-exports |

`_CREATURE_TYPES_CACHE:2607` moves with `cost_reduction` into `goldfish/profiles.py` and
must be added to `common.clear_memo()` — P2-03 already does that, which is
another reason for the phase order.

**Do this as pure moves.** No renames, no signature changes, no behaviour
changes, nothing "tidied on the way past". A move-only commit is reviewable by
`git diff -M`; a move-plus-tidy commit is not, and this is the one file in the
repo where an accidental change is most expensive.

### P6-03 · `model_version` hashes the package

`sha256` over the sorted concatenation of every `.py` file in the package
directory. Keep the docstring's argument and extend it: the version is coarse on
purpose, and the unit is now the package. Note in the docstring that this
commit's own regeneration was the one-time cost of the split.

### P6-04 · The proof: a byte-identical fleet regen

1. `manamap pilot regen --jobs 8` across the fleet (72 targets, ~109 s).
2. Diff every `goldfish_metrics.json` against P6-01's baseline **ignoring
   `meta.model_version`**. Any other difference means the split changed
   behaviour and must be found before going further. This is the only
   acceptable proof, and it is the reason the split is worth attempting at all:
   seeded determinism makes a 5,830-line refactor verifiable.
3. `cache-rerecord`, gated on P6-01's snapshot.
4. Commit code and regenerated data **together**, because here the regeneration
   *is* the proof (ground rule 8's stated exception). Put the diff summary in
   the commit body: N files, M bytes changed, all in `meta.model_version`.

Watch for the trap the audit found in the history: a branched write needs a
branched read, and `regen` must not measure a champion and file it under a
branch. That bug has shipped three times, once inside the commit fixing the
class.

### P6-05 · `run()` stops raising SystemExit at its callers

`run` raises `SystemExit` at :5229, :5308, :5359, :5380 and :5450 — declaration
validation, mostly — and is called **in-process** by `benchmark.py:213`,
`diagnostic.py:488`, `deck_branch.py:497` and `calibrate.py:164`. So a typo in
one deck's `goldfish_targets.json` kills whatever command is running, mid-regen,
with a message about a file the pilot was not asking about.

Fix: a typed `DeclarationError` raised by `run`, caught and converted to
`SystemExit` by `main` only. The four callers decide for themselves — `regen`
should skip the deck and report it at the end rather than abort the sweep.

This is one instance of a class: **259 `SystemExit` raises in pilot library
code**, ~60 inside non-`main` functions (`deck_branch.stage/unstage/commit/propose`,
`deck_versions.tag/set_paper`, `common.resolve_out_path:314`,
`common.report_errors:429`). Fix the goldfish instance here because it has
measurable consequences; file the class as its own issue rather than widening
this phase.

**Gate.** A test that a malformed declaration on one deck leaves a multi-deck
`regen` running and reports the deck at the end.

### P6-06 · Metric hygiene learns the `model_*` flags

Two measured gaps.

`test_metric_hygiene.test_every_signal_the_model_sets_is_read_by_something`
AST-sweeps the keys `classify()` emits. It does **not** cover the 12 `model_*`
declaration flags, which `run` reads at :5293–5446 and threads into
`simulate_once` as 26 keyword parameters.

`model_coverage.CHANNELS:45` lists **7** flags and omits
`model_commander_animate`, `model_commander_combat_reveal` and `model_deaths`.
So a new commander flag can be declared in a deck's targets, read by `run`,
acted on by the simulator, and **invisible to `model-coverage`** — the command
whose whole job is saying what the model cannot see — with nothing failing.

This matters because the flags are how the two worst measurement errors in the
project's history were introduced and found: eminence absent entirely
(understating bodies at turn ten by 50%) and the attack tutor firing 5.70 times
a game against Forge's 1.22 (which took kill-by-t8 from 0.501 to 0.173).

**Fix.** Extend the hygiene test to assert every `model_*` flag `run` reads is
(a) named in `model_coverage.CHANNELS`, (b) documented in one of the
`*_ASSUMPTIONS` lists, and (c) actually read inside `simulate_once` — the
"a flag the model sets is a claim the model must act on" rule that
`treasure_doubler` violated when it shipped set-and-unread and fifteen
candidates returned byte-identical results.

Complete `CHANNELS` with the three missing flags. Re-run `model-coverage` across
the fleet afterwards and record the new DARK/never-cast figures, since adding
channels changes what "seen" means — the audit notes PLAN.md already flags that
coverage "over-claims `seen` through a token clause on an unread trigger".

### P6-07 · `simulate_once`'s 239 locals become a state object

The deep half, and it should be a separate sub-phase with its own byte-identical
proof. `simulate_once` is 1,999 lines with 239 assigned locals, 10 nested
closures (`draw_n`, `discard_n`, `creature_entered`, `spend`,
`_commander_arrives`, `_cast_triggers`, `_free_creature_enters`,
`_engine_permanent`, …) and a 33-key return dict. Its phases are marked only by
comments: upkeep treasure and draw at :3233/:3284, commander/partner/attack-tutor
at :3545–3602, reducers/ETB/rocks/tutors/extra-combat/doublers/draw/drain at
:3698–3919, bodies at :4139, discard and draw payoffs at :4409, combat and
stockpile at :4887.

**Approach.** The closures become methods on a `GameState` dataclass holding the
locals; the comment-marked phases become `cast_rocks`, `cast_tutors`,
`cast_bodies`, `combat`, `end_step`. Seeded determinism is preserved **as long
as the `rng` call order inside the turn loop is untouched** — which is the whole
constraint, and the byte-identical fleet diff is what proves it held.

**Do it one phase at a time**, with a full fleet diff after each. Extracting all
eleven phases in one commit and then finding a difference gives no bisection
point. Eleven commits each proven byte-identical is slower and is the only way
this is safe.

**Stop condition.** If two consecutive phase extractions produce a diff that
cannot be explained within an hour, stop and leave `simulate_once` as it is.
A 1,999-line function that produces correct, reproducible numbers is better
debt than a decomposed one that produces different ones. Record the stop in
this file as `DROPPED` with the reason — that is a legitimate outcome and worth
more than a silent abandonment.

---

## Risk register

| risk | phase | mitigation |
|---|---|---|
| A new gate fires on correct data | 1, 3, 4 | Ground rule 1: measure against all 12 decks before committing, record the hit count |
| The import-closure cache serves a stale pass | 2 | Controls 1–3 in P2-02; fall back to `SRC` if Control 1 cannot pass |
| A validator lands and reds three branches | 3 | Sweep every `net_change.json` and `goldfish_targets.json` before P3-01.2 and P3-05.1 |
| The sha change alters a tracked stamp | 4 | P4-01 proves every decklist is LF-only first; any CRLF fix is a separate commit |
| Six registries do not reduce to one cleanly | 4 | P4-03's control keeps the old literals as expected values for one commit |
| The magazine delete breaks the manifest | 5 | P5-01 lands first and its gate is a byte-identical `index.json` |
| CI's determinism gate stops guarding anything | 5 | P5-03 retargets it and proves it still fails on a deliberate edit |
| The goldfish split changes a figure | 6 | P6-04's byte-identical fleet diff; P6-07 one phase per commit |
| The goldfish split costs LLM spend | 6 | P6-01 snapshot + P6-04 `cache-rerecord`, never `cache-record` |
| A phase drags and the audit goes stale | all | Phase 1 is days, not weeks. Re-run the audit's measurements at the end of Phase 3 |

## What this plan does not do

Named so nobody has to wonder whether they were forgotten.

- **The PRD phase issues** (#5, #6, #7, #8, #9, #10, #11, #13, #14) — product
  work, not debt. #6 unblocks the benchmark trio and #11 unblocks #14 when the
  PRD calls for them. Their `§`/`Phase` citations resolve against the
  **superseded** `prd-2026-08.md`, which P1-08 leaves in place for exactly that
  reason.
- **The embedding chain** (#12 → #40 → #38) — research, strictly in that order,
  and #38 keys on signatures a retrain invalidates.
- **The parallelism pair** (#29, #30) — `candidates` and `card-value` run one
  10,000-game goldfish per card serially and are embarrassingly parallel. Worth
  doing, independent of everything here, and cheaper after P6-02 makes the
  simulator importable in pieces.
- **The VAE/CardBERT experiments** (~3,600 LOC in `training/`, four tracked
  artifacts over 30 MB, six undocumented CLI commands) — they are either a
  documented research appendix or a delete, and that is the pilot's call, not a
  debt paydown. They should not stay half-in.
- **The Forge scheduler** (#26) — one JVM per game would make #27's replay
  property true and decide what `run_id` should contain. Independent.
- **`registry.add_pilot_parser`'s 754 lines** and the 21 unreasoned
  `except Exception` (#23) — both real, both noted as classes in P3-04 and
  P6-05, neither in scope. File them rather than letting them widen a phase.
