# PLAN — current state and what's next

*The resume-here doc. `README.md` orients, `CLAUDE.md` carries the gotchas, this says what
exists and what is open. Superseded plans live in `docs/history/`.*

Last updated **2026-08-14**. Everything below is committed and pushed except where marked.
Every figure was derived from the repo at write time — **do not quote one from memory**;
the commands that print them are named beside each.

## What this is

Two products in one repo, sharing a data layer and a CLI.

**The card tool** opens on a single card and grows a graph as you click. Underneath it,
34,890 oracle cards are embedded twice — once for layout, once for function — projected to
2D, and served as a static site from `viz/`. The atlas drifts at altitude and settles as you
zoom in.

**Pilot's Manual** turns one Commander deck into a self-contained web issue under a
three-tier evidence contract. The deck can be produced by a deterministic builder from a
written brief, and the issue now opens on two pictures of the deck: its **constellation**
(what shape it is) and its **engine flow** (how it runs).

**Nine issues published**, one per deck:
[001 Goblin Storm](https://macrae.github.io/mana-map/manuals/goblin-storm.html) ·
[002 Hapatra](https://macrae.github.io/mana-map/manuals/hapatra.html) ·
[003 Sisay](https://macrae.github.io/mana-map/manuals/sisay.html) ·
[004 Heliod](https://macrae.github.io/mana-map/manuals/heliod.html) ·
[005 Ur-Dragon](https://macrae.github.io/mana-map/manuals/ur-dragon.html) ·
[006 Edgar](https://macrae.github.io/mana-map/manuals/edgar-vampires.html) ·
[007 Gishath](https://macrae.github.io/mana-map/manuals/gishath.html) ·
[008 Yawgmoth](https://macrae.github.io/mana-map/manuals/yawgmoth-swarm.html) ·
[009 Radagast](https://macrae.github.io/mana-map/manuals/radagast.html) ·
[newsstand](https://macrae.github.io/mana-map/manuals/index.html)

Scale: 55 `manamap pilot` subcommands, 18 top-level subcommands, 15 agents, 19 skills,
10 cache-gated routines (plus `stack:`/`decision:`/`prescription:` per artifact), and the magazine's department list as `issue_spec.DEPARTMENTS`
gives it — **`OPTIONAL_DEPARTMENTS` is empty**, both migrations having landed on all nine
decks. Test counts live in `docs/testing.md`.

## Start here on any deck

```bash
manamap pilot deck-status <slug>
```

It reports every lifecycle stage as present, missing or **STALE**, and flags stages added
recently that an older deck will not have. `/publish-deck` is the runbook that sequences the
whole thing; `pilot/deck_status.py:STAGES` is the machine-readable version, and **a new
phase belongs there** or the next person will not find it.

Staleness is an ERROR; incompleteness is a state. A half-built deck is work in progress. A
deck whose artifacts disagree about which decklist they describe is confident and wrong.

## The evidence contract

| | Tier | Granted by |
|---|---|---|
| ✓ | rules-verified | A stack artifact whose every step cites a real CR rule verbatim (`validate_stack.py`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | Deterministic Python over committed artifacts. Same inputs, same bytes, no LLM. |
| ★ | coaching | Labelled judgment. Useful, and never disguised as measurement. |

**Agents return JSON and the renderer emits HTML**, so regenerating a manual from unchanged
artifacts is byte-identical. That is why a refactor cannot disturb a published issue.

## The decks

Run `deck-status` per deck for the live picture. As of this writing:

| Deck | Vol | Stacks | Bracket | Map | Engine | Front of book |
|---|---|---|---|---|---|---|
| `goblin-storm` | 001 | 5/5 | 4 | named | `pass` | yes |
| `hapatra` | 002 | 1/1 | 4 | named | `pass` | yes |
| `sisay` | 003 | 1/3 | 4 | named | `pass` | yes |
| `heliod` | 004 | 6/6 | 4 | named | `pass` | yes |
| `ur-dragon` | 005 | 6/6 | 4 | named | `pass` | yes |
| `edgar-vampires` | 006 | 7/9 | 4 | named | `pass` | yes |
| `gishath` | 007 | 5/5 | 4 | named | `pass` | yes |
| `yawgmoth-swarm` | 008 | 11/14 | 4 | named | `pass` | yes |
| `radagast` | 009 | 7/7 | 3 | named | `pass` | yes |

*Stacks* is checker-passed / total; a failed artifact is kept as an open question and never
publishes. **The fleet is at parity**: every deck carries a bracket target, a named
constellation, a critic-passed engine model, an Editor's Letter, a Pilot's Log opening on a
hot take, and the same seventeen departments in the same order.

## What shipped in the 2026-08-13 cycle

Recorded here because a capability nobody knows about does not propagate. All of it is in
`deck_status.STAGES` and the gates, so new decks inherit it without anyone remembering.

**The constellation.** `deck-map` re-lays-out one deck's cards from the ability embeddings
and cuts two levels of cities and neighbourhoods; `deck-cartographer` names them for the job
their cards do. Radagast's read THE WIDE BOARD, MANA ON LEGS, THE TRAPS, THE BODYGUARDS,
GREEN ON CURVE, THE SEARCH PARTY, A CARD PER BODY. The 99 groups by city and shares the
map's ink by construction.

**The engine model.** The cartographer's own notes exposed the limit that motivated it: a
card is clustered by what it SAYS and an engine is what cards DO TO EACH OTHER, and on
radagast only **4 of 10** declared components sit in a single city. `engine-facts` joins
everything that already answers part of the question; `deck-engineer` ⇄ `engine-critic`
reasons over it into `engine.json` — eight closed stages, drawn as a flow whose arrows are
solid when a checker-passed stack proves them and dashed when they are a reading.

**Two magazine lints and two components.** Internal taxonomy ids in reader copy (68
occurrences across all eight published issues, now 0) and deks that open by asking the
reader a question (14, now 0). `coach-gauge` renders a ★ judgment as stars rather than a
percentage; `stat-slab` runs the issue's signature number once, full width.

**Ambient motion in the atlas**, in the projection rather than the data, with an exact
inverse so hit-testing keeps working.

**`deck-status` + `/publish-deck`** — the lifecycle, checked and sequenced.

## Open work

### 0. THE WORKBENCH PIVOT — in progress, read this first

Decided 2026-08-19. The magazine is no longer the product; a **lab bench for one
pilot's paper decks** is — versions, a captain's log, stats, goldfish, and later a
one-opponent T5 simulation. The manual survives as a compact technical page rendered
from the same artifacts. `docs/agent-audit-2026-08-19.md` is the audit of all 18
agents against that brief, with four fates and a Sprint 0 of eight items.

**Step 1 is done**: `.claude/agents-common.md` holds the contract that was pasted into
twelve charters (~1,000 lines), hashed with every agent in `agent_prompt_sha256`; L10
is repealed in every charter (history is a workbench *input* now); the step-6
cosmetics (engineer/critic "columnists", "Judge's Desk A-004", the coach's stale
"Fetch Quests", pipeline-runner's "13 steps") rode along so the fleet MISSes once.
`validate_issue`'s own L10 lint stays until the manual is simplified — it is
magazine-only and goes with the renderer.

**The cache board is red fleet-wide and has NOT been re-recorded.** A charter edit
disqualifies STALE_OK by construction, and the rule is not bent here. The evidence
artifacts are still gated by their own validators and tests (`deck-status` reads 19/19
on radagast); the MISS only means the *next* spawn of each routine is a real spawn.
Magazine routines clear themselves in step 2 (retirement). Decision pending on whether
the evidence routines (stacks, engine, diagnosis, frame) are re-spawned, re-blessed
with a written reason as the 2026-08 embedding rebuild was, or left red until their
next real run — see the commit message and the audit.

**Step 2 is done**: `magazine-editor` and `pilot-panel` are deleted (charters, the
`issue-plan` and `panel-prose` routines, the `design-issue` skill, the `prose:shape`
and `cards:printing` cache tokens only they consumed); `deck-cartographer` survives but
`map-names` is no longer a `deck_status` stage; `panel` and `plan` left `STAGES` with
the keyed-stage mechanism that existed only for `panel`. `build-manual` renders with
department defaults when no plan exists, so a new deck needs no editor; the nine
tracked `issue_plan.json` and the panel keys are **frozen legacy inputs** the current
renderer still reads, gated by `validate-issue`, until the manual is simplified.

**Step 3 is done**: `manual-writer` + `pilot-coach` → one **`pilot-notes`** agent in one
technical voice, owning five keys (`how_it_wins`, `mulligan`, `combo_lines`,
`threat_assessment`, `matchups`) plus `decisions/` and `tutor_guide.json` outright. The
three keys it does not own — `card_roles`, `mana_base`, `upgrades` — are retired and
**frozen on the published decks** (decided 2026-08-19): no routine owns them, `merge-prose`
never touches them, and the renderer still prints them until the manual is simplified.
`coach-prose` + `writer-prose` → one `pilot-notes` routine; the graphs left its inputs
(only the retired keys read them) and `deck:engine.json?` joined (the notes argue in
stage labels). `validate_issue`'s voice lint is unchanged and the new charter carries its
bans, so new prose stays satisfiable under the legacy gate.

**Step 4 is done — the captain's log, the MVP's heart.** `manamap pilot deck-notes
<slug> add "…" [--result] [--opponents] [--tag]` appends to an AUTHORED `log.jsonl`
stamped with the decklist sha as it stood; `list`/`show` read it back. The `debrief`
agent (charter + `/debrief` skill + `debrief` cache routine, N/A until something is
logged) writes `log_annotations.json` by entry id; `merge-debrief` carries earlier
entries and rejects ids the log lacks; `validate-debrief` holds the annotation to the
note and the 99 (verbatim opponent evidence, cards in deck or note, lines verified only
by a passing stack, routes in a closed set incl. `diagnose`, stages from `engine.json`).
`deck-status` has a `log` stage reporting `N logged, M debriefed`. Seventeen tests.
Nothing is logged on any deck yet — the first real entry is yours to write.

**Step 5 is done — prescriptions, the researcher analyst.** `manamap pilot prescribe
<slug> "…"` opens one question to the doctor under `prescriptions/<id>-….json`
(accumulating, never overwritten; the id is the prompt's hash). `deck-doctor` gained
MODE prescribe — the diagnosis contract scoped to the question, `add_candidates` ranked
and capped at ten (The Short List's rule, relocated), reading the captain's log — and
`deck-skeptic` reviews it the same way. `validate-prescription` reuses the diagnosis
validator's functions; stale prescriptions are form-checked only. Cache routine
`prescription:<id>` with `prompt:self`; `cache-record` refuses without a passing skeptic.
`short-list-analyst`, `/short-list` and `the-ten` are retired; the nine `considering.json`
are frozen legacy. Both doctor modes read `log_annotations.json`. Twelve tests + a
per-deck gate. Follow-up worth doing: `deck-history pending` should read open
prescriptions' `add_candidates` as a source beside the legacy `considering.json`.

**Step 7 is done — the `game_state` v2 schema, docs only** (`docs/pilot.md` → *Game
state v2*): one `seat` object for you and every opponent (archetype, commander zone +
casts, life, hand as a list or `{unknown: n}`, library count, mana available/open/pool,
a board of strings-or-objects), `turn`/`active_seat`/`phase`/`step`/`priority` in the
CR's own words, and an `actions[]` list (cast, activate, play_land, attack, block, pass,
special) that is the difference between resolving a stack and resolving a turn.
Additive (`version: 2` per artifact; every v1 string still accepted), no rules engine,
no probabilities, no migration of the 49 passing stacks, and **no consumer until the
simulation branch** — the first v2 artifact is authored by hand for a real question.
`debrief`'s `opponents[]` and `prescribe`'s pod use the same seat words.

**MVP Sprint 1 is done — deck versioning** (`deck-version <slug> list|show|tag|restore`,
`pilot/deck_versions.py`): versions derived from git by reusing `deck-history`'s walk (a
content change, not a commit — a comment edit adds a sha to its version), the captain's
log joined to them by the stamped sha (games + W/L per list; an uncommitted working copy
reports unmatched, never guessed), authored tags in `deck_versions.json`, `show` diffs
against the working list, `restore` dry-runs unless `--write`, and `deck-status` prints
the current version in its header. **The version list is deliberately not a tracked
file** — a commit's sha is unknown inside the commit; the viz history viewer will get its
copy at deploy time. Six tests on a throwaway git repo. The MVP loop is closed: log ↔
versions ↔ prescriptions.

Next: the audit's item 8 (`card-search` CLI, then maybe `deck-analyst` MODE query), MVP
Sprint 3 (the unified workbench view — `deck-info`), then the feature branches
(simplified manual; simulation).

### 1. Phase 3 — DONE. Radagast carries the whole v4 shape.

All four parts shipped. The magazine now opens on **The Editor's Letter** (Margot Stet,
the masthead's only unbadged name) and **The Pilot's Log** (three columnists arguing in
the engine model's stage names), then the engine flow, then the constellation and a 99
grouped by its cities.

| | |
|---|---|
| Engine loop closed | critic `pass`, `deck-status radagast` **19/19** |
| Fourth persona | Margot Stet — no tier, no glyph, and `badge()` still raises |
| Two departments | all nine carry them; seventeen departments each, `--strict` clean |
| Voice separation | fleet-wide **0**; the blind-attribution test passes |

**What it cost to learn, in one line each** — the detail is in CLAUDE.md:

- A field is only as revisable as it is short. One stage failed four consecutive
  revisions at 2,554 characters; the cap is 1,800 and the artifact failed it on arrival.
- Run a proposed check against the whole fleet before keeping it. The voice lint shipped
  matching `"very "` inside `"every "` — 13 false hits — and then lost three more bans to
  the hedge/intensifier problem.
- A critic's findings become mechanical checks or its work is re-spent every run.
- The dashed line reaches the prose: the panel may discuss an unverified line, never
  assert it.

### 2. The v4 shape is on all nine decks — DONE

Every deck now carries the identical seventeen departments in canonical order, and
`validate-issue <slug> --strict` exits 0 on all nine. Per deck: `analyze-engine`
(engineer ⇄ critic) → `pilot-panel` → `merge-prose` → a `magazine-editor` re-plan that
opts in the two front-of-book departments, merges Act III and turns on the constellation,
the schematic and the not-modelled rail → `build-manual`.

Every engine passed its critic. Round 1 failed on all nine, which is the loop working:
the critic caught hapatra's central exchange rate stated the wrong way round, sisay citing
the meld rule for an MDFC, edgar mis-describing Mondrak, and the refutation defect
reappearing on radagast. Several engineers rebutted correctly rather than weakening —
edgar's ten-becomes-nine DFC trap, gishath's Courtyard defence, heliod's fodder argument —
which is what the charter asks for and the reason the rebuttal instruction is in it.

**Nine hot takes, nine different openings.** Eight decks wrote one and five opened *"Here
is the thing…"* — a formula I introduced by shipping radagast first. The rule is in the
`pilot-panel` charter and
`tests/test_pilot_voice_lint.py::test_no_two_decks_open_their_hot_take_the_same_way`
enforces it across the fleet, because a formula is invisible in one issue and obvious by
the second.

### 3. The Coach-department merge — DONE, and the ids are DELETED

`politics-table` + `know-your-enemy` + `fetch-quests` → one **At the Table**, on all nine.
The three originals are gone from `DEPARTMENTS`, `ACTS`, `INTENSITY`, `MODE`, `ACCENT`,
`PROSE_KEY_DEPARTMENT`, the renderer and the dispatch, and **`OPTIONAL_DEPARTMENTS` is
empty** for the first time since it was added. `test_the_optional_set_is_empty_until_
something_is_being_piloted` now fails while it has members, which is the reminder to
finish a migration rather than let a transitional id outlive it.

Two things the emptying broke quietly, both now covered: `PROSE_KEY_DEPARTMENT` still
pointed `threat_assessment` and `matchups` at deleted ids, which makes `voices_for` return
nothing and the voice lint **pass by finding nothing**; and three renderer tests that
iterated `OPTIONAL_DEPARTMENTS` became vacuous, so they inject a synthetic member instead.

**The merge did not shorten the issue, and that measurement still stands.** What it fixed
is editorial: the reader met the same byline, colour and furniture three times before the
argument had moved once.

### 3a. Length: the budget bought back the additions, and no more

Measured with `manamap pilot issue-length <slug> --rendered`, against a 40-screen target:

| deck | screens | words | visible |
|---|---:|---:|---:|
| hapatra | 63.2 | 26,855 | 23,375 |
| radagast | 71.3 | 39,291 | 30,351 |
| **yawgmoth-swarm** | **96.5** | **104,068** | **70,445** |

Radagast went **74.5 → 71.3** screens while *gaining* two departments (6.8 screens), the
constellation and the schematic. The Kill fell 17.6 → 14.6 and Judge's Desk collapsed to
1.1 — so the cuts were worth about ten screens and the additions spent seven of them.

**The target is not met and the remaining overage is content, not packaging.** On radagast
four departments are 57% of the scroll: the-kill 14.6, at-the-table 9.7, the-99 9.4,
whats-your-play 6.7. Each is the thing itself — seven stack theatres, three arguments plus
five threat boxes, seventy card tiles, ten branch cards.

**Yawgmoth-swarm was the outlier and it is FIXED.** Its Kill was **44,119 words in one
department**, 42% of the issue, because it has **eleven** checker-passed stacks against
radagast's seven and its loops run 11–14 steps each (153 total steps against 51) — and The
Kill staged every one. `the-kill.features` now names which lines get a theatre; the rest
print under *Also on the record*. Yawgmoth features four (002, 004, 012, 011) chosen from
its own artifacts rather than by taste: 002 is `engine.json`'s most-cited evidence at three
`verified_by` references and answers the word the goldfish declaration used (NOT unbounded),
004 is the sharpest refutation in the fleet and §7.6 celebrates those, 012 is the only
resolution that ends "not a draw — you win", and 011 carries two engine lines plus the
mana-ability finding.

| | before | after |
|---|---:|---:|
| The Kill, words | 44,119 | **19,104** |
| The Kill, share of scroll | — | **20.4%** (radagast's share on seven stacks) |
| Issue total | 96.5 screens | **88.4** |

**The first cut was wrong and measuring it said so.** It printed a bare pointer row and
dropped the authored `combo_lines` intro — but a rendered stack is ~4,000 words and its
intro is **77–144**, so the intro costs nothing and the theatre is the whole expense. An
index that dropped the argument would keep the department's title and cut the thing it
names. Rows keep the intro and the result; only the staging is rationed.

**Word count is a bad proxy for scroll in this department.** −25,015 words bought −8.1
screens, because the theatre stacks its plates in Z: word-heavy, pixel-light. `issue-length`
already reports two numbers that disagree on purpose, and this is a third way they can.

Left open, and pre-existing rather than introduced here: `final_state.summary` runs 122–762
words on yawgmoth and renders ONLY in The Kill — Judge's Desk's `render_after_block` prints
life and battlefield rows, not the summary. So it cannot simply be dropped from a row, and
it visibly restates the intro's holding at length. Moving it into the collapsed case file
would cost no scroll and fix the duplication; it changes all nine issues, so it is a
decision rather than a tidy-up.

### 3b. `issue-plan` is MISS fleet-wide, and it is bookkeeping rather than work

Adding `the-kill.features` changed the plan schema, so `magazine-editor.md` and STYLEv3
both moved and every `issue-plan` entry went MISS. That is the cache telling the truth —
those eight plans were written by editors that had never heard of the key. It is not eight
respawns of work: **every other deck has seven or fewer presentable stacks** (edgar 7,
radagast 7, heliod 6, ur-dragon 6, goblin-storm 5, gishath 5, hapatra 1, sisay 1), and
STYLEv3 now says one theatre per line is right up to about seven. Omitting the key is the
correct plan for all eight. Re-spawn one only when its stack count grows past seven, or when
it is being re-planned for another reason anyway.

### 3c. Two deliberate reds, and one charter left stale on purpose

Beyond §3b's fleet-wide `issue-plan` MISS, radagast's `panel-prose` is red and **will not
be recorded**: the `pilot-panel` charter gained the no-formula rule *after* radagast's panel
was recorded, and a charter edit disqualifies a re-bless by construction. Everything else
is green.

*(Resolved 2026-08-19: the stale "Fetch Quests" name in the coach's charter was fixed in
step 1, when every routine MISSed anyway, and the charter itself was folded into
`pilot-notes` in step 3. The deferral argument above is kept in git, not here.)*

### 3e. `main` is branch-protected, and the launch is one merge away

**Correction to something this document and I both got wrong.** `gh repo view`
does not report branch protection, so an audit read "no branch protection" and the
plan filed it under out-of-scope as "meaningless with one maintainer". It is on,
and it is strict: **1 approving review, code-owner review, require-last-push
approval, and required signed commits.** `enforce_admins` is off, so an admin can
bypass in the UI; nothing else can.

That is a good setting to have discovered rather than a problem — but it means the
solo "commit straight to main" workflow recorded elsewhere no longer applies, and
CONTRIBUTING should say so.

**State at the moment of writing:** `origin/main` carries the licence, NOTICE,
CONTRIBUTING, the Makefile, CI and `docs/README.md`. It does **not** carry the two
commits that fix the red CI run, make GitHub detect the licence as MIT, chain the
nine volumes, gate four unwatched validators and close the dead case pointers.
Both sit in **PR #2**, which contains PR #1's commit as well, so the launch is:

1. merge **PR #2** (green); close PR #1 as superseded
2. flip visibility to public

Flipping before that publishes a default branch whose latest CI run is red, whose
licence reads "Other", and whose issues end on "TO BE ANNOUNCED" six times.
Deliberately not done for that reason, on a repo whose whole pitch is rigour.

### 3d. Found while shipping — what a survey of §4–§12 turned up

A read of the whole open-work list against the tree, 2026-08-15. The corrections are
worth more than the confirmations:

- **`next_issue` was six decks, not five, and the chain was wrong on a seventh.** Fixed:
  every volume now names its successor, derived from the artifacts, 009 wrapping to 001.
  Vol 003 had pointed at Edgar (Vol 006) since before Heliod existed.
- **Two tracked artifacts were failing their own validator**, and both are fixed —
  including the two-validator conflict above, which is the more interesting one.
- **"Twelve stack artifacts" with L10 preambles is three different numbers**: 14 by
  preamble, 16 by any reader-facing field, and **4 in fields the renderer actually
  prints**. `scenario.extras` is read by nobody but `scenario_facts`. The four that reach
  a reader are the set worth doing (gishath 002, ur-dragon 003, radagast 007,
  yawgmoth 010); the other twelve are invisible and can wait.
- **§5's cross-reference problem is more than doubled and reaches print.** edgar's
  presentable stacks point at withheld 001 in 11 places and **yawgmoth's five stacks point
  at withheld 001 in eight** — 21 and 9 occurrences in the published HTML. Sharpest:
  **two of yawgmoth's four staged theatres send the reader to a stack the issue does not
  contain.** `validate_issue` checks that `features` names presentable stacks and nothing
  checks that a presentable stack's PROSE does not.
- **§8's "four strategy-DB gaps" is 49**, across all nine frames. Aristocrats/sacrifice is
  requested by four decks and yawgmoth calls it "this deck's entire architecture" — the
  highest-value single write-up in the corpus.
- **radagast has seven `open_questions`, not six.** **49 presentable stacks, not 54** (that
  figure was in CLAUDE.md and a renderer docstring; both corrected).
- **`deck-status` reports presence, staleness AND validity** (validity added 2026-08-18). It read all nine green
  while two of them were failing their own validators. Worth stating wherever it is sold
  as the first thing to run.
- Untested modules are six, not three: `validate_deck_map`, `query_strategy` and
  `download_rules` join §9's list. No dead functions anywhere in `pilot/`.

### 4. Verification backlog

- **Sisay 001** (the tutor chain) is the highest-value fix in the fleet: it would promote the
  ladder arithmetic and the summoning-sickness answer from ★ to ✓ across three sections at
  once. 003 needs a fresh run, not a patch.
- **Grafdigger's Cage** — three of yawgmoth's four kills rest on an oracle reading no checker
  has settled.
- **Hapatra's eleven unresolved combo lines**, headed by whether Mikaeus's `+1/+1` anthem
  switches off the deck's own token loops. If it does, its two flagship engines are mutually
  exclusive.
- **Radagast's `open_questions`** — six, emitted by the engineer with a `settled_by` each.
  Whether the other three finishers also carry the kill is the load-bearing one; only
  Craterhoof has a stack.
- **Queued**: Roaming Throne × Zada, the Past in Flames ritual rebuild, sisay's other
  Najeela pairs.

### 5. Stack artifact staleness

Twelve stack artifacts carry preambles that violate L10 and name retired concepts, and
edgar's presentable stacks cross-reference a non-presentable one. **Do not regex it** — a dry
run destroyed substance in four files. Needs a per-file read, and it invalidates
`stacks:passing` fleet-wide, so do it in one pass and re-record. ~200k.

### 6. `deck-recon` on six decks

Never run there. Its staleness is **time, not inputs** — a decklist edit does not change what
strong lists for a commander run — so it needs dated web passes judged against
`RECON_MAX_AGE_DAYS` (120). ~600k.

### 7. Known-wrong artifacts

- **hapatra's `bracket_report.json` contradicts a verified stack in print**: an inflated
  two-card-infinite count naming a refuted pairing as its driver. The floor of 4 holds on the
  11 Game Changers alone. Either annotate the artifact or teach the engine to consult passing
  stacks.
- **`build_plan.json` is not reproducible from today's data** — re-running `build-deck` on
  hapatra yields a different 99, because the embeddings, roles and synergy graph it scored
  against have been regenerated. Nothing tests this.
- **Five decks carry `next_issue: TO BE ANNOUNCED`.**
- **`manabase.pip_requirements` cannot see a double-faced card's pips** — it reads
  `card["mana_cost"]`, which Scryfall leaves empty for transform/MDFC layouts, and
  never falls back to `card_faces[0]["mana_cost"]`. This is the same trap that once
  made `cards.json` colours read empty for every DFC; that was fixed for COLOURS
  and not for PIPS. Measured across the fleet: **10 spells on 7 decks**, and the
  totals move edgar 76→79, yawgmoth 76→80, sisay 81→84, heliod 77→79, gishath
  96→97, hapatra 56→57, radagast 82→83. Found by `engine-critic` on edgar, where
  the engine model quoting `mana_analysis`'s "39 of 76 pips" disagreed with
  `cards.json`'s 41 of 79. The fix is three lines; the cost is that it changes the
  maths, so all nine `mana_analysis.json` must be regenerated (a tracked artifact
  with a staleness test) and any prose quoting a pip figure re-checked. Deliberately
  NOT done inside the fleet-parity pass — changing it mid-flight would have
  invalidated figures that a fleet of agents was verifying at the time.

### 8. Strategy-DB gaps

Four, all flagged by strategic frames and all load-bearing there: **auditing a combo list**;
**aristocrats/sacrifice engines** (outlet, fodder, converter — now partly answered by the
engine model's stage vocabulary, which is the better home for it); **counters-matter**; and
**tutor sequencing**, doubly wanted since Fetch Quests is a whole section with no pillar.

### 9. Unit tests for the new subsystems · **partly paid**

**Done (sprint step 0):** `deck_map.json` and `engine.json` are now gated by
`test_pilot_tracked_artifacts_validate.py` (45 → 55 cases), and both gates were proven to
fire by sabotaging a real artifact and watching them fail. `tests/test_pilot_deck_map.py`
adds determinism, orientation, the unit box, the balance bound, completeness and unique
naming. Suite 1,530 → 1,547.

**One correction came out of writing them, and it matters more than the tests.** The record
credited Ward linkage with shrinking radagast's oversized city. It did not: at the k a
divisor picks, Ward is 38/71 and average linkage 37/71 — *average is better*. Ward only
leads once k grows (32% vs 49% at k=7), so the win belongs to **the balance bound**. Ward's
other apparent advantage — no one-card cities — does not generalise either; on
outlier-bearing data it strands strays exactly as average does. The test file asserts the
bound and documents why it **refuses** to assert the linkage: three attempts each came back
narrower than the last, and a test asserting it would look like a fact about the algorithm
while being a fact about one deck. CLAUDE.md is corrected too.

**A second correction, from closing the engine loop.** `stages[4].what_it_does` failed
**four consecutive revisions** — each fixed the defect it was sent to fix and introduced a
new one. The cause was not the agent: the field had reached 2,554 characters by accreting
narration of its own previous drafts, against 836–1,645 for the six stages that were never
revised at all. **A field is only as revisable as it is short.** `validate-engine` now caps
`what_it_does` at 1,800 characters, the charter says why, and the fifth revision — cut to
1,704 by deleting draft narration rather than findings — passed. The limit is measured, not
fitted: the data failed it when it was added.

The step-0 gate earned itself the same day: `test_pilot_tracked_artifacts_validate.py`
caught the tracked `engine.json` violating the new limit before anything else did.

**Paid since:** the constellation's two rendering rules are gated
(`test_pilot_deck_map.py` — the diffuse-lobe suppression proven against a compact-lobe
control, and the card-labelling completeness contract), the Act III merge has two
(`test_pilot_build_manual.py`, including the pre-merge fallback), and both browser failures
are fixed and green on three consecutive full runs. `test_docs_section_count.py` grew three
inventory guards — step numbers against `pipeline.STEPS`, the pilot command list against
`PILOT_STEPS`, and tracked per-deck files against the docs. That last one immediately found
`data/decks/radagast/None`, 25 KB of superseded deck map written by an early run that handed
`resolve_out_path` a literal `None`; it had been tracked for weeks because nothing ever
compared the directory against the docs. Deleted.

**Still owed:**

- `merge_deck_map`, `engine_facts` and `deck_status` have no direct unit tests. The
  tracked-artifact gate exercises the first indirectly; the other two do not appear in any
  test.
- No regression floor on the balance bound's *effect* — the test asserts the invariant holds
  on a synthetic fixture, not that real decks stay balanced.

`analyze-engine` has since run on all nine decks, so this debt was paid in arrears rather than in advance — which is the wrong order and is why it is still listed. The gap it left is real: `engine_facts` and `deck_status` produced nine decks' worth of output with no unit test under them.

### 9b. Going public, and the inner loop — DONE 2026-08-15

The repo has a licence, a setup command and CI, and `pytest` takes 22 s instead
of ten minutes.

**Speed.** The single biggest cost in the fast suite was a wrong tree walk: two
doc guards globbed the whole repository per question — 179 times in one — over
38,669 files of which 37,653 are in `.venv`, then discarded the `.venv` hits.
`tests/repo_tree.py` does one pruned walk and both files went **12.4 s → 1.9 s**.
A bare `pytest` is now `-m 'not browser' -n auto`, and four regenerate-and-compare
files are served from a content-hash cache when nothing they read has moved.

| | |
|---|---:|
| serial, before | 86.5 s |
| `make test` cold / warm | 36 s / **22 s** |
| `make test-fresh` | 29 s |
| **fresh clone, cold** | **20 s** |

**The cache's five safety properties were proven, not argued** — key covers the
code's source, records only on pass, corruption invalidates rather than hides,
gitignored so CI runs everything, hits printed. The proofs are in the commit and
in CLAUDE.md.

**Two defects found only by doing the thing.** `make manuals` caught
`manuals/index.html` referencing a stylesheet hash two generations stale, because
`test_pilot_manual_freshness` covered the nine issues and not the newsstand that
links them. And cloning into an empty directory and running `make setup && make
test` — which nobody had ever done — turned up **23 failures**, all correct tests
with missing gates on `cards.csv` and the strategy DB. Neither could fail here.

**Still open, deliberately.** 49 fixed `wait_for_timeout` calls remain in the
browser suite, 53.6 s of them; four were converted and the rest are left because
a bad condition is worse than a sleep and the wall-clock payoff under `-n 4` is
about 3 s. `docs/testing.md` records the reasoning. `.claude/agents/pilot-coach.md`
still keeps one stale department name on purpose (see §3c).

### 10. Codebase hygiene

Still open and genuinely worth doing:

- **Browser-suite runtime**: ~73s of unconditional `wait_for_timeout`, and most tests
  re-parse the projection because pages are function-scoped. Replace the sleeps with
  condition waits first; only consider page reuse if that misses the budget, since cross-test
  state is this repo's known enemy.
- ~~`test_the_spotlight_actually_dims_the_canvas` fails on `main`~~ — **FIXED 2026-08-14,
  and the feature was never broken.** The test counted green over the whole canvas, but
  clearing a spotlight rests the line you were looking at *and* un-mutes the other eleven
  verified lines; on goblin-storm those cancel (868 px spotlit vs 833 cleared, ratio 0.96)
  while the line's own box goes 1024 → 301. It reads `Force.spotlitRows` and measures near
  the line now, with a threshold measured on both sides. The sibling flake,
  `test_canvas_draws_density_contours` (~1 run in 3), was a `canvas_page` race: `setCamera`
  no-ops while `baseFit` is null, so the test measured the fitted view believing it had
  zoomed. Fixed in the fixture — every test on it had the same hole. Full browser suite
  green on three consecutive runs.
- **Leave `.git` alone** — history rewriting breaks every clone to reclaim ~76 MB.

Closed deliberately: **`config.py` is NOT split** (most-imported file in the repo; the
frozen/mutable boundary is a rule, not a filing system), **int8 embeddings** measured at
97.3% top-10 agreement with no first-paint win to weigh against it — a user-facing quality
trade belongs to a person — and **five flagged "duplications" were false**.

### 11. Deck versioning — the remaining third

`HISTORY.md` and a validated `decklist_sha256` exist. An optional **`status`** on
`issue.json` now exists too (`issue_spec.ISSUE_STATUSES`: `broken-down` / `superseded` /
`retired`) — it banners the issue and mutes its newsstand card without editing or deleting
it, first used on hapatra Vol. 002. Still open: **`supersedes`**, the *pointer* half — a
`superseded` issue cannot yet say WHICH volume corrects it, and `build_index` still has
nothing to key on besides the slug, so it emits at most one entry per deck and a second
issue for one deck would silently overwrite `manuals/<slug>.html`.

### 12. Frontend engine port — not started

`manabase` is trivial, `bracket` and `goldfish` are easy, `build_deck` is hardest because
pandas is load-bearing in pool filtering. `viz/js/engine/constants.js` must be **generated
from `config.py`**, never hand-edited, with a parity test. Goldfish determinism needs an
MT19937 port; the honest fallback is labelling a browser-computed goldfish an *estimate* that
never overwrites a ◆ artifact.

### 13. Ur-Dragon two-engine rebuild — proposed, not applied

A 20-swap rebuild into Dragons + Treasure was measured on 2026-08-17 and **nothing was
applied**: `data/decks/ur-dragon/` is untouched and the build lives only in a scratch
`MANAMAP_DATA_DIR`. The list and its figures are in the memory store
(`ur-dragon-deck.md`), deliberately not duplicated here — swap history is derived, and a
second hand-kept copy would disagree with it.

What matters for planning is the **debt applying it creates**. It is a full issue
regeneration, not a swap: all 20 cuts are named across seven tracked artifacts,
`decisions/002` is about Tiamat end to end, three of `engine.json`'s `lines[]` break
(`wincon → conversion`, `protection → wincon`, `ignition → fuel`), and stack **005's
checker verified Hellkite Courser as "maindeck"** — a card fact inside a ✓ artifact that
the rebuild falsifies. The rules holding survives (Sneak Attack still cannot reach the
command zone) and §5.1 forbids editing a passing artifact's text post-hoc, so the fix is
regeneration, not a patch. Until then Vol. 005 describes a deck the pilot may no longer
be holding.

Also open and cheap: the deck's only verified kills are a burst (stack 004) and a bounded
loop (006), while the new combat clock measures an ordinary damage race — **neither**. A
turn-seven race scenario against three seats with a flier or reach blocker would close the
gap between "faster board" and "wins more", which is currently a correlate with no
verified mechanism.

## Decisions that bind

### The frontend stays LLM-free

The deployed static site and the local checkout run the same code. The viz is exploration
plus artifact reads; the agent loop stays in Claude Code, reached by an exported brief. No
local bridge, because a bridge means the deployed site and your machine run different code
and only one of them is the one you test.

The costing behind it: the deterministic layer is 47 subcommands answering in 0.2–2.2s with
JSON out and zero LLM calls. What genuinely needs an agent is artifact-shaped and expensive —
the cheapest routine was `coach-prose` at ~54.5k tokens (now folded into `pilot-notes`), `candidate-pool` is ~235k.

### Similarity comes from the function space, always

`embeddings.npy` is the **layout** space (colour and type) and feeds `projection_2d.json`
only. `embeddings_ability.npy` is the **function** space and is the sole source of similarity
— Find Similar, the walk, drill and the deck map all read it. Held-out: 27.87 effective
dimensions, recall@10 0.245, median rank 78. Similarity is exactly
`0.7·cos_learned + 0.3·cos_text` by fixed weight, so the model cannot discard the text.

**Do not tune on the golden set.** At ~50 dev / ~160 test queries everything in
W ∈ [0.15, 0.6] is inside noise and the splits disagree. W=0.3 was chosen a priori.

Still open: neighbour spread 0.0315 against a 0.05 target, held as a failing
`xfail(strict=True)` rather than lowering the threshold to match the result.

### Synergy is complementary, not similar

24 rules over mechanical tags (blink → ETB), ranked by **playability**, not embedding
similarity. Coverage is uneven and the UI says so: similar 100%, synergy 76.1%, obsolescence
22.5%. **"Anti-cards" do not exist** — across 4.5M pairs the minimum cosine is +0.344, median
+0.714, so "orthogonal" is not a place.

### The clusters are an input to engine analysis, never the analysis

Measured before it was built and re-measured after: only 4 of 10 of radagast's declared
components sit in a single city. A city name is the wrong address for a component, and a
disagreement between the map and the engine is a *finding* — it means that part of the engine
is not visible in card text.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML — that is what keeps rebuilds byte-identical.
- `issue.json` is authored, never generated.
- Costume never earns the badge: a section cannot claim a tier it was not granted.
- Record the cache **after** validation, never before. Never `cache-record` to make a board
  green, and never record a routine whose critic verdict is `fail`.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Never transcribe the section list or its count into a prompt** — read
  `issue_spec.DEPARTMENTS`.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped.
- **A validator that fires on correct data is worse than none** — measure a proposed check
  against the whole fleet before keeping it. Four were scoped down or deleted this cycle.
- **A critic's findings become mechanical checks**, or its work is re-spent every run.
- **Name what a gate cannot see** rather than papering it with string matching.
