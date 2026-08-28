# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what this is for; `CLAUDE.md` carries the
gotchas; this says what exists and what is open. The magazine era's plan is archived
verbatim in git at `git show 23e8cec:docs/history/PLAN-2026-08-magazine-era.md`.*

Last updated **2026-08-28**. Everything below is committed and pushed to `main` except
where marked. Every figure was derived from the repo at write time — **do not quote one
from memory**; the command that prints it is named beside it.

## What this is

A **workbench for crafting, experimenting, researching and analysing Commander decks**,
built around one idea: a claim about a deck is worth what the experiment behind it is
worth. **Simulation is the centre** — Forge for the real rules against a real pod, a
seeded Monte Carlo goldfish for the questions that are about a curve rather than a table —
and `experiment` is the flagship: two versions, one table, one artifact, the delta and the
overlap sentence. Around it sit a deterministic builder, a rules-citation loop for lines
that must be proven, dated reconnaissance, deterministic card mining, and a frontend that
surfaces the results.

Optimised for one player (the maintainer, in Orinda); open-sourced, not externally
supported. The magazine that used to be the product is a frozen legacy renderer until the
compact deck page replaces it. The card atlas in `viz/` is unchanged and live; the **deck
page** (`viz/deck.html?deck=<slug>`) is new and is the workbench surface.

Scale (derived; `tests/test_docs_counts.py` polices these): 83 `manamap pilot`
subcommands, 21 top-level subcommands, 15 agents, 19 skills, 10 static cache routines
(plus `stack:`/`decision:`/`prescription:` per artifact). Test counts live in
`docs/testing.md` only.

## Start here on any deck

```bash
manamap pilot deck-info <slug>        # where it stands, and a derived NEXT
manamap pilot deck-status <slug>      # lifecycle: present / missing / STALE / INVALID, gates run
```

Staleness is an ERROR; incompleteness is a state. A half-built deck is work in progress.
A deck whose artifacts disagree about which decklist they describe is confident and wrong.

## The evidence contract

| | tier | granted by |
|---|---|---|
| ✓ | rules-verified | a stack artifact whose every step cites a real CR rule verbatim (`validate_stack`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | deterministic Python over committed artifacts; **seeded** where randomness is involved (goldfish, Forge runs), **sampled** said out loud where it was not (the first Forge run). |
| ★ | coaching | labelled judgment. Useful, never disguised as measurement. |

Agents return JSON and validators check it; no agent writes prose that claims a tier it
was not granted; a figure travels with its interval, its N and its limits.

## The decks

`deck-info <slug>` per deck for the live picture; `deck-status --all` for the fleet. As of
this writing (derived from the stack, bracket, engine, sim and log artifacts):

| deck | stacks ✓/total | bracket floor/target | engine | sim runs | logged | status |
|---|---|---|---|---|---|---|
| `goblin-storm` | 5/5 | 4/4 | pass | 0 | 0 | |
| `hapatra` | 1/1 | 4/4 | pass | 0 | 0 | `broken-down` (cards live in yawgmoth) |
| `sisay` | 1/3 | 4/4 | pass | 0 | 0 | `retired` — **not the pilot's deck** |
| `heliod` | 6/6 | 4/4 | pass | 0 | 0 | white short by 16, not 5 — see the DFC fix |
| `ur-dragon` | 6/6 | 4/4 | pass | 0 | 0 | two-engine rebuild proposed, not applied |
| `edgar-vampires` | 11/11 (9 presentable) | 4/4 | pass | 1 | 1 | **v1.0.0 baselined, sleeved; both loops pass; 1 game logged** |
| `gishath` | 5/5 | 4/4 | pass | 0 | 0 | |
| `yawgmoth-swarm` | 14/14 (11 presentable) | 4/4 | pass | 0 | 0 | paper rebuild in progress |
| `radagast` | 8/8 | 1/3 | pass | **2** | 0 | `broken-down` (2026-08-21) |
| `kianne` | 0/0 | 4/4 | — | **2** | 0 | **concept abandoned** — voltron is 10 decks of 970; the shell is sound, the win condition is not |
| `kinnan` | 0/0 | 4/4 | — | 0 | 0 | deterministic baseline built; recon done; commander not owned |

**Eight Forge runs and six experiments exist**, across four decks — edgar (3 runs, 2
experiments), kianne (2, 2), radagast (2, 2), ur-dragon (1). The other seven have not been
simulated at all.

**Two decks have a real table logged**: edgar and ur-dragon, one game each, both losses,
both debriefed, each feeding a prescription. That is enough to have proved the loop works
end to end and nowhere near enough to conclude anything about either deck — and the other
nine still have zero. The next entry is the pilot's to write.

**Five decks are not marked as built in paper** (heliod, yawgmoth-swarm, gishath, kianne,
kinnan). That is a third state, distinct from the three that no longer exist as cardboard:
nobody has said either way, and `deck-info` now says so instead of assuming.

Three decks no longer exist as cardboard (`hapatra`, `sisay`, `radagast`). Their
artifacts stay exactly as published; `deck-info` states the status and withholds the
suggestions that would need a deck to shuffle.

## Where it stands (2026-08-25)

| | | where |
|---|---|---|
| Agent audit + Sprint 0 | 18 → 15 agents; shared contract (`.claude/agents-common.md`); L10 repealed; magazine editor/panel/short-list retired; writer + coach → `pilot-notes`; `debrief` new; doctor MODE prescribe | `docs/agent-audit-2026-08-19.md` |
| MVP Sprints 1–3 | `deck-version`, `deck-notes` + `/debrief`, `prescribe`, `deck-info` | `docs/pilot.md` |
| Simulation S0–S5 | Forge spike + verdict; seeded harness; parser with CIs; `validate-sim`; the pod; the bridge `sim-scenario` → `game_state` v2; the doctor reads the table | `docs/simulation.md` |
| The chain, once for real | stack 008 — a board lifted from a simulated game, resolved + checker-passed in 3 iterations; matched Forge's log line for line | `docs/simulation.md` |
| `experiment` + AI profiles | the controlled A/B, one accumulating artifact; `--profile` on both commands (Default stays default, measured) | `docs/simulation.md` |
| Commander damage | per DEFENDER through the parser, the record and `experiment`'s delta; all runs migrated | `sim/parse.py` |
| The distribution | `mean_ci` carries median/min/max beside the mean — a skewed arm read mean 17.42, median 0 | `sim/parse.py` |
| `card-search` | deterministic corpus mining: identity, oracle/name regex, role, cmc, `--owned` | `docs/pilot.md` |
| The collection | `pilot/collection.py`, the one reader of `COLLECTION_DIR`, memoized | `docs/pilot.md` |
| `validate-recon` | the gate `deck_recon.json` never had; its first catch was a data gap, not an agent error | `pilot/validate_recon.py` |
| Builder curve + combos | role quota crossed with a **cited** mana-value target; `complete_combos` finishes a line the deck half-holds | `pilot/build_deck.py` |
| **The deck page** | `viz/deck.html` — nine workbench panels over `info.json`, sim figures with intervals, lifecycle flag | `docs/viz.md` |
| **The Pilot's Manual** | `manuals/p/<slug>.html` from `build-page` — the compact technical page, no `<script>`, rebuilds byte-identically. The magazine is unlinked from every live surface | `docs/manual-v5-spec.md` |
| **The first real games** | edgar v1.0.0 and ur-dragon v1.0.0, one logged game each, both debriefed, both feeding a prescription | `data/decks/*/log.jsonl` |
| **The workbench landing page** | `viz/workbench.html` — racks by whether a deck is SLEEVED, plus a fleet table sorted four ways over every `info.json`. Three labelled links per deck | `docs/viz.md` |
| **The fusion** | an open verified line prints its prose (50 of 50 covered); engine arrows on the 5 of 196 edges whose direction is a fact; a curve bar is a control | `docs/viz.md` |
| **Seed a walk from named cards** | textarea + `?cards=`; the enumeration separates, never the comma (9.2% of names contain one); *Start here* vs *Add to walk* | `docs/viz.md` |
| **The version policy** | PATCH/MINOR/MAJOR by capability, every slug from v1.0.0; releases sort numerically, near-misses refused, re-tagging needs `--force` | `docs/pilot.md` |
| **The paper lock's third state** | UNLOCKED is not dead. Four of eleven decks are unlocked and now say so; three rehearsal locks withdrawn | `docs/pilot.md` |

## Open work

### RULE — a branch is graded on what the deck PRODUCES, never on an authored file

**Set 2026-08-28, by measurement.** `net-change` carried an ENGINE LIFT: kill
rate in the games where every component marked `required` in
`goldfish_targets.json` was assembled by T3, minus the rate where it was not,
with a Newcombe interval on the difference. The statistics were right. The input
was not: **that file is authored, and the same hand writes the declaration and
reads the verdict.** Three defensible declarations of one Ur-Dragon list, same
10,000 games, same seed, against kill-by-T8 — **+0.007 (spans zero), −0.036
(REAL), +0.014 (REAL)**. One of them says at an interval excluding zero that
assembling the engine makes the deck win less.

**Deleted, with the guard in the same commit.** `deck_branch.MEMBERSHIP_AXES`
refuses `engine_online_*` and `any_route_*` as branch objectives and names the
output axes instead; a test walks every tracked `branch.json` so an old one
cannot survive it. `goldfish_targets.json` stays — it drives the `*_assisted`
figures and the target table, which are hypergeometric and real. It just does
not get to grade anything. Full record: `docs/gotchas-bench.md`.

Ur-Dragon's objective moved from `engine_online_5 >= 0.22` (met **4.4x over**
while the lift spanned zero) to **`damage_8 >= 40.0`** — the opponent's starting
life, a number with meaning outside this branch. v1.0.1 reads 30.81 and misses.

### DONE — the merge request: `propose`

**2026-08-28.** A branch is now `OPEN` / `PROPOSED · BLOCKED` / `READY` / `STALE`
/ `OUTRUN` / `MERGED`, all derived and none stored, so a proposal un-blocks
itself as cards land in a box. `deck-branch <slug> propose <name> --as v1.0.2`
freezes the decision (which list, which report, which grade) and hands over a
**pull list** split by what each bucket costs. `validate-branch` gates it —
`branch.json` was the last tracked pilot artifact without one.

Ur-Dragon's `eminence-v3` is proposed as **v1.0.2**, blocked on **6 cards**
(down from 12: four sat in decks that do not physically exist, and five more the
pilot had already agreed to proxy). Full record: `docs/gotchas-bench.md`.

### OPEN — three diagnoses are stale and need the doctor, not a patch

`gishath`, `goblin-storm` and `yawgmoth-swarm` fail
`validate-diagnosis` on `axes[].colour-sources.measured.value`: the `manabase`
correctness fixes moved the audit's figure underneath a diagnosis whose PROSE
names the old number ("Green 26, red 20, white 20 against a 36 target"; "Four
sources above the Karsten yardstick"). goblin-storm's reading even describes the
colour-identity fallback that no longer exists — the number moved *because* the
doctor's complaint was fixed.

**Patching `measured.value` is forbidden** — it would leave prose contradicting
its own figure, which is "a fresh claim under an old byline". These need a
`/diagnose-deck` re-run each, which is an agent spend and the pilot's call.
Until then `make test` is 3 red on exactly these three. **`diagnosis.json`
carries `as_of_decklist_sha256` but no AUDIT stamp**, which is why a code change
leaves it looking current — the same class `meta.model_version` solved for the
goldfish, unsolved here.

### RULE — a retired deck is not a downstream target

**Set by the pilot, 2026-08-27:** *"if a deck is deprecated, broken down,
exclude it from these downstream tasks."*

A `manabase` correctness fix moved the colour-source figure on six decks, and
three of them — `hapatra` and `radagast` (broken-down), `sisay` (retired, and
not the pilot's deck) — got an agent re-run each before the rule was stated.
That is real tokens spent regenerating a document about a deck nobody will play.

**A retired deck's artifacts are HISTORY, NOT CLAIMS.** Nothing derives from
them and nothing plays the list, so holding them to today's model is the "gate
that reddens history" `validate_prescription` already refused to be.
`deck_info.STATE_RETIRED` buckets broken-down, superseded and retired together
and is the one place that decides; `tests/test_pilot_tracked_artifacts_validate`
now skips a deck with a `lifecycle` block, and any fleet fan-out should do the
same.

**The related rule, same day:** every optimisation happens in a BRANCH. Nothing
touches a main `decklist.txt` — measurements of the existing list are not
optimisations, and regenerating them after a model change is required, but the
list itself moves only through `deck-branch merge`.

### ISSUE — `land_colors` credits mana it cannot actually make

**Opened 2026-08-27. Found by the `yawgmoth-swarm` doctor mid-run, which is the
loop working: an agent re-derived a figure and disbelieved it.**

Two stacked defects, both the class the reminder-text fix already closed once —
**text describing ANOTHER object's ability read as this card's own.**

1. **A quoted token ability counts as a mana source.** `Pawn of Ulamog` creates
   an Eldrazi Spawn with `"Sacrifice this token: Add {C}."` —
   `nonland_producer_kind` matches the quoted clause and calls it `ramp:dork`.
   It is also one-shot and self-sacrificing, which is the Jeweled Lotus rule
   `goldfish._CONSUMING_COST` already enforces and `manabase` does not.
2. **The colour then comes from COLOUR IDENTITY, not from the ability.**
   `land_colors` ends `if not produced and not restricted: produced.update(
   color_identity)`. `{C}` is not a coloured symbol, so nothing parses and the
   fallback credits Pawn of Ulamog with **B — because it is a black card.**

**AND THE FALLBACK IS THE BIG ONE — it is systematically wrong for LANDS.**
Measured across the corpus: **60 land entries** get their colours only from that
fallback, and the pattern is always the same — the land taps for `{C}` and its
colour identity comes from an ACTIVATION COST it has nothing to do with
producing. `Goblin Burrows` is `{T}: Add {C}` plus `{1}{R}, {T}: pump a Goblin`,
and reads as a **red source**. So does `Kher Keep`. `Kor Haven` reads white off
`{1}{W}, {T}: prevent damage`; `Blighted Woodland` reads green off a `{3}{G}`
sacrifice. None of them makes a single coloured mana. Two doctors found this
independently on two different decks in the same batch, which is the corroboration
— goblin-storm's true repeatable red land count is **31, not 33**.

Whether the fallback has any legitimate case is the open question: it exists for
a land whose production text does not parse, and nobody has enumerated those. The
fix must not be "delete the fallback" until that set is known, which is the same
discipline that kept the quoted-text class from being a blanket strip.

**Scope, measured:** 8 further cards read mana colours only from text inside
quotation marks. They are NOT uniformly wrong — `Worldknit` and `Paradise
Mantle` grant `{T}: Add one mana of any color` to permanents you control, which
is real fixing. The wrong ones are token abilities. So the fix is not "strip
quotes"; it is to separate *granting an ability to something you control* from
*describing a token you may create*.

**Not fixed on discovery, on purpose.** Six `deck-doctor` runs were in flight
against the current model. Changing `land_colors` mid-batch would have made
every one of them stale on arrival and wasted the spend. A model change and a
regeneration of the artifacts that depend on it belong in one commit — which is
the same rule that made this batch necessary in the first place.

### ISSUE — unit tests must not depend on an experimental deck

**Opened 2026-08-27, by the pilot, and it is a rule rather than a chore.**

> *"we shouldn't be writing tests for decks that are experimental/feature
> branches — we can have tests to assert the functions that support that work,
> but we don't need to test the actual deck via unit tests. Stats, math, etc. is
> what we will apply to those decks. Not until they are merged and pinned do
> they get tests."*

**What happened.** Twenty tests across nine files hardcoded
`ur-dragon/treasure-v2` as a data fixture — `close`'s component search, `assess`'s
triage, the diagnostic's magnitude series, the branch-scoping controls, the
net-change report. The treasure refactor was measured, found worse, and deleted,
which is the tool working exactly as designed. All twenty failed. Deleting a
branch is a first-class pilot action and the suite punished it.

**The rule.** A branch is a candidate 99 that is *supposed* to change and
*supposed* to be thrown away. Only a **merged and pinned** deck is stable enough
to assert against. What a test may assert about branch machinery is the
FUNCTION — that a branched write lands in the branch directory, that reads fall
back and writes never do, that a swap is one-for-one — and those need a
**synthetic branch in a `tmp_path` sandbox**, which
`tests/test_pilot_branch_lifecycle.py::sandbox` already demonstrates.

**What is done and what is not.** `conftest.requires_branch` / `A_BRANCH` now
take *whichever* branch exists rather than one name, so the suite no longer
names a specific experiment — that is a stopgap, not the fix. The fix is to
rebuild these tests on synthetic fixtures and delete the assertions that were
only ever about the treasure deck's contents (the `MULTIPLIER` component, the
treasure cards `assess` was triaging). Until then they are gated and some of
them will skip.

**Related invariant** (below): a tracked artifact needs a gate in the same
commit. That is still true — but the gate belongs on the artifact's SHAPE and
its producing function, never on one experimental deck's numbers.

### Next, in order

1. **Paper check-ins, one deck at a time** — **only the pilot can do this**, and it is
   the highest-value thing available. Five decks (heliod, yawgmoth-swarm, gishath,
   kianne, kinnan) are **not marked as built in paper**, so nothing knows whether they
   exist as cardboard; Dinosaurs and Shrines were never checked in at all. `check-in`
   takes a typed list and refuses rather than guesses, then `deck-version paper` locks
   it and drift is computed on every swap from then on. Two decks (edgar, ur-dragon)
   have one logged game each — which is two, not a sample.
2. **The versions deploy-time step** — a Pages workflow checking out with
   `fetch-depth: 0` and running `deck-version list --json` per deck into `versions.json`.
   The producer already exists and the deck page's panel is already written; it renders
   nothing until the artifact does. Needs the Pages source flipped from "branch" to
   "GitHub Actions" in repo settings, which only the maintainer can do.
3. **An agent in the pilot's seat** for a handful of seeded games on one question — the
   evidence for it is in: no Forge AI profile flies a hold-up deck better than Default,
   so the AI is the thing limiting the measurement rather than the configuration.
4. **The version-bump classifier** — `deck-version bump` measuring a diff and PROPOSING
   major/minor/patch with its evidence, for the pilot to confirm. The policy is written
   (`docs/pilot.md`); the classifier needs three things that do not exist: a diff between
   two *arbitrary* versions (every diff today is consecutive or against the working tree),
   `quantity_changes` carried into `versions()` — `history()` computes it and `versions()`
   drops it, so 36 to 37 Forests reads as no change at all — and a classifier reporting
   **evidence, never intent**, since `deck_history` is explicit that *why* a card moved is
   not knowable from a commit.
5. **Content-addressed cache busting for `deck.html`** — it went from nine artifact
   fetches to fifteen, and `manuals/magazine.css`'s `?v=<sha8>` is the pattern to copy.

**Done since the last revision:** the Pilot's Manual (`build-page`) and the magazine
unlinked from every live surface; edgar and ur-dragon pinned at v1.0.0 with a real game
each, debriefed and prescribed; the workbench landing page and its fleet table;
verified-line prose, engine arrows and clickable group bars in Build; seeding a walk from
named cards and `?cards=`; the semver policy and its three tag guards; the paper lock's
third state and the withdrawal of three rehearsal locks (2026-08-23 → 25). Before that:
`experiment` and AI profiles (2026-08-19); `card-search`,
commander damage per defender, the collection primitive, `validate-recon`, deck lifecycle
in `deck-info` (2026-08-21); the builder's curve quota and combo completion, `mean_ci`'s
distribution, `deck-info --write`, the manifest's instanced files, the deck page, the DFC
pip fix, and `deck-status`'s gate blind spot (2026-08-22).

### Known gaps, named in the artifacts

- **Forge's AI pilots the deck** — "poor to ok in control, pretty bad for combo" (its own
  words, quoted in every run record). radagast 0/20 vs the pod at 12.5 combat damage a
  game against 45.6 among its stablemates. A lower bound on the pilot; a true picture of
  the table's clock.
- **Bridge approximations**: token types unknown from the log (tokens filed as
  `other_permanents` by `scenario-facts`), hand sizes are estimates, continuous effects
  (a Craterhoof pump) are not tracked and must be authored into the scenario. Every one is
  written into `extras.reconstruction_notes`.
- **Parser**: damage figures see damage only; drain kills show in `life_by_turn` and
  `eliminated_how`, never in a damage total. **Commander damage is now measured**
  (2026-08-21) — per DEFENDER, because CR 903.10a asks for 21 from one commander on one
  player and `combat_damage_dealt_to_players` sums every source and every seat at once.
  The commander names ride IN the record (`seats[].commander`), never looked up from
  disk at validate time; a record without the field re-derives exactly as before, and
  `simulate <slug> --analyze <run>` is the migration.
- ~~**The deterministic builder cannot produce a curve SHAPE.**~~ **FIXED 2026-08-22.**
  It scored every card independently and took the top N while `curve_fit` penalised
  each point above `DECK_CURVE_SWEET_SPOT = 3`, so the top N were always cheap: the
  first kinnan baseline was 64 nonland cards with curve `{0:1, 1:11, 2:28, 3:24}`,
  **nothing above mana value 3**, and 29 of them mana producers — a legal deck that
  ramps into nothing, which `validate_build` passed because it checks form. The role
  quota in `fill_slots` is now crossed with a mana-value quota derived from
  `DECK_AXIS_TARGETS["curve"]`, the target `deck_audit` already measured against and
  the builder never read, so no new uncited constant. Rebuilt: `{0:1, 1:9, 2:15,
  3:16, 4:10, 5:6, 6:4, 7:1, 8:1}`. Combo blindness went with it — `complete_combos`
  reads real lines from `combo_details` (not the flat `combo_partners` map, which
  cannot tell a completion from a coincidence) and swaps in the one missing card of a
  line the deck half-holds. kinnan went from 23 partners and 0 completions to **4
  contained combos and 2 two-card infinites**, including Kinnan + Pili-Pala +
  Enduring Vitality. And `build()` now has end-to-end tests: there were none.
- ~~**DFC pips**~~ — **FIXED 2026-08-22**, and it was two defects rather than one.
  `pip_requirements` read `card["mana_cost"]`, which Scryfall leaves EMPTY on
  transform/MDFC layouts (counting zero pips) and which holds BOTH halves on
  adventure/split layouts (counting double). `common.front_field` is the one shared
  reader now, replacing `deck_facts._front`, which had solved this for colours and never
  for pips. It produced a real finding: **heliod's commander is `{2}{W}{W}` and that
  second pip was invisible**, so its white target read 22 when it should read 36 — short
  by 16 against 20 sources, not by 5, with white rather than blue the binding colour.
  Seven `mana_analysis.json` regenerated. Same-class defect still unfixed in
  `build_deck.castability` (`build_deck.py`, `getattr(row, "mana_cost", "")`).
- **hapatra's `bracket_report.json`** contradicts a verified stack: two of its three
  `drivers` cite lines that stack 001 refuted or explicitly declined to rest on, and the
  "19 two-card infinites" figure is inflated by six. The contradiction is duplicated
  verbatim into `build_plan.json`. Blocked on a schema question — the refutation is prose
  inside `resolution.final_state.summary`, and there is no machine-readable `refutes`
  field for a gate to read.
- **`build_plan.json` is not reproducible** from today's data, and is now *further* from
  it: the builder gained a mana-value quota and `complete_combos` on 2026-08-22, so
  re-running produces a different 99 by design. Open question whether historical build
  plans should be reproducible at all — they are records of a build that happened, like
  `log.jsonl`.
- **A diagnosis can go stale without its decklist moving.** The DFC fix changed the audit
  underneath heliod's `diagnosis.json`, which had cited the old figures correctly when
  written. `validate_diagnosis` re-derives every axis, so it failed — right, but there is
  no staleness *class* for "the measurement code moved", the way there is for "an older
  decklist". The only route is a re-spawn.

### Verification backlog (✓ work)

- **Sisay 001** (the tutor chain) — highest-value promotion from ★ to ✓ in the fleet.
- **Grafdigger's Cage** — three of yawgmoth's kills rest on an oracle reading no checker has settled.
- **Hapatra's Mikaeus +1/+1 anthem** vs its token loops — if it switches them off, its two engines are mutually exclusive.
- **Radagast's `open_questions`** — six from the engineer; only Craterhoof has a stack. **Boards for these can now be lifted from sims** (`sim-scenario`).
- Queued: Roaming Throne × Zada; the Past in Flames rebuild; sisay's other Najeela pairs.

### Still owed

Sized 2026-08-22; each is small and independent unless noted.

**QUEUED 2026-08-24, both ready to run, both precisely specified.** They came out of
ur-dragon's terminal rounds and were deliberately NOT acted on there: a change made after
the last adversary has finished is unreviewed by construction.

- **`/resolve-stack ur-dragon 007`** — the scenario is written, preflighted
  (`validate-stack --scenario-only`: OK) and committed as
  `stacks/queued/007-cascade-without-panharmonicon.json` — in `queued/` rather than
  `stacks/`, because `validate-stack`'s glob is non-recursive and the citation-contract
  test requires every tracked stack to carry a PASSING resolution, so a staged scenario in
  `stacks/` turns the suite red. **Move it up one directory to run it.** That tension is
  real and worth noticing: `--scenario-only` exists to preflight in place, and a preflighted
  scenario cannot then be committed where it was preflighted. It is stack **002's exact board minus
  Panharmonicon**, derived from 002 rather than invented, and answers the three things
  `diagnosis.json`'s `open_questions[0]` asks: how many token copies and how much damage
  without the doubler; whether it is still lethal to the 32-life seat 002 kills with 64;
  and — the deck's central question — whether ANY version kills a fresh seat at **40**,
  which neither proven board has been shown to do (002 leaves the 40-life seat alive at 8).
  It prices `cut_candidates[5]`, which currently rests on a mana argument alone with its
  decisive evidence *named and absent*: nothing on the record says whether cutting
  Panharmonicon costs 002's **lethality** or only its **margin**. One rules domain, no
  combat; 002's nontoken finding is inherited and stated in `extras.context`.

- **Scourge of the Throne in THE COMBAT KILL's multiplier leg** — `engine-critic`'s
  terminal round found the two-card leg (Atarka, Thrakkus) arguably omits it: an additional
  combat phase doubles the turn's Dragon combat damage, and `engine.json` itself calls it
  that kill's mechanism, which is an internal contradiction. It differs from the other two
  in being **conditional** (it must attack the player with the most life), which is why it
  is a judgement rather than an obvious omission. Recorded in `goldfish_targets.json`'s
  note. **Run it with a critic attached** — this declaration moved three times on
  2026-08-24 and every move cascaded into `engine.json` and `diagnosis.json`.

- **`diagnosis.json` for ur-dragon carries a terminal `fail`** and is NOT cache-recorded,
  per the rule that a fail is never recorded. Two text-level defects, both discharged,
  neither moving a swap. It clears on the next diagnose pass, which the two items above
  should precede — both change figures the diagnosis quotes.

- **Six of nine `strategic_frame.json` are unstamped** and now say so in `deck-status`
  ("unstamped — staleness cannot be checked"). That is a third state, not a softer STALE:
  they may be current and simply not say so. Each is one `strategy-researcher` MODE consult
  to stamp, and worth doing opportunistically rather than as a sweep — ur-dragon's turned
  out to be asserting proof that had left the deck.

- **`merge_deck_map` / `engine_facts` have ZERO test coverage** — no test file imports
  either. ~300 lines, blocked on nothing. `merge_deck_map`'s whole reason to exist is the
  `OWNED = ("label", "gloss")` whitelist, and nothing asserts it.
- **No regression floor on the balance bound.** Nothing reads the nine tracked
  `deck_map.json`. Note edgar-vampires sits at **35.05%** against a 35% bound, so the
  floor must encode the `MAX_CITIES` escape the synthetic test already uses.
- **`deck-recon` on four LIVE decks** — gishath, goblin-storm, heliod, ur-dragon. It is
  *absence*, not staleness: nothing is stale by `RECON_MAX_AGE_DAYS` (oldest 19 days
  against 120). hapatra and sisay are dead cardboard and should be skipped — a perishable
  meta artifact for a deck nobody can shuffle is ~100k spent on nothing.
- **`deck-history pending` should read prescriptions' adds.** Blocked twice over: "open"
  is the wrong predicate (an open prescription has *no* adds; the wanted state is
  answered-but-unapplied), and **zero prescription files exist fleet-wide**.
- **`supersedes`** — no scaffolding at all; the status half exists (`DECK_STATUSES`) and
  the pointer does not. Blocked on a decision: it lives in the frozen magazine layer.
- **Strategy-DB gaps** — 49 across the frames; aristocrats/sacrifice first.
- **Ur-Dragon two-engine rebuild** — proposed, measured, not applied.
- **`build_deck.castability` reads `getattr(row, "mana_cost", "")`** — the same defect the
  DFC fix just closed in `manabase`, still open one module over.

### Legacy, frozen — and what unfreezes it

The magazine renderer (`build_manual`, `issue_spec`, `design`, `validate_issue`, STYLEv3)
still renders the nine pages from artifacts nothing regenerates (`issue_plan.json`, the
panel keys, `card_roles`/`mana_base`/`upgrades`, `considering.json`). They are edited by
nobody and deleted in one commit when the compact page lands (`docs/manual-v5-spec.md`
§"What gets unfrozen"). Code and docs about it carry a LEGACY banner and are otherwise
left accurate.

## Decisions that bind

### The frontend stays LLM-free
The deployed site and the local checkout run the same code. The agent loop stays in
Claude Code, reached by commands and a brief. No local bridge.

### Forge is the engine; we build the harness, the parser and the bridge
Measured before chosen: (a) every log line parses, (b) 4-seat Commander runs, (c) `-s`
makes it byte-replayable. Writing our own rules engine is shelved for one narrow
deterministic case. The goldfish stays as the seeded resource model.

### The cache board is red fleet-wide, and deliberately not re-recorded
The shared-contract commit MISSed every routine; charter edits disqualify STALE_OK by
construction. Artifacts are gated by validators and tests, not by the cache; each routine
clears on its next real spawn. Never `cache-record` to make a board green.

### Versions are derived from git and never committed; tags are authored
A commit's sha is unknown inside the commit, so a generated version list would be one
behind forever. `deck_versions.json` (tags) is the one version datum a browser can read.

### A prescription accumulates; stale is not wrong
One file per question, keyed by the prompt's hash; an older-decklist prescription is
form-checked only.

### Similarity comes from the function space, always
`embeddings_ability.npy` is the sole source of similarity (Find Similar, the walk, drill,
the deck map). `embeddings.npy` feeds `projection_2d.json` only. Do not tune on the golden set.

### Synergy is complementary, not similar
24 rules over mechanical tags, ranked by playability. "Anti-cards" do not exist.

### The clusters are an input to engine analysis, never the analysis
A city name is the wrong address for a component; a disagreement between the map and the
engine is a finding.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML.
- `issue.json`, `log.jsonl`, `deck_versions.json`, a prescription's prompt: **authored**,
  never regenerated. A derived artifact may be regenerated; an authored one may not be rewritten.
- Costume never earns the badge.
- Record the cache **after** validation, never before; never record a `fail`.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped.
- **A validator that fires on correct data is worse than none** — measure a proposed check
  against the whole fleet before keeping it.
- **A critic's findings become mechanical checks**, or its work is re-spent every run.
- **Name what a gate cannot see** rather than papering it with string matching.
- A sim figure travels with its interval, its N and its limits — or it does not travel.
- **A measure computed from an authored file is not evidence, however tight its
  interval.** A branch objective names an OUTPUT the deck produces.
- **Every figure carries its definition in the report that prints it** — a number a
  reader has to look up gets guessed at, and the guesses go one way.
- **When you delete a producer, grep the validator** for the only place its contract
  was enforced. Removing the engine lift silently took "an unavailable block owes a
  reason" with it.
