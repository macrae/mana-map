# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what this is for; `CLAUDE.md` carries the
gotchas; this says what exists and what is open. The magazine era's plan is archived
verbatim in git at `git show 23e8cec:docs/history/PLAN-2026-08-magazine-era.md`.*

Last updated **2026-09-11**. Everything below is committed and pushed to `main` except
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

Scale (derived; `tests/test_docs_counts.py` polices these): 100 pilot subcommands,
28 top-level subcommands, 17 agents, 21 skills, 10 static cache routines
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

| deck | stacks ✓/total | sim runs | logged | record | branches | status |
|---|---|---|---|---|---|---|
| `edgar-vampires` | 11/11 | 9 | **4** | 0W 4L | 6 | ◆ **SLEEVED V2**; direction changed 08-28 |
| `ur-dragon` | 7/7 | 4 | **6** | 2W 4L | 6 | ◆ **SLEEVED V4** (v1.2.1); `final-v1` PROPOSED as v1.3.0 |
| `heliod` | 1/1 | 4 | **2** | 0W 2L | 2 | ◆ **SLEEVED V10** (v1.2.1); `splendor-v1` PROPOSED as v1.2.2 |
| `gishath` | 5/5 | 3 | **1** | 1W 0L | 2 | ◆ **SLEEVED V4** (v1.0.0); `mana-v1` PROPOSED as v1.1.0 |
| `goblin-storm` | 5/5 | 4 | **2** | 0W 2L | 0 | ◆ **SLEEVED V1**; 0.031 at standard-v3 |
| `sharknado` | 0/0 | 0 | 0 | — | 1 | on the bench; proposals withdrawn 09-11 |
| `ingris-infect` | 0/0 | 0 | 0 | — | 3 | on the bench; the commander is legal 2026-10-02 |
| `zur-enchantress` | 0/0 | 2 | 0 | — | 8 | **`broken-down` 2026-09-10** — a failed experiment, called by the pilot |
| `yawgmoth-swarm` | 14/14 | 1 | 0 | — | 0 | `broken-down` |
| `hapatra` | 1/1 | 1 | 0 | — | 0 | `broken-down` (cards live in yawgmoth) |
| `radagast` | 8/8 | 3 | 0 | — | 0 | `broken-down` (2026-08-21) |
| `sisay` | 1/3 | 1 | 0 | — | 0 | `retired` — **not the pilot's deck** |

*Derived from the artifacts 2026-09-12. `deck-info <slug>` for the live picture.*

**Deleted 2026-09-01 — `kianne`, `kinnan`, `blar`.** All three were deterministic
whole-format baselines: never sleeved, never played, never published, and
`deck-delete` refuses any deck that was any of the three. They were not inert.
`deck_branch._deck_holders` walks every `cards.json` on disk and treats a build plan
as a HOLDER of the cards it names, so Edgar's `bloodline-v4` was `mergeable: false`
on *"unsleeve The Ozolith from kianne"* — a deck that has never existed in paper.
After the deletion and a `regen --only net-change`, that branch reads **"3 to buy;
93 of 96 already owned"**, which is the truth. The measurements taken on them stand
where they are cited (`docs/gotchas-bench.md`'s mean-vs-median lesson, the builder's
curve and combo-completion findings) — the measurement happened, and removing the
deck does not unmake it. `kinnan`'s build plan is now `tests/fixtures/`, owned by the
two regression tests that assert it rather than borrowed from a live deck.
Recoverable: `git show <the deletion commit>~1:data/decks/kianne/decklist.txt`.

**32 Forge runs exist across nine decks** — edgar (9), heliod (4), goblin-storm (4),
ur-dragon (4), gishath (3), radagast (3), zur (2), hapatra (1), sisay (1), yawgmoth (1).
The two bench decks have not been simulated at all.

**Five decks have a real table logged — fifteen games, 3W 12L.** ur-dragon (6),
edgar (4), goblin-storm (2), heliod (2), gishath (1). **Every one is debriefed**; the
`log_annotations.json` files were written 2026-09-11.

The night of **2026-08-28** at Alex's (Moraga Way, Orinda; three-player pod with Alex and
Stuart) put four of those on the board in one sitting — goblin-storm, edgar, heliod and
gishath, in that order, **1W 3L with the Dinosaurs taking it**. Three of those decks had
never been logged at all.

**Two decks are not marked as built in paper** (`sharknado`, `ingris-infect`). That is a
third state, distinct from the five that no longer exist as cardboard: nobody has said
either way, and `deck-info` says so instead of assuming.

**heliod and gishath were paper-locked on 09-07 and 09-11**, which closed the
PLACEHOLDER state this section used to describe: both had been played before anyone had
asserted what was sleeved, and a check-in is what settles that. All five played decks now
carry a lock.

Five decks no longer exist as cardboard (`hapatra`, `sisay`, `radagast`,
`yawgmoth-swarm`, `zur-enchantress`). Their artifacts stay exactly as published; `deck-info` states the
status and withholds the suggestions that would need a deck to shuffle.

## Where it stands (2026-08-25)

| | | where |
|---|---|---|
| **The workbench gets verbs** | the lifecycle leaves `issue.json` for `deck_versions.json`; `deck-state` and `deck-delete`; `validate-deck-versions` (closes #24); the archive rack folds; the version stamps on the art; the fleet controls, absent without a local server | `docs/pilot.md` |
| Agent audit + Sprint 0 | 18 → 17 agents; shared contract (`.claude/agents-common.md`); L10 repealed; magazine editor/panel/short-list retired; writer + coach → `pilot-notes`; `debrief` new; doctor MODE prescribe | `docs/history/agent-audit-2026-08-19.md` |
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
| **The Pilot's Manual** | `manuals/p/<slug>.html` from `build-page` — the compact technical page, no `<script>`, rebuilds byte-identically. The magazine is unlinked from every live surface | `docs/history/manual-v5-spec.md` |
| **The first real games** | edgar v1.0.0 and ur-dragon v1.0.0, one logged game each, both debriefed, both feeding a prescription | `data/decks/*/log.jsonl` |
| **The workbench landing page** | `viz/workbench.html` — racks by whether a deck is SLEEVED, plus a fleet table sorted four ways over every `info.json`. Three labelled links per deck | `docs/viz.md` |
| **The fusion** | an open verified line prints its prose (50 of 50 covered); engine arrows on the 5 of 196 edges whose direction is a fact; a curve bar is a control | `docs/viz.md` |
| **Seed a walk from named cards** | textarea + `?cards=`; the enumeration separates, never the comma (9.2% of names contain one); *Start here* vs *Add to walk* | `docs/viz.md` |
| **The version policy** | PATCH/MINOR/MAJOR by capability, every slug from v1.0.0; releases sort numerically, near-misses refused, re-tagging needs `--force` | `docs/pilot.md` |
| **The paper lock's third state** | UNLOCKED is not dead. Four of eleven decks are unlocked and now say so; three rehearsal locks withdrawn | `docs/pilot.md` |

## The measurement audit (2026-09-04/05)

Two days spent making the goldfish honest, prompted by one deck refusing to
close. Every entry below is a defect that was reporting a number, not a missing
feature. The full record is in `docs/gotchas-bench.md`.

| what was wrong | what it cost |
|---|---|
| **The commander's attack tutor fired EVERY turn** | reported 5.70 fires a game against Forge's measured 1.22. Correcting it: kill-by-t8 0.501 -> 0.173, board power 27.30 -> 16.64. **More than half of one day's measured gains were this assumption** |
| **A card can be read correctly and never PLAYED** | five instances, caught as a class on the fourth. Two Shrines measured as exactly nothing; a SLEEVED deck ran its sacrifice engine on 2 of 4 outlets; four of six attack enablers were uncastable, so the model could not start its own engine |
| **Lifegain-drains was unmodelled entirely** | the deck's declared third pillar scored ZERO. No drain metric existed anywhere; the only occurrence of the word was a target LABEL |
| **A token per enchantment priced as one token** | Archon of Sun's Grace, the best card on its branch, valued at a single 2/2 forever |
| **Six lands counted as coloured sources that cannot pay on curve** | every colour in zur-enchantress short, while the headline read at or above target on two of three |
| **`^Game Outcome: .*draw` matched a DECKING LOSS** | 12 games recorded as draws while carrying winners; one run's accounting came to 72 of 60 |
| **A game nobody won was dropped from the summary** | a simultaneous loss — all four seats at 0 life — vanished; another run came to 59 of 60 |
| **A validator printed `OK ◆` over checks it never ran** | membership and win-line checks skipped in silence whenever `cards.json` was absent |
| **`model-coverage` pointed at a switch connected to nothing** | 21 cards across 7 decks told the pilot to set a flag that returns byte-identical figures |

What the deck itself is now measured at: **v2.3.1**, kill-by-t8 0.278 as a
CEILING, and 3 wins in 60 Forge games against a table whose own null is 0.159 —
an interval that excludes zero. The engine runs when Zur attacks, and Forge
attacks about once in four eligible turns. **That gap is the pilot, and it is
the largest single lever left.** No games have been logged.

## Open work

### OPEN — the swap game (2026-09-12): three staged, four assessed, five Edgar patches undecided

The pilot proposed cards one at a time and the bench answered additive / subtractive /
wash, with the cut named. Nothing here is measurable: nothing dies in the goldfish, it
is never targeted, and the Forge AI will not sacrifice or hold protection in response,
so every verdict rests on the log and on theory, and says so.

- **heliod/splendor-v1** now carries two swaps, re-proposed as v1.2.2: Gleaming Splendor
  for Alhammarret's Archive, and **Greater Auramancy for Lightning Greaves** — Heliod is
  an enchantment creature, so the shroud covers him as Greaves did and every other
  enchantment besides (game 001 was lost to a Swords redirected onto him). Two to buy.
  **Open from it:** Karametra's Blessing can no longer target Heliod once Auramancy
  resolves; its slot wants a wipe answer that does not target. The pilot is still
  iterating — stage and commit only, no regen, until they say the list is done.
- **edgar-vampires/elenda-v1** OPEN: Elenda, the Dusk Rose for Gifted Aetherborn and
  Blade of the Bloodchief for Unbounded Potential. Both owned, both feed the nine
  sacrifice outlets. Inevitable Defeat (subtractive: fourth colour pip at four mana,
  interaction already 12 against 5–6) and Duty Beyond Death (a wash: indestructible
  on a wipe turn undoes the drain package's best play) were declined.
- **ur-dragon**: Balefire Dragon and Ancient Silver Dragon assessed against final-v1
  and **not staged**. Silver is double blue at eight in the deck's worst colour
  (69% on curve against 93% white). Balefire is a late-game card in a deck whose
  median kill is turn seven; revisit after v1.3.0 is sleeved and has played, cut
  Scourge of the Throne if it goes in. Dragon curve on final-v1: 24 Dragons,
  mean MV 5.46, median 5.
- **Edgar's five single-card patch branches** (fear-v1, march-v1, mound-v1,
  newblood-v1, sergeant-v1, all opened 2026-09-11) are still OPEN. Three were
  measured at 40 games and read as noise; newblood and march were cancelled before
  they ran. Decide or delete — a Forge batch at 40 games cannot resolve a single
  card, so the honest options are theory (as above) or 400 games per arm.
- Filed from the session: #45 (net-change stamps a stale sha), #46 (staging on a
  PROPOSED branch, no amend path), #47 (goldfish_targets shared across branches),
  #48 (`propose --reason` writes an empty why; conftest needs the cache plugin).

### The round robin and `standard-v3` (2026-09-09 → 10) — DONE

The full record, with the round-robin table, the calibration and the reason
`standard` and `standard-v2` were both wrong, is **`docs/simulation.md`,
"2026-09-10: the round robin, and `standard-v3`"**. It was duplicated here and
there for two days; the simulation doc is the home.

### IN FLIGHT — the PRD v2 build-out (2026-09-03)

**New investors, and the vision changed.** `docs/prd.md` is the September PRD,
now tracked; `docs/prd-2026-08.md` is the superseded one that ~27 `PRD-v1 §N`
citations across 15 source files resolve against. Neither had ever been in the
repo, so every one of those citations pointed at nothing.

**The four blocking decisions are resolved** and recorded in `prd.md`'s intake
notes: bracket is a HARD CONSTRAINT that refuses (which is what `enforce_bracket`
already did); the dev batch is the GOLDFISH, not Forge; twelve opponents, ~6 real
and ~6 from EDHREC; and old and new paths run in parallel, with nothing deleted
before its replacement is live.

#### The measurement that inverted an epic

Epic D dispositions the goldfish as "folded into Forge as the no-interaction
batch". Measured at intake, **that would delete half the metrics catalog.**
Forge emits exactly two zone transitions and there are **zero `from Library`
lines** and **zero `to Battlefield` lines** in a 100-game run, so cards drawn,
turns with empty hand, draw-engine uptime, missed land drops, mulligan-conditioned
keeps, mean available mana and post-wipe BOARD recovery are not unimplemented —
they are unrecoverable. The goldfish simulates the library and hand; it is the
only engine that can answer the Mana and Card-flow groups at all.

The timing follows from the same place. A Forge game is **~100s at the median on
four jobs**, so a twelve-minute dev build buys 15-20 games against an MDE of
**42 points at n=20**. A dev-stage Forge batch is statistically inert. The
goldfish is 10,000 seeded games in ~4s and the benchmark adds 2.3s, so
`manamap pilot build` finishes a Zur deck in **9.4 seconds**.

**B-1's reproducibility criterion was rewritten** rather than met: Forge's seed
fixes the shuffle only, AI evaluation is budgeted in wall time, and the
2026-09-02 measurement has the same seed producing four different games and a
different winner. No test can assert what the engine does not do.

#### DONE

| | |
|---|---|
| `manamap pilot build` | Epic A as one command — intent, anchor, build, resolve, measure, land. Composes `commander-search`, `archetypes`, `brew`, `build_deck`, `manabase`, `bracket`, `goldfish`, `benchmark`. A `--brief` yields a bracket and a style, the style matched only against the commander's REAL archetypes; every other prose key is stored and consumed by nothing, and the report says so |
| `validate-brief` | the gate `brief.json` never had. Every check measured against all four briefs first and firing on none; inert keys and the EDHREC theme lookup REPORTED, never failed |
| `experiment`, four defects | it raised `KeyError` on `ci95_a` on **every real run**, after writing its artifact; `_run_arm` had no timeout; and the pod ran on Default while `simulate` has run it on Experimental since 2026-08-30, so the two measured different populations. `experiment_id` carries `profile_tag` now, so no record on disk is renamed or reinterpreted |
| `manuals/p/` has one writer | `build_page` and `poh` both wrote it, and `deck-branch merge` plus `make demo` both called the superseded one over the live one. `build-page` has no default output any more |

#### The opponent library — and the three seats that stay OPEN

Eleven opponent seats now, from four. Three are the playgroup as the captain's
log actually names it — **Oliver on Krenko** (goblin-storm 002), **Tom on
Purphoros** (same entry, and edgar-vampires 004), **Alex on Tannuk, Steadfast
Second** (ur-dragon 001, named exactly) — and all three are mono-red, which is
an observation rather than a choice: *"the red density took the game over."*

**Three of the playgroup's decks are deliberately NOT built**, because the log
never names their commanders and a guess in a file called `playgroup` is an
invention that goes invisible:

| whose | what the log says | what is missing |
|---|---|---|
| Tom | "one of his blue-black decks" (edgar 001) | the commander |
| Tom | "Tom had no creatures (enchantment deck)" (edgar 004) | the commander |
| Alex | "Alex came in with his fight deck" (edgar 004), "his fight deck" (ur-dragon 003) | the commander |

Alex's Sauron deck (edgar 001) is a fourth: "his Lord of the Rings Sauron deck"
names a character with several legal commanders and no way to choose between
them. **Ask the pilot; do not infer.**

Four EDHREC seats cover archetypes the playgroup does not run —
`jarad-graveyard` (which PRD §6 B-2 names by commander), `muldrotha-value`,
`sythis-enchantress`, `talrand-spells`. They are labelled COVERAGE in their own
notes and claim no bracket, because none was verified.

Five pods ship: `standard` (the benchmark table), `playgroup` (observed seats
only), `value-chains` (the axis the pod nights were lost on), `five-player`
(Oliver's table size, which nothing has ever measured) and `vito-era` (kept so a
pre-2026-09-02 record stays reproducible).

#### NEXT, in order

1. ~~**The metrics registry.**~~ **DONE** — `manamap pilot metrics`. 27 figures,
   13 published, 2 opt-in, 2 derivable, **10 unavailable with a measured reason**.
   `--problems` answers PRD §2 directly: **2 of 6 pod-night problems are fully
   answerable, 2 partly, 2 not at all**, and all three RESILIENCE metrics are
   unavailable.
2. ~~**`mulligan` parsed and thrown away.**~~ **DONE**, and it found a rules gap:
   Forge gives the first mulligan free.
3. ~~**The pod object.**~~ **DONE** — `data/pods/`, `--pod`, per-seat AI profiles.
4. ~~**Report a batch by pod composition and seat.**~~ **DONE.** `seat effect`
   and `placement` are published — and the seat figure had to be rebuilt from
   `Turn: Turn 1 (seat)`, because `-d` position reports BACKWARDS: Forge gives
   the first turn to the previous game's loser, so edgar started **81%** of a
   400-game run. Records carry a `pod` block, which is how we now know **14 of
   19 tracked runs faced `vito-era`**.
5. ~~**Environments** (Epic C).~~ **DONE** — `promote` / `demote`, a `dev`/
   `bench`/`sleeved` ladder where two rungs are DERIVED, a per-requirement gate,
   and C-3's three ownership buckets with the holding deck named.

#### THE TABLE WAS NEVER LEVEL, AND THE ONE THAT REPLACED IT IS NOT EITHER

`pods <name> --calibration` pools every tracked run that faced a table. Set by
the pilot, 2026-09-03: *"the exact pod players is less important than just
having calibrated teams playing against each other."*

    standard   giada-angels 0.572 (2.29x fair)   baylen-tokens 0.052 (0.21x)
               SUBJECT 0.159        3 runs, 290 decided games
    vito-era   vito 0.448   giada-angels 0.401   baylen-tokens 0.016 (0.06x)
               SUBJECT 0.135       14 runs, 761 decided games

`vito-era` was dropped on 2026-09-02 for being unfair. **The dominant seat moved
and the table did not level out**, and `baylen-tokens` is the floor in BOTH —
which is a fact about that deck under Forge's AI rather than about either table.

**So the null for `standard` is 0.159, not 0.25**, and `simulate --list` prints
it beside the rate. goblin-storm's 0.031 there is genuinely bad; edgar's 0.144 is
at the typical subject rate. Every win rate this bench has published was read
against a quarter that never existed.

Two nulls, and they differ: this one pools OUR decks and describes the fleet as
much as the table; `pod-control` (an opponent's own average deck in the subject
chair, **0.099**) is the neutral one and has been run once.

#### ZUR CHANGES DIRECTION — commander damage is abandoned (2026-09-04)

**Set by the pilot, on a measurement.** Forty seeded Forge games against the
five-player playgroup table:

    win rate            0.000 (0 of 39)     ci95 [0.000, 0.090]
    commander damage    0.35 a game · best single game 2 · 0 of 39 reached 21
    combat damage       2.33 a game  (edgar on the SAME table: 32.30)
    Zur ITSELF attacked 13 times in 40 games — 0.33 a game
    fetch trigger fired 15 times in 40 games — 0.38 a game

So the deck did not lose because its kill was slow. **It never ran its engine.**
Zur's trigger is on ATTACK rather than on connect, so a human pilot fetches every
turn regardless of blockers — Forge's AI will not send a 1/4 into an open board,
which is a real understatement of THIS deck and the mirror of the goldfish's own
blind spot. The truth is between the two and neither number is the ceiling.

`deck-doctor` MODE recon reached the same mechanism independently and added the
structural objection: Voltron is Zur's SMALLEST theme (369 decks against
Enchantress's 1,207), and *"if you're taking out one opponent per turn, but there
are three others, you will lose"*. It flagged the Draftsim disagreement as
`contested` and settleable by `simulate` rather than by more recon, which is
exactly right.

**The new plan, in the pilot's words: (1) counter target spell, (2) stax, (3)
enchantment-creature trickery for asymmetric table damage.** The four
swing-enablers stay but now serve the ENGINE rather than the kill.
`branches/pillars-v1` is the recut — 14 out, 14 in, bracket 3 held at exactly 3
Game Changers and zero two-card infinites.

**THE GOLDFISH CANNOT GRADE IT, AND SAYS SO.** `output.available: false` — *"no
magnitude series … that is an absent measurement, not a zero"* — plus
*"death-triggered DRAIN is not modelled at all"* and *"it cannot see interaction,
removal or any alternate win"*. Stax needs opponents, counterspells need
opponents, drain needs things to die. The branch missed its objective
(`interaction_6` 0.7393 against 0.80) and every other row came back noise; the
objective was NOT moved afterwards, because the same hand writing the declaration
and reading the verdict is what the engine lift was deleted for. `experiment`
against the pod is the instrument that can see all three pillars.

### OPEN — what the build-out still needs

1. **THE PILOT'S CALL: name three playgroup commanders.** The log names these
   decks only as archetypes, and a guess in a file called `playgroup` is an
   invention that stops looking like one immediately:
   Tom's *"one of his blue-black decks"*, Tom's *"enchantment deck"*, Alex's
   *"fight deck"* — and Alex's *"Lord of the Rings Sauron deck"*, which names a
   character with several legal commanders.
2. **One pipeline definition.** The rebuild chain is written out three times —
   `deck_branch.py` twice and `regen.STAGES` once — with `goldfish` and
   `mana-analysis` duplicated between two of them.
3. ~~**The workbench reads the stage.**~~ **DONE.**
4. **Epic D**: the inventory is CHECKED IN (`docs/agent-inventory.md`) and it
   makes the epic smaller than the PRD assumes — the editorial layer was retired
   on 2026-08-19, so this is a re-grouping of 17 charters rather than an
   excavation. Two charters are never spawned by a skill (`pipeline-runner`,
   `viz-dev`) and are NOT therefore deletable: both are invocable by name, and
   D-2 forbids deleting a capability before it has a new home. **Two magazine
   couplings block a clean retirement**, both in skills — `write-manual`'s build
   half and `author-decision`'s step 5 each call `build-manual` — and
   Nothing validates `manual_prose.json` any more — its only gate was the
   magazine's `validate_issue`, deleted 2026-09-13 with the renderer.
5. **Epic E**: mostly link wiring — `?cards=`, `?deck=` and `?ref=` already open
   the Atlas seeded, and the library is persisted and cross-surface.


### The speed sprint (2026-08-30/31) — DONE

Moved to **`docs/gotchas-bench.md`, "The speed sprint"** on 2026-09-12 — the
measurements (the warm worker's 43x on `query-rules`, `regen`'s 72 targets in
109s at `--jobs 8`) belong with the other measurements, not in a status file.

### The embedding architecture (2026-08-31 → 09-01) — PARKED

Moved to **`docs/architecture.md`** on 2026-09-12: the VAE half was already
there, and the CardBERT and VICReg record now sits beside it. Nothing cut over;
CardBERT survives as a toggle on the atlas. What is still open is filed —
**#12** (the eval measures a task the product never performs), **#40** (VICReg
is built, tested and unrun), **#38** (the CardBERT regions carry raw mechanical
labels while both other maps are named), strictly in that order.

### OPEN — what the speed sprint deliberately did not do

- **The five stale `diagnosis.json` files** — edgar-vampires, gishath,
  goblin-storm, heliod, yawgmoth-swarm. Their colour-source figure moved when
  fetchlands started resolving; two were already stale before that. Each needs a
  `/diagnose-deck` run, and prose is never hand-patched to green a gate. This is
  the only thing failing `make test` (plus one `deck_info` test downstream of
  heliod's).
- **Forge job count and work stealing.** `jobs` defaults to `cpu_count - 1` = 7
  on a machine with 4 performance cores, with a static split and no stealing —
  straggler tails of +4061s / +6734s / +1304s on the three biggest runs. Left
  alone because `run_id` does not encode `jobs`, so changing the default changes
  the SAMPLE a given run id produces.
- **`CODE = (SRC,)` over-invalidation.** An edit to any of ~130 files under
  `src/manamap/` invalidates the regenerate-and-compare cache for **236
  parametrized cases** across three files. The comment prices this at 20s; it is
  now 5-8x that. A per-subpackage key would keep the conservative property.
- **`card_value` and `candidates` sweeps are embarrassingly parallel** and still
  serial — each `_measure` builds a fresh generator, so a pool would be
  bit-identical. `candidates` runs one full 10k-game goldfish PER CANDIDATE.
- **Digest-based staleness.** Freshness tests still re-run the real producer;
  stamping an input digest would make them O(1) with one canary per artifact.
- **Agent fan-out.** The `open_questions` work queue dispatches `/resolve-stack`
  one at a time, `/write-manual` runs analyst then researcher serially, and
  `/diagnose-deck` runs recon then diagnose serially — all independent work.
  `resolve-stack/SKILL.md` already names this under "Scale-out note".
- **A resolved-path memo survives `monkeypatch` teardown** in `test_serve.py`, so
  a later test reading a real deck sees a tmp directory. Worked around by keeping
  `test_serve_cli.py` separate; the memo is the actual defect.
- **`stats.mde_proportion` overflows** above n ~ 1000 (exact binomial). A
  multiplicative PMF recurrence would fix it and is not bit-identical, so it was
  not done alongside the hoist.


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

### The merge request: `propose` — DONE

Documented in **`docs/pilot.md`**. What is still open about it is filed: **#46**
(staging on a PROPOSED branch silently makes it STALE, and there is no amend
path) and **#48** (`propose --reason` writes an empty `why`).

### A granted mana ability belongs to whoever received it (2026-08-31) — DONE

Moved to **`docs/gotchas-bench.md`** on 2026-09-12.

### OPEN — five diagnoses are stale and need the doctor, not a patch

`edgar-vampires`, `gishath`, `goblin-storm`, `heliod` and `yawgmoth-swarm` fail
`validate-diagnosis` on `axes[].colour-sources.measured.value`: the `manabase`
correctness fixes moved the audit's figure underneath a diagnosis whose PROSE
names the old number ("Green 26, red 20, white 20 against a 36 target"; "Four
sources above the Karsten yardstick"). goblin-storm's reading even describes the
colour-identity fallback that no longer exists — the number moved *because* the
doctor's complaint was fixed.

**Patching `measured.value` is forbidden** — it would leave prose contradicting
its own figure, which is "a fresh claim under an old byline". These need a
`/diagnose-deck` re-run each, which is an agent spend and the pilot's call.
Until then `make test` is 5 red on exactly these five, plus one `deck_info`
test downstream of heliod's. **gishath's is now doubly stale** — the granted-mana
fix moved its figures again on 2026-08-31. **`diagnosis.json`
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

### `land_colors` credits mana it cannot actually make — **#17**, partly fixed

The reminder-text and colour-identity halves are fixed; quoted token abilities
are untouched. The measurements are in **`docs/gotchas-bench.md`** (the fetchland
family, Archway Commons, the gated lands). Read the issue for what remains.

### Unit tests must not depend on an experimental deck — **#18, CLOSED**

Closed 2026-08-31: the branch-state tests were rebuilt on `tmp_path` fixtures.
The live successor is **#35** — three tests still name one deck as an example of
a property — sequenced as part of Phase 2 of `docs/paydown-plan.md`.

### The red tests, and the issue each one lives in

**`docs/known-issues.md` is the board.** It is verified against a real run and
dated; this file carried a second copy that said THIRTEEN while the board said
nine, which is the reason it is now one place and not two.

### THE ORDER OF TASKS — set by the pilot, 2026-09-01

**Modelling first, agents after.** Six logged games are un-debriefed and stay that
way until the embedding work clears. This is a deliberate sequencing decision, not
a backlog: `/debrief` is an agent spend against a model that is actively moving,
and a debrief written now would be annotating figures that the next commit
changes.

1. **Finish the `VIEW_WEIGHT` ablation** — `vw025` done, `vw050` running, `vw100`
   preserved. Report the three-way table.
2. **`project-spaces` on all five spaces** and LOOK at them — visual inspection is
   part of evaluation, and a space can win recall@10 by concentrating while losing
   everything that makes an atlas navigable.
3. **`train-cardbert --objective vicreg`** — built, tested, waiting on 1.
4. **Then decide what, if anything, cuts over.** The honest current answer is that
   CardBERT is a THEME-and-CENTROID space complementary to the function space, not
   a replacement for it.
5. **`/debrief` the six un-debriefed entries** — edgar 002/003, ur-dragon 002,
   goblin-storm 001, heliod 001, gishath 001. One batch, five decks.
6. **Edgar's direction change**, which the 08-28 log records and which supersedes
   the token-conversion axis: swarm typal → LORDS AND PAYOFFS, tokens demoted from
   win condition to dig engine, a mill package as a second route through the deck,
   W/B lifegain into lifedrain. This is a branch, never a `decklist.txt` edit.
7. **Heliod: protection first.** Losing the commander takes the engine with it and
   there is no answer today to a counter war plus redirected removal.

### Next, in order (the standing backlog)

1. **Paper check-ins, one deck at a time** — **only the pilot can do this**, and it is
   the highest-value thing available. **heliod and gishath lists are promised** and hold
   v1.0.0 placeholders until they land; three more (yawgmoth-swarm, kianne, kinnan) are
   still **not marked as built in paper**, so nothing knows whether they exist as
   cardboard. **ur-dragon v1.0.2 paper is on the way**; v1.0.1 is what is being played. `check-in`
   takes a typed list and refuses rather than guesses, then `deck-version paper` locks
   it and drift is computed on every swap from then on. Two decks (edgar, ur-dragon)
   have one logged game each — which is two, not a sample.
2. ~~**The versions deploy-time step**~~ — **DONE 2026-09-02, differently.** The step
   was never built and the panel rendered nothing in production for months, so
   `versions.json` is simply TRACKED now: regenerated by `make manuals`, gated by
   `test_versions_json_matches_a_fresh_run`, and byte-compared by CI. No Pages workflow
   and no repo-settings change needed. The one rule is that a commit never carries both
   a decklist and its version list — enforced by `tests/test_pilot_commit_protocol.py`.
   Also fixed in the same pass: **CI checked out at depth 1**, so every `git log --follow`
   returned nothing and `build-index` wrote `unresolved` for all three sleeved decks —
   the byte-diff gate could never have passed. `fetch-depth: 0` on both jobs.
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
   fetches to fifteen, and the handbook stylesheet's `?v=<sha8>` is the pattern to copy.

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
nobody and deleted in one commit when the compact page lands (`docs/history/manual-v5-spec.md`
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
- **A model's embedding must be in its own objective.** `[CLS]` received no
  gradient for a whole 40-epoch run and shipped as a random projection; nothing
  failed, because every other metric was measuring the imputation heads.
- **A guard that cannot fail is a claim, not a guard.** A bug probe that MISSES
  is a finding: either the test is vacuous or the code is dead. Both have
  happened, one commit apart.
- **Never train on what the eval measures.** EDHREC co-occurrence is genuinely
  external supervision AND is what `theme_groups` builds the theme eval from.
- **Visual inspection is part of evaluation.** A space can win recall@10 by
  concentrating and lose everything that makes an atlas navigable.
