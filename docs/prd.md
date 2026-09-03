# ManaMap — Product Requirements

*Engineering handoff. September 2026. Single user: Sean MacRae. No hosting, auth,
or multi-tenancy in scope. Supersedes [`prd-2026-08.md`](prd-2026-08.md), which is
kept verbatim because source citations resolve against its numbering as
`PRD-v1 §N`.*

**Ship auto-build. Everything else in this document exists to make auto-build's
output trustworthy enough to act on.**

The tool already changes outcomes. Ur-Dragon and Edgar Markov both went through
the measure-and-refactor loop and both visibly improved at the Sept 1 pod. What's
missing is the ability to go from an idea to a testable deck in minutes instead of
a weekend, and to know whether a change actually worked.

> **Read the [Intake notes](#intake-notes-2026-09-03) at the bottom before
> planning against this document.** The four blocking decisions are resolved
> there, and §1's account of the fleet is stale in six places.

## 1. Where the six decks stand

| Deck | Role | Pipeline state | Next |
|---|---|---|---|
| Goblin Storm (Zada) | Aggro / storm | Loaded, locked v1.0.0 | None — precon, no edits |
| The Ur-Dragon | Ramp / dragons | Loaded, v1.0.2 | Haste + protection density |
| Edgar Markov | Midrange aggro | Loaded, v1.0.1 | Card draw validated, continue |
| Heliod | Control / group hug | Not loaded | Commander protection rebuild |
| Gishath | Big-creature beatdown | Not loaded | First tuning pass |
| Zur the Enchanter | Esper tempo engine | In design | Build from scratch |

Goal state: all six locked. A locked deck is one where an off game reads as
variance, not as a deck failure, and no change is made without a measured reason.

## 2. What's broken

**Simulation doesn't resemble the table.** Forge runs a fixed four-player pod
against three static opponents, one of which reportedly wins around 60% of games
(figure as reported, not verified). That's a stable benchmark and nothing more. It
can't tell you how a deck performs against the actual meta — which, per the Sept 1
log, is red-dense, removal-heavy, and won by non-combat damage.

**Simulation metrics and deck-building questions aren't the same set.** Every real
problem from recent pod nights is a metric that doesn't currently exist:

| Observed problem | Deck | Metric needed |
|---|---|---|
| Ran out of gas late | Edgar | Turns with empty hand; draw-engine uptime |
| 3–5 lands behind curve | Edgar, Ur-Dragon | Missed land drops by turn; mean available mana |
| Threat telegraphed before lethal | Ur-Dragon | Turns between visible threat and lethal |
| Kept a hand with no goblins | Goblin Storm | Opener composition against archetype gate |
| Board dies to wipes, no value | Edgar, Gishath | Value generated on creature death; post-wipe recovery |
| Lost commander, engine offline | Heliod | Commander uptime; protection available when targeted |

If the simulation doesn't measure it, the builder can't optimize for it and the
swap can't be validated.

**The sub-agent layer has drifted.** The harness was built to produce a retro
magazine. It carries editorial voice definitions, writer teams, editor and coach
roles, and department structures that no longer serve the product. Underneath that
sits genuinely valuable work — stack resolution against the rules vector DB,
strategy research agents, combo verification — which needs to survive the cleanup.

**There's no fast path from idea to testable deck.** Building a deck is still
mostly manual. The pieces to automate it exist (card embeddings, rules DB, EDHREC
access, simulation) but nothing composes them into a single command.

## 3. Product model: three environments

A deck is a first-class versioned object that moves through three environments.
Requirements tighten at each promotion. This is the spine of the product — build
order, agent invocation, and UI lanes all derive from it.

### DEV — Workbench

Brewing. Physical card ownership not assumed. Fast, cheap, disposable iterations.
Most decks here will be thrown away — the environment is optimized for throughput,
not rigor.

*Build produces:* 100-card list, mana and curve stats, unit-test pass, short
simulation batch. No pilot's manual, no exhaustive combo search.

### STAGING — Bench

Candidate decks earning their way to the table. Full analytical treatment.
Comparison against a control version is available here.

*Promotion gate:* full simulation batch complete; combo audit run (infinite lines,
known combos, standout cards); mulligan and tutor guidance generated; no unresolved
rules questions on flagged interactions.

### SLEEVED — Prod

Physically sleeved and playable tonight. Version pinned. Changes require a feature
branch and a measured reason.

*Promotion gate:* card ownership reconciled against bulk pools; order list
generated for anything missing; pilot's manual and dossier complete; benchmark
score recorded.

### Three pillars mapped to environments

| Pillar | Dev | Staging | Sleeved |
|---|---|---|---|
| Version control & build | Branch freely, auto-build, cheap merges | Branch requires a stated hypothesis | Branch triggers ownership check |
| Simulation & insights | Short batch, headline metrics | Full batch, significance testing, A/B vs. control | Benchmark score for deck selection |
| Discovery | Primary surface — Atlas exploration into basket | Targeted — fill a named gap | Read-only — card lookup from manual |

## 4. Agent architecture

One entry point. Sean talks to Spen Botsum; Spen decides whether to answer directly
or delegate. Spen is biased toward answering himself and only calls a specialist
when he clearly lacks the information.

```
                    ┌─────────────────┐
     CLI / UI  ───▶ │  SPEN BOTSUM    │ ◀── card embed queries
                    │  router + voice │ ◀── deck version store
                    │  stats literacy │ ◀── pod night logs
                    └────────┬────────┘
                             │ delegates only when needed
       ┌─────────────┬───────┼───────┬─────────────┐
       ▼             ▼       ▼       ▼             ▼
  AUTO-BUILD      FORGE   STACK    BUILD        SCOUT
  assembles       runs    resolves runs CI on   researches
  100 cards       sims,   rules    merge,       archetypes,
  from a          reports questions babysits,   combos,
  brief           metrics w/ adver- escalates   meta shifts
                          sarial
                          checker
```

| Agent | Owns | Returns to Spen |
|---|---|---|
| **Spen Botsum** | Intent routing, conversation, statistical interpretation, direct card/rules/deck lookups | — |
| **Auto-Build** | Brief → 100-card list. Color identity, curve, category depth, bracket fit | Decklist, composition report, flagged gaps |
| **Forge** | Pod assembly, opponent AI profiles, game execution, log parsing, metric computation, significance tests | Metric table, effect sizes, confidence, sample adequacy |
| **Stack** | Rules adjudication via vector DB plus adversarial verification loop | Verdict, cited rules, confidence |
| **Build** | CI on merge: schema validation, legality, ownership check, artifact generation, failure triage | Pass/fail, failure cause, suggested fix |
| **Scout** | External research — EDHREC lists, archetype patterns, combo databases, meta shifts | Findings with sources |

The five specialists replace the current editorial agent set. Voice, style, and
department agents are retired; the writing capability they contained migrates into
pilot's-manual generation as a function of the Build agent, not as a standalone
editorial team.

## 5. Epic A — Auto-build

The marquee feature. One command takes a brief and returns a playable v0.0.1 in
the workbench.

**Inputs** — any combination, all optional except one:

- **Description.** Free text: what the deck should do, how it should win, what it must avoid.
- **Commander.** Named legend. Fixes color identity.
- **Card set.** A list of cards to include, typically from bulk or a scan.
- **Bracket.** Target power level.
- **Format.** Commander for MVP; architecture stays format-agnostic.

At least one of description, commander, or card set must be present. If no
commander is given, the builder derives color identity from the description and
card set and proposes three candidate commanders before proceeding.

**Stages**

1. **Resolve intent** — parse brief into archetype, win condition, color identity, bracket, hard includes.
2. **Anchor** — select or confirm commander; pull comparable EDHREC lists via Scout for the archetype centroid.
3. **Populate** — fill categories to depth using card embeddings, honoring hard includes and owned-card preference.
4. **Balance** — mana base, curve, color pip-to-source per Karsten, interaction density, category depth rule.
5. **Verify** — legality, singleton, color identity, bracket rules (game changers, tutors, fast mana).
6. **Test** — short simulation batch; report headline metrics.
7. **Land** — commit as v0.0.1 in dev with the brief stored as the version message.

### A-1 — As Sean, I describe a deck in a sentence and get a legal, playable 100-card list in the workbench.

- Given a description with no commander, the builder proposes three commanders with a one-line rationale each and waits for selection.
- Given a commander, the builder returns a complete 100-card list that passes singleton, color identity, and format legality checks.
- The list is committed as v0.0.1 with the original brief as the commit message.
- Composition report shows category depth, curve, land count, and pip-to-source coverage per color.
- Any category the builder could not fill to depth is flagged explicitly rather than silently padded.

### A-2 — As Sean, I hand the builder a pile of cards I already own and it builds around them.

- Given a card set, every named card appears in the output list or is reported as excluded with a reason (color identity, legality, bracket).
- Remaining slots prefer cards in known bulk pools over cards requiring purchase, and the report states how many slots are unowned.
- Given a card set and no commander, the builder derives color identity from the set and proposes commanders within it.

### A-3 — As Sean, a bracket target is respected so the deck is legal for the table I'm building it for.

- Bracket rules are encoded as a checkable rule set (game changer count, tutor density, fast mana, two-card infinites, mass land denial).
- The verification stage reports bracket compliance per rule, not as a single pass/fail.
- ~~Blocked on decision 1~~ — **resolved: hard constraint, refuse on failure.** See intake notes.

### A-4 — As Sean, the build finishes fast enough that I keep iterating instead of walking away.

- Dev build completes inside the agreed budget end to end, including the short simulation batch.
- Progress is streamed: current stage, elapsed, and estimated remaining.
- Pilot's manual generation, exhaustive combo search, and full simulation batches are excluded from dev builds by default.

### Auto-build CLI

```
$ mm build --brief "Zur esper enchantment tempo, bracket 3,
             board by t4-5, threat of lethal t6-7" \
           --include "Zur the Enchanter"

SPEN  Reading brief. Esper, enchantment/artifact engine,
      B3 optimized, tempo curve. Zur as anchor.

  [1/7] intent          resolved
  [2/7] anchor          Zur confirmed · 4 EDHREC lists pulled
  [3/7] populate        ████████████████░░░░  84 / 99
  [4/7] balance         pending
  [5/7] verify          pending
  [6/7] test            pending
  [7/7] land            pending

SPEN  Built. v0.0.1 committed to workbench.

      lands 36 · avg mv 2.9 · interaction 11 · draw 9
      pips W 24 / U 31 / B 27 — sources 18 / 22 / 20
      keepable sevens  79%
      commander by t5  73%
      board by t4      61%   ← brief target t4-5

      flagged  removal at depth 6, rule wants 8
      flagged  4 slots unowned (~$31)

      Want me to fill removal from the Black/Green
      bulk pool, or open the Atlas around Zur?
```

## 6. Epic B — Simulation overhaul

Two problems: the pod isn't representative, and the metrics don't match the
questions. Both must be fixed before auto-build's output can be trusted.

### B1 — Validate the existing engine first

Current Forge behavior is not well understood. Before extending it, establish
ground truth.

**B-1 — As an engineer, I can see exactly what goes into and comes out of a Forge simulation.**

- ~~Unit tests cover a fixed seed producing identical results across runs.~~ **Rewritten — see intake notes.** Forge's seed fixes the shuffle only; AI evaluation is budgeted in wall time, so games are not reproducible. The meetable criteria are seed plumbing, run-id derivation, and `validate-sim` re-deriving `analysis` from stored logs byte-for-byte.
- A known-outcome scenario (scripted opening hands, deterministic draw order) resolves as expected.
- Log schema is documented: what events are emitted, at what granularity, with what fields.
- Any metric currently reported is traced back to the log events that produce it.
- Known gaps between Forge's rules coverage and real play are listed.

### B2 — Representative pods

**B-2 — As Sean, my deck is simulated against opponents that resemble my actual table.**

- An opponent library of twelve decks exists, each with commander, decklist, bracket, and archetype tag.
- The library includes real playgroup archetypes: Krenko red tokens, Purphoros pingers, Jarad Golgari, graveyard midrange, enchantment control, fight-based green.
- Remaining slots pull from EDHREC at specified brackets to cover archetypes not represented in the playgroup.
- Pod size is configurable at three, four, or five players and randomized across a batch by default.
- Opponents are sampled without replacement per game; seat order is randomized.
- Each opponent deck carries an AI strategy profile: aggression, threat assessment, removal priority, interaction timing, politics behavior.
- Batch results report performance by pod composition and by seat position, not just in aggregate.

### B3 — Metrics that match the questions

**B-3 — As Sean, every metric I care about when tuning a deck is measured in simulation.**

- The full metrics catalog (appendix) is emitted per game and aggregated per batch.
- Token generation is broken out by token type, not counted as one number.
- Damage is attributed by source type: combat, direct, drain, ping, commander.
- Creature size is reported as a distribution with percentiles, not a mean.
- Every metric has a written definition and the log events it derives from.
- Adding a new metric does not require changing the log format.

> **Constraint discovered at intake:** roughly half the catalog is unrecoverable
> from Forge logs and is answerable only by the goldfish. See intake notes.

**B-4 — As Sean, I'm told whether a change actually moved the metric I was targeting.**

- A run can be declared against a control version; results report the difference per metric with effect size and confidence interval.
- Spen states required sample size for a target effect before the batch runs, and flags when a batch is underpowered.
- Metrics that moved but weren't targeted are surfaced separately as side effects.
- A change that shows no detectable effect is reported as "not detected at this sample size," never as "no effect."

Statistical non-detection is not a verdict on the card. A swap can be correct for
reasons simulation can't see — rules interaction, threat optics, politics. That's
what the Stack agent is for, and its verdict stands alongside the metrics rather
than beneath them.

## 7. Epic C — Version control and build pipeline

### Deck object

| Field | Dev | Staging | Sleeved |
|---|---|---|---|
| Decklist, commander, colors | required | required | required |
| Version, parent, branch, message | required | required | required |
| Composition stats | required | required | required |
| Short sim batch | required | superseded | superseded |
| Full sim batch | optional | required | required |
| Combo audit | optional | required | required |
| Mulligan / tutor guidance | optional | required | required |
| Ownership reconciliation | optional | optional | required |
| Pilot's manual, dossier | optional | optional | required |
| Benchmark score | optional | optional | required |

**C-1 — As Sean, I branch a deck, change it, and merge it back with the history intact.**

- Branch from any version; branch inherits the parent's decklist and stats as its baseline.
- A branch off a staging or sleeved deck automatically registers its parent as the control for A/B comparison.
- Merge writes a new version with a message, a diff of cards in and out, and the metric deltas.
- Version history is viewable per deck with messages and deltas.
- Versioning is semantic: major for an axis change, minor for engine changes, patch for mana base and swaps.

**C-2 — As Sean, merging triggers a build automatically and tells me if it failed and why.**

- Merge invokes the Build agent with the protocol for the destination environment.
- The build runs end to end without supervision and reports stage-level status.
- On failure, the Build agent attempts triage and retry once, then escalates to Spen with the cause.
- Spen resolves what he can and surfaces the rest with a specific ask, never a raw stack trace.
- A failed build leaves the previous version intact and playable.

**C-3 — As Sean, promoting to sleeved tells me exactly which cards I need to acquire.**

- Promotion produces a checklist of every card in the list, grouped: owned in another deck, in a named bulk pool, unowned.
- Unowned cards render as an order list with quantity and current price.
- Cards can be checked off manually; the system does not attempt full inventory tracking.
- Promotion is blocked until every card is marked acquired or explicitly waived.
- Pulling a card that lives in another sleeved deck raises a conflict warning naming that deck.

## 8. Epic D — Sub-agent audit and migration

The harness contains real value buried in obsolete framing. The migration is a
translation, not a rewrite. Nothing gets deleted before its useful capability has a
new home.

| Existing | Disposition | Becomes |
|---|---|---|
| Stack resolver + rules RAG | Keep | Stack agent, unchanged capability, new interface |
| Adversarial validator | Keep | Stack agent verification loop |
| Deck data agent (Scryfall) | Keep | Shared service under Auto-Build and Build |
| Combo audit agents | Keep | Staging gate under Build |
| Strategy research agents | Repurpose | Scout — archetype and meta research |
| Coach voice | Repurpose | Piloting guidance generation under Build |
| Goldfish / Monte Carlo sims | ~~Repurpose~~ **REJECTED** | ~~Folded into Forge~~ — kept as a separate engine. See intake notes |
| Editorial voice and style defs | Retire | — |
| Writer teams, editor roles | Retire | — |
| Magazine department structure | Retire | — |
| Issue planning / newsstand | Retire | — |

**D-1 — As an engineer, I have a complete inventory of the existing harness before anything is touched.**

- Every agent, skill, prompt, and context file is listed with its file path, invocation points, and dependencies.
- Each entry is classified keep, repurpose, or retire, with a one-line reason.
- Every front-end surface that calls into the harness is identified and mapped to the agents it depends on.
- The inventory is a checked-in artifact, not a one-time report.

**D-2 — As Sean, the refactor doesn't break anything I'm currently using.**

- Regression tests capture current output for each live surface before migration begins.
- Integration tests cover each Spen-to-specialist path.
- Migration proceeds one agent at a time; each lands green before the next begins.
- Existing manuals and dossiers remain viewable throughout.
- Rollback is possible at any single step.

## 9. Epic E — Discovery substrate

Card Embed is the foundation the other pillars sit on. Any card shown anywhere in
the product is a link into the Atlas at that card's position.

**E-1 — As Sean, I can ask Spen embedding questions in the CLI and get an answer plus a link.**

- Spen can query the embedding store directly — nearest neighbors, cards within a color identity near a centroid, cards similar to a set.
- Results render in the CLI with name, mana cost, type, and a one-line reason for the match.
- Every result includes a deep link opening the Atlas centered on that card.
- Queries return inside two seconds for a single-card neighborhood.

**E-2 — As Sean, every card reference anywhere in the product opens the Atlas at that card.**

- Card names in pilot's manuals, dossiers, decklists, composition reports, and simulation output are links.
- The Atlas opens centered on the card with its neighborhood loaded.
- From that view, cards can be added to a basket and the basket handed to auto-build.

**E-3 — As Sean, the map shows relationships beyond raw similarity.**

- Edge types beyond embedding distance are supported: known combo, synergy, functional reprint, archetype co-occurrence.
- Edges are toggleable as overlays.
- Combo edges are sourced from the combo audit and carry their verification status.

## 10. Non-functional requirements

| Operation | Budget | Feedback |
|---|---|---|
| Spen direct answer | < 3s | Streamed text |
| Embedding query | < 2s | Result list |
| Stack resolution | < 60s | Stage-by-stage |
| Dev auto-build | seconds, not minutes (see intake notes) | Stage bar + elapsed + ETA |
| Staging full build | Not bounded | Stage bar, interim metrics, resumable |

**CLI**

- Every operation over three seconds shows a progress indicator with stage name and elapsed time.
- Long operations stream interim results rather than blocking to completion.
- Any operation can be backgrounded and its status queried.
- Metric output is aligned in columns and uses consistent precision.
- Color is used semantically — flags, deltas, verdicts — and degrades cleanly when unsupported.
- Spen ends substantive responses with the next available action, not an open question.

## 11. Mocks

### Workbench — landing

```
manamap · workbench

SLEEVED — playable tonight
  Goblin Storm   v1.0.0 · locked   bench 74  ██████████████░░░░░
  Ur-Dragon      v1.0.2            bench 68  █████████████░░░░░░
  Edgar Markov   v1.0.1            bench 71  ██████████████░░░░░
  Gishath        v1.0.0            not measured

BENCH — staging
  Heliod         v4.2 · 2 gates open    needs combo audit
  Edgar          draw-engine branch     vs control · running

WORKBENCH — dev
  Zur            v0.0.1                 removal depth 6/8
  Ur-Dragon haste v0.0.3                3 builds today
  + auto-build                          start from a brief
```

### Promotion gate

```
$ mm promote edgar/draw-engine --to sleeved

SPEN  Checking gates for sleeved.

  full sim batch      ✓  400 games, 12-deck pool
  combo audit         ✓  2 verified, 0 unresolved
  piloting guidance   ✓
  pilot's manual      ✓
  ownership           ✗  3 cards not accounted for

      Welcoming Vampire        Black bulk pool
      Cordial Vampire          unowned          $4.10
      Bloodletter of Aclazotz  in Blech  ← conflict

      Blech is sleeved. Pulling this card
      takes that deck to 99.

  [c] check off owned   [o] order list   [w] waive
```

### Comparison against control

```
$ mm compare edgar/draw-engine --control edgar@v1.0.1

SPEN  400 games each, same 12-deck pool, seats randomized.
      You targeted turns with empty hand.

  metric                 control   branch    delta
  ─────────────────────────────────────────────────
  turns w/ empty hand      3.10     1.40     -1.70  ✓
  cards drawn / game      11.20    14.60     +3.40  ✓
  mean avail mana t6       5.34     5.29     -0.05
  bodies on board t6       3.80     3.10     -0.70
  value on creature death  0.90     1.30     +0.40
  win rate                 0.19     0.24     +0.05

SPEN  Target moved and held at n=400. Board width
      dropped — you traded bodies for draw, which is
      the trade you asked for. Win rate is up but the
      interval crosses zero; don't read it yet.

      Side effect worth a look: value-on-death is up
      even though bodies are down. The draw creatures
      are doing double duty.
```

## 12. Sequencing

*Superseded by the intake notes — auto-build goes first. Retained as the PRD's own
proposal.*

1. **Ground truth** — Forge validation (B-1) and harness inventory (D-1).
2. **Measurement** — metrics catalog (B-3), opponent library and pod randomization (B-2), control comparison (B-4).
3. **Spine** — deck object and environments (C-1), build pipeline on merge (C-2), Spen routing to Forge and Stack.
4. **Auto-build** — A-1 through A-4.
5. **Migration and discovery** — D-2 migration under regression cover; E-1 through E-3.
6. **Load the remaining three** — Heliod, Gishath, and Zur through the full pipeline.

## 13. Risks and open questions

| Risk | Mitigation |
|---|---|
| Forge's rules coverage is narrower than assumed, invalidating metrics | B-1 establishes coverage gaps before anything depends on it |
| Auto-build produces legal but incoherent decks | Category depth rule and archetype centroid anchoring; every build reports flagged gaps rather than hiding them |
| Migration breaks a surface in daily use | Regression capture before migration; one agent at a time; rollback at each step |
| Simulation win rates don't predict table results | Pod night logs are the ground truth; batch results should be checked against real outcomes before being trusted |
| Opponent AI profiles are too crude to be meaningful | Start with the six real playgroup archetypes where behavior is observed, not invented |

**Still open**

- Does the benchmark score need to be a single number, or is a metric profile better for choosing what to play?
- How does card scanning get cards into the system — does it exist today, and at what fidelity?
- Should staging support more than one branch against the same control at once?

## 14. Appendix — metrics catalog

Every metric below is emitted per game and aggregated per batch. Definitions are
binding: the same definition is used by simulation, by the builder's objective, and
by the comparison report.

| Group | Metric | Definition |
|---|---|---|
| Mana | Missed land drops | Turns 1–8 where no land was played and one was available to play |
| Mana | Mean available mana by turn | Untapped mana at start of main phase, per turn |
| Mana | Color screw rate | Games where a castable card was held for missing a color, by color |
| Mana | Keepable sevens | Share of opening sevens meeting the deck's stated keep rule |
| Mana | Mulligan rate | Mean mulligans taken per game |
| Card flow | Cards drawn per game | Total draws beyond the natural draw step |
| Card flow | Turns with empty hand | Count of turns ending with zero cards in hand |
| Card flow | Draw-engine uptime | Share of turns with at least one active repeatable draw source |
| Board | Bodies by turn | Creature count on board at end of each turn |
| Board | Creature power distribution | P25 / P50 / P75 / max power of creatures on board |
| Board | Tokens by type | Created per game, split: creature, treasure, blood, clue, food, other |
| Board | Counter frequency | +1/+1 and −1/−1 counters placed per game, by source |
| Board | Anthem-adjusted power | Total board power with static pump effects applied |
| Speed | Commander resolve turn | Turn the commander first resolves; share resolved by turn 5 |
| Speed | First payoff turn | Turn the deck's stated engine first produces its payoff |
| Speed | Turn to lethal, goldfish | Unopposed kill turn, no interaction |
| Speed | Threat-to-lethal gap | Turns between board reading as lethal-capable and lethal landing |
| Resilience | Post-wipe recovery | Turns to return to pre-wipe board power |
| Resilience | Value on creature death | Cards drawn, damage dealt, and life gained triggered by own creatures dying |
| Resilience | Commander uptime | Share of turns after first resolve with the commander on battlefield |
| Interaction | Removal used vs. held | Interaction cast against interaction still in hand at loss |
| Interaction | Protection available at threat | Share of turns holding a protection effect and the mana to cast it |
| Interaction | Opposing threats answered | Opponent permanents removed per game |
| Outcome | Win rate | Wins over games, reported with interval |
| Outcome | Placement | Finishing position distribution |
| Outcome | Damage by source | Dealt and taken, split: combat, direct, drain, ping, commander |
| Outcome | Seat effect | Win rate by turn order position |

Damage-by-source and value-on-creature-death exist because of a specific
observation from the Sept 1 pod: consistent winners at this table win through
non-combat damage and value chains that survive wipes. A deck-building tool that
only measures board presence will optimize toward the losing axis.

---

## Intake notes (2026-09-03)

*Added on receipt, from a survey of the repository. The PRD above is otherwise
unedited; every correction is here rather than in the body, so the document that
was handed over stays legible as what was handed over.*

### The four blocking decisions, resolved

| Decision | Resolution |
|---|---|
| **1. Bracket** — constraint or objective | **Hard constraint, refuse on failure.** This is already what `pilot/bracket.py` and `build_deck.enforce_bracket` do, and A-3's "report per rule, not a single pass/fail" is already satisfied by `bracket.assess`'s `drivers[]`. Keeps the repo's binding invariant that a bracket floor is *what the contents are consistent with, never a verdict*. |
| **2. Dev build budget** | **Seconds, not minutes — because the dev batch is the goldfish, not Forge.** See below. |
| **3. Opponent library** | **Twelve: ~6 from the captain's log, ~6 from EDHREC** via the existing `pilot fetch-opponent`. |
| **4. Migration parallelism** | **Run old and new in parallel; delete only once the replacement is live.** The repo's own precedent: the magazine renderer was frozen and unlinked from every live surface, not deleted. |

### Why the goldfish is not folded into Forge

Epic D dispositions the goldfish as "folded into Forge as the no-interaction
batch". Measured at intake, that would delete half the metrics catalog.

Forge emits exactly two zone transitions — Battlefield→Graveyard and
Battlefield→Exile. Across a 100-game pod run there are **zero `from Library`
lines** and **zero `to Battlefield` lines**. So cards drawn per game, turns with
empty hand, draw-engine uptime, mulligan-conditioned keeps, mean available mana,
missed land drops and post-wipe *board* recovery are not unimplemented — they are
**unrecoverable**. The goldfish simulates the library and hand, so it is the only
engine that can answer the Mana and Card-flow groups at all.

The timing follows from the same measurement. A Forge game runs ~100s median at
four parallel jobs, so a 12-minute dev build buys ~15–20 games — against a minimum
detectable effect of 42 percentage points at n=20. A dev-stage Forge batch is
statistically inert. The goldfish runs 10,000 seeded games in ~4s and is
byte-deterministic; `pilot benchmark` adds ~2.3s. **Dev batch ≈ 6 seconds.** Forge
moves to the staging gate as a backgrounded 400-game job, where the same table
gives an MDE of 8.5 points.

### Why B-1's reproducibility criterion was rewritten

Measured 2026-09-02: the same deck, pod, seed, clock and job count run twice
produced four different games, one with a different winner. Forge budgets AI
evaluation in **wall time**, so piloting quality tracks machine load; the seed
fixes the shuffle only. No test can assert what the engine does not do. What is
true, tested, and kept: seed plumbing, run-id derivation, and `validate-sim`
re-deriving `analysis` from the stored logs byte-for-byte. The goldfish *is*
byte-deterministic.

### Where §1 and §2 are stale

| The PRD says | The repository says |
|---|---|
| six decks | **ten** live deck directories, plus three deleted 2026-09-01 |
| Heliod "not loaded" | 6/6 verified stacks, one logged game, a Pilot's Operating Handbook |
| Gishath "not loaded" | 5/5 stacks, and the fleet's only win |
| Zur "in design, build from scratch" | **V6**, 100 cards, committed and pushed 2026-09-03 |
| "one opponent wins around 60%" | `vito` won **0.447**, and was dropped as the default pod on 2026-09-02 for being the only bracket-4 seat at a bracket-3 table |
| editorial agents carry "writer teams, editor and coach roles" | retired 2026-08-19 — `magazine-editor`, `manual-writer`, `pilot-coach`, `pilot-panel`, `short-list-analyst`, `upgrade-scout` are all deleted. What is still frozen is the **Python renderer**, not the agent set |
| "nothing composes them into a single command" | `pilot brew <slug> --from <file> --build` already goes from a card pile to a written decklist. What is missing is the measurement, the progress stream, the flagged-gaps report and the v0.0.1 landing |

Also: the playgroup archetype list in B-2 reads "Cranko red tokens". The captain's
log names **Krenko** — Oliver on Krenko token-and-ping, and Tom on Purphoros doing
the same. Corrected in the body.

### Two `PRD` references in `goldfish.py` are deliberately left ambiguous

`model_version()` is a sha over the whole of `goldfish.py`, so a **comment edit
moves the digest on every deck** and restamps the fleet — the most expensive line
in the repo, measured and recorded in `docs/gotchas-bench.md`. The citation sweep
that qualified every other `PRD §N` as `PRD-v1 §N` therefore skipped
`goldfish.py:1884` and `:2679`. They refer to PRD-v1. Requalify them the next time
that file changes for a reason worth restamping.

### What already exists, and should not be rebuilt

`pilot commander-search` (A-1's three commanders), `pilot archetypes` (A stage 2's
EDHREC centroid and role template), `pilot assess` (A-2's card-pile triage),
`build_deck.fill_slots` (A stages 3–4, role quota crossed with a cited curve
quota), `manabase.build` (Karsten pip-to-source), `pilot benchmark` (four measures
under a frozen harness), `sim/experiment.py` (B-4's control comparison, with
Newcombe/Welch/permutation/bootstrap and exact power), `deck_branch` (C-1's six
derived branch states), `deck_branch._deck_holders` (C-3's conflict warning), and
`net_change.METRICS` (B-3's definition registry, already tested both ways).

`.claude/skills/publish-deck/SKILL.md` is already the router §4 describes — a
13-phase runbook whose own text says *"None of them knew the sequence, and that is
the failure this runbook exists to stop."*
