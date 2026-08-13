---
name: analyze-engine
description: Work out how a deck actually runs — ignition, fuel, fodder, conversion, output, win-con — and produce engine.json. Runs the deck-engineer ⇄ engine-critic loop with its mechanical gate, then dispatches the open questions. Use after `deck-map`, when a deck needs its machinery understood rather than described.
---

# Analyse a deck's engine (the reasoning loop)

Turns a finished deck into `data/decks/<slug>/engine.json` (tracked): eight
possible stages, the lines between them, and — the part that costs the most and
matters the most — which of those lines a rules checker has actually proved.

**The clusters are an input, not the answer.** `deck-map` groups cards by what
they SAY; an engine is what cards DO TO EACH OTHER. Measured on radagast, only
**4 of 10** declared components sit in a single city. Naming a component after a
city is wrong in a way that reads as correct, which is the whole reason this loop
exists.

## Loop (max ENGINE_MAX_ITERATIONS = 3, from config.py)

0. **The brief** — free, and it is the doctor's-orders step of this loop:

   ```bash
   .venv/bin/manamap pilot engine-facts <slug>          # read the summary
   .venv/bin/manamap pilot engine-facts <slug> --json   # the arrays
   ```

   It joins `deck_audit.engine_activation` (components already priced
   hypergeometrically), the verified pairings from checker-passed stacks, the
   contained combo lines, and the scatter table. **Read its `notes[]` yourself
   before spawning** — a deck with zero verified pairings has no fact tier at all,
   and an engineer that does not know this will assert lines it cannot support.

1. **Cache gate** — this loop costs ~120k per engineer pass and ~140k per critic:

   ```bash
   .venv/bin/manamap pilot cache-status <slug> --routine deck-engine
   ```

   exit 0 = current, report and **do not spawn**. exit 1 = run it. exit 2 = a
   required input is missing.

2. **Model**: spawn `deck-engineer` with the slug. It writes
   `.agent-out/deck-engineer.json`; copy that to `engine.json`.

3. **Mechanical gate**: `.venv/bin/manamap pilot validate-engine <slug>`. On
   failure, re-spawn the engineer with the errors — do NOT hand-fix, and do not
   proceed to the critic until form passes.

4. **Attack it**: spawn `engine-critic`. Merge its block into `engine.json` under
   `critic`, setting `iterations`. **Preserve the critic block when you merge a
   revision** — the round-N verdict is the context round N+1 was written against.

5. **Iterate**: verdict `fail` and iterations < 3 → re-spawn the engineer with the
   findings, and tell it to **rebut rather than weaken**. A model that defends a
   claim with evidence, or withdraws it outright, is worth more than one that
   softens the wording until nobody objects. Both real revisions here produced
   rebuttals the critic then judged correct — including one against the critic's
   own arithmetic.

6. **Record**, last, only after the verdict is `pass`:
   `.venv/bin/manamap pilot cache-record <slug> --routine deck-engine`.
   A `fail` model is still saved — it documents what could not be grounded — but
   say so plainly rather than presenting it as clean, and never `cache-record` it.

7. **Dispatch the work queue.** Subagents cannot spawn subagents, so the engineer
   emits `open_questions` and *you* run them:
   - `settled_by: "resolve-stack"` → `/resolve-stack`. Preflight free with
     `validate-stack --scenario-only`. This is the common one: an engine line with
     no `verified_by` is exactly a scenario waiting to be written.
   - `settled_by: "goldfish"` → a `goldfish_targets.json` edit plus a re-run.
     `proposed_goldfish_edits` is where the engineer puts these; **it never edits
     the declaration itself**, and applying one is a human act.
   - `settled_by: "research-strategy"` → `/research-strategy`. Price it first: a
     `strategy.md` edit MISSes six routines on every deck in the fleet.

## What the gate can and cannot see

`validate-engine` re-derives every figure, enforces the closed stage set, requires
every card in the 99 to be placed or explicitly unassigned, and checks that a
line's `verified_by` names a checker-passed stack **whose scenario actually names
the line's cards** — plus that the line's `via` cards live in the stages the arrow
connects, a check added after the critic found three lines that failed it.

**It cannot check that the stack SUPPORTS the line.** Two radagast lines passed
every mechanical check while citing a stack that showed the opposite: one claimed
Castle Garenbrig paid for Craterhoof, citing a stack that leaves Garenbrig
untapped; another cited a stack whose resolution verifies three negatives. Both
render as solid green — the renderer's mark for proof. That inference is the
critic's whole job, and it is why this loop does not shortcut to the validator.

## Notes

- **Tier discipline.** A stage's membership and every rate are ◆. A line with a
  `verified_by` is ✓. A line without one is ★ — the analyst's reading — and the
  renderer draws it dashed for exactly that reason.
- **A decklist edit invalidates everything**, `deck_map.json` included. Re-run
  `deck-map` and its naming pass before this loop, not after.
- **`engine.json` is tracked**; `engine-facts` output is a view and is never
  committed, same rule as `deck-facts` and `deck-audit`.

**Agent output arrives as a path, not inline JSON.** Read the file, validate it,
then merge.
