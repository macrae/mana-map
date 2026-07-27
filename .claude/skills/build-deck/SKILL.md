---
name: build-deck
description: Build a Commander deck from a brief — deterministic 99, then deck-analyst → deck-architect → deck-critic to improve it, all gated on the bracket engine and the citation contract. Use when the user wants a deck built or rebuilt for a commander at a target power bracket.
---

# Build a Commander deck (the construction loop)

Turns `data/decks/<slug>/brief.json` into a validated 99, a `decklist.txt`, and a
fetched `cards.json` the rest of the pilot subsystem can consume. Schema reference:
`docs/pilot.md`; design rationale: `docs/deck-builder-v2.md`.

**The deterministic builder always runs and always succeeds on its own.** The agents
improve a baseline they can be measured against; if you skip them entirely you still
get a legal, bracket-compliant, goldfishable deck. Never present an agent pass as the
thing that made the deck legal — the code did that.

## Loop (max DECK_BUILD_MAX_ITERATIONS = 3, from config.py)

0. **Brief**: `data/decks/<slug>/brief.json` must exist with at least `slug`,
   `commander`, `bracket` (1–5). Colour identity is *derived* from the commander,
   never authored. Budget is not supported — prices are stripped from the card data,
   so say "budget unsupported" rather than approximating it.

1. **Baseline** (free, no agents): `.venv/bin/manamap pilot build-deck <slug> --write-decklist`.
   Writes `build_plan.json` and `decklist.txt`. Then, all still free:

   ```bash
   .venv/bin/manamap pilot validate-build <slug>
   .venv/bin/manamap pilot fetch-deck  <slug>
   .venv/bin/manamap pilot validate-deck <slug>   # the real legality gate
   .venv/bin/manamap pilot deck-facts  <slug>     # the brief every agent will read
   ```

   If any of these fails, stop — it is a code or brief problem, not something an agent
   fixes. **`validate-deck` runs here, not at step 8.** It used to sit after the
   analyst, architect and critic had all run, so a 100-card or colour-identity failure
   cost ~530k tokens to discover something the deterministic path knew for free.

2. **Cache gate** — the agent passes cost real tokens, so never re-run them blindly:
   `.venv/bin/manamap pilot cache-status <slug> --routine candidate-pool`
   (then `deck-build`)
   - **exit 0** — inputs unchanged since the artifact was recorded. Report the
     recorded result and **do not spawn.** `--force` to override.
   - **exit 1** — run that step.
   - **exit 2** — a required input is missing; fix that first.

3. **Pool**: spawn `deck-analyst` with the slug, the brief, and the target bracket.
   It writes `.agent-out/deck-analyst.json`; read that and copy it to `candidate_pool.json`. This is the architect's sandbox — the
   architect may not name a card that isn't in it, so a thin pool caps the ceiling.

4. **Architect**: spawn `deck-architect` with the brief, `build_plan.json`,
   `candidate_pool.json`, and the slug. Merge its output into `build_plan.json`
   (`archetype`, `gameplan`, `role_budget`, `role_budget_citations`, `swaps`,
   `engines`, `keep`, `gaps`) and apply its `swaps` to `slots`.

5. **Mechanical gate**: `.venv/bin/manamap pilot validate-build <slug>`. On failure,
   re-spawn the architect with the validator errors — do NOT hand-fix content
   yourself beyond mechanical formatting. Do not proceed to the critic until form
   passes.

6. **Bracket gate**: `.venv/bin/manamap pilot bracket-check <slug> --target <N>`.
   Exit 1 means the deck is out of tier; the cut candidates it prints go back to the
   architect.

7. **Critic**: spawn `deck-critic`. Merge its `critic` block into `build_plan.json`.
   If verdict is `fail` and iterations < 3, re-spawn the architect with the findings.
   Else save as-is — a `fail` plan is still saved (it documents what couldn't be
   grounded), but say so plainly in the report rather than presenting it as clean.

8. **Re-materialise**: the architect's swaps changed the 99, so regenerate and re-gate
   — `.venv/bin/manamap pilot build-deck <slug> --write-decklist`, then
   `fetch-deck <slug>`, then `validate-deck <slug>`. Step 1 already proved the
   *baseline* was legal; this proves the *edited* deck still is. Cheap either way, and
   now it can only fail on something an agent actually did.

9. **Simulate**: author `goldfish_targets.json` from `build_plan.engines` (the deck
   declared its own gameplan, so it knows what to test), then
   `.venv/bin/manamap pilot goldfish <slug>`. A low assembly rate is a build defect,
   not a mystery — feed it back to the architect if you have iterations left.

10. **Record**: `.venv/bin/manamap pilot cache-record <slug> --routine <R>` for each
    routine, **last**, only after its artifact is written and validated. Recording
    before validating poisons the cache.

11. **Report**: commander, computed bracket floor vs target, what was cut for bracket
    and why, the critic verdict and iteration count, goldfish assembly rates, and the
    architect's `gaps`. Surface the gaps as work for other skills — pool shortfalls
    are a `deck-analyst` problem, uncitable ratios are a `/research-strategy` problem,
    and unresolved engines are `/resolve-stack` scenarios.

## Notes

- The deck flows straight into the manual pipeline from step 8 — a built deck is an
  ordinary deck. `/write-manual` and `/design-issue` need nothing special from here.
- Engines the architect names are **candidates**, never facts. Promoting one to ✓
  means running `/resolve-stack` on it like any other line.
- If the user asks for a bracket the commander can't reach without its best cards,
  the builder raises rather than silently shipping off-tier. That is the intended
  behaviour; relay the message rather than lowering the target for them.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with a short summary. Read the file, validate it, then merge — never ask for the JSON in the reply. A 133 KB `candidate_pool.json` returned inline costs ~35k tokens of context for nothing.
