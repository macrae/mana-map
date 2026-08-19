---
name: resolve-stack
description: Resolve a stack scenario with rules citations and adversarial verification — the resolver→validator→checker quality loop. Use when the user wants a combo line or rules interaction resolved and saved as a verified artifact for a deck — a ✓ line on its deck page and in its engine model.
---

# Resolve a stack scenario (the quality loop)

Runs the resolver→checker loop for one scenario and saves the artifact at
`data/decks/<slug>/stacks/NNN-<kebab>.json`. Schema reference: `docs/pilot.md`.

## Loop (max RESOLVE_MAX_ITERATIONS = 3, from config.py)

1. **Scenario**: create or receive the scenario block (`id`, `slug`, `deck`, `title`, `scenario{board, hand, mana_available, stack[], extras, question}`). Stack is ordered, `pos` 0 = bottom. Number scenarios `NNN` zero-padded in authoring order.

   ### The scenario format (this is a spec, not a suggestion)

   Everything below was a de-facto convention nobody wrote down, and each gap cost
   real rounds. `validate-stack --scenario-only` now enforces the parts it can, for
   free, before a ~35k-token spawn.

   **`hand`: a JSON list, `[]` when empty.** All 42 committed stacks use a list;
   26 are empty. Never prose. A placeholder sentence written into `hand` was read
   by `build_index.line_cards` as a card name and shipped into the deck manifest.

   **`board`: `{you: [...], opponents: [{life, board}]}`.** Seven decks use this;
   yawgmoth-swarm's `opponent_a..d` is the outlier and needs a compatibility read
   in every consumer. New scenarios use the list shape.

   **A permanent already sacrificed to pay a cost stays LISTED, annotated:**

   ```
   "Fume Spitter (1/1) — already sacrificed to pay the cost of the ability now on the stack"
   ```

   It is on the board list and **NOT on the battlefield**. This is the single most
   consequential reading in the corpus: it sets the body count, and every engine in
   these decks is bounded by bodies. Undeclared, it cost a resolver two separate
   arguments *inside resolution prose* for the reading it had already used, and a
   checker flagged the same scenario as self-contradictory. If you omit the
   annotation the board says a creature is present that the stack says is gone.

   **`mana_available`: symbols first, optional gloss in parentheses.**

   ```
   "{B}{B}{B} from three untapped Swamps"      good
   "{0}"                                        none available
   ""                                           NO — 16 of 42 files do this
   ```

   `""` and `"{0}"` mean the same thing on some boards and opposite things on
   others. Do not normalise the existing files — a scenario edit is a fingerprint
   input, so tidying 42 of them costs 42 respawns. Apply this going forward.

   **Every card named on a board, in hand, or on the stack must resolve against
   `cards.json`** — except tokens, and except an opponent's permanent that is
   deliberately not yours. The preflight errors on anything else, because a
   scenario naming a card the deck does not have describes a line nobody can play.

   **`extras` is non-normative.** `note_for_the_resolver`, `assumptions` and
   `life_totals` are scaffolding for the agent, not part of the question. Say
   anything there that helps; it does not change the rules problem.

   **A v2 game state is a second scenario form** (`docs/pilot.md` → *Game state v2*;
   `pilot/game_state.py` is the vocabulary): `seats[]` that hold priority, CR step names,
   and an `actions[]` list resolved left to right. It is consumed now — `validate-stack`
   and `scenario-facts` read `version: 2`, and `manamap pilot sim-scenario <slug> <run>
   --game G --turn T --step S --stack` WRITES one, lifted from a Forge game, with
   `question` empty for you to pose (stack 008 on radagast is the first, checker-passed).
   Author v1 for a board you describe yourself; v2 for a board a simulation surfaced.
   The same citation contract, the same loop, the same scope budget apply to both.

   ### Before you write one: `manamap pilot scenario-facts <slug> [--stack NNN]`

   The deterministic brief for this scenario — board split into creature bodies /
   other permanents / lands / the already-paid cost payment, opponent seats and
   life, the per-opponent vs pod-total arithmetic, which named cards are actually
   in the 99, and which sibling scenarios are comparable and how they differ.
   Read it instead of recalling figures. Five errors reached agent briefs in one
   session and every one was a correct-sounding number remembered rather than
   looked up — most damagingly a pod total quoted as a per-seat figure, which
   overstates a kill by 4×.

   **Keep it to one rules domain.** This is the single biggest lever on cost and
   success, and it is entirely in your hands. The checker's verdict is atomic over the
   whole artifact, so every extra citation is another chance for the whole thing to
   fail. Measured across the first three published decks: every artifact at **≤32
   citations passed in one or two rounds** (goblin-storm, 5 for 5); every artifact at
   **≥59 needed four rounds or failed** (hapatra 59@4; sisay 84 fail, 82 fail, 116@3).
   goblin-storm got 5 verified lines out of 6 rounds; sisay got 1 out of 9.

   A scenario asking `(a)`–`(e)` across summoning sickness, layers, priority and mana
   pools is five chances to fail in one file — and when it fails, correct answers die
   with it. Sisay 003's (a)–(d) were verified correct in all three passes and were
   discarded because (e) was weak. **Split multi-part questions into separate
   scenarios**; they resolve faster, cost less, and fail independently.

1a. **Preflight** — free, and it runs before you spend anything:

   ```bash
   .venv/bin/manamap pilot validate-stack <slug> --scenario-only
   ```

   Checks the scenario's form without needing a resolution, and warns when the question
   is over the sub-question budget. An empty `scenario.stack` once aborted three
   resolutions *after* all three had run, because nothing looked at the scenario until
   it had an answer attached. Fix anything it reports before step 1b.

1b. **Cache gate** (the loop costs 65–130k tokens — never re-run it blindly):
   `.venv/bin/manamap pilot cache-status <slug> --routine stack:<NNN>`
   - **exit 0** — the scenario block, `cards.json`, the CR version, and both agent
     prompts are unchanged since this artifact was resolved. Report the recorded
     verdict and iteration count and **stop — spawn neither agent.** A recorded `fail`
     is a HIT too: identical inputs reproduce the same failure, and the loop already
     spent its iterations. Use `--force` to retry anyway.
   - **exit 1** — run the loop below.
   - **exit 2** — the scenario file or `cards.json` is missing; fix that first.

   Only the scenario block is fingerprinted, so the `resolution` and `checker` blocks
   the loop writes into the same file never self-invalidate.
2. **Resolve**: spawn the `stack-resolver` agent with the scenario JSON and deck slug. It writes `data/decks/<slug>/.agent-out/stack-resolver-<NNN>.json`; read that file and merge its `resolution` block into the stack file.
3. **Mechanical gate**: `.venv/bin/manamap pilot validate-stack <slug> --stack <NNN>`. On failure, re-spawn the resolver with the validator errors — do NOT proceed to the checker until form passes.
4. **Check**: spawn the `rules-checker` agent for the file. Merge its `checker` block (set `iterations` to the loop count) into the file.
5. **Iterate**: if verdict is `fail` and iterations < 3, re-spawn `stack-resolver` with the findings attached. Else save as-is — a `fail` artifact is saved too (it documents an open question), but it can never appear in a manual.
6. **Record**: once the `checker` block is merged and
   `.venv/bin/manamap pilot validate-stack <slug> --stack <NNN>` passes form, run
   `.venv/bin/manamap pilot cache-record <slug> --routine stack:<NNN>`. The record
   stores the verdict and iteration count beside the fingerprint. Record **last**,
   after the artifact is written and validated.
7. **Report**: verdict, iterations used, and the artifact path. If it failed after 3 iterations, summarize the unresolved findings for the user.

Also set `rules_version` in the artifact from `data/rules/.rules-meta.json` `effective_date`.

## Scale-out note

For batch-resolving many scenarios (full manual prep), the Workflow tool can fan out resolver→checker pipelines per scenario. v1 keeps it sequential — a quality loop with human-reviewable intermediate states.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with a short summary. Read the file, validate it, then merge — never ask for the JSON in the reply. A 133 KB `candidate_pool.json` returned inline costs ~35k tokens of context for nothing.
