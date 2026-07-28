---
name: analyse-sideboard
description: Analyse a deck's sideboard against its manual — which swaps are worth making, what to cut, what lines they open, and whether anything belongs in the 99 permanently. When the deck has NO sideboard, routes to the upgrade-scout instead — top-10 pool cards that would uplevel the deck, rendered as Upgrade Watch's "On the Lookout" section. Use when the user wants a sideboard read or a pool-upgrade read, with or without pilot feedback.
---

# Analyse a sideboard (or scout the pool when there isn't one)

Produces the Upgrade Watch department's data layer — one of two mutually
exclusive tracked artifacts, decided by whether the deck has a real sideboard:

- **Has one** → `data/decks/<slug>/sideboard_analysis.json` via `sideboard-analyst`
  (pool = that sideboard only; never searches the card pool).
- **Empty (or accessories only)** → `data/decks/<slug>/upgrade_watch.json` via
  `upgrade-scout` (pool = the whole card database; every claim must trace to a
  tracked index — see the agent's charter). Schema reference: `docs/pilot.md`.

Neither agent rebuilds the deck or publishes a new decklist. Applying a swap or
acquiring a lookout card is a separate job.

## Loop

1. **Preconditions** — free, and they pick the branch:

   ```bash
   .venv/bin/manamap pilot sideboard-facts <slug>
   ```

   `available: true` → sideboard branch (steps 2–9 below).
   `available: false` → **empty-sideboard branch**: jump to "The empty-sideboard
   branch" at the bottom. Do not stop — a deck with no bench gets the pool scout.

2. **Pilot feedback sets the agent's appetite**: if the user has described how the deck
   plays or what they want from it — "draws too few cards", "clunky on turn three", "want
   it a bracket lower", "bracket 4 is fine, maximize power" — write it to
   `data/decks/<slug>/pilot_feedback.md` before spawning. It is a cache input, so it must
   exist first. The stated appetite is the swap budget; absent feedback, the agent does a
   conservative unprompted analysis.

2a. **Sequence after the strategic frame.** Run this skill only once
   `data/decks/<slug>/strategic_frame.json` exists — the frame is the declared pivot for
   bracket-moving swaps, and an analysis produced without it argues archetype questions
   from thinner evidence (ur-dragon's first pass flagged exactly this in its own gaps).

3. **Cache gate** — never spawn blindly:
   `.venv/bin/manamap pilot cache-status <slug> --routine sideboard-analysis`
   - **exit 0** (`HIT`/`EDITED`) — current; report it and **do not spawn**. `--force` to
     override.
   - **exit 1** — run step 4.
   - **exit 2** — a required input is missing; fix that first.

4. **Spawn** `sideboard-analyst` with the slug. It writes
   `data/decks/<slug>/.agent-out/sideboard-analyst.json` and returns the path.

5. **Merge + validate**: copy the scratchpad file to
   `data/decks/<slug>/sideboard_analysis.json`, then
   `.venv/bin/manamap pilot validate-sideboard <slug>`. On failure, re-spawn with the
   validator errors — do not hand-fix content beyond mechanical formatting. The validator
   recomputes every claimed bracket delta, so a mismatch is a real disagreement, not a typo.

6. **Record**, last, only after validation passes:
   `.venv/bin/manamap pilot cache-record <slug> --routine sideboard-analysis`.

7. **Build**: `.venv/bin/manamap pilot build-manual <slug>`. The Upgrade Watch section
   renders straight from the artifact — there is no prose key to merge.

8. **Report**: the swaps proposed and their conditions, any line the sideboard opens, the
   long-term-default verdicts, and the `gaps`.

9. **Resolve, then re-analyse.** Every `"needs a stack scenario"` entry is a work queue,
   not a dead end: offer the user the `/resolve-stack` runs, and when a scenario passes,
   the analysis is stale by its own admission — re-run this skill (the new stack artifact
   is evidence the previous pass declared itself missing). The loop is
   analyse → resolve the blocking scenarios → re-analyse.

## Notes

- **Tier discipline.** Computed deltas are ◆; the recommendation to make a swap is ★. The
  section may never imply a swap is verified, and a line the sideboard opens is a candidate
  until a stack artifact passes the checker — once one has, the line carries
  `"status": "verified"` plus its `stack_artifact` path, and the validator confirms the
  artifact's verdict.
- **Editing a sideboard invalidates everything.** `is_sideboard` is in
  `CARD_SEMANTIC_FIELDS`, so adding or removing a sideboard card changes the deck's card
  digest and MISSes every routine on that deck — prose, coach, editor, stacks, decisions.
  Expect it; do not re-record blindly.
- "Nothing in this sideboard earns a slot" is a complete answer when the evidence says so.
  The swap count is set by the evidence and the pilot's stated appetite, never by a
  preference for small diffs.

## The empty-sideboard branch (upgrade scout)

Same shape as the main loop, different agent, artifact and validator:

1. `.venv/bin/manamap pilot upgrade-facts <slug>` — the deterministic pool brief
   (obsolescence upgrades, combo openers, synergy candidates, role-budget diff).
   `available: false` here means either the deck HAS a sideboard (use the main
   loop) or `cards.csv` is absent (fresh clone — needs a pipeline run first).
2. Pilot feedback: same rule as step 2 above — write it first, it sets appetite.
   Sequencing: same as 2a — run after `strategic_frame.json` exists.
3. Cache gate: `.venv/bin/manamap pilot cache-status <slug> --routine upgrade-watch`
   (same exit semantics; the routine is N/A for decks that have a sideboard —
   `sideboard-analysis` and `upgrade-watch` partition every deck).
4. Spawn `upgrade-scout`; it writes `data/decks/<slug>/.agent-out/upgrade-scout.json`.
5. Merge to `data/decks/<slug>/upgrade_watch.json`, then
   `.venv/bin/manamap pilot validate-upgrade-watch <slug>` — it re-checks every
   obsolescence and synergy claim against the tracked indexes and every combo
   claim's status. On failure, re-spawn with the errors.
6. Record last: `.venv/bin/manamap pilot cache-record <slug> --routine upgrade-watch`.
7. Build: `build-manual` renders the "On the Lookout" section from the artifact.
8. Report the ten picks in rank order with their evidence; combo lines opened are
   `/resolve-stack` candidates, exactly like opened lines in the main loop.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to
`data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with a short
summary. Read the file, validate it, then merge — never ask for the JSON in the reply.
