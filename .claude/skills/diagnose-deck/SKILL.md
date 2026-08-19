---
name: diagnose-deck
description: Diagnose a finished Commander deck — cited axis targets, engine activation, a ranked add list and an argued cut list — via deck-doctor ⇄ deck-skeptic, gated on deck-audit and the citation contract. Use when the user wants a deck improved rather than described.
---

# Diagnose a deck (the improvement loop)

Turns a finished deck into `data/decks/<slug>/diagnosis.json` (tracked): what it is
good at, what actually limits it, what to lean into, what to add, and the cuts
nobody wants to make. Schema reference: the `deck-doctor` charter.

**The deterministic audit always runs and always answers on its own.**
`manamap pilot deck-audit <slug>` costs nothing, cites every target it sets, and
names the engine's thinnest component. If you skip the agents entirely you still
get the whole measurement. The agents read it and decide what binds — never
present an agent pass as the thing that produced the numbers.

**Analysis-only.** Nothing in this loop edits `decklist.txt` or `cards.json`.
Applying a swap is a separate, human act.

**One question instead of the whole reading → `/prescribe`.** Same doctor, same
skeptic, same contract, scoped to a prompt, accumulating under `prescriptions/`. Both
modes read the captain's log (`log_annotations.json`); if games are logged but not
debriefed, run `/debrief` first.

## Loop (max DIAGNOSE_MAX_ITERATIONS = 3, from config.py)

0. **Facts** — free, run them all; they are the doctor's brief and yours:

   ```bash
   .venv/bin/manamap pilot deck-audit      <slug>
   .venv/bin/manamap pilot deck-facts      <slug>
   ```

   Read `deck-audit`'s `notes` yourself before spawning anything. An axis that
   reads UNDER with a probe note under it is a question, not a finding — and
   putting the wrong one in a brief is how five errors reached agents in a single
   session.

1. **Freshness gate** — free, and the cheapest failure to avoid. `deck-audit`'s
   `freshness` block compares each derived artifact's stamped `decklist_sha256`
   against `cards.json`. If `goldfish_metrics.json` or `mana_analysis.json` is not
   `current`, re-run it **before** spawning:

   ```bash
   .venv/bin/manamap pilot validate-deck   <slug>
   .venv/bin/manamap pilot bracket-check   <slug>
   .venv/bin/manamap pilot goldfish        <slug>
   .venv/bin/manamap pilot mana-analysis   <slug>   # AFTER goldfish; it embeds its figures
   ```

   A diagnosis of stale numbers is worse than no diagnosis: it is confident and
   wrong, and every downstream artifact inherits it.

2. **Cache gate** — the agent passes cost real tokens, so never re-run them blindly:
   `.venv/bin/manamap pilot cache-status <slug> --routine deck-recon`
   (then `deck-diagnosis`)
   - **exit 0** (`HIT`/`EDITED`) — current; report and **do not spawn.** `--force`
     to override.
   - **exit 1** — run that step.
   - **exit 2** — a required input is missing; fix that first.

3. **Recon** (optional, age-gated): spawn `deck-doctor` with **`MODE: recon`** and
   the slug. `deck-recon` takes almost no cache inputs on purpose — a decklist edit
   does not change what strong lists for this commander run — so its staleness is
   **age**, not inputs. Compare `deck-audit`'s `freshness.recon.as_of` against
   `RECON_MAX_AGE_DAYS` (120) yourself; the audit never reads the clock. Skip this
   step entirely for a deck where the pilot only wants the artifact-grounded read.

   It writes `.agent-out/deck-doctor-recon.json`; copy that to `deck_recon.json`.

4. **Diagnose**: spawn `deck-doctor` with **`MODE: diagnose`** and the slug. It
   writes `.agent-out/deck-doctor.json`; read that and copy it to `diagnosis.json`.

5. **Mechanical gate**: `.venv/bin/manamap pilot validate-diagnosis <slug>`. On
   failure, re-spawn the doctor with the validator errors — do NOT hand-fix content
   beyond mechanical formatting. Do not proceed to the skeptic until form passes.
   The validator re-derives every axis figure from `deck-audit`, recomputes every
   bracket delta, and computes for itself whether a proposed cut strands a
   checker-passed stack.

6. **Skeptic**: spawn `deck-skeptic`. Merge its block into `diagnosis.json` under
   `skeptic`. If the verdict is `fail` and iterations < 3, re-spawn the doctor with
   the findings. Else save as-is — a `fail` diagnosis is still saved (it documents
   what could not be grounded), but say so plainly in the report rather than
   presenting it as clean.

7. **Record**, last, only after validation passes:
   `.venv/bin/manamap pilot cache-record <slug> --routine deck-diagnosis`
   (and `deck-recon` if you ran it). Recording before validating poisons the cache.

8. **Route the open questions.** This is the step that makes the loop compose.
   Subagents cannot spawn subagents — every charter here is `Bash, Read, Grep,
   Glob` — so the doctor emits a work queue and *you* dispatch it:
   - `settled_by: "resolve-stack"` → `/resolve-stack` per line. Preflight free with
     `validate-stack --scenario-only` before any spawn.
   - `settled_by: "research-strategy"` → `/research-strategy`. **Price it first:**
     editing `strategy.md` changes `strategy:doc` and MISSes six routines on every
     deck in the fleet. Take a `cache-snapshot` before the pass.
   - `settled_by: "goldfish"` → a `goldfish_targets.json` edit plus a re-run. A
     component the pilot cares about that no target names is invisible to the
     engine block, and widening a target is a decklist-free change.

9. **Report**: the archetype and where it came from, the one axis that binds, the
   engine's thinnest component with its measured rate, the ranked adds, the cut
   list **with its `painful` entries named**, the skeptic verdict and iteration
   count, and the open questions as dispatched work.

## Notes

- **Tier discipline.** Every axis measurement is ◆ (deterministic, reproducible,
  re-derived by the validator). Every verdict, ranking and prescription is ★. A
  line a proposed add would open stays a candidate until a stack artifact passes.
- **The diagnosis is a working artifact, not a page.** It may name a weakness
  plainly and compare the deck to what it could be — it feeds `/prescribe` and the
  pilot; the deck page never reads it.
- **Recon is evidence, not authority.** "Most lists run this" is a fact about other
  people's decks. Whether it belongs in this one is decided against this deck's
  measured axes, and a recon-sourced add still needs a `closes`.
- **A decklist edit invalidates everything.** `cards.json`'s semantic fields feed
  `cards:semantic`, so any change to the 99 MISSes every routine on the deck, this
  one included. Expect it; do not re-record blindly.
- **Do not commit `deck-audit` output.** It is a view, like `deck-facts`, and it
  embeds goldfish and bracket figures that go stale the moment the decklist moves.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to
`data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path with
a short summary. Read the file, validate it, then merge — never ask for the JSON in
the reply.
