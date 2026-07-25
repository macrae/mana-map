---
name: resolve-stack
description: Resolve a stack scenario with rules citations and adversarial verification — the resolver→validator→checker quality loop. Use when the user wants a combo line or rules interaction resolved and saved as a verified artifact for a deck's pilot's manual.
---

# Resolve a stack scenario (the quality loop)

Runs the resolver→checker loop for one scenario and saves the artifact at
`data/decks/<slug>/stacks/NNN-<kebab>.json`. Schema reference: `docs/pilot.md`.

## Loop (max RESOLVE_MAX_ITERATIONS = 3, from config.py)

1. **Scenario**: create or receive the scenario block (`id`, `slug`, `deck`, `title`, `scenario{board, hand, mana_available, stack[], extras, question}`). Stack is ordered, `pos` 0 = bottom. Number scenarios `NNN` zero-padded in authoring order.
1a. **Cache gate** (the loop costs 65–130k tokens — never re-run it blindly):
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
2. **Resolve**: spawn the `stack-resolver` agent with the scenario JSON and deck slug. Write its `resolution` block into the stack file.
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
