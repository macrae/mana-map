---
name: resolve-stack
description: Resolve a stack scenario with rules citations and adversarial verification — the resolver→validator→checker quality loop. Use when the user wants a combo line or rules interaction resolved and saved as a verified artifact for a deck's pilot's manual.
---

# Resolve a stack scenario (the quality loop)

Runs the resolver→checker loop for one scenario and saves the artifact at
`data/decks/<slug>/stacks/NNN-<kebab>.json`. Schema reference: `docs/pilot.md`.

## Loop (max RESOLVE_MAX_ITERATIONS = 3, from config.py)

1. **Scenario**: create or receive the scenario block (`id`, `slug`, `deck`, `title`, `scenario{board, hand, mana_available, stack[], extras, question}`). Stack is ordered, `pos` 0 = bottom. Number scenarios `NNN` zero-padded in authoring order.
2. **Resolve**: spawn the `stack-resolver` agent with the scenario JSON and deck slug. Write its `resolution` block into the stack file.
3. **Mechanical gate**: `.venv/bin/manamap pilot validate-stack <slug> --stack <NNN>`. On failure, re-spawn the resolver with the validator errors — do NOT proceed to the checker until form passes.
4. **Check**: spawn the `rules-checker` agent for the file. Merge its `checker` block (set `iterations` to the loop count) into the file.
5. **Iterate**: if verdict is `fail` and iterations < 3, re-spawn `stack-resolver` with the findings attached. Else save as-is — a `fail` artifact is saved too (it documents an open question), but it can never appear in a manual.
6. **Report**: verdict, iterations used, and the artifact path. If it failed after 3 iterations, summarize the unresolved findings for the user.

Also set `rules_version` in the artifact from `data/rules/.rules-meta.json` `effective_date`.

## Scale-out note

For batch-resolving many scenarios (full manual prep), the Workflow tool can fan out resolver→checker pipelines per scenario. v1 keeps it sequential — a quality loop with human-reviewable intermediate states.
