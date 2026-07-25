---
name: strategy-lookup
description: Query the strategy companion — semantic search for discovering relevant strategic theory, exact lookup for grounding claims, and consult-mode spawns for board-state/deck questions. Also covers rebuilding the strategy DB after doc edits.
---

# Strategy companion lookup

The strategy DB mirrors the rules DB: semantic discovery vs exact verification.

```bash
.venv/bin/manamap pilot query-strategy "when do I become the archenemy" --json   # semantic top-6
.venv/bin/manamap pilot lookup-strategy strategy:multiplayer.threat-deflection --json  # exact fetch
.venv/bin/manamap pilot validate-strategy      # form-check doc + changelog
.venv/bin/manamap pilot build-strategy-db      # rebuild after ANY doc edit (staleness-guarded)
```

- **Discovering** theory for a question → `query-strategy`, several phrasings.
- **Grounding** a claim you're about to make → `lookup-strategy` (exact only;
  cite the id, e.g. `strategy:pivot-point`).
- Section IDs are citation IDs: decision-branch citations may use
  `{"rule": "strategy:<id>", "quote": "<verbatim doc text>"}` — validate-stack
  enforces the same verbatim-quote contract as CR citations.
- Editing `data/strategy/strategy.md` by hand is fine (it's tracked) — run
  `validate-strategy` then `build-strategy-db` afterward, and add a CHANGELOG
  entry (the validator requires bullets to name real section ids).

## Consulting the strategist

For board-state, sequencing, card-package, or whole-deck strategic questions,
spawn the `strategy-researcher` agent with `MODE: consult` and the question plus
the deck slug (if any). For a full deck assessment ask for "the strategic frame"
— it returns the strategic_frame JSON schema (see the agent definition); save it
to `data/decks/<slug>/strategic_frame.json`.
