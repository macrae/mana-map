---
name: analyse-sideboard
description: Generate The Short List — the ten cards most worth the pilot's sleeves (bench-first, pool-filled) — as data/decks/<slug>/considering.json and the manual's Short List section.
---

# The Short List (the ten)

Produces `data/decks/<slug>/considering.json` (tracked) and the manual's
**The Short List** section. One artifact for every deck: a bench bigger than ten
is pruned to its best ten, a smaller or empty bench is topped up from the whole
card pool, and the analyst gives the ten a once-over (gaps, strictly-better
alternatives, obsolescence). Analysis-only — the physical sideboard in
`cards.json` is never rewritten. Schema reference: the `sideboard-analyst`
charter.

## Loop

1. **Facts** — free, run all three; they are the analyst's brief and yours:

   ```bash
   .venv/bin/manamap pilot deck-facts <slug>
   .venv/bin/manamap pilot sideboard-facts <slug>
   .venv/bin/manamap pilot upgrade-facts <slug>
   ```

   `sideboard-facts` reports `available: false` for an empty/accessory-only
   bench — that is fine now; the pool fills all ten. `upgrade-facts` needs
   `data/cards.csv` (a pipeline run); without it a fresh clone can only rank
   the bench.

2. **Optional pilot feedback**: if the user has described how the deck plays,
   write it to `data/decks/<slug>/pilot_feedback.md` **before** spawning (it is
   a cache input). Absent, the analyst uses the standing forward-looking
   half-step posture.

3. **Cache gate** — never spawn blindly:
   `.venv/bin/manamap pilot cache-status <slug> --routine the-ten`
   - **exit 0** (`HIT`/`EDITED`) — current; report and **do not spawn**.
     `--force` to override.
   - **exit 1** — run step 4.
   - **exit 2** — a required input is missing; fix that first.

4. **Spawn** `sideboard-analyst` with the slug. It writes
   `data/decks/<slug>/.agent-out/sideboard-analyst.json` and returns the path.

5. **Merge + validate**: copy the scratchpad file to
   `data/decks/<slug>/considering.json`, then
   `.venv/bin/manamap pilot validate-considering <slug>`. On failure, re-spawn
   with the validator errors — do not hand-fix content beyond mechanical
   formatting. The validator enforces exactly ten, source membership,
   obsolescence/synergy claims against the indexes, and recomputes every
   claimed bracket delta.

6. **Record**, last, only after validation passes:
   `.venv/bin/manamap pilot cache-record <slug> --routine the-ten`.

7. **Build**: `.venv/bin/manamap pilot build-manual <slug>`. The Short List
   renders straight from the artifact — no prose key to merge (the writer's
   `upgrades` key is the section's opening copy and is cached separately).

8. **Report**: the bench/pool split, the ranked ten, any line a pick opens
   (candidates until a stack passes — surface as `/resolve-stack` fodder), and
   the `gaps`.

## Notes

- **Tier discipline.** Computed evidence is ◆; every ranking and verdict is ★.
  A line a pick would open stays a candidate until a stack artifact passes.
- **Editing a sideboard invalidates everything.** `is_sideboard` is in
  `CARD_SEMANTIC_FIELDS`: adding or removing a bench card MISSes every routine
  on the deck. Expect it; do not re-record blindly.
- **Legacy artifacts**: `sideboard_analysis.json` / `upgrade_watch.json` are
  superseded by `considering.json`; the renderer falls back to them only for
  decks not yet regenerated. When you produce a deck's first `considering.json`,
  `git rm` its legacy artifact in the same change.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to
`data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path
with a short summary. Read the file, validate it, then merge — never ask for
the JSON in the reply.
