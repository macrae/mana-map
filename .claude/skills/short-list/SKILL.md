---
name: short-list
description: Generate The Short List — ten cards worth knowing about that could play well with a deck, scouted from the whole card pool — as data/decks/<slug>/considering.json and the manual's Short List section.
---

# The Short List (the ten)

Produces `data/decks/<slug>/considering.json` (tracked) and the manual's
**The Short List** section. One artifact for every deck: ten cards worth knowing
about that could play well with it, scouted from the whole card pool and given a
once-over (gaps, strictly-better alternatives, obsolescence).

**Ownership is not a criterion.** There is no sideboard. A card is on the list
because it is worth knowing about, not because the pilot already has it — the
list used to rank owned cards first, which made an inventory question into a
selection rule. Analysis-only: `cards.json` and `decklist.txt` are never
rewritten. Schema reference: the `short-list-analyst` charter.

## Loop

1. **Facts** — free, run all three; they are the analyst's brief and yours:

   ```bash
   .venv/bin/manamap pilot deck-facts <slug>
   .venv/bin/manamap pilot deck-audit <slug>
   ```

   `deck-audit` names what limits the deck and which pool cards would join the
   engine's thinnest component — the sharpest starting point for the ten. It
   needs `data/cards.csv` (a pipeline run) to read the pool.

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

4. **Spawn** `short-list-analyst` with the slug. It writes
   `data/decks/<slug>/.agent-out/short-list-analyst.json` and returns the path.

5. **Merge + validate**: copy the scratchpad file to
   `data/decks/<slug>/considering.json`, then
   `.venv/bin/manamap pilot validate-considering <slug>`. On failure, re-spawn
   with the validator errors — do not hand-fix content beyond mechanical
   formatting. The validator enforces exactly ten, that no pick is already in
   the deck, obsolescence/synergy claims against the indexes, and recomputes
   every claimed bracket delta.

6. **Record**, last, only after validation passes:
   `.venv/bin/manamap pilot cache-record <slug> --routine the-ten`.

7. **Build**: `.venv/bin/manamap pilot build-manual <slug>`. The Short List
   renders straight from the artifact — no prose key to merge (the writer's
   `upgrades` key is the section's opening copy and is cached separately).

8. **Report**: the ranked ten, any line a pick opens
   (candidates until a stack passes — surface as `/resolve-stack` fodder), and
   the `gaps`.

## Notes

- **Tier discipline.** Computed evidence is ◆; every ranking and verdict is ★.
  A line a pick would open stays a candidate until a stack artifact passes.
- **A decklist edit invalidates everything.** `cards.json`'s semantic fields feed
  `cards:semantic`, so any change to the 99 MISSes every routine on the deck, this
  one included. Expect it; do not re-record blindly.

**Agent output arrives as a path, not inline JSON.** Every deck agent writes to
`data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and returns that path
with a short summary. Read the file, validate it, then merge — never ask for
the JSON in the reply.
