---
name: poh-procedures
description: Write a deck's procedures — emergency procedures, normal procedures, rules of engagement (poh_procedures.json) — numbered checklists grounded in the deck's own games, shown on the deck page's "Play it" chapter. Use when a deck's procedures are missing or name cards it no longer runs.
---

# The procedures (`poh_procedures.json`)

Emergency procedures, normal procedures and handling — written by a person with
an agent's first draft, and read on the deck page's **Play it** chapter
(`viz/deck.html?deck=<slug>`: the turns, the emergencies, the table read). The
rendered Pilot's Operating Handbook that first carried them was retired on
2026-11-22; the file and its name stayed.

## Loop

0. **See what the deck page already says** — free, and it is the brief:
   `.venv/bin/manamap pilot deck-info <slug>` (its handbook line says whether the
   procedures exist and which list they were written for).

1. **Cache gate**:

   ```bash
   .venv/bin/manamap pilot cache-status <slug> --routine poh-procedures
   ```

   exit 0 = current, do not spawn. exit 1 = run it. exit 2 = a required input is
   missing.

2. **Model**: spawn `poh-procedures`. It reads the engine model, the audit, the
   diagnosis and — the part that matters — **the captain's log**, so a procedure
   for a wipe is grounded in the games that ended in one. It writes
   `.agent-out/poh-procedures.json`.

3. **Install, then validate, then recompose the page's data**:

   ```bash
   .venv/bin/manamap pilot install-agent <slug> --routine poh-procedures
   .venv/bin/manamap pilot validate-poh-procedures <slug>
   .venv/bin/manamap pilot deck-info <slug> --write && .venv/bin/manamap pilot build-index
   ```

4. **Record**, only once the validator passes:

   ```bash
   .venv/bin/manamap pilot cache-record <slug> --routine poh-procedures
   ```

5. **Read it as a pilot, not as a reviewer.** The real test of section 3 is
   whether the page for `wipe` tells you something you would act on. Read it
   against the games in the log that ended that way — the cause vocabulary joins
   them by construction. A page that restates the Limitations section, or that
   hedges every step, has failed even if every gate is green.

## What the gate can and cannot see

`validate-poh-procedures` holds every condition to `deck_notes.CAUSES`, every phase to
the ones the page draws, and every card named to the 99. It cannot tell whether a
checklist is *followable* — that is the whole content of the section and it is not
mechanically decidable.

## Notes

- **A decklist edit DOES stale this**, unlike the captain's log: a procedure
  naming a card the deck no longer runs fails at the table. `cards:semantic` is
  a declared input and a swap MISSes the routine.
- Editing `poh_spec.EMERGENCY_CONDITIONS` or `NORMAL_PHASES` is a fleet-wide
  re-spawn, not a re-bless — it changes what the agent may write.
- The conditions mirror `deck_notes.CAUSES` deliberately. Keep them mirrored, or
  a page stops joining to the games that prove it.

**Agent output arrives as a path, not inline JSON.** Install it, re-render, then
validate.
