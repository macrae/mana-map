---
name: poh-procedures
description: Write the authored half of a deck's Pilot's Operating Handbook — emergency procedures, normal procedures, rules of engagement. Numbered checklists grounded in the deck's own games. Use after `build-poh` renders the data sections.
---

# The handbook's authored half

Sections 0, 1, 2, 5 and 6 regenerate from tracked artifacts. Sections **3
(Emergency)**, **4 (Normal)** and **7 (Handling)** are written by a person with
an agent's first draft, and they are what make the book a handbook rather than a
report.

## Loop

0. **See what the data half says** — free, and it is the brief:

   ```bash
   .venv/bin/manamap pilot build-poh <slug>
   .venv/bin/manamap pilot validate-poh <slug>   # NOTEs the sections still missing
   ```

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

3. **Install, then validate** — in that order, because the validator reads the
   rendered page:

   ```bash
   .venv/bin/manamap pilot install-agent <slug> --routine poh-procedures
   .venv/bin/manamap pilot build-poh <slug>
   .venv/bin/manamap pilot validate-poh <slug>
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

`validate-poh` checks the callout cap per page, that every cross-reference
resolves, that no `<script>` and no build date reached the output. It cannot tell
whether a checklist is *followable* — that is the whole content of the section
and it is not mechanically decidable.

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
