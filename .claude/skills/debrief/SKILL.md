---
name: debrief
description: Turn the pilot's captain's-log entries for a deck into structured annotations — spawn the debrief agent for the un-debriefed games, validate, merge, record, and route the open questions it raises. Use after `manamap pilot deck-notes <slug> add`, or whenever `deck-status` shows logged games that are not yet debriefed.
---

# Debrief the captain's log

The pilot writes; this reads. `data/decks/<slug>/log.jsonl` is authored and
append-only (`manamap pilot deck-notes <slug> add "…" [--result win|loss|draw]
[--opponents N] [--tag T]`). `log_annotations.json` is the `debrief` agent's
derived reading, keyed by entry id, and everything below exists to keep the
second honest to the first.

1. **What is outstanding.** `.venv/bin/manamap pilot deck-notes <slug> list` marks
   each entry `✓ debriefed` or `—`. `deck-status <slug>` shows the same on the `log`
   row. Nothing logged → stop; there is nothing to read.
2. **Cache gate.** `.venv/bin/manamap pilot cache-status <slug> --routine debrief`.
   Exit 0 → every logged game is already read under the current deck; do not spawn.
   Exit 1 → the log grew (or the deck moved under it); continue. Exit 2 → N/A or a
   missing input; the message says which.
3. **Spawn `debrief`**, scoped to the un-debriefed ids (name them in the prompt). The
   agent returns a path. It may name nothing the pilot did not — that is the whole
   charter — and it routes what it cannot settle.
4. **Merge, then validate** — in that order, because the validator reads the tracked
   file: `.venv/bin/manamap pilot merge-debrief <slug>` (by id; ids not in the log are
   rejected and reported; earlier annotations are carried), then
   `.venv/bin/manamap pilot validate-debrief <slug>`. A FAIL goes back to the agent
   with the errors; do not hand-patch an annotation.
5. **Record.** `.venv/bin/manamap pilot cache-record <slug> --routine debrief` — only
   after the validator passes.
6. **Route the open questions.** The agent's summary lists each with its
   `settled_by`: `resolve-stack` → author a scenario (`/resolve-stack`); `goldfish` →
   check `goldfish_targets.json` declares what the question measures, then re-run;
   `diagnose` → `/diagnose-deck` (the doctor reads the log); `research-strategy` →
   queue the topic. A `decisions[].worth_a_spread` is a candidate for
   `/author-decision`. Report what you routed and what you left for the user.

**Agent output arrives as a path, not inline JSON.** The agent writes
`data/decks/<slug>/.agent-out/debrief.json` (gitignored) and returns that path with a
short summary. Read the file, merge, validate — never ask for the JSON in the reply.
