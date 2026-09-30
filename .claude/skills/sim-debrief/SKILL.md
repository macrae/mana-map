---
name: sim-debrief
description: Turn a deck's simulated runs into a reading a pilot can act on — compute the findings skeleton, spawn the sim-debrief agent for the runs with no prose, merge (which recomputes the skeleton and takes the prose), validate, and route the open questions it raises. The simulated counterpart of /debrief; the captain's log is never touched. Use after `manamap pilot simulate` or a campaign run, or whenever `sim-findings <slug>` shows runs with no prose.
---

# Debrief the simulated games

Forge writes run records; `sim_findings.json` is the computed skeleton over them — per
run, findings with ids, intervals and sources — and the `sim-debrief` agent writes prose
that may cite only those ids. Everything below keeps the second honest to the first. A
Forge game has no pilot, so nothing here goes near `log.jsonl`.

1. **The skeleton first.** `.venv/bin/manamap pilot sim-findings <slug> --write`
   (add `--boards` where the run's logs are on this machine: it adds recurring board
   shapes from `sim-boards`, which need the logs). Read the output: which runs have
   prose and which do not, and which runs are on the CURRENT list — a run of an older
   list is evidence about that list.
2. **Nothing to read?** No runs → stop; `simulate <slug> --pod standard-v3 --games N`
   is the next thing, not this.
3. **Spawn `sim-debrief`**, scoped to the runs with no prose (name their ids in the
   prompt). The agent returns a path. It may cite only finding ids and quote only their
   figures — that is the whole charter — and it routes what it cannot settle.
4. **Merge, then validate** — in that order, because the validator reads the tracked
   file: `.venv/bin/manamap pilot merge-sim-findings <slug>` (the skeleton is recomputed
   from the records and only `prose` is taken, per run; a run id the records do not hold
   is rejected and reported; the merge REFUSES before writing when a number in the prose
   is not the record's), then `.venv/bin/manamap pilot validate-sim-findings <slug>`.
   A refusal goes back to the agent with the errors; do not hand-patch prose.
5. **Route the open questions.** The agent's summary lists each with its `settled_by`:
   `resolve-stack` → lift the board it names (`sim-boards … --lift --stack`, or
   `sim-scenario`) and run `/resolve-stack`; `experiment` → `experiment <slug> --a … --b …
   --pod …`; `campaign` → add an entry to the queue and `campaign <name> plan`;
   `poh-proposal` → the handbook proposal flow (D3); `diagnose` → `/diagnose-deck` or
   `/prescribe`; `unsettled` → say so to the user. Report what you routed and what you
   left.

**Agent output arrives as a path, not inline JSON.** The agent writes
`data/decks/<slug>/.agent-out/sim-debrief.json` (gitignored) and returns that path with
a short summary. Read the file, merge, validate — never ask for the JSON in the reply.
