---
name: publish-deck
description: The deck lifecycle end to end — brief to published issue, every phase in dependency order with its gate. Use when starting a new deck, when regenerating one, or to find out what a half-finished deck still needs. This is the runbook the individual skills hang off.
---

# Publish a deck (the lifecycle)

Every other skill here knows its own phase. **None of them knew the sequence**, and
that is the failure this runbook exists to stop: a capability added in one
development cycle is reachable only by somebody who remembers it exists, so a deck
built the following month silently inherits the old pipeline. Three capabilities
went in during August 2026 and every deck built before them is missing all three.

## Start here, always

```bash
.venv/bin/manamap pilot deck-status <slug>
```

It reports every lifecycle stage as present, missing or **STALE**, and it flags
stages added recently that an older deck will not have. `pilot/deck_status.py:STAGES`
is the machine-readable version of everything below — **when you add a phase to
the lifecycle, add it there**, or the next person will not find it.

Read its exit code: staleness is an ERROR, incompleteness is a state. A half-built
deck is work in progress. A deck whose artifacts disagree about which decklist
they describe is confident and wrong, which is worse and looks finished.

## The order, and why it is this order

**1 — The 99.** `brief.json` → `/build-deck` (architect ⇄ critic) → `fetch-deck` →
`validate-deck`. Use a **Moxfield export** for the decklist: printings, collector
numbers and foil markers resolve first, and a name-only list yields default
reprints and a visibly weaker issue.

**2 — Measure it.** `bracket-check` → author `goldfish_targets.json` → `goldfish`
→ `mana-analysis`. **Order matters twice**: `mana-analysis` embeds goldfish
figures, so it runs after; and `goldfish_targets` is the engine DECLARATION every
later phase reads, so a lazy one poisons the rest. Run
`validate-goldfish-targets` — a fleet survey once found it wrong on six of eight
decks, including two decks whose primary win line had no target at all, so the
simulator never measured how they actually win.

**3 — Understand it.** `/research-strategy` consult → `strategic_frame.json`.

**4 — Map it.** `deck-map` → `validate-deck-map`. The constellation: the deck
re-laid-out from its own cards and clustered into cities. Naming them
(`deck-cartographer` → `merge-deck-map`) is **optional** since the workbench pivot —
the deterministic fallback names are honest, and it is not a `deck-status` stage.

**5 — Engine it.** `/analyze-engine` — `deck-engineer` ⇄ `engine-critic`, gated by
`validate-engine`. **This is where the clusters become an input rather than an
answer.** A card is clustered by what it SAYS; an engine is what cards DO TO EACH
OTHER, and on radagast only 4 of 10 declared components sit in a single city.
Expect a real loop: the first radagast model prescribed the board that LOSES.

**6 — Prove the lines.** `/resolve-stack` per scenario, dispatching whatever
`engine.json`'s `open_questions` asked for. A line without a `verified_by` is a
claim; this is how it becomes a fact. **One rules domain per scenario** — every
artifact at ≤32 citations passed in one or two rounds; every one at ≥59 needed
four or failed, taking correct answers down with it.

**7 — Furnish it.** `/short-list` → `considering.json`, then `short-list-art` →
`considering_art.json`, the tracked sidecar that lets the one department whose
subject is OUTSIDE the deck actually show you the cards. `tutor-guide` → At the
Table's tutor subhead. `/author-decision` → the What's Your Play spreads.

**8 — Write it.** `/write-manual` (writer + coach, merged by key ownership via
`merge-prose`). The front of book (`pilot-panel`) and the magazine editor
(`/design-issue`) are **retired** with the workbench pivot
(`docs/agent-audit-2026-08-19.md`); the published decks keep their tracked
`issue_plan.json` and panel keys as frozen legacy inputs until the manual is
simplified.

**9 — Render it.** Author `issue.json` → `build-manual` → `build-index`. Without a
plan, `build-manual` renders with department defaults. `validate-issue --strict`
still gates the legacy plans on already-published decks.

**10 — Diagnose it**, when the deck is to be improved rather than described:
`/diagnose-deck`. Deliberately NOT a consumer of `engine.json` — the doctor keeps
its own engine view, and two consumers of one artifact is a migration.

## Gates, in the order they catch things

```bash
manamap pilot deck-status <slug>              # completeness + staleness, first and last
manamap pilot validate-deck <slug>            # 100 cards, commander, singleton
manamap pilot validate-goldfish-targets <slug>
manamap pilot validate-deck-map <slug>        # names distinct, membership untouched
manamap pilot validate-engine <slug>          # stages, completeness, verified_by
manamap pilot validate-considering <slug>
manamap pilot validate-tutor-guide <slug>
manamap pilot validate-issue <slug>           # legacy plans only (published decks)
.venv/bin/python -m pytest -m "not browser" -n auto
```

**Read exit codes directly.** `| tail` swallows them, which has burned this repo
four times, once in the session that wrote this file.

## What this cycle added, and what it cost to learn

New decks inherit these because they are in `deck_status.STAGES` and in the gates
above, not because anyone remembers them.

- **`deck-map`** (+ the optional `deck-cartographer`) — the constellation, and
  cities named for the job their cards do. Ward not average linkage (average put 37 of 71 cards in
  one city); city count grown until the largest holds under 35%; territories drawn
  per neighbourhood because a spread city's hull swallowed the map.
- **`analyze-engine`** — the eight-stage model. It caught a prescription for the
  losing board, three arrows between stages containing neither endpoint, and two
  citations pointing at stacks that showed the opposite.
- **Two `validate-issue` lints** — internal taxonomy ids in reader copy (68
  occurrences across all eight published issues) and deks that open by asking the
  reader a question (14).
- **`coach-gauge` and `stat-slab`** — a ★ judgment renders as stars, never a
  percentage; the issue's signature number runs once, full width.

**Four rules this cycle earned, in the order they will bite again:**

1. **A validator that fires on correct data is worse than none.** Four separate
   checks were written, run, found to fire on accurate artifacts, and either
   scoped down or deleted. Always run a proposed check against the whole fleet
   before keeping it.
2. **A critic's findings become mechanical checks, or its work is re-spent every
   run.** `validate-engine`'s "a line's via cards must live in the stages it
   connects" exists because a critic found it by hand once.
3. **Name what a gate cannot see.** `validate-engine` checks that a stack NAMES a
   line's cards and can never check that it SUPPORTS the line. That is documented
   rather than papered over with string matching, because the same wrong line
   survives a rephrase.
4. **A failing artifact is saved and reported as failing.** Never `cache-record`
   to make a board green.

## Cache discipline

`cache-status` before spawning, `cache-record` **after** validating, `impact`
before any re-bless. **A charter edit invalidates that agent's routines by design
and disqualifies STALE_OK — so charter edits land BEFORE recording, never after.**
Never record a routine whose critic verdict is `fail`.
