---
name: publish-deck
description: The deck lifecycle end to end — from a list to a measured, proven, noted deck that the bench can version, simulate, log, debrief and prescribe — every phase in dependency order with its gate. Use when starting a new deck, when regenerating one, or to find out what a half-finished deck still needs. This is the runbook the individual skills hang off.
---

# The deck lifecycle (the runbook)

Every other skill here knows its own phase. **None of them knew the sequence**, and
that is the failure this runbook exists to stop: a capability added in one
development cycle is reachable only by somebody who remembers it exists, so a deck
built the following month silently inherits the old pipeline. The vision this serves
is `docs/vision.md`: a lab bench for one pilot's paper decks — version it, measure it,
prove its lines, play it, log it, ask it questions. The page it renders today is the
**legacy magazine renderer**, frozen until the compact deck page
(`docs/manual-v5-spec.md`) replaces it; nothing in this runbook depends on which.

## Start here, always

```bash
.venv/bin/manamap pilot deck-status <slug>     # every stage: present / missing / STALE / INVALID
.venv/bin/manamap pilot deck-info   <slug>     # the workbench view, and a derived NEXT
```

`deck-status` reports every lifecycle stage and flags stages added recently that an
older deck will not have; it also runs each artifact's validator, because a green
dashboard over a red gate is worse than none. `pilot/deck_status.py:STAGES` is the
machine-readable version of the sequence below — **when you add a phase to the
lifecycle, add it there**, or the next person will not find it. `deck-info` composes
status, version, the log's record, sim runs, the doctor's verdicts and every open
question into one screen and names the next command.

Read exit codes: staleness is an ERROR, incompleteness is a state. A half-built deck
is work in progress. A deck whose artifacts disagree about which decklist they
describe is confident and wrong, which is worse and looks finished.

## The order, and why it is this order

**1 — The 99.** `brief.json` → `/build-deck` (architect ⇄ critic) → `fetch-deck` →
`validate-deck`. Or a list you already hold: `decklist.txt` in the repo's format (a
Moxfield export is best — printings, collector numbers and foil markers resolve
first; a name-only list yields default reprints). **Commit the list**: every change to
`decklist.txt` is a commit, and that is what `deck-version` numbers (`V1`, `V2`…) and
what the captain's log stamps a game against. An uncommitted working copy is a list
the bench cannot name.

**2 — Measure it.** `bracket-check` → author `goldfish_targets.json` → `goldfish` →
`mana-analysis`. **Order matters twice**: `mana-analysis` embeds goldfish figures, so
it runs after; and `goldfish_targets` is the engine DECLARATION every later phase
reads, so a lazy one poisons the rest. Run `validate-goldfish-targets` — a fleet
survey once found it wrong on six of eight decks, including two whose primary win
line had no target at all, so the simulator never measured how they actually win.

**3 — Understand it.** `/research-strategy` consult → `strategic_frame.json`.

**4 — Map it.** `deck-map` → `validate-deck-map`. The constellation: the deck
re-laid-out from its own cards and clustered into cities. Naming them
(`deck-cartographer` → `merge-deck-map`) is **optional** — the deterministic fallback
names are honest, and it is not a `deck-status` stage.

**5 — Engine it.** `/analyze-engine` — `deck-engineer` ⇄ `engine-critic`, gated by
`validate-engine`. **This is where the clusters become an input rather than an
answer.** A card is clustered by what it SAYS; an engine is what cards DO TO EACH
OTHER, and on radagast only 4 of 10 declared components sit in a single city. Expect a
real loop: the first radagast model prescribed the board that LOSES.

**6 — Prove the lines.** `/resolve-stack` per scenario, dispatching whatever
`engine.json`'s `open_questions` asked for — and, once the deck has been simulated
(phase 10), whatever board a run surfaced: `sim-scenario <slug> <run> --game G --turn
T --step S --stack` lifts it into a `game_state` v2 scenario for you to pose the
question on. A line without a `verified_by` is a claim; this is how it becomes a fact.
**One rules domain per scenario** — every artifact at ≤32 citations passed in one or
two rounds; every one at ≥59 needed three or four or failed, taking correct answers
down with it (stack 008, lifted from a simulated board, passed at 62 in three).

**7 — Coach it.** `tutor-guide` (one wish per tutor; `N/A` on a tutorless list) and
`/author-decision` (What's Your Play spreads). Both are `pilot-notes` under their own
routines, both validated (`validate-tutor-guide`, `validate-stack` for decisions).

**8 — Write the notes.** `/write-manual` — one `pilot-notes` spawn for the five prose
keys (`how_it_wins`, `mulligan`, `combo_lines`, `threat_assessment`, `matchups`),
merged by key ownership via `merge-prose` so the frozen legacy keys on a published
deck survive untouched.

**9 — Render the page** (legacy). Author `issue.json` (identity: deck name,
commander, status) → `build-manual` → `build-index`. This is the magazine renderer,
kept frozen until manual-v5; without an `issue_plan.json` it renders with defaults,
which is what a new deck gets. `validate-issue` gates the legacy plans on the
already-published decks.

**10 — Simulate it.** `fetch-opponent "<commander>" --as <slug>` for each seat at
your table (`data/opponents/`; once, then reuse), then `simulate <slug> --vs a --vs b
--vs c --games N` — N seeded Commander games in Forge, headless, one tracked run
record under `sim/` with win rate and interval, who kills you and how, the kill curve,
the token figures, and the assumptions (every seat is Forge's AI; it says so).
`validate-sim` re-proves the analysis from the logs where they exist. A run is what
phase 6 lifts boards from and what `/prescribe` cites.

**11 — Play it, and log it.** `deck-notes <slug> add "…" --result win|loss
--opponents N [--tag …]` after each game — your words, stamped with the list you held.
Then `/debrief` turns the un-debriefed entries into structure (seats, cards that
over/under-performed, takeaways, questions routed to the loop that settles them).
Nothing here needs the page; this is the bench's reason to exist.

**12 — Ask it.** `/prescribe <slug> "<one question>"` — the doctor (`MODE: prescribe`)
⇄ the skeptic, scoped to your question, reading the log, the sim and the audit:
ranked adds that close a named axis, cuts priced, accumulated under `prescriptions/`.
`/diagnose-deck` is the whole reading when you want all of it rather than an answer.

**13 — Change it, on a branch, and PROPOSE it.** A candidate 99 lives at
`branches/<name>/` — `deck-branch <slug> new … --objective "<measure> <op> <n>"`,
then `stage --out X --in Y`, then `net-change --branch <name>`, which is the report
a purchase rests on. When you accept it:

```bash
manamap pilot deck-branch <slug> propose <name> --as v1.0.2 --why "…" [--proxy] [--ordered "…"]
```

**That is the merge request, and it is the phase this runbook was missing.** The
decision is frozen (which list, which report, which grade); the blocker is LIVE and
is recomputed from your boxes on every read, so a proposal un-blocks itself when a
card lands in one. `deck-branch <slug> show <name>` prints the pull list — buy,
unsleeve, proxy, free — and `deck-info` names the state. Merging is still your act:

```bash
manamap pilot deck-branch <slug> merge <name> --write   # refuses until the cards exist
git add … && git commit                                 # this is what mints the version
manamap pilot deck-version <slug> tag v1.0.2 && … paper # the tag, then the sleeves
```

Then phase 1 again (commit → a new version), phase 2 (re-measure), and the log keeps
going.

## Gates, in the order they catch things

```bash
manamap pilot deck-status <slug>              # completeness + staleness + validity, first and last
manamap pilot validate-deck <slug>            # 100 cards, commander, singleton
manamap pilot validate-goldfish-targets <slug>
manamap pilot validate-deck-map <slug>        # names distinct, membership untouched
manamap pilot validate-engine <slug>          # stages, completeness, verified_by
manamap pilot validate-stack <slug>           # the citation contract, every stack and decision
manamap pilot validate-tutor-guide <slug>
manamap pilot validate-sim <slug>             # run records; analysis re-derived from logs
manamap pilot validate-debrief <slug>         # the log's annotations, held to the log
manamap pilot validate-prescription <slug>    # every question the doctor answered
manamap pilot validate-branch <slug> --branch N  # the objective is falsifiable; a
                                              #   proposal freezes what it accepted
manamap pilot validate-considering <slug>     # LEGACY: frozen Short Lists on published decks
manamap pilot validate-issue <slug>           # LEGACY: the magazine plan on published decks
.venv/bin/python -m pytest -m "not browser and not forge" -n auto
```

**Read exit codes directly.** `| tail` swallows them, which has burned this repo
four times, once in the session that wrote this file.

## History: what earlier cycles added, and what it cost to learn

Kept because the lessons still bind; the features are in `deck_status.STAGES` and
the gates above, not in anyone's memory.

- **`deck-map`** (+ the optional `deck-cartographer`) — the constellation, cities
  named for the job their cards do. Ward not average linkage (average put 37 of 71
  cards in one city); city count grown until the largest holds under 35%; territories
  drawn per neighbourhood because a spread city's hull swallowed the map.
- **`analyze-engine`** — the eight-stage model. It caught a prescription for the
  losing board, three arrows between stages containing neither endpoint, and two
  citations pointing at stacks that showed the opposite.
- **The legacy renderer's lints** — internal taxonomy ids in reader copy (68
  occurrences across eight pages) and deks that open with a question (14) — still
  run under `validate-issue` on the frozen pages; the taxonomy-id rule carries over to
  the pilot notes.
- **The workbench pivot (2026-08-19)** — the log, versions, prescriptions, the
  simulation harness and the bridge; the magazine agents retired (see
  `docs/agent-audit-2026-08-19.md`).

**Four rules these cycles earned, in the order they will bite again:**

1. **A validator that fires on correct data is worse than none.** Four separate
   checks were written, run, found to fire on accurate artifacts, and either scoped
   down or deleted. Always run a proposed check against the whole fleet before
   keeping it.
2. **A critic's findings become mechanical checks, or its work is re-spent every
   run.** `validate-engine`'s "a line's via cards must live in the stages it
   connects" exists because a critic found it by hand once.
3. **Name what a gate cannot see.** `validate-engine` checks that a stack NAMES a
   line's cards and can never check that it SUPPORTS the line. That is documented
   rather than papered over with string matching, because the same wrong line
   survives a rephrase.
4. **A failing artifact is saved and reported as failing.** Never `cache-record` to
   make a board green.

## Cache discipline

`cache-status` before spawning, `cache-record` **after** validating, `impact` before
any re-bless. **A charter edit invalidates that agent's routines by design and
disqualifies STALE_OK — so charter edits land BEFORE recording, never after.** Never
record a routine whose critic verdict is `fail`. The board has been red fleet-wide
since the 2026-08-19 charter consolidation, deliberately; it clears as each routine
next really spawns.
