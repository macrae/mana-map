# The documentation, sorted

About 11,980 lines, and 1,412 of them are history — five design records, a brief and a decklist,
kept because live things cite them. It used to be roughly two thirds history; the magazine era was deleted
on 2026-08-25 and lives in git. This page says which doc is which, so you do not read a
design record as a description of the code. **Start with the vision; everything else is
written against it.**

## Start here

| | | |
|---|---:|---|
| **[prd.md](prd.md)** | 635 | **The product requirements (Sept 2026)** — the three environments, the five epics, the metrics catalog, and the four blocking decisions resolved in its **Intake notes**. Says what is being BUILT; `vision.md` says what the bench IS. Where they disagree on the present tense, `vision.md` wins; on the future tense, this does. |
| [prd-2026-08.md](prd-2026-08.md) | 404 | The superseded PRD, kept verbatim: ~27 citations across 15 source files read `PRD-v1 §N` and resolve here. |
| **[vision.md](vision.md)** | 168 | Who this is for (a deck scientist and pilot), what the bench does end to end, the evidence contract, what is live / legacy / next, and the vocabulary. If a doc disagrees with this page, this page wins. |
| **[pilot.md](pilot.md)** | 1781 | The bench reference: the evidence and citation contracts, every `manamap pilot` command, the per-deck artifacts, then each piece — status, `deck-info`, versions (incl. **the paper lock's three states and what a version bump means**), the captain's log and debrief, prescriptions, simulation, goldfish, scenarios and game state v2, the resolve loop, the rules and strategy DBs, facts, audit, diagnosis, engine, constellation — and a LEGACY block on the magazine renderer last. |
| **[simulation.md](simulation.md)** | 629 | Forge is the engine: the spike and its three criteria, the verdict, S1–S5 (harness, parser, the pod, the v2 bridge, the doctor reading the table), the tiers under seeding, the first runs and what they say, and the chain run once for real. |
| **[pipeline.md](pipeline.md)** | 43 | The 15 card-pipeline steps: command, inputs, outputs, runtime, when to re-run what. |
| **[data-artifacts.md](data-artifacts.md)** | 86 | Every file in `data/`: producer, size, tracked or not, who reads it — including the per-deck bench artifacts, the pod, and what is frozen legacy. Read before touching anything under `data/`. |
| **[known-issues.md](known-issues.md)** | 899 | **THE INVENTORY OF WHAT IS BROKEN.** The first four entries are the nine red tests and the parser bug that drops 88% of noncombat damage; the rest are gaps NO TEST FAILS ON — undecidable staleness on 19 artifacts, 50 undispatched open questions, 11 logged games and 0 debriefs, two uncriticised engine models. §12 lists what is already FIXED so it is not re-investigated, and a closing part tracks DEBT and OPPORTUNITIES — things that are not red but are wrong, wasteful or half-built. |
| **[audit-2026-09-12.md](audit-2026-09-12.md)** | 482 | **THE INVENTORY BEFORE GROOMING.** A read-only audit at `bccca716`: the project by the numbers against what the docs claim, the twelve-deck fleet with its lifecycle and branches, the MEASURED red board (13 failing, five of them on no board), all 42 open issues triaged into nine themes with the one that unblocks each, and the debt by area — the magazine's single remaining dependency edge, `goldfish.simulate_once` at 1,999 lines, six definitions of the decklist sha, five artifact registries. Ends with a recommended grooming order. Archive once acted on. |
| **[paydown-plan.md](paydown-plan.md)** | 1442 | **THE PAYDOWN PLAN AND ITS TRACKER.** Six phases in order — housekeeping and doc corrections, the daily-loop test cost (#28/#48/#31), the branch-lifecycle cluster (#45-#48, #25), the three predicates, the magazine delete (#4), the goldfish decomposition. Every task has an id, a gate, a proof and a status; the tracker table at the top is updated in the same commit as the work. Opens with the eight ground rules from the gotchas pages that constrain it, and closes with a risk register and what is deliberately out of scope. |
| **[testing.md](testing.md)** | 649 | How the suite is organised, the markers (incl. `forge`), the cache, and the lessons. **The only place that states test counts.** |

## The gotchas — every measurement this project has paid for

Extracted from `CLAUDE.md` verbatim in the 2026-08-27 compaction: that file went
**239 KB to 25 KB** because it loads into every session, and 218 of its bullets
were the full record of a defect rather than the rule a session needs before it
touches code. **Nothing was reworded and no number was lost** — the split is
asserted both ways, by numeric token and by verbatim bullet. `CLAUDE.md` keeps a
digest of the rules that bite whatever you are touching; these hold the evidence.

| | | |
|---|---:|---|
| [gotchas-viz.md](gotchas-viz.md) | 83 | The canvas renderer, the force graph, the three modes, the library and its piles, the shell, the atlas drift. Read before touching `viz/`. |
| [gotchas-bench.md](gotchas-bench.md) | 2102 | Agents and the invocation cache, Forge and the goldfish model, branches, the diagnostic layer, `deck-audit`, versions, the captain's log. Read before touching `src/manamap/pilot/` or `src/manamap/sim/`. |
| [gotchas-analysis.md](gotchas-analysis.md) | 26 | Synergy, the obsolescence index and its audit, card roles, region clustering. Read before touching `src/manamap/analysis/`. |
| [gotchas-evidence.md](gotchas-evidence.md) | 50 | Stacks, citations, `engine.json`, the deck map, and every validator's reasoning — **including the checks prototyped and REJECTED for firing on correct data**. Read before adding a validator or a claim. |
| [gotchas-magazine-legacy.md](gotchas-magazine-legacy.md) | 30 | The frozen renderer. Its code is not extended; the layout and prose lessons outlive it. |

## Reference, by subsystem

| | | |
|---|---:|---|
| [architecture.md](architecture.md) | 755 | The two embedding models, how a card is decomposed, tag and role taxonomies, synergy rules, power-creep criteria, region clustering. |
| [viz.md](viz.md) | 1491 | The frontend: the four PAGES (workbench, atlas, dossier, branch), the three modes, the `window.MM` contract, the canvas renderer, seeding a walk from named cards, and what an open verified line prints. Read before any `viz/` change. |
| [agent-cost.md](agent-cost.md) | 304 | Where LLM spend lives, per-routine token sizing (current first, legacy measurements after), and how the invocation cache decides what to re-run. |
| [agent-inventory.md](agent-inventory.md) | 122 | **The harness as it stands** — every agent and skill with its path, what it owns, which skill spawns it and how a five-specialist consolidation would re-home it, plus the front-end surfaces that depend on each. PRD §8 D-1 asks for this as a CHECKED-IN artifact rather than a report. Read before touching a charter. |
| [agent-audit-2026-08-19.md](history/agent-audit-2026-08-19.md) | 400 | The pivot's audit of the agents (18 then; 17 now): four fates, per-agent strengths and enrichment, the Sprint 0 order of work — all since executed. Read before touching a charter. |
| [manual-v5-spec.md](history/manual-v5-spec.md) | 139 | DRAFT: the compact deck page that replaces the magazine — what survives section by section, the section order, what the renderer and its gates lose, the phases. Waiting on the pilot's strikes. |

The LEGACY magazine renderer's constitution, `STYLEv3.md`, was **deleted 2026-08-25**
(`git show 23e8cec:STYLEv3.md`). It governed `build_manual.py`, `design.py`, `issue_spec.py`
and `validate_issue.py`, which still render the nine frozen pages; nothing in it ever
applied to the bench, and the compact Pilot's Manual that replaced the magazine
(`build_page.py`) was never written against it.

## History — records, not description

Seven files. They document decisions and reasoning of their era, parts of them describe code
that was never written or has since been deleted, and they are **excluded from the docs
guards** for exactly that reason: a design record deliberately quotes the numbers of its
own time, and rewriting it would destroy the thing it is kept for.

| | | |
|---|---:|---|
| [history/deck-builder-v2.md](history/deck-builder-v2.md) | 461 | The deck builder's design: bracket engine, role taxonomy, the architect ⇄ critic loop, and where the implementation departed from the plan. **Load-bearing**: `deck-doctor`'s charter cites it for the hole recon exists to fill (no per-commander inclusion rates in any bulk data), and for why perishable meta claims stay out of `strategy.md`. |
| [history/frontend-v2.md](history/frontend-v2.md) | 330 | A proposed deck-building surface, superseded first by the dossier and then by the bench. Cited by the `refresh-corpus` skill. |
| [history/naya-treasure-shell.md](history/naya-treasure-shell.md) | 159 | A Naya treasure shell costed against the Ur-Dragon fork, 2026-08-26. Opens by correcting the fork memo. **Nothing in it was applied.** Archived 2026-09-12. |
| [history/ur-dragon-fork.md](history/ur-dragon-fork.md) | 156 | The proposal to fork Ur-Dragon into two decks, 2026-08-26. Not applied; the deck was instead patched and is now sleeved at v1.2.1. Archived 2026-09-12. |
| [history/ur-dragon-refactor.md](history/ur-dragon-refactor.md) | 306 | The twenty-swap two-engine rebuild, 2026-08-26. Not applied. Archived 2026-09-12. |
| history/ur-dragon-treasure.decklist.txt · history/ur-dragon-treasure-brief.json | — | The list and brief the two Ur-Dragon memos cost out. Archived 2026-09-12. |

**Deleted 2026-08-25** — the magazine era, ~428 KB: `STYLEv3.md` (the constitution),
`STYLE-v1-visual-research.md`, `STYLE-v2-editorial-method.md`, both
`magazine-feedback-*.md`, `PLAN-2026-08-magazine-era.md` and an older `PLAN.md`. The
magazine is not the product and has not been since the pivot; keeping its constitution and
its editorial theory in the tree made a frozen renderer look like a live subsystem to
anyone reading the docs.

Nothing is lost — git holds all of it. `git show 23e8cec:STYLEv3.md` reads the
constitution, and `git show 23e8cec:docs/history/<file>` any of the rest. **The code that
carried `STYLEv3 §N` citations was itself deleted on 2026-09-13** with the rest of the
magazine renderer; the citations went with it, and both halves now read out of git.

## The files at the root that are not obviously docs

**[../CLAUDE.md](../CLAUDE.md)** is an instruction file for Claude Code, and also the
densest engineering knowledge here: paragraph-length post-mortems of real defects, each
with the measurement that settled it. If you want to know *why* something is the way it
is, look here first.

**[../PLAN.md](../PLAN.md)** is the resume-here doc: current state, open work, decisions
that bind, invariants that must not erode. It is candid about what is broken — that is
deliberate, and it is the fastest way to find something worth doing.
