# The documentation, sorted

About 7,500 lines, and 791 of them are history — two design records, kept because live
things cite them. It used to be roughly two thirds history; the magazine era was deleted
on 2026-08-25 and lives in git. This page says which doc is which, so you do not read a
design record as a description of the code. **Start with the vision; everything else is
written against it.**

## Start here

| | | |
|---|---:|---|
| **[vision.md](vision.md)** | 154 | Who this is for (a deck scientist and pilot), what the bench does end to end, the evidence contract, what is live / legacy / next, and the vocabulary. If a doc disagrees with this page, this page wins. |
| **[pilot.md](pilot.md)** | 1,431 | The bench reference: the evidence and citation contracts, every `manamap pilot` command, the per-deck artifacts, then each piece — status, `deck-info`, versions (incl. **the paper lock's three states and what a version bump means**), the captain's log and debrief, prescriptions, simulation, goldfish, scenarios and game state v2, the resolve loop, the rules and strategy DBs, facts, audit, diagnosis, engine, constellation — and a LEGACY block on the magazine renderer last. |
| **[simulation.md](simulation.md)** | 254 | Forge is the engine: the spike and its three criteria, the verdict, S1–S5 (harness, parser, the pod, the v2 bridge, the doctor reading the table), the tiers under seeding, the first runs and what they say, and the chain run once for real. |
| **[pipeline.md](pipeline.md)** | 43 | The 15 card-pipeline steps: command, inputs, outputs, runtime, when to re-run what. |
| **[data-artifacts.md](data-artifacts.md)** | 85 | Every file in `data/`: producer, size, tracked or not, who reads it — including the per-deck bench artifacts, the pod, and what is frozen legacy. Read before touching anything under `data/`. |
| **[testing.md](testing.md)** | 553 | How the suite is organised, the markers (incl. `forge`), the cache, and the lessons. **The only place that states test counts.** |

## Reference, by subsystem

| | | |
|---|---:|---|
| [architecture.md](architecture.md) | 316 | The two embedding models, how a card is decomposed, tag and role taxonomies, synergy rules, power-creep criteria, region clustering. |
| [viz.md](viz.md) | 1,405 | The frontend: the three PAGES (workbench, atlas, dossier), the three modes, the `window.MM` contract, the canvas renderer, seeding a walk from named cards, and what an open verified line prints. Read before any `viz/` change. |
| [agent-cost.md](agent-cost.md) | 243 | Where LLM spend lives, per-routine token sizing (current first, legacy measurements after), and how the invocation cache decides what to re-run. |
| [agent-audit-2026-08-19.md](agent-audit-2026-08-19.md) | 400 | The pivot's audit of the agents (18 then; 15 now): four fates, per-agent strengths and enrichment, the Sprint 0 order of work — all since executed. Read before touching a charter. |
| [manual-v5-spec.md](manual-v5-spec.md) | 115 | DRAFT: the compact deck page that replaces the magazine — what survives section by section, the section order, what the renderer and its gates lose, the phases. Waiting on the pilot's strikes. |

The LEGACY magazine renderer's constitution, `STYLEv3.md`, was **deleted 2026-08-25**
(`git show 23e8cec:STYLEv3.md`). It governed `build_manual.py`, `design.py`, `issue_spec.py`
and `validate_issue.py`, which still render the nine frozen pages; nothing in it ever
applied to the bench, and the compact Pilot's Manual that replaced the magazine
(`build_page.py`) was never written against it.

## History — records, not description

Two files. They document decisions and reasoning of their era, parts of them describe code
that was never written or has since been deleted, and they are **excluded from the docs
guards** for exactly that reason: a design record deliberately quotes the numbers of its
own time, and rewriting it would destroy the thing it is kept for.

| | | |
|---|---:|---|
| [history/deck-builder-v2.md](history/deck-builder-v2.md) | 461 | The deck builder's design: bracket engine, role taxonomy, the architect ⇄ critic loop, and where the implementation departed from the plan. **Load-bearing**: `deck-doctor`'s charter cites it for the hole recon exists to fill (no per-commander inclusion rates in any bulk data), and for why perishable meta claims stay out of `strategy.md`. |
| [history/frontend-v2.md](history/frontend-v2.md) | 330 | A proposed deck-building surface, superseded first by the dossier and then by the bench. Cited by the `refresh-corpus` skill. |

**Deleted 2026-08-25** — the magazine era, ~428 KB: `STYLEv3.md` (the constitution),
`STYLE-v1-visual-research.md`, `STYLE-v2-editorial-method.md`, both
`magazine-feedback-*.md`, `PLAN-2026-08-magazine-era.md` and an older `PLAN.md`. The
magazine is not the product and has not been since the pivot; keeping its constitution and
its editorial theory in the tree made a frozen renderer look like a live subsystem to
anyone reading the docs.

Nothing is lost — git holds all of it. `git show 23e8cec:STYLEv3.md` reads the
constitution, and `git show 23e8cec:docs/history/<file>` any of the rest. **The `STYLEv3
§N` citations in `issue_spec.py`, `design.py`, `validate_issue.py` and `build_index.py`
were deliberately left in place**: they say which clause a piece of frozen code implements,
which is still true, and they now resolve through git rather than through the tree.

## The files at the root that are not obviously docs

**[../CLAUDE.md](../CLAUDE.md)** is an instruction file for Claude Code, and also the
densest engineering knowledge here: paragraph-length post-mortems of real defects, each
with the measurement that settled it. If you want to know *why* something is the way it
is, look here first.

**[../PLAN.md](../PLAN.md)** is the resume-here doc: current state, open work, decisions
that bind, invariants that must not erode. It is candid about what is broken — that is
deliberate, and it is the fastest way to find something worth doing.
