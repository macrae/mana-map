# The documentation, sorted

About 20,000 lines, and roughly two thirds of it is history. This page says which is
which, so you do not read a design record — or the magazine era — as a description of
the code. **Start with the vision; everything else is written against it.**

## Start here

| | | |
|---|---:|---|
| **[vision.md](vision.md)** | 82 | Who this is for (a deck scientist and pilot), what the bench does end to end, the evidence contract, what is live / legacy / next, and the vocabulary. If a doc disagrees with this page, this page wins. |
| **[pilot.md](pilot.md)** | 1,114 | The bench reference: the evidence and citation contracts, every `manamap pilot` command, the per-deck artifacts, then each piece — status, `deck-info`, versions, the captain's log and debrief, prescriptions, simulation, goldfish, scenarios and game state v2, the resolve loop, the rules and strategy DBs, facts, audit, diagnosis, engine, constellation — and a LEGACY block on the magazine renderer last. |
| **[simulation.md](simulation.md)** | 169 | Forge is the engine: the spike and its three criteria, the verdict, S1–S5 (harness, parser, the pod, the v2 bridge, the doctor reading the table), the tiers under seeding, the first runs and what they say, and the chain run once for real. |
| **[pipeline.md](pipeline.md)** | 43 | The 15 card-pipeline steps: command, inputs, outputs, runtime, when to re-run what. |
| **[data-artifacts.md](data-artifacts.md)** | 80 | Every file in `data/`: producer, size, tracked or not, who reads it — including the per-deck bench artifacts, the pod, and what is frozen legacy. Read before touching anything under `data/`. |
| **[testing.md](testing.md)** | 543 | How the suite is organised, the markers (incl. `forge`), the cache, and the lessons. **The only place that states test counts.** |

## Reference, by subsystem

| | | |
|---|---:|---|
| [architecture.md](architecture.md) | 316 | The two embedding models, how a card is decomposed, tag and role taxonomies, synergy rules, power-creep criteria, region clustering. |
| [viz.md](viz.md) | 1,208 | The frontend: the three modes, the `window.MM` contract, the canvas renderer, the deck dossier, Pages deployment. Read before any `viz/` change. |
| [agent-cost.md](agent-cost.md) | 243 | Where LLM spend lives, per-routine token sizing (current first, legacy measurements after), and how the invocation cache decides what to re-run. |
| [agent-audit-2026-08-19.md](agent-audit-2026-08-19.md) | 400 | The pivot's audit of the agents (18 then; 15 now): four fates, per-agent strengths and enrichment, the Sprint 0 order of work — all since executed. Read before touching a charter. |
| [manual-v5-spec.md](manual-v5-spec.md) | 100 | DRAFT: the compact deck page that replaces the magazine — what survives section by section, the eight-section order, what the renderer and its gates lose, the phases. Waiting on the pilot's strikes. |

The LEGACY magazine renderer's constitution is **[../STYLEv3.md](../STYLEv3.md)** (1,116
lines), banner-marked SUPERSEDED: it governs `build_manual.py`, `design.py`,
`issue_spec.py` and `validate_issue.py` only, which still render the nine frozen pages
until manual-v5 replaces them. Nothing in it applies to the bench.

## History — records, not description

These document decisions and reasoning of their era. Parts of them describe code that was
never written or has since been deleted. They are kept because the reasoning is worth more
than the accuracy, and they are **excluded from the docs guards** for exactly that reason.

| | | |
|---|---:|---|
| [history/deck-builder-v2.md](history/deck-builder-v2.md) | 461 | The deck builder's design: bracket engine, role taxonomy, the architect ⇄ critic loop, and where the implementation departed from the plan. Moved 2026-08-19. |
| [history/frontend-v2.md](history/frontend-v2.md) | 330 | A proposed deck-building surface, superseded first by the dossier and then by the bench. Moved 2026-08-19. |
| [history/magazine-feedback-2026-08.md](history/magazine-feedback-2026-08.md) | 185 | Founder feedback on the magazine, verbatim, with what shipped against each thread. Moved 2026-08-19. |
| [history/magazine-feedback-2026-08-13.md](history/magazine-feedback-2026-08-13.md) | 255 | The next round. Its §0 shows the previous round's "still open" list becoming this round's complaints. Moved 2026-08-19. |
| [history/PLAN-2026-08-magazine-era.md](history/PLAN-2026-08-magazine-era.md) | 713 | The plan as it stood when nine magazine issues were the product, up to the pivot. |
| [history/PLAN.md](history/PLAN.md), [history/STYLE-v1-visual-research.md](history/STYLE-v1-visual-research.md), [history/STYLE-v2-editorial-method.md](history/STYLE-v2-editorial-method.md) | 12,385 | An older planning doc, the 1990s games-magazine visual research the design language came from, and an editorial-method treatise. Archive. |

## The files at the root that are not obviously docs

**[../CLAUDE.md](../CLAUDE.md)** is an instruction file for Claude Code, and also the
densest engineering knowledge here: paragraph-length post-mortems of real defects, each
with the measurement that settled it. If you want to know *why* something is the way it
is, look here first.

**[../PLAN.md](../PLAN.md)** is the resume-here doc: current state, open work, decisions
that bind, invariants that must not erode. It is candid about what is broken — that is
deliberate, and it is the fastest way to find something worth doing.
