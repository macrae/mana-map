# The documentation, sorted

There is a lot of it — about 19,000 lines — and roughly a third describes things
that were planned, superseded or abandoned. This page says which is which, so
you do not read a design record as a description of the code.

## Start here

| | | |
|---|---:|---|
| **[pipeline.md](pipeline.md)** | 43 | The 15 steps: command, inputs, outputs, runtime, when to re-run what. The shortest useful orientation in the repo. |
| **[data-artifacts.md](data-artifacts.md)** | 78 | Every file in `data/`: what produces it, how big it is, whether it is tracked, who reads it. Read before touching anything under `data/`. |
| **[testing.md](testing.md)** | 472 | How the suite is organised, the markers, and eight lessons about testing this thing. **The only place that states test counts.** |

## Reference, by subsystem

| | | |
|---|---:|---|
| [architecture.md](architecture.md) | 316 | The two embedding models, how a card is decomposed, tag and role taxonomies, synergy rules, power-creep criteria, region clustering. |
| [viz.md](viz.md) | 1,206 | The frontend: the three modes, the `window.MM` contract, the canvas renderer, the deck dossier, Pages deployment. The largest doc, and the one to read before any `viz/` change. |
| [pilot.md](pilot.md) | 769 | The magazine subsystem: the evidence contract, the citation contract, the rules and strategy RAG databases, the resolve loop, and what every per-deck artifact means. |
| [simulation.md](simulation.md) | 111 | Simulation design (S0) and the Forge spike: three criteria measured, verdict (Forge is the engine; we build the harness, parser and v2 bridge), tiers under sampling, what the LLM does and does not do, phases S1–S5. |
| [agent-audit-2026-08-19.md](agent-audit-2026-08-19.md) | 399 | The workbench pivot's audit of the agents (18 then, 16 after step 2): four fates (keep/fold/retire/new), per-agent strengths and enrichment, and the Sprint 0 order of work. Read before touching a charter. |
| [agent-cost.md](agent-cost.md) | 279 | Where LLM spend lives, per-routine token sizing, and how the invocation cache decides what to re-run. Only relevant if you are driving the agent phases. |

The magazine's editorial and design constitution is **[../STYLEv3.md](../STYLEv3.md)**
(1,116 lines). Its stated primary reader is an agent, but it is also the spec for
`build_manual.py`, `design.py` and `issue_spec.py` — read §5 before changing what
a department renders.

## Design records — history, not description

These document decisions and reasoning. Parts of them describe code that was
never written or has since been deleted. They are kept because the reasoning is
worth more than the accuracy, and they are **excluded from the docs guards** for
exactly that reason.

| | | |
|---|---:|---|
| [deck-builder-v2.md](deck-builder-v2.md) | 459 | The deck builder's design: bracket engine, role taxonomy, the architect ⇄ critic loop, and where the implementation departed from the plan. |
| [frontend-v2.md](frontend-v2.md) | 328 | A proposed deck-building surface, **largely superseded 2026-07-31**. Its own header says so. |
| [magazine-feedback-2026-08.md](magazine-feedback-2026-08.md) | 183 | Editorial feedback, verbatim, with what shipped against each thread. |
| [magazine-feedback-2026-08-13.md](magazine-feedback-2026-08-13.md) | 253 | The next round. Its §0 shows the previous round's "still open" list becoming this round's complaints, which is the most useful thing in either file. |
| [history/](history/) | 12,385 | An outdated planning doc, the 1990s games-magazine visual research the design language came from, and an editorial-method treatise on how good educational magazines teach. Archive. |

## The two files at the root that are not obviously docs

**[../CLAUDE.md](../CLAUDE.md)** is an instruction file for Claude Code, and it
is also the densest engineering knowledge here: roughly a hundred
paragraph-length post-mortems of real defects, each with the measurement that
settled it. If you want to know *why* something is the way it is, look here
first. The filename is misleading about its audience.

**[../PLAN.md](../PLAN.md)** is the resume-here doc: current state, open work,
decisions that bind, invariants that must not erode. It is candid about what is
broken — that is deliberate, and it is the fastest way to find something worth
doing.
