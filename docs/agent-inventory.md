# The agent harness, inventoried

*PRD §8 D-1: "Every agent, skill, prompt, and context file is listed with its
file path, invocation points, and dependencies. Each entry is classified keep,
repurpose, or retire, with a one-line reason. **The inventory is a checked-in
artifact, not a one-time report.**" Produced 2026-09-03; every claim below was
read off the tree rather than remembered.*

## The premise the PRD got wrong, and it makes Epic D smaller

PRD §2 says the sub-agent layer "carries editorial voice definitions, writer
teams, editor and coach roles, and department structures". **That layer was
retired on 2026-08-19** (`docs/history/agent-audit-2026-08-19.md`). `git log
--diff-filter=D -- .claude/agents` shows six charters already deleted:
`magazine-editor`, `manual-writer`, `pilot-coach`, `pilot-panel`,
`short-list-analyst`, `upgrade-scout`, plus the `design-issue` and `short-list`
skills. `pilot-notes` is the fold of the deleted writer and coach.

The **Python renderer** was frozen at the time and was deleted on 2026-09-13; the
manifest half of `build_index.py` lives on as `deck_manifest.py`, which writes the
`data/decks/index.json` the whole frontend reads.

So Epic D was a **re-grouping of 18 charters**, not an excavation.

## Agents — 24 charters in `.claude/agents/`

Every one opens by reading `.claude/agents-common.md` (the shared contract);
`pipeline-runner` and `viz-dev` are exempt there by name.

| Agent | Owns | Spawned by | Class |
|---|---|---|---|
| `captains-log` | `read` (the deck-level roll-up) + the night summaries of `captains_log.json` | `captains-log` | keep → Build (piloting guidance) |
| `debrief` | `log_annotations.json` — a structured reading of each logged game | `debrief`, `captains-log`, `diagnose-deck`, `prescribe`, `publish-deck` | keep → Build |
| `sim-debrief` | the `prose` of `sim_findings.json` — a reading of each simulated run that may cite only the computed findings' ids and quote only their figures; the captain's log is never touched (added 2026-09-30) | `sim-debrief` | keep → Build (the simulated counterpart of `debrief`) |
| `deck-analyst` | `candidate_pool.json` | `build-deck`, `write-manual` | keep → shared service under Auto-Build |
| `deck-architect` | `build_plan.json` | `build-deck` | keep → Auto-Build |
| `deck-critic` | adversarial verifier for build plans | `build-deck` | keep → Auto-Build's verify loop |
| `deck-doctor` | `deck_recon.json`, `diagnosis.json`, `prescriptions/`, branch objectives — four modes, 421 lines, the largest charter | `diagnose-deck`, `prescribe` | keep → Scout (recon) + Build (diagnose) |
| `deck-engineer` | `engine.json` | `analyze-engine`, `publish-deck` | keep → Build |
| `engine-critic` | adversarial verifier for `engine.json` | `analyze-engine`, `publish-deck` | keep → Build's verify loop |
| `deck-skeptic` | adversarial verifier for diagnoses and prescriptions | `diagnose-deck`, `prescribe` | keep → Build's verify loop |
| `stack-resolver` | cited stack resolutions | `resolve-stack` | **keep unchanged → Stack** |
| `rules-checker` | adversarial verifier for resolutions | `resolve-stack`, `rules-lookup` | **keep unchanged → Stack** |
| `strategy-researcher` | `data/strategy/strategy.md` + `strategic_frame.json`. The ONLY agent with Write/Edit scope | `research-strategy`, `strategy-lookup`, `write-manual` | repurpose → Scout |
| `pilot-notes` | five keys of `manual_prose.json`, `decisions/`, `tutor_guide.json`. The fold of the deleted writer + coach | `author-decision`, `publish-deck`, `refresh-corpus`, `write-manual` | keep, magazine-descended → Build |
| `poh-procedures` | `poh_procedures.json` — the handbook's authored half | `poh-procedures` | keep → Build |
| `deck-cartographer` | city names on `deck_map.json`, names only | `publish-deck` | ambiguous — demoted to OPTIONAL by the 2026-08-19 audit; not a `deck_status.STAGES` row |
| `pipeline-runner` | runs pipeline steps via the CLI | **no skill spawns it** | see below |
| `viz-dev` | frontend work under `viz/` | **no skill spawns it** | see below |
| `context-keeper` | the prose of `CONTEXT.md` — Summary, How it plays, Cards by role, Pilot notes, Open questions — through `context --install`'s strict gate; never a number, never a generated block. Modes seed / log / deck-change. `sla_s: 120`, Sonnet (PRD v2, 2026-10-07) | `jarvis` | **PRD v2 roster: Context Keeper** — will absorb `captains-log` and `debrief` |
| `data-analyst` | nothing — filters, sorts and computes over the 99, the goldfish, the logs with read-only commands; answers in ≤10 lines plus the commands it ran. `sla_s: 30`, Haiku (PRD v2) | `jarvis` | **PRD v2 roster: Data Analyst** |
| `incubation-pod` | nothing tracked — at most three testable hypotheses per run, applied to `data/queue.jsonl` by `queue apply`; modes incubate / rebut. `sla_s: 180`, Sonnet, web (PRD v2 Phase 2) | `incubate` | **PRD v2 roster: Incubation Pod** |
| `challenger` | nothing tracked — one round of promote / revise / drop per hypothesis, settled by `queue apply`. `sla_s: 90`, Sonnet (PRD v2 Phase 2) | `incubate` | **PRD v2 roster: Challenger** |
| `test-preflight` | nothing — a GO / WAIT / FIX verdict over `suite_report preflight` (collisions, plugins, dirty source, harness changes since the last report); exempt from `agents-common.md` (added 2026-10-05) | `test-report` | infrastructure |
| `test-debrief` | nothing tracked — a reading of `suite_report diff` that may cite only finding ids and quote only their figures, plus a proposed diff to docs/testing.md's hand-written table; the generated block is code's (added 2026-10-05) | `test-report` | infrastructure (the test-suite counterpart of `sim-debrief`) |

### The two "orphans" are invocable, so they are not deletable

`pipeline-runner` and `viz-dev` are referenced only by their own files,
`agents-common.md`'s exemption list, `docs/history/agent-audit-2026-08-19.md` and
PLAN.md's counts. No skill spawns them and no code names them.

**That is not the same as dead.** Both are registered agent types and can be
invoked directly by name, which is a capability deleting them would remove —
and D-2 is explicit that "nothing gets deleted before its useful capability has
a new home". They are recorded here as *never spawned by a skill* and left in
place; retiring them is a decision, not a cleanup.

## Skills — 26 in `.claude/skills/`

Deck-facing, and the ones a consolidation has to re-home:

| Skill | Spawns | Note |
|---|---|---|
| `publish-deck` | debrief, deck-cartographer, deck-engineer, engine-critic, pilot-notes | **The router that already exists** — 13 ordered phases, and its own text says "None of them knew the sequence, and that is the failure this runbook exists to stop." Its only magazine coupling was phase 9 (`build-manual`), now `build-poh` |
| `build-deck` | deck-analyst → deck-architect → deck-critic | the agent build loop, gated on `validate-build` / `bracket-check` |
| `diagnose-deck` | deck-doctor ⇄ deck-skeptic | |
| `prescribe` | deck-doctor (MODE prescribe) ⇄ deck-skeptic | |
| `analyze-engine` | deck-engineer ⇄ engine-critic | |
| `resolve-stack` | stack-resolver ⇄ rules-checker | max 3 iterations |
| `write-manual` | deck-analyst → strategy-researcher → pilot-notes | its build half renders the handbook (`build-poh`) since the magazine was deleted |
| `author-decision` | pilot-notes | step 5 rebuilds the handbook (`build-poh`) |
| `debrief`, `captains-log`, `poh-procedures`, `research-strategy`, `strategy-lookup`, `rules-lookup`, `build-deck-db` | as named above | |
| `sim-debrief` | sim-debrief | `sim-findings --write` → spawn for the runs with no prose → `merge-sim-findings` (recomputes the skeleton, takes the prose, refuses what does not hold) → `validate-sim-findings` → route the open questions |

Infrastructure, not deck-facing and not part of the consolidation:
`run-pipeline`, `run-tests`, `retrain`, `refresh-corpus`, `regen-analysis`,
`serve-viz`, `print-proxies` (added 2026-10-06: a proxy sheet for a branch's adds minus what the pilot holds, via `manamap pilot proxies`), and `test-report` (added 2026-10-05), which spawns `test-preflight` →
`make test-report` → `test-debrief` over the findings `manamap.suite_report` computes.

## PRD v2 (2026-10-07): Jarvis and the roster

`/jarvis` (`.claude/skills/jarvis/SKILL.md`) is the one entry point: it reads the deck's
`CONTEXT.md` first (`docs/deck-context.md`), answers from it, and routes the rest. The PRD's
seven sub-agents each have one job and a response-time target (`sla_s:` in the charter, shown
live by the job band, logged to `.progress/sla-log.jsonl`, read by `manamap pilot sla-report`).
Phase 1 shipped `data-analyst` and `context-keeper`; Phase 2 added `incubation-pod` and `challenger` with the queue (`docs/queue.md`). Card Scout (from `deck-analyst`),
Strategist (from `deck-doctor` + `strategy-researcher`), Rules Checker (single question, from
`rules-lookup`) and the Incubation Pod + Challenger are Phase 2; Scenario Sim is Phase 3.
Nothing in the tables above is deleted until every deck's context is approved.

## Front-end surfaces that depend on the harness

D-1's third clause. Every one reads a **committed artifact**, never an agent —
the pipeline and pilot commands make zero LLM calls and the deployed site makes
none either. The two local exceptions — `serve.py`'s `ask` bridge and `mm ask` —
are opt-in and never run on the deployed site.

| Surface | Artifacts it needs | Agents behind them |
|---|---|---|
| `viz/workbench.html` | every `info.json`, `data/decks/index.json` | diagnosis, engine, prescriptions (counts only) |
| `viz/deck.html` | `info.json` + the per-deck artifacts | deck-doctor, deck-engineer, pilot-notes, captains-log, debrief |
| `viz/branch.html` | `branch.json`, `net_change.json` | none — both deterministic |
| `viz/index.html` | the corpus artifacts | none |
| `manuals/p/<slug>.html` | `poh_procedures.json` | poh-procedures |

`serve.py`'s `ask` endpoint is the one exception in the whole repo: it shells out
to `claude -p`, deliberately, so the local Build page can ask for an agent.

## What a five-specialist consolidation actually has to move

- **Stack** — `stack-resolver` + `rules-checker`, unchanged capability. The
  cleanest lift in the set.
- **Scout** — `strategy-researcher` (MODE research) + `deck-doctor` (MODE recon).
- **Auto-Build** — `deck-analyst` + `deck-architect` + `deck-critic`.
- **Build** — `deck-doctor` (diagnose/prescribe) + `deck-skeptic` +
  `deck-engineer` + `engine-critic` + `pilot-notes` + `poh-procedures` +
  `debrief` + `captains-log`. The largest bucket by far, and the one where
  "piloting guidance generation" lands.
- **Spen** — `publish-deck` is the router today and knows the sequence.

**The two magazine couplings are gone** (2026-09-13): `write-manual`'s build half and
`author-decision`'s step 5 called `build-manual`, and both call `build-poh` now.
`validate_poh.py` is still the **only** validator that touches `manual_prose.json`,
so retiring the handbook would leave the router's prose output ungated.

## Cost, per routine

`docs/agent-cost.md` carries the measured token counts and is the file to read
before planning a batch. The headline: the resolve loop is the outlier at
~570–600k for a full stack, a diagnosis is 200–300k, and the deterministic half
of the bench costs zero.
