# The agent harness, inventoried

*PRD §8 D-1: "Every agent, skill, prompt, and context file is listed with its
file path, invocation points, and dependencies. Each entry is classified keep,
repurpose, or retire, with a one-line reason. **The inventory is a checked-in
artifact, not a one-time report.**" Produced 2026-09-03; every claim below was
read off the tree rather than remembered.*

## The premise the PRD got wrong, and it makes Epic D smaller

PRD §2 says the sub-agent layer "carries editorial voice definitions, writer
teams, editor and coach roles, and department structures". **That layer was
retired on 2026-08-19** (`docs/agent-audit-2026-08-19.md`). `git log
--diff-filter=D -- .claude/agents` shows six charters already deleted:
`magazine-editor`, `manual-writer`, `pilot-coach`, `pilot-panel`,
`short-list-analyst`, `upgrade-scout`, plus the `design-issue` and `short-list`
skills. `pilot-notes` is the fold of the deleted writer and coach.

What is still frozen is the **Python renderer**, not the agent set — and
`build_index.py`, which `CLAUDE.md` listed under that heading until 2026-09-03,
is fully live and writes the manifest the whole frontend reads.

So Epic D is a **re-grouping of 17 charters**, not an excavation.

## Agents — 17 charters in `.claude/agents/`

Every one opens by reading `.claude/agents-common.md` (the shared contract);
`pipeline-runner` and `viz-dev` are exempt there by name.

| Agent | Owns | Spawned by | Class |
|---|---|---|---|
| `captains-log` | the six prose sections of `captains_log.json` | `captains-log` | keep → Build (piloting guidance) |
| `debrief` | `log_annotations.json` — a structured reading of each logged game | `debrief`, `captains-log`, `diagnose-deck`, `prescribe`, `publish-deck` | keep → Build |
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

### The two "orphans" are invocable, so they are not deletable

`pipeline-runner` and `viz-dev` are referenced only by their own files,
`agents-common.md`'s exemption list, `docs/agent-audit-2026-08-19.md` and
PLAN.md's counts. No skill spawns them and no code names them.

**That is not the same as dead.** Both are registered agent types and can be
invoked directly by name, which is a capability deleting them would remove —
and D-2 is explicit that "nothing gets deleted before its useful capability has
a new home". They are recorded here as *never spawned by a skill* and left in
place; retiring them is a decision, not a cleanup.

## Skills — 21 in `.claude/skills/`

Deck-facing, and the ones a consolidation has to re-home:

| Skill | Spawns | Note |
|---|---|---|
| `publish-deck` | debrief, deck-cartographer, deck-engineer, engine-critic, pilot-notes | **The router that already exists** — 13 ordered phases, and its own text says "None of them knew the sequence, and that is the failure this runbook exists to stop." Its only magazine coupling was phase 9 (`build-manual`) |
| `build-deck` | deck-analyst → deck-architect → deck-critic | the agent build loop, gated on `validate-build` / `bracket-check` |
| `diagnose-deck` | deck-doctor ⇄ deck-skeptic | |
| `prescribe` | deck-doctor (MODE prescribe) ⇄ deck-skeptic | |
| `analyze-engine` | deck-engineer ⇄ engine-critic | |
| `resolve-stack` | stack-resolver ⇄ rules-checker | max 3 iterations |
| `write-manual` | deck-analyst → strategy-researcher → pilot-notes | **its build half renders the frozen magazine** |
| `author-decision` | pilot-notes | **step 5 is `build-manual`** — the other magazine coupling |
| `debrief`, `captains-log`, `poh-procedures`, `research-strategy`, `strategy-lookup`, `rules-lookup`, `build-deck-db` | as named above | |

Infrastructure, not deck-facing and not part of the consolidation:
`run-pipeline`, `run-tests`, `retrain`, `refresh-corpus`, `regen-analysis`,
`serve-viz`.

## Front-end surfaces that depend on the harness

D-1's third clause. Every one reads a **committed artifact**, never an agent —
the Python makes zero LLM calls and the deployed site makes none either.

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

**Two magazine couplings block a clean retirement**, both in skills rather than
agents: `write-manual`'s build half and `author-decision`'s step 5 both call
`build-manual`. And `validate_issue.py` is the **only** validator that touches
`manual_prose.json`, so retiring it leaves the router's prose output ungated.

## Cost, per routine

`docs/agent-cost.md` carries the measured token counts and is the file to read
before planning a batch. The headline: the resolve loop is the outlier at
~570–600k for a full stack, a diagnosis is 200–300k, and the deterministic half
of the bench costs zero.
