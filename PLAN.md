# PLAN — current state and what's next

*The resume-here doc. `docs/vision.md` says what this is for; `CLAUDE.md` carries the
gotchas; this says what exists and what is open. The magazine era's plan is archived
verbatim at `docs/history/PLAN-2026-08-magazine-era.md`.*

Last updated **2026-08-19**. Everything below is committed and pushed to `main` except
where marked. Every figure was derived from the repo at write time — **do not quote one
from memory**; the command that prints it is named beside it.

## What this is

A **lab bench for one pilot's paper Commander decks** — versions, a captain's log, stats,
a seeded goldfish, a real-rules simulator against the pilot's own table, and a set of
agents that turn a question into a priced, checked answer. Optimised for one player
(the maintainer, in Orinda); open-sourced, not externally supported. The magazine that
used to be the product is a frozen legacy renderer until the compact deck page replaces
it. The card atlas in `viz/` is unchanged and live.

Scale (derived; `tests/test_docs_counts.py` polices these): 63 `manamap pilot`
subcommands, 18 top-level subcommands, 15 agents, 19 skills, 10 static cache routines
(plus `stack:`/`decision:`/`prescription:` per artifact). Test counts live in
`docs/testing.md` only.

## Start here on any deck

```bash
manamap pilot deck-info <slug>        # where it stands, and a derived NEXT
manamap pilot deck-status <slug>      # lifecycle: present / missing / STALE / INVALID, gates run
```

Staleness is an ERROR; incompleteness is a state. A half-built deck is work in progress.
A deck whose artifacts disagree about which decklist they describe is confident and wrong.

## The evidence contract

| | tier | granted by |
|---|---|---|
| ✓ | rules-verified | a stack artifact whose every step cites a real CR rule verbatim (`validate_stack`), then survives the adversarial `rules-checker`. Only a `pass` publishes. |
| ◆ | data-derived | deterministic Python over committed artifacts; **seeded** where randomness is involved (goldfish, Forge runs), **sampled** said out loud where it was not (the first Forge run). |
| ★ | coaching | labelled judgment. Useful, never disguised as measurement. |

Agents return JSON and validators check it; no agent writes prose that claims a tier it
was not granted; a figure travels with its interval, its N and its limits.

## The decks

`deck-info <slug>` per deck for the live picture; `deck-status --all` for the fleet. As of
this writing (derived from the stack, bracket, engine, sim and log artifacts):

| deck | stacks ✓/total | bracket floor/target | engine | sim runs | logged games | status |
|---|---|---|---|---|---|---|
| `goblin-storm` | 5/5 | 4/4 | pass | 0 | 0 | |
| `hapatra` | 1/1 | 4/4 | pass | 0 | 0 | `broken-down` (cards live in yawgmoth) |
| `sisay` | 1/3 | 4/4 | pass | 0 | 0 | `retired` |
| `heliod` | 6/6 | 4/4 | pass | 0 | 0 | |
| `ur-dragon` | 6/6 | 4/4 | pass | 0 | 0 | |
| `edgar-vampires` | 9/9 (7 presentable) | 4/4 | pass | 0 | 0 | paper rebuild in progress (v4 LOCK) |
| `gishath` | 5/5 | 4/4 | pass | 0 | 0 | |
| `yawgmoth-swarm` | 14/14 (11 presentable) | 4/4 | pass | 0 | 0 | paper rebuild in progress |
| `radagast` | **8/8** | 1/3 | pass | **2** | 0 | `broken-down` (2026-08-21; cards pulled for the next build) |

**Nothing is logged on any deck yet.** The log, debrief, prescriptions and versions are
built and tested with zero real entries — the first one is the pilot's to write, and it
will teach the agents more than another sprint would.

Three decks no longer exist as cardboard (`hapatra`, `sisay`, `radagast`). Their
artifacts stay exactly as published; `deck-info` states the status and withholds the
suggestions that would need a deck to shuffle.

## Where the pivot stands (2026-08-19)

| | | where |
|---|---|---|
| Agent audit + Sprint 0 | 18 → 15 agents; shared contract (`.claude/agents-common.md`); L10 repealed; magazine editor/panel/short-list retired; writer + coach → `pilot-notes`; `debrief` new; doctor MODE prescribe | `docs/agent-audit-2026-08-19.md` |
| MVP Sprints 1–3 | `deck-version`, `deck-notes` + `/debrief`, `prescribe`, `deck-info` | `docs/pilot.md` |
| Simulation S0–S5 | Forge spike + verdict; seeded harness `simulate`; parser with CIs; `validate-sim`; the pod (`fetch-opponent`, `data/opponents/`); the bridge `sim-scenario` → `game_state` v2 (`validate-stack`/`scenario-facts` read v2); the doctor reads the table | `docs/simulation.md` |
| The chain, once for real | stack 008 — a board lifted from game 1 of the first run, resolved + checker-passed in 3 iterations; matched Forge's log line for line | `docs/simulation.md` |
| Compact deck page | spec drafted, awaiting the pilot's strikes | `docs/manual-v5-spec.md` (branch `manual-v5` for the work) |

## Open work

### Next, in order

1. ~~`experiment`~~ — **DONE 2026-08-19**: `experiment <slug> --a <ref> --b <ref> --vs …`,
   one accumulating artifact with both arms, the delta and the overlap sentence; arms never
   touch the deck dir; `--profile` on it and on `simulate` (measured: aggro profiles make a
   hold-up deck worse; Default stays default). First tracked one: radagast V1 vs V5 —
   win-rate delta is noise, damage +27.6/game and token share 0 → 0.19 are not.
2. **The first real logged games** — any deck, `deck-notes add`, then `/debrief` and
   `/prescribe`. Only the pilot can do this.
3. ~~**Forge AI profiles**~~ — **DONE 2026-08-19**, alongside `experiment`: `-a` is per-seat
   in `-d` order and `--profile` rides on both commands. Measured on radagast's seat vs a
   Default edgar, 6 seeded games each — Default 3/6, Experimental 2/6, Reckless 2/6. No
   profile flies a hold-up deck better than Default, so Default stays the default and the
   AI caveat stands unchanged. That is the evidence for 5.
4. **The deck page in the viz** — notes, versions (a deploy-time JSON, since the version
   list cannot be committed in the commit that creates it), sim, prescriptions, the manual
   as a tab. The magazine gets simplified *into* this (`docs/manual-v5-spec.md`), not rewritten.
5. **An agent in the pilot's seat** for a handful of seeded games on one question — only
   once 1–3 show the AI's play is the thing limiting the measurement.
6. ~~`card-search`~~ — **DONE 2026-08-21**: deterministic card mining over `cards.csv`
   (identity derived from a deck's commander, oracle/name regex, role, type, cmc,
   Game-Changer filter), the deck's own cards excluded by default. Built because
   kianne's audit ended in "which cards fix this" and nothing could answer it. Neighbour
   search is NOT in it — that would be a second retrieval opinion beside the synergy
   graph. The audit's item 8.

### Known gaps, named in the artifacts

- **Forge's AI pilots the deck** — "poor to ok in control, pretty bad for combo" (its own
  words, quoted in every run record). radagast 0/20 vs the pod at 12.5 combat damage a
  game against 45.6 among its stablemates. A lower bound on the pilot; a true picture of
  the table's clock.
- **Bridge approximations**: token types unknown from the log (tokens filed as
  `other_permanents` by `scenario-facts`), hand sizes are estimates, continuous effects
  (a Craterhoof pump) are not tracked and must be authored into the scenario. Every one is
  written into `extras.reconstruction_notes`.
- **Parser**: damage figures see damage only; drain kills show in `life_by_turn` and
  `eliminated_how`, never in a damage total. **Commander damage is now measured**
  (2026-08-21) — per DEFENDER, because CR 903.10a asks for 21 from one commander on one
  player and `combat_damage_dealt_to_players` sums every source and every seat at once.
  The commander names ride IN the record (`seats[].commander`), never looked up from
  disk at validate time; a record without the field re-derives exactly as before, and
  `simulate <slug> --analyze <run>` is the migration.
- ~~**The deterministic builder cannot produce a curve SHAPE.**~~ **FIXED 2026-08-22.**
  It scored every card independently and took the top N while `curve_fit` penalised
  each point above `DECK_CURVE_SWEET_SPOT = 3`, so the top N were always cheap: the
  first kinnan baseline was 64 nonland cards with curve `{0:1, 1:11, 2:28, 3:24}`,
  **nothing above mana value 3**, and 29 of them mana producers — a legal deck that
  ramps into nothing, which `validate_build` passed because it checks form. The role
  quota in `fill_slots` is now crossed with a mana-value quota derived from
  `DECK_AXIS_TARGETS["curve"]`, the target `deck_audit` already measured against and
  the builder never read, so no new uncited constant. Rebuilt: `{0:1, 1:9, 2:15,
  3:16, 4:10, 5:6, 6:4, 7:1, 8:1}`. Combo blindness went with it — `complete_combos`
  reads real lines from `combo_details` (not the flat `combo_partners` map, which
  cannot tell a completion from a coincidence) and swaps in the one missing card of a
  line the deck half-holds. kinnan went from 23 partners and 0 completions to **4
  contained combos and 2 two-card infinites**, including Kinnan + Pili-Pala +
  Enduring Vitality. And `build()` now has end-to-end tests: there were none.
- **DFC pips**: `manabase.pip_requirements` reads `card["mana_cost"]`, empty for
  transform/MDFC layouts; 10 spells on 7 decks. Three-line fix that changes every
  `mana_analysis.json` — deliberately not done mid-flight.
- **hapatra's `bracket_report.json`** contradicts a verified stack (an inflated two-card-
  infinite count). **`build_plan.json` is not reproducible** from today's data.

### Verification backlog (✓ work)

- **Sisay 001** (the tutor chain) — highest-value promotion from ★ to ✓ in the fleet.
- **Grafdigger's Cage** — three of yawgmoth's kills rest on an oracle reading no checker has settled.
- **Hapatra's Mikaeus +1/+1 anthem** vs its token loops — if it switches them off, its two engines are mutually exclusive.
- **Radagast's `open_questions`** — six from the engineer; only Craterhoof has a stack. **Boards for these can now be lifted from sims** (`sim-scenario`).
- Queued: Roaming Throne × Zada; the Past in Flames rebuild; sisay's other Najeela pairs.

### Still owed

- `deck-recon` on six decks (time-based staleness, ~600k).
- Strategy-DB gaps — 49 across the frames; aristocrats/sacrifice first.
- Unit tests for `merge_deck_map`, `engine_facts`; no regression floor on the balance
  bound's effect on real decks.
- Ur-Dragon two-engine rebuild — proposed, measured, not applied (`ur-dragon-deck` memory).
- `deck-history pending` should read open prescriptions' adds beside the legacy
  `considering.json`.
- Versioning: `deck-version` replaced the hand-kept `HISTORY.md`; the `supersedes` pointer
  for a deck page's status is still open.

### Legacy, frozen — and what unfreezes it

The magazine renderer (`build_manual`, `issue_spec`, `design`, `validate_issue`, STYLEv3)
still renders the nine pages from artifacts nothing regenerates (`issue_plan.json`, the
panel keys, `card_roles`/`mana_base`/`upgrades`, `considering.json`). They are edited by
nobody and deleted in one commit when the compact page lands (`docs/manual-v5-spec.md`
§"What gets unfrozen"). Code and docs about it carry a LEGACY banner and are otherwise
left accurate.

## Decisions that bind

### The frontend stays LLM-free
The deployed site and the local checkout run the same code. The agent loop stays in
Claude Code, reached by commands and a brief. No local bridge.

### Forge is the engine; we build the harness, the parser and the bridge
Measured before chosen: (a) every log line parses, (b) 4-seat Commander runs, (c) `-s`
makes it byte-replayable. Writing our own rules engine is shelved for one narrow
deterministic case. The goldfish stays as the seeded resource model.

### The cache board is red fleet-wide, and deliberately not re-recorded
The shared-contract commit MISSed every routine; charter edits disqualify STALE_OK by
construction. Artifacts are gated by validators and tests, not by the cache; each routine
clears on its next real spawn. Never `cache-record` to make a board green.

### Versions are derived from git and never committed; tags are authored
A commit's sha is unknown inside the commit, so a generated version list would be one
behind forever. `deck_versions.json` (tags) is the one version datum a browser can read.

### A prescription accumulates; stale is not wrong
One file per question, keyed by the prompt's hash; an older-decklist prescription is
form-checked only.

### Similarity comes from the function space, always
`embeddings_ability.npy` is the sole source of similarity (Find Similar, the walk, drill,
the deck map). `embeddings.npy` feeds `projection_2d.json` only. Do not tune on the golden set.

### Synergy is complementary, not similar
24 rules over mechanical tags, ranked by playability. "Anti-cards" do not exist.

### The clusters are an input to engine analysis, never the analysis
A city name is the wrong address for a component; a disagreement between the map and the
engine is a finding.

## Invariants that must not erode

- Only checker-passed stacks publish; failed artifacts are kept as open questions.
- Agents return JSON and never write HTML.
- `issue.json`, `log.jsonl`, `deck_versions.json`, a prescription's prompt: **authored**,
  never regenerated. A derived artifact may be regenerated; an authored one may not be rewritten.
- Costume never earns the badge.
- Record the cache **after** validation, never before; never record a `fail`.
- Charter edits invalidate before they inform — make them **before** `cache-record`.
- A bracket **floor** is what the contents are consistent with, never a verdict.
- The deterministic builder must always produce a complete legal 99 with no agent involved.
- **Count copies, not decklist entries**, for anything the shuffler would see.
- `--out` on a per-deck command must be slug-scoped.
- **A validator that fires on correct data is worse than none** — measure a proposed check
  against the whole fleet before keeping it.
- **A critic's findings become mechanical checks**, or its work is re-spent every run.
- **Name what a gate cannot see** rather than papering it with string matching.
- A sim figure travels with its interval, its N and its limits — or it does not travel.
