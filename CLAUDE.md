# CLAUDE.md — Mana Map

**A workbench for crafting, experimenting, researching and analysing Commander decks**
(`docs/vision.md` is the page everything is written against), on top of an MTG card
embedding pipeline: ~35,000 oracle cards from Scryfall, two small neural nets (128-dim), a
2D projection, and an interactive card map served from `viz/`.

**The decision loop is the goldfish, paired.** An idea becomes an answer in about ten
seconds: `manamap pilot try <slug> --out A --in B` runs both lists through the seeded Monte
Carlo goldfish **game by game on the same seeds**, so the interval is on the difference and
the noise of two independent samples cancels. A swap that survives becomes a branch, and
`net-change` grades it on one pre-registered objective plus twelve Holm-corrected rows.
Since **2026-10-04 Forge is a targeted probe, never a gate**: `forge-cast-check` answers
"does the AI actually play this card", and a pod run is optional evidence whose loss is a
warning (`forge_warning`), not a block. A board can be lifted out of a Forge game
(`sim-scenario`) and proven with rules citations (`/resolve-stack`). Around that: a
deterministic builder, `deck-audit`'s 16 cited axes, `card-search` over the corpus, dated
`deck-recon`, versions from git, a captain's log, the pilot's keep list
(`protected.json`), and agents that turn a question into a priced, checked answer.

**PRD v2 (2026-10-07) is replacing the engine model, the handbook and the dossier** with one
living **Deck Context** per deck (`data/decks/<slug>/CONTEXT.md`, `docs/deck-context.md`) and
**Jarvis** (`/jarvis`), the one entry point that answers from it or routes to a small roster of
sub-agents with response-time targets. **Deck questions go through `/jarvis`.** Phase 1 is in;
nothing old is deleted until every deck's context is approved.

**Six pages over one data layer**: the landing page (`viz/workbench.html`), the card
atlas (`viz/index.html`), the deck page (`viz/deck.html?deck=<slug>`), the branch workbench
(`viz/branch.html`), Curate (`viz/library.html`) and the embedding-space appendix
(`viz/spaces.html`) — all rendering committed artifacts, figures with their intervals. The
Pilot's Operating Handbook (`src/manamap/pilot/poh.py`) renders `manuals/p/<slug>.html`; the
magazine before it was **deleted 2026-09-13** (record: `docs/gotchas-magazine-legacy.md`).
Runs locally on a Mac. **The pipeline and the pilot commands make zero LLM calls**; two
opt-in exceptions do — `serve.py`'s `ask` bridge (shells out to `claude -p`) and `mm ask`
(Sven Botstrom, an optional extra the core install does not pull).

## Layout

```
src/manamap/          # the Python package (pip install -e ".[dev]")
  config.py           # ALL constants: paths, hyperparams, tag patterns, synergy rules,
                      #   and the goldfish's AUTHORED RATES (seats, Tithe pay, Geyser)
  mechanical_tags.py  # regex tags — a FROZEN, model-facing vocabulary
  progress.py         # live .progress/<job>.json + heartbeat (regen, simulate, pytest) for
                      #   the Claude Code job band; display only, never a result
  pipeline.py, cli.py # the 15-step STEPS registry; `manamap` console script
  ingest/ training/ export/ analysis/   # the card pipeline (docs/pipeline.md)
  sim/                # Forge, OUTSIDE the repo engine: forge.py (seeded harness), parse.py
                      #   (logs -> facts -> aggregates + CIs), stats.py (exact power, no
                      #   scipy), engine_casts.py (did the AI cast it, or hold it),
                      #   bridge.py / board_series.py (a board out of a game), pods
                      #   and opponents (docs/simulation.md)
  pilot/              # the bench, one evidence contract across all of it
                      # shared:   card_pool.py (THE ONLY reader of cards.csv),
                      #           collection.py (THE ONLY reader of the boxes),
                      #           common.py (paths, hashing, lifecycle — has a file map),
                      #           registry.py (every subcommand), model_coverage.py
                      #           (what the goldfish CANNOT see), card_search.py
                      # BUILD:    build_deck.py, manabase.py, bracket.py, pool_facts.py
                      # MEASURE:  goldfish.py + goldfish_{profiles,library,turn}.py (the
                      #           seeded Monte Carlo; every channel opt-in per deck),
                      #           diagnostic.py (paired readings), mana_analysis.py
                      # DECIDE:   try_swap.py (`try`), net_change.py, deck_branch.py,
                      #           protected.py (the keep list), decisions ledger
                      # PROVE:    validate_stack.py (the citation contract), the rules
                      #           and strategy DBs, game_state.py, scenario_facts.py
                      # DIAGNOSE: deck_audit.py, deck_status.py, engine_facts.py,
                      #           validate_diagnosis.py, prescribe.py
                      # LOG:      deck_notes.py (append-only), deck_versions.py,
                      #           deck_state.py (THE ONE lifecycle writer), deck_delete.py
                      # PAGE:     deck_info.py (`--write` -> info.json), poh*.py,
                      #           deck_manifest.py (data/decks/index.json)
tests/                # two tiers; counts and runtimes ONLY in docs/testing.md
data/                 # artifacts; mostly gitignored, viz-served files tracked
  decks/<slug>/       #   one deck: decklist, cards.json, measurements, branches/, log
  pods/ opponents/    #   the named Forge tables and their seats (standard-v3 default)
  collection/         #   the PHYSICAL boxes (MANAMAP_COLLECTION_DIR overrides)
tools/claude-plugins/ # the job-band Claude Code plugin (live progress above the prompt), installed
                      #   from this folder at project scope; reads .progress/ (manamap.progress)
viz/                  # static frontend: the six pages above, d3 from CDN, window.MM;
                      #   index.html has three modes — discover / explore / build
                      #   (docs/viz.md)
docs/                 # docs/README.md indexes them, with line counts a test asserts
```

## Environment

- **Python 3.10** via conda `py310` → `.venv` in project root. PyTorch has NO wheels for 3.14.
- Install (macOS order matters — pacmap needs prebuilt numba wheels first), or `make setup`:
  ```bash
  .venv/bin/pip install llvmlite==0.41.1 numba==0.58.1
  .venv/bin/pip install -e ".[dev]"
  ```
- Training device: MPS → CUDA → CPU fallback.
- Version pins that matter: `sentence-transformers<4`, `numpy<2` (PyTorch 2.2.2 compat).

## Commands

`manamap pilot --help` lists all 139 pilot subcommands; `docs/pilot.md` is the reference.
The annotated block this file used to carry, measurements included, is kept verbatim at the
end of `docs/gotchas-bench.md`.

```bash
manamap run [--from STEP]     # the 15-step card pipeline; `manamap --help` lists all 28 top-level subcommands
manamap synergy && manamap power-creep && manamap cluster-regions && manamap card-roles
                              # analysis-only refresh, no retrain

# ── THE LOOP ──────────────────────────────────────────────────────────────
manamap pilot deck-info <slug>              # START HERE: where a deck stands and a derived NEXT
manamap pilot try <slug> --out "A" --in "B" [--out C --in D …] [--each] [--stage NAME]
                              # ~10 s, nothing written: every card in and out (roles, what the
                              # goldfish can SEE of it, what Forge's AI did with it — never a
                              # reason to cut), the keep list, colour sources before/after, the
                              # paired net-change rows, one line. `--stage` writes a branch.
manamap pilot deck-branch <slug> new|stage|propose|withdraw|reject|merge …
                              # a candidate 99; `propose <name> --as v1.2.0` accepts it and
                              # waits for cardboard. Decisions go in the append-only ledger.
manamap pilot net-change <slug> --branch <name> --write
                              # ONE primary (the objective) + twelve exploratory rows,
                              # Holm-corrected, paired per game. A Forge loss is a WARNING.
manamap pilot card-search --deck <slug> --oracle REGEX     # mine the corpus
manamap pilot prices <slug> [--branch B] --write            # the list's prices as DATED evidence (prices.json)
manamap pilot scan-candidates <slug> [--dimension …] --write  # one pass along the deck's axes
manamap pilot model-coverage <slug>         # what the goldfish cannot see: seen / DARK / invisible
data/decks/<slug>/protected.json            # THE PILOT'S KEEP LIST, hand-written only;
                              # stage/new/propose/merge, try, the builder and the diagnosis
                              # gates all refuse to cut a card it names (`validate-protected`)

# ── BUILD, LOG, VERSION ───────────────────────────────────────────────────
manamap pilot build <slug> --commander "<name>" [--brief "…"]   # brief -> a legal, measured 99 in ~10 s
manamap pilot check-in <slug> --from <file>  # a PAPER list -> decklist.txt; refuses a wrong one
manamap pilot deck-version <slug> [list|show|tag|restore|paper] # `paper` marks what is SLEEVED
manamap pilot deck-state <slug> archive|retire|supersede|revive --reason "…"
manamap pilot deck-notes <slug> add "…" --result win|loss --cause <code>   # the captain's log
manamap pilot regen [--only STAGE] [--slug S] [--jobs 8]
                              # rebuild the fleet after a MODEL change, in dependency order;
                              # then `make manuals`. Bit-identical across --jobs.

# ── FORGE: THE PROBE ──────────────────────────────────────────────────────
manamap pilot forge-cast-check <slug> --card "<name>"   # does the AI PLAY it? (two-seat shell)
manamap pilot forge-cast-check <slug> --branch B --adds --write   # every add, before any arm
manamap pilot simulate <slug> --pod standard-v3 --games N   # optional; prints its MDE first
manamap pilot experiment <slug> --a V1 --b working --pod <name> --games N [--looks K] [--aa]
manamap pilot sim-scenario <slug> <run> --game G --turn T --stack   # a board -> /resolve-stack

# ── AGENTS (Claude Code skills, 26 in .claude/skills/, 28 charters in .claude/agents/) ──
# /publish-deck sequences the lifecycle; /diagnose-deck /prescribe /resolve-stack
# /analyze-engine /build-deck /debrief /captains-log /sim-debrief /poh-procedures
# /write-manual /research-strategy /refresh-corpus /test-report (pre/post-test agents) /print-proxies
# /jarvis — THE ENTRY POINT (PRD v2): context-keeper writes CONTEXT.md, data-analyst computes (<30 s)
# /incubate — feedback -> incubation-pod proposes, challenger argues once -> data/queue.jsonl (`manamap pilot queue`)

make test                     # the UNIT tier: no tracked data, ~1 min (runtimes: docs/testing.md)
make regression               # the tracked fleet + corpus, every producer re-run
make integration              # browser + Forge + the pages rebuilt byte-identically
make prepush                  # before every push, BY AREA: docs 10 s, deck ~1 min, viz ~8, python ~10, full ~20
make check-deck SLUG=<slug>   # a deck's own checks while iterating on its list (~1 min)
make test-browser             # playwright; local only
make test-report              # unit tier + coverage, ~2 min -> data/test_reports/ (FULL=1: + regression)
.venv/bin/pytest -n0 -k NAME  # one test

manamap serve                 # viz + a LOCAL /api, and a WARM WORKER: read-only `manamap
                              # pilot` commands route through it and skip the cold start
                              # (query-rules 6.9 s -> 0.16 s). Fails open. Restart it after
                              # a code change. MANAMAP_NO_DAEMON=1 opts out.
python -m http.server 8000    # or plain static, FROM REPO ROOT
# http://localhost:8000/viz/workbench.html   THE LANDING PAGE — start here
```

`.mcp.json` registers a read-only MCP server (`manamap.mcp_server`): deck_state, fleet,
search_docs, search_code, stats, run_command, command_help — structured data from the warm
daemon. It cannot write.

## Gotchas

One line per rule; the measurement behind each is in the page it points at. CLAUDE.md's
previous full-length digest is kept verbatim at the end of `docs/gotchas-bench.md`.

### Pipeline, data and models

- **Frozen config**: changing `MECHANICAL_TAGS` (or any model-facing dim in `config.py`) invalidates `model_ability.pt` — retrain steps 3–5. `ROLE_PATTERNS` is separate because roles change often and tags must not; editing roles needs only `manamap card-roles` then `manamap viz-index`.
- **Roles ≠ mechanical tags**: tags say what a card is LIKE; roles say what JOB it does in a 99 (one `ramp` tag, five `ramp:*` roles — a Signet is not a Dark Ritual).
- **Index alignment**: `projection[i]` == `cards.csv[i]` == `embeddings[i]`. After the card count changes, re-run from the changed step onward, never partially.
- **No Git LFS on `data/`**: GitHub Pages serves LFS pointers, which would break the viz.
- **Two embedding spaces, two jobs**: `embeddings.npy` is LAYOUT (colour/type) and feeds the default map only; `embeddings_ability.npy` is FUNCTION and is the sole source of similarity, whichever map is shown.
- **The obsolescence index publishes a measure, not a verdict** (`strength` 0–1; it shipped as "Obsoleted By" with 36.5% of pairs failing). → `docs/gotchas-analysis.md`
- **A trigger pattern's `.*` sits where the subject noun lives** — "a Goblin you control dies" and "another creature dies" read the same. → `docs/gotchas-analysis.md`

### Evidence

- **A validator that fires on correct data is worse than none — measure it against the whole fleet first.** → `docs/gotchas-evidence.md`
- **Absent means ABSENT, never zero.** An unmeasured figure is a missing key with a reason. → `docs/gotchas-bench.md`
- **Every rate carries its interval; a comparison carries the interval on the DIFFERENCE; a family carries its correction** (net-change: one primary, twelve Holm-corrected rows). → `docs/gotchas-bench.md`
- **Pair what can be paired.** Every goldfish game has its own seed (`f"{seed}:{i}"`, harness v2) and two lists are aligned slot by slot, so `try` and `net-change` read a paired interval — roughly half the width of two independent samples. An A/A reads exactly zero. → `docs/simulation.md`
- **A mean is not a result.** Carry median, min and max beside it. → `docs/gotchas-bench.md`
- **Never `cache-record` to make a board green, and never hand-patch an agent's prose to pass a gate.** → `docs/gotchas-bench.md`
- **Every figure carries its definition in the report that prints it** (`net_change.METRICS` is the registry; a test holds it to `ROWS`). → `docs/pilot.md`
- **A measure computed from an authored file is not evidence**, however tight its interval; aim a branch at an OUTPUT. → `docs/gotchas-bench.md`
- **External prices are dated evidence in `prices.json`; agents quote it, never a live lookup** (`manamap pilot prices <slug> --write`; form gated, freshness never). → `docs/pilot.md`
- **An authored rate driving a figure must name where it came from** — the attack tutor read 5.70 fires a game where Forge resolved 1.22. The goldfish's authored rates live in `config.py` and `try` names them. → `docs/gotchas-bench.md`

### Forge, the probe

- **No Forge rate or mean is graded without an A/A at the same N.** The champion read its own life removed as 74.39, 48.77 and 59.53 across three samples; at 200/arm one list reads ±0.09 apart from itself (MDE ~0.14). → `docs/gotchas-bench.md`
- **A 100-game run is not a result** (18/100, then 50/400 on one list; MDE 42 points at 20/arm, 8.5 at 400). → `docs/gotchas-bench.md`
- **Unflagged is not castable — prove every add with `forge-cast-check` before an arm.** `AI:RemoveDeck:All` means the AI never casts the card; X costs, curse-less −X/−X and no-`AILogic$` shapes refuse for their own reasons. `data/forge_overrides/unflag.txt` lists the overrides. → `docs/gotchas-bench.md`
- **A Forge result on a deck whose engine the AI never cast is a FLOOR** — read `engine_casts` and HELD-WHILE-CASTABLE before any rate. → `docs/gotchas-bench.md`
- **The AI will not sacrifice for a benefit its evaluator cannot price, nor target its own commander for a copy** — such a run measures a different deck. → `docs/gotchas-bench.md`
- **A Forge record describes the list it PLAYED, not the list on disk** (`forge.list_mismatch`). → `docs/gotchas-bench.md`
- **A lifted board is an inference**; `validate-lift` re-lifts every committed board. → `docs/gotchas-bench.md`
- **A harness change can move the table** — an `ai` patch for our card can change how the pod plays. → `docs/gotchas-bench.md`

### The goldfish

- **It has no blockers and no removal, so its verdict on board QUALITY is not evidence** (a go-wide refactor it preferred lost 31/400 to 50/400 in Forge). Damage is measured against ONE seat at 40 life; three seats (`GOLDFISH_OPPONENTS`) only for what we gain off opponents' draws.
- **Read `docs/simulation.md`'s channel table before trusting a figure.** Every channel is opt-in per deck in `goldfish_targets.json`.
- **A card the model cannot read looks exactly like a card that does not help.** Check `model-coverage` before trusting a null; measure the ceiling with the omission reversed and LABEL it. → `docs/gotchas-bench.md`
- **A card can be read correctly and never cast** — teach the casting predicate in the SAME commit as the ability (`model_coverage.never_cast`). → `docs/gotchas-bench.md`
- **A commander's ability is not modelled until somebody models it** (Edgar's eminence, Zur's animate were both absent). → `docs/gotchas-bench.md`
- **A creature that may not attack does not attack** — Defender and "can't attack unless" gate it (`attack_gate`); Kefnet was a free 5/5 flyer every turn. → `docs/simulation.md`
- **A flag the model sets is a claim the model must ACT ON** (`tests/test_metric_hygiene.py`).
- **A one-turn grant is not a permanent doubler; a channel placed where it cannot fire is not a channel.** → `docs/gotchas-bench.md`
- **A model change makes every derived artifact stale.** `meta.model_version` hashes all four goldfish files BYTE FOR BYTE — a comment edit there means `manamap pilot regen --jobs 8 && make manuals`. → `docs/gotchas-bench.md`
- **The goldfish cannot rank two lands that make the same colours**; `mana-analysis` and `mana-fit` are the evidence for a land swap. → `docs/gotchas-bench.md`
- **Adding a metric requires re-running the independence check** (`pytest -m fleet`; one pair is over the line today — known-issues 9b).

### Changing a matcher

- **Widening a pattern needs a CORPUS SWEEP in the same commit** — newly matched, newly dropped, the extreme tail read card by card. → `docs/gotchas-bench.md`
- **A condition is scoped to the clause it attaches to** (Archway Commons read as an untapped five-colour land). → `docs/gotchas-bench.md`
- **A fetchland's colours are a property of the deck, not the card** — `land_colors(card, pool=…)`. → `docs/gotchas-bench.md`
- **A land whose only coloured mode costs extra is not a coloured source on curve** (`sources.gated`). → `docs/gotchas-bench.md`

### Branches, paths and artifacts

- **A branched write needs a branched READ**, and every branch measurement records the branch's own `decklist_sha256`. → `docs/gotchas-bench.md`
- **A branch may not declare a model channel its deck does not** — that is two simulators, not an A/B. → `docs/gotchas-bench.md`
- **`--out` on a per-deck command is slug-scoped**; a shell redirect cannot be policed. → `docs/gotchas-bench.md`
- **A new tracked artifact needs a gate in the same commit** — a validator, a freshness test, or both — and a `deck_status.VALIDATED` entry. → `docs/gotchas-evidence.md`
- **One predicate, one home.** `common.deck_is_apart` decides "is this deck a pile"; `regen.is_retired` and `regen.is_pinned` read it. → `docs/gotchas-bench.md`
- **A decided branch is not an experiment**: `deck_branch.branch_state` derives six states and stores none. → `docs/pilot.md`
- **Count COPIES, not decklist entries** — `common.expand_copies()`. → `docs/gotchas-bench.md`

### Tests

- **A test that re-derives the rule is testing itself.** Drive the production function and prove the test by re-introducing the bug. → `docs/testing.md`
- **Assert behaviour, not a pinned simulated figure**; a corpus count is a band (`assert_corpus_count`). → `docs/testing.md`
- **A loop over a possibly-empty collection needs `assert checked >= N`.**
- **A cache key missing an input serves a PASS the code never earned.** → `docs/testing.md`
- **Never edit source while the suite or `regen` is running** — the run reads a mix of both.

### The frontend

- **Cache-bust `?v=N` on every script and CSS tag in `viz/index.html` AND `viz/deck.html` after any JS/CSS change**; `index.html`'s thirteen busts (twelve scripts, one stylesheet) move together. Bump `DATA_VERSION` when a consumer would draw a DIFFERENT CONCLUSION from the bytes.
- **`viz/` and `data/` must stay top-level siblings**; every fetch is `../data/<file>`. Serve from the repo root.
- **A renderer kept behind a flag is a renderer nobody is testing.** → `docs/gotchas-viz.md`
- **Claude Code runs a CACHED copy of the job band, keyed on its version** — an edit to `tools/claude-plugins/job-band` without a `version` bump never runs (the SLA log stayed empty all of 2026-10-07). Bump it, `claude plugin marketplace update mana-map && claude plugin update job-band@mana-map`, restart; `manamap pilot sla-report` warns when the running band is stale.

### The full record, by subsystem

These do not load into every session; read the one that covers what you are about to touch.

| page | read before touching |
|---|---|
| `docs/gotchas-bench.md` | `src/manamap/pilot/`, `src/manamap/sim/` |
| `docs/gotchas-viz.md` | anything under `viz/` |
| `docs/gotchas-analysis.md` | `src/manamap/analysis/` |
| `docs/gotchas-evidence.md` | a validator, a citation, `engine.json` |
| `docs/gotchas-magazine-legacy.md` | nothing — the renderer is deleted; this is its record |

## Sleeved is built automatically; on the bench is triggered by hand

A deck with a **paper lock** (`deck_versions.paper`) is one the pilot plays, so `regen`
keeps its whole chain complete without being asked — measurements, then the agent
artifacts, then the handbook, then the dossier, in that order. A deck **on the bench**
changes daily and its stages run when the pilot asks; the dossier says per section what
is missing. `regen.is_pinned()` is the predicate. REFRESH is every live deck (an artifact
that exists is rebuilt wherever it lives); BOOTSTRAP — creating a missing one — is
sleeved decks only.

## Data artifacts

- **The deck manifest is generated**: `manamap pilot build-index` writes `data/decks/index.json`, which every page fetches; a test asserts it matches the artifacts.
- **`mana_analysis.json` is tracked and staleness-tested**: run `manamap pilot mana-analysis <slug>` AFTER `goldfish`, since it embeds goldfish figures.
- **CI** (`.github/workflows/test.yml`) runs the unit tier, its isolation proof and the regression tier on every push and PR, then `make manuals` and `git diff --exit-code -- manuals/ data/decks/` — the determinism claim asserted from outside the code. An `integration` job runs the browser suite on every push; Forge stays local. A weekly `corpus-gates` job downloads the corpus (the network leg) and runs the gates a push cannot reach.

## Pointers

- **`docs/vision.md`** — START HERE: what the bench is, the loop, the evidence contract, the vocabulary
- **`docs/simulation.md`** — both engines: the goldfish as the decision instrument and its channel table, then Forge
- **`docs/pilot.md`** — every command and artifact, the evidence and citation contracts
- **`docs/testing.md`** — the tiers, markers, the cache, how to write a test; the ONLY place counts and runtimes are stated
- **`docs/known-issues.md`** — what is wrong that no test fails on
- `docs/prd.md` — what is being built (Sept 2026); `PLAN.md` — current state and next steps
- `/publish-deck` — the deck lifecycle end to end, every phase with its gate
- `docs/architecture.md`, `docs/pipeline.md`, `docs/data-artifacts.md`, `docs/viz.md` — the card pipeline, every `data/` file, the frontend
- `docs/agent-inventory.md` — every agent and skill; read before touching a charter. `docs/agent-cost.md` — where LLM spend lives
- `docs/README.md` — indexes every doc, current reference above history
