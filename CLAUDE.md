# CLAUDE.md — Mana Map

**A workbench for crafting, experimenting, researching and analysing Commander decks**
(`docs/vision.md` is the page everything is written against), on top of an MTG card
embedding pipeline: ~34,900 oracle cards from Scryfall, two small neural nets (128-dim), a
2D projection, and an interactive card map served from `viz/`.

**Simulation is the centre.** Forge runs headless and **seeded** against the pilot's own
pod (`simulate`), `experiment` is the controlled A/B of two versions on one table, and a
seeded Monte Carlo goldfish answers the questions that are about a curve rather than a
table. A board can be lifted out of a simulated game (`sim-scenario`) and proven with rules
citations (`/resolve-stack`). Around that: a deterministic builder, `deck-audit`'s 16 cited
axes, `card-search` over the corpus, dated `deck-recon`, versions from git, a captain's log,
and agents that turn a question into a priced, checked answer.

**Four pages over one data layer**: the landing page (`viz/workbench.html`), the card
atlas (`viz/index.html`), the **deck page** (`viz/deck.html?deck=<slug>`) and the branch
workbench (`viz/branch.html`) — all rendering committed artifacts, with sim figures that
carry their intervals. The magazine that used to be the product is a
**frozen legacy renderer** until the compact deck page (`docs/manual-v5-spec.md`) replaces
it. Runs locally on a Mac; the Python makes zero LLM calls.

## Layout

```
src/manamap/          # the Python package (pip install -e ".[dev]")
  config.py           # ALL constants: paths, hyperparams, tag patterns, synergy rules
  mechanical_tags.py  # regex tag extraction (shared by ingest + analysis)
  pipeline.py         # ordered STEPS registry + runner
  cli.py              # `manamap` console script
  ingest/             # download, extract, preprocess, download_combos, process_combos
  training/           # model, train, train_ability, embed, common
  export/             # reduce (PaCMAP), export_embeddings (.bin for JS),
                      #   viz_index (discovery index + neighbours.bin)
  sim/                # Forge harness (forge.py): .dck from decklist, N games across
                      #   stats.py: Newcombe/Welch/permutation/bootstrap + EXACT power
                      #   (no scipy); threat.py: who the pod attacks (opponent modelling)
                      #   JVMs; parse.py: logs → events → facts → aggregates + CIs;
                      #   validate_sim.py re-proves a record against its logs;
                      #   bridge.py lifts a board into a game_state v2 scenario;
                      #   opponents.py fetches a pod seat from EDHREC.
                      #   ◆ SEEDED run records under data/decks/<slug>/sim/
                      #   (docs/simulation.md — the engine lives OUTSIDE the repo)
  analysis/           # synergy, power_creep, cluster_regions, card_roles,
                      #   eval_embeddings (step 15, the quality gate), common
  pilot/              # the bench: BUILD / PROVE+MEASURE / PAGE (legacy) / DIAGNOSE / LOG+ASK —
                      # one evidence contract across all of it
                      # ---- shared ----
                      #   card_pool.py     THE ONLY reader of cards.csv; one parse,
                      #                    several views (frame/pool/flags/names/oracle)
                      #   collection.py    THE ONLY reader of COLLECTION_DIR; "do I own
                      #                    this" = a BOX, never deck membership
                      #   card_search.py   deterministic corpus mining: identity, oracle/
                      #                    name regex, role, cmc, --owned/--unowned
                      #   common.py        paths, memos, DFC faces, citation ids,
                      #                    resolve_out_path, the validator CLI tail
                      #   registry.py      subcommand table + argparse wiring
                      # ---- BUILD — a brief -> a legal, tier-conditioned 99 ----
                      #   pool_facts.py    a BOX OF CARDS -> which deck to build
                      #   build_deck.py    pool -> score -> fill -> enforce_bracket
                      #   manabase.py      hypergeometric colour-source math
                      #   bracket.py       computed bracket floor + evidence
                      #   validate_build.py
                      # ---- PROVE + MEASURE — a deck -> evidence ----
                      #   check_in.py      a PAPER list -> decklist.txt; refuses a
                      #                    silently-wrong list rather than guessing
                      #   fetch_deck.py    decklist -> cards.json (Scryfall, printings)
                      #   validate_deck.py 100 / commander / singleton / identity
                      #   validate_recon.py  deck_recon.json: cards real, legal, in
                      #                    identity; ownership falsified against the boxes
                      #   download_rules / build_rules_db / query_rules  CR + RAG
                      #   build_strategy_db / query_strategy
                      #   validate_strategic_frame.py  frame form + line flags
                      #   validate_stack.py   the citation contract
                      #   goldfish.py      seeded Monte Carlo (resource development;
                      #                    Treasure + one-opponent combat opt-in)
                      #   mana_analysis.py colour sources / castability — deterministic
                      #   game_state.py    the game_state v2 vocabulary + form check
                      #   merge_prose.py   pilot-notes' five keys in; frozen legacy keys untouched
                      #   agent_cache.py / impact.py / card_refs.py  incremental regen
                      #   validate_tutor_guide / validate_strategy
                      # ---- PAGE ----
                      #   poh.py / poh_spec.py / poh_design.py / validate_poh.py
                      #                    THE PILOT'S OPERATING HANDBOOK — LIVE.
                      #                    Owns manuals/p/<slug>.html
                      #   build_index.py   LIVE, and NOT legacy however it reads:
                      #                    it writes data/decks/index.json, the
                      #                    manifest the whole frontend fetches,
                      #                    and six modules import `line_cards`
                      #   LEGACY, frozen (docs/manual-v5-spec.md): issue_spec.py /
                      #   design.py / build_manual.py / validate_issue.py /
                      #   validate_considering.py / issue_length.py /
                      #   artist_credits.py / short_list_art.py, plus build_page.py
                      #   — superseded by the handbook, and it has no default
                      #   output because it used to clobber one.
                      #   design.py and issue_length.py are imported by LIVE code
                      # ---- DIAGNOSE — a finished deck -> what limits it ----
                      #   deck_status.py   IS THIS DECK FINISHED? lifecycle +
                      #                    staleness; STAGES is the sequence
                      #   deck_audit.py    16 cited axes + engine activation
                      #   deck_map.py      the deck's OWN constellation: local
                      #                    layout + cities/neighbourhoods
                      #   merge_deck_map.py  names ONLY; membership is measured
                      #   engine_facts.py  the deterministic engine brief
                      #   validate_engine.py  stages, completeness, verified_by
                      #   validate_deck_map.py
                      #   deck_facts.py / deck_history.py / scenario_facts.py
                      # ---- LOG + ASK — what happened at the table, and what to do ----
                      #   deck_notes.py    the captain's log: log.jsonl, AUTHORED,
                      #                    append-only, stamped with the decklist sha
                      #   validate_debrief.py / merge_debrief.py  the debrief
                      #                    agent's reading, held to the log by id
                      #   deck_info.py     THE WORKBENCH VIEW: one deck, one screen, and
                      #                    what to do next — composes, computes nothing.
                      #                    `--write` emits info.json, which the DECK PAGE
                      #                    fetches (committed, staleness-gated, no versions)
                      #   deck_branch.py   a candidate 99 you cannot yet sleeve: stage,
                      #                    commit, measure, PROPOSE (the merge request —
                      #                    decision frozen, blocker live), merge
                      #   deck_state.py    archive / retire / supersede / revive;
                      #                    THE ONE WRITER of the lifecycle, which
                      #                    lives in deck_versions.json beside `paper`
                      #                    because a deck in a pile is not sleeved
                      #   deck_delete.py   the only destructive fleet verb; refuses a
                      #                    deck that was sleeved, played or published
                      #   validate_deck_versions.py  the lifecycle, the lock, and the
                      #                    invariant that they cannot both be set
                      #   deck_versions.py every list the deck has been, numbered from
                      #                    git (reuses deck_history), TAGGED in an authored
                      #                    file, JOINED to the log by decklist sha
                      #   prescribe.py     one QUESTION to the doctor: authored prompt +
                      #                    the doctor ⇄ skeptic answer, accumulating
                      #   validate_prescription.py  the diagnosis contract, scoped;
                      #                    stale (older decklist) = form only
                      #   diagnosis_report.py   the diagnosis, rendered readable
                      #   validate_diagnosis.py / validate_goldfish_targets.py
tests/                # pytest suite; counts in docs/testing.md. Markers in
                      # conftest.py: requires_data/rules/deck/strategy/roles;
                      # `-m browser` needs playwright + chromium
data/                 # artifacts; mostly gitignored, viz-served files tracked
  opponents/          # THE POD: opponent seats for `simulate --vs`, from EDHREC's
                      #   average deck (`fetch-opponent`) or authored; tracked
  collection/         # a PHYSICAL card collection (COLLECTION_DIR); the only
                      #   ownership question left, and it is about cardboard.
                      #   MANAMAP_COLLECTION_DIR overrides it
viz/                  # static frontend. FOUR pages, one data layer:
                      #   workbench.html  THE LANDING PAGE — every deck, racked by
                      #                   SLEEVED / waiting on cardboard / on the bench /
                      #                   history, or one fleet table sorted by played /
                      #                   needs-logs / needs-analysis / optimisations /
                      #                   waiting-on-cardboard. Reads every info.json.
                      #   deck.html       one deck's dossier: case file / log / next /
                      #                   status / versions / audit / engine / sim +
                      #                   experiments / prescriptions / questions, and
                      #                   branches LAST, over info.json
                      #   branch.html     one candidate 99: the PROPOSAL, the verdict,
                      #                   the measured table with each row's definition,
                      #                   reward/risk/cost, the bill
                      #   index.html      the atlas + the graph
                      # THREE modes on index.html: discover (the FRONT DOOR — one random
                      # card OR cards you name, click a relation, grow a graph) / explore
                      # (the 34K atlas, live-lit with what you hold) / build (a deck or
                      # pool: graph by default, map by toggle), plus drill, orthogonal.
                      #   workbench.js  the landing page (no MM, no map — like deck-view)
                      #   branch-view.js  the branch workbench, over branch.json +
                      #                 net_change.json
                      #   discovery.js  landing, relations, library, deck load, import,
                      #                 seedFromRows (named cards / ?cards=), brief
                      #   build.js      Build mode — a lens over a deck/pool (window.Build)
                      #   session.js    focus, LIBRARY (persisted), commander — one each
                      #   stage.js      shared canvas primitives for both renderers
                      #   force.js      the graph engine (canvas + d3-force)
                      #   decklist.js   Moxfield parser, fixture-locked to the Python one
                      #   render/canvas.js  the map — the ONLY renderer; Plotly
                      #                    is gone. Drifts at altitude (see below)
                      #   deck-view.js  the dossier + the interactive constellation
                      # d3 from CDN, IIFE, window.MM — see docs/viz.md
docs/                 # reference docs; docs/README.md indexes and sorts them
```

## Environment

- **Python 3.10** via conda `py310` → `.venv` in project root. PyTorch has NO wheels for 3.14.
- Install (macOS order matters — pacmap needs prebuilt numba wheels first):
  ```bash
  .venv/bin/pip install llvmlite==0.41.1 numba==0.58.1
  .venv/bin/pip install -e ".[dev]"
  ```
- Training device: MPS → CUDA → CPU fallback.
- Version pins that matter: `sentence-transformers<4`, `numpy<2` (PyTorch 2.2.2 compat).

## Commands

```bash
manamap run                   # full 15-step pipeline (steps 1 & 7 need internet)
manamap run --from STEP       # resume from a step
manamap <step>                # single step; see `manamap --help` for all 18 subcommands
manamap synergy && manamap power-creep && manamap cluster-regions && manamap card-roles
                              # fast analysis-only refresh (no retrain)
manamap pilot <cmd>           # the bench (97 pilot subcommands); `manamap pilot --help`

manamap pilot deck-info <slug>                          # START HERE: where a deck stands + a derived NEXT
manamap pilot build <slug> --commander "<name>" [--brief "…"] [--from FILE]
                              # THE ONE COMMAND (PRD Epic A): brief -> a legal, MEASURED 99
                              # on the bench, six stages in ~10s. Omit --commander and it
                              # proposes three and halts. The dev batch is the GOLDFISH,
                              # not Forge: a 12-minute Forge batch is ~20 games, whose MDE
                              # is 42 points. `simulate` against a pod is the staging gate.
manamap pilot validate-brief <slug> [--themes]          # the gate brief.json never had
manamap pilot check-in <slug> --from <file>             # a PAPER list -> decklist.txt: diff, refuse, apply
manamap pilot deck-version <slug> [list|show|tag|restore|paper]  # every list from git, joined to the log;
                                                        #   `paper` marks the version you have SLEEVED
manamap pilot deck-state <slug> [archive|retire|supersede|revive] --reason "…"
                              # IS THIS STILL A DECK OR A PILE OF CARDS. Writes
                              # deck_versions.json's `lifecycle`, WITHDRAWS the paper
                              # lock (the two contradict), and rewrites info.json —
                              # which it must, since `regen` skips archived decks
manamap pilot deck-delete <slug>                        # only a deck that was never sleeved,
                              # never played and never published; git rm, staged not committed
manamap pilot deck-notes <slug> add "…" --result win|loss --cause <code>
                              # the captain's log (authored). `--cause` is a CLOSED
                              # vocabulary (deck_notes.CAUSES) so the dossier's priors
                              # table can COUNT how games end; it lands in the sidecar
                              # log_causes.json because log.jsonl is append-only
manamap pilot deck-notes <slug> cause <id> --cause <code>   # file one after the fact
manamap pilot simulate <slug> --vs giada-angels --vs baylen-tokens --vs abaddon --games N
                              # Forge, seeded, against THE STANDARD POD — three
                              # bracket-3 decks with ZERO combos between them.
                              # `vito` was the default until 2026-09-02 and is a
                              # bracket-4 pile of 13 two-card infinites that won
                              # 0.447; naming it is now a deliberate act.
                              # A clock-out is `truncated`, has NO winner, and is
                              # excluded from the rate — it used to be awarded to
                              # the last seat, which our deck can never be.
manamap pilot fetch-opponent "<commander>" --as <slug>  # a pod seat under data/opponents/
manamap pilot sim-scenario <slug> <run> --game G --turn T --stack   # lift a board -> /resolve-stack
manamap pilot prescribe <slug> "<question>"             # open a question to the doctor (then /prescribe)
manamap pilot experiment <slug> --a V1 --b working --vs <pod> --games N   # THE CONTROLLED A/B
manamap pilot net-change <slug> --branch <name> --write  # what a branch costs and buys
manamap pilot deck-branch <slug> propose <name> --as v1.0.2   # accept it; wait for cards
manamap pilot card-search --deck <slug> --oracle REGEX [--owned]         # mine the corpus
manamap pilot model-coverage <slug>                     # WHAT THE MODEL CANNOT SEE, before the games:
                              # seen / DARK (feeds a channel that is OFF) / invisible.
                              # 236 DARK cards across the fleet when it shipped; goldfish
                              # and net-change now print the headline as a PREFLIGHT.
manamap pilot regen [--only STAGE] [--slug S] [--jobs N] [--dry-run]
                              # REBUILD THE FLEET after a model change, in dependency
                              # order (goldfish -> mana-analysis -> net-change ->
                              # diagnose -> benchmark -> deck-info), parallel across
                              # TARGETS. 72 targets in 109s at --jobs 8; the goldfish
                              # stage alone 83.6s -> 23.7s. BIT-IDENTICAL: games inside
                              # one run are never split, only decks are.
                              # A MISSING artifact is CREATED, not skipped -- but only
                              # on a SLEEVED deck (`regen.BOOTSTRAP` + `is_pinned`).
manamap pilot deck-info <slug> --write                  # write info.json for the deck page
manamap pilot build-page <slug> && manamap pilot build-index   # the Pilot's Manual + the manifest
# agents (Claude Code skills): /publish-deck /debrief /prescribe /resolve-stack /analyze-engine /diagnose-deck

make test                     # THE INNER LOOP — non-browser, -n auto, cached. ~22s
make test-fresh               # same with nothing cached; trust this one. ~29s
make test-browser             # the playwright suite, ~4 min
.venv/bin/pytest -n0 -k NAME  # one test, no worker startup
.venv/bin/pytest -m ""        # literally everything, ~10 min

manamap serve                 # viz + a LOCAL /api the deployed site does not have
                              # ALSO A WARM WORKER: with it running, every read-only
                              # `manamap pilot <cmd>` routes through /api/cli and skips
                              # the cold start. query-rules 6.93s -> 0.16s (43x),
                              # deck-facts 1.44s -> 0.14s, deck-audit 2.26s -> 0.59s;
                              # output byte-identical. Fails OPEN — no server, or any
                              # error at all, and the command runs locally as before.
                              # MANAMAP_NO_DAEMON=1 opts out; MANAMAP_DAEMON=host:port
                              # points elsewhere. Restart the server after a code change:
                              # it holds the old modules until you do.
python -m http.server 8000    # or plain static, FROM REPO ROOT (no Build agents)
# http://localhost:8000/viz/workbench.html          THE LANDING PAGE — start here
# http://localhost:8000/viz/index.html              the card map (3 modes)
# http://localhost:8000/viz/index.html?cards=1)%20Sol%20Ring,%202)%20Zur%20the%20Enchanter
#                                                   a walk seeded from cards you name
# http://localhost:8000/viz/deck.html?deck=heliod   a deck's dossier
# http://localhost:8000/viz/branch.html?deck=ur-dragon&branch=eminence-v3
#                                                   a candidate 99 and its net change
# http://localhost:8000/manuals/p/heliod.html       its Pilot's Manual (printable, no JS)
# http://localhost:8000/manuals/index.html          legacy magazine rack (frozen, unlinked)
```

## Gotchas

Grouped by what they are about; the legacy renderer's lessons are last and are kept because they were measured, not because the code will grow.

### Pipeline, data and models

- **Frozen config**: changing `MECHANICAL_TAGS` (or any model-facing dim in `config.py`) invalidates `model_ability.pt` — retrain steps 3–5. Don't touch config values in refactors. `ROLE_PATTERNS` is a **separate** dict for exactly this reason: roles change often, tags must not. Editing roles needs only `manamap card-roles` (step 13) — then `manamap viz-index` (14), which bakes role tags into `viz_index.json`.
- **Roles ≠ mechanical tags**: `MECHANICAL_TAGS` is a retrieval vocabulary ("what is this card like"); `ROLE_PATTERNS` answers "what job does it do in a 99". One `ramp` tag versus five `ramp:rock|dork|land|ritual|cost-reduction` roles is the canonical difference — a curve model that conflates a Signet with a Dark Ritual is wrong.
- **Index alignment**: `projection[i]` == `cards.csv[i]` == `embeddings[i]`. Never partially regenerate after the card count changes; re-run from the changed step onward.
- **No Git LFS on `data/`**: GitHub Pages serves LFS pointers, which would break the deployed viz. Large tracked JSON/bin files are intentional.
- **The obsolescence index publishes a MEASURE, not a verdict.** `compare_with[].strength` runs 0.0–1.0, multiplicative so two problems compound, and the pilot sets the line. It shipped for months as **"Obsoleted By"** with **36.5% of 22,753 pairs failing a purely mechanical check** — costs reported as advantages, restrictions invisible, 8.2% commander-illegal. The retrieval half was always fine (82% share a real role); the judgement half was not. `manamap eval-obsolescence` is the harness, and a change that does not move its separation figure did not do anything. → `docs/gotchas-analysis.md`
- **A trigger pattern's `.*` sits where the subject noun lives.** `when .* dies` makes *"whenever a Goblin you control dies"* and *"whenever another creature dies"* byte-identical, so a tribal deck's payoff reads as a generic one. The gate is the substring the regex throws away — recover it outside `MECHANICAL_TAGS`, which is model-facing. → `docs/gotchas-analysis.md`
- **Two embedding spaces, two different jobs.** `embeddings.npy` is the **layout** space (colour/type) and feeds `projection_2d.json` only; `embeddings_ability.npy` is the **function** space and is the sole source of similarity — Find Similar, the walk and drill all read it whichever map is displayed. Similarity must never follow the displayed map: the layout space knows only colour and type, so asking it for neighbours returns arbitrary same-colour cards.

### The rules that bite whatever you are touching

Each of these cost something to learn. The full record — the measurement, the
wrong first attempt, the number — is in the page named beside it.

**Evidence**
- **A validator that fires on correct data is worse than no validator, and the only way to know is to MEASURE IT AGAINST THE WHOLE FLEET FIRST.** Six proposed checks have been prototyped and rejected on this ground; one fired on 27% of correct authored data, another on 29 of 91 components. → `docs/gotchas-evidence.md`
- **Absent means ABSENT, never zero.** A figure nobody measured must be a missing key with a stated reason. `0.0` is a measurement, and a reader cannot tell it from one. → `docs/gotchas-bench.md`
- **Every rate carries its interval, and a comparison carries the interval on the DIFFERENCE.** Two marginal intervals overlapping implies nothing at all. → `docs/gotchas-bench.md`
- **Never `cache-record` to make a board green**, and never hand-patch an agent's prose to make a gate pass. Editing prose to satisfy a check puts a fresh claim under an old byline. → `docs/gotchas-bench.md`
- **A mean is not a result.** Carry median, min and max: a mean of 17.42 against 2.25 read as a sevenfold win when the median was 0 in both arms and two games were the whole difference. → `docs/gotchas-bench.md`
- **THE GOLDFISH HAS NO BLOCKERS, so its verdict on board QUALITY is not evidence.** With eminence, the token doublers, the sacrifice engine and four draw channels all finally modelled, it still preferred a go-wide Edgar refactor on damage, kill rate and card advantage — and Forge, 400 games per arm against the pilot's own pod, gave the refactor **31/400 against the champion's 50/400**, a difference whose interval EXCLUDES ZERO. The mechanism is one number: combat damage dealt to players fell **29.07 → 18.20**, because 1/1 tokens do not connect and the refactor had cut every lord. That one missing assumption outweighed every other gap closed the same day. Judge a go-wide or token strategy in FORGE from the start. → `docs/gotchas-bench.md`
- **THE FORGE AI WILL NOT SACRIFICE FOR A BENEFIT ITS EVALUATOR CANNOT PRICE**, so a Forge result on a sacrifice deck is a FLOOR. `Indulgent Aristocrat` puts +1/+1 COUNTERS on the board and activates 0.41/cast under `--profile Experimental` against 0.07 under Default; `Ashnod's Altar` makes colourless MANA and is **0 for 59 castings under BOTH**, while `Viscera Seer` (scry) was cast 0 times in 500 games. Cost is not the discriminator -- the Aristocrat costs {2} and the Altar is free. Prefer an outlet with a visible BOARD payoff, and a TRIGGER over an ACTIVATION. Check `activated <card>` against `cast <card>` before trusting any result that rests on one. -> `docs/gotchas-bench.md`
- **A 100-GAME FORGE RUN IS NOT A RESULT.** The same champion read 18/100 and then **50/400**, so the first estimate of a refactor's cost was more than double the powered one. MDE against an 0.18 baseline: **42 points at 20 games/arm, 17.5 at 100, 8.5 at 400**. → `docs/gotchas-bench.md`
- **A COMMANDER'S ABILITY IS NOT AUTOMATICALLY MODELLED.** `command_zone_reduction` reads a commander for COST REDUCTION only; Edgar Markov's eminence MINTS A TOKEN on every other Vampire cast and was absent entirely — the deck's whole axis, understating bodies at turn ten by 50%. `deck-audit`'s engine brief had described it in prose the whole time. Before trusting a figure on a deck, check that the model reads the commander. → `docs/gotchas-bench.md`
- **A MEASURE COMPUTED FROM AN AUTHORED FILE IS NOT EVIDENCE, however tight its interval.** The engine lift split games by the `required` flags in `goldfish_targets.json` — which the same hand writes. Three defensible declarations of one Ur-Dragon list, same 10,000 games, same seed, gave **+0.007 (spans zero), −0.036 (REAL) and +0.014 (REAL)** against kill-by-T8; one of them said, at an interval excluding zero, that assembling the engine made the deck win LESS. Deleted 2026-08-28, and `deck_branch.MEMBERSHIP_AXES` now refuses `engine_online_*` and `any_route_*` as branch objectives. Aim a branch at an OUTPUT the deck produces. → `docs/gotchas-bench.md`
- **Every figure carries its definition, in the report that prints it.** A number a reader has to look up elsewhere gets guessed at, and the guesses go one way: a mean read as a rate, a clock read as a win rate, a hoard read as mana. All three have happened. `net_change.METRICS` is the registry and a test asserts it matches `ROWS` exactly, in both directions. → `docs/pilot.md`

**Changing a matcher or a model**
- **Widening a pattern needs a CORPUS SWEEP in the same commit** — newly matched, newly dropped, and the extreme tail read card by card. Skipped once, it billed Jeweled Lotus three mana every turn forever and counted `Add {R}, {G}, or {W}` as three. → `docs/gotchas-bench.md`
- **A CONDITION IS SCOPED TO THE CLAUSE IT ATTACHES TO.** `enters_tapped_unconditionally` searched the whole oracle text for "unless", so Archway Commons — *"This land enters tapped. When this land enters, sacrifice it unless you pay {1}"* — read as an UNTAPPED five-colour source and `mana-fit` offered it as one. Eleven lands share the wording. The obvious fix is worse and the sweep is what says so: scoping to the SENTENCE flags all ten shocklands, whose idiom spans two. → `docs/gotchas-bench.md`
- **A FETCHLAND'S COLOURS ARE A PROPERTY OF THE DECK, NOT OF THE CARD**, so a function
  that takes only a card cannot answer the question and must not pretend to. `land_colors`
  credits basic types in the type line and symbols in an `add` clause; a fetch has neither,
  so all sixteen true fetches in the corpus read as producing NOTHING — and `goldfish` built
  every land's colours from the same call, modelling four fetches as four colourless lands.
  Measured on ur-dragon/landbase-v1: `mana-fit` reported **every colour worse** on a change
  that left colour access flat (W +1, U −1, B +2, R 0, G 0) and **halved the recurring life,
  8 → 4 per tap-cycle**. `land_colors(card, pool=…)` takes the deck; without `pool` it is
  byte-identical, which is what keeps a caller that has no deck reproducible. The sweep's
  load-bearing split is one word: **`a Mountain card` finds a shockland, `a basic Mountain
  card` cannot** — 16 true fetches against 20 Panorama-shaped ones that read almost
  identically. → `docs/gotchas-bench.md`
- **The goldfish CANNOT rank two lands that make the same colours.** It plays the first land in hand and credits its colours the same turn — there is no tapped state and no choice of which land to play. A twelve-land `candidates` sweep returned exactly two distinct readings, with always-tapped Grand Coliseum tying never-tapped Forbidden Orchard. `mana-analysis` and `mana-fit` are deterministic for exactly this reason and are the whole of the evidence for a land swap. → `docs/gotchas-bench.md`
- **A flag the model sets is a claim the model must ACT ON.** `treasure_doubler` shipped set-and-unread; fifteen candidates returned byte-identical −0.026. `tests/test_metric_hygiene.py` checks this now.
- **A model change makes every derived artifact stale.** `meta.model_version` (a sha over `goldfish.py`) makes that decidable; the three prose validators REPORT it and never fail on it. Regenerate the fleet after any model change. The 39 figures already stale predate stamping and report as unknown, not stale. → `docs/gotchas-bench.md`
- **Adding a metric requires re-running the independence check.** Three magnitude axes shipped that were one axis at r = 0.92–0.98. → `tests/test_metric_hygiene.py`

**Branches, paths and artifacts**
- **A branched write needs a branched READ.** Three instances now, the third committed inside the commit fixing the class: `goldfish.main` measured the champion and filed it under the branch, understating turn-10 hoard by 4×. Every branch measurement must record the branch's own `decklist_sha256`. → `docs/gotchas-bench.md`
- **`--out` on a per-deck command is slug-scoped, and a shell redirect cannot be policed.** Concurrent agents overwrote each other's views seven times across two sessions. → `docs/gotchas-bench.md`
- **A new tracked artifact needs a gate in the same commit** — a validator, a freshness test, or both — and a `deck_status.VALIDATED` entry so the status command sees what the tests see. → `docs/gotchas-evidence.md`
- **ONE PREDICATE, ONE HOME.** Four modules had grown their own answer to "is this deck in a pile" — `common.UNPLAYABLE_STATUSES`, `deck_info.STATE_RETIRED`, `net_change.FREE_TO_RAID` and `deck_branch._deck_holders`, which carried the status and did nothing with it. None disagreed yet and it was already costing something: `deck-branch merge` refused Ur-Dragon on 12 cards, 4 of which sit in decks that do not physically exist. `common.deck_is_apart` decides; everything else reads the row. → `docs/gotchas-bench.md`
- **A DECIDED BRANCH IS NOT AN EXPERIMENT, and until `propose` shipped they rendered identically.** A branch had two observable states — the directory exists, or `merged` is present — and `delete` was the only reader of `merged`. `deck_branch.branch_state` derives six and stores none, so a proposal un-blocks itself when a card lands in a box. `base_version` had been written since branches shipped and **no code had ever compared it to anything**; that comparison is `PROPOSED · OUTRUN`. → `docs/pilot.md`
- **Count COPIES, not decklist entries.** `cards.json` stores basics as one entry with `quantity: N`; counting entries once published "18 lands" for a 33-land deck. Use `common.expand_copies()`. → `docs/gotchas-bench.md`

**Tests**
- **A test that re-derives the rule is testing itself.** Drive the production function, and prove the test by RE-INTRODUCING the bug it was written for. Four such tests shipped, one guarding the flagship metric. → `docs/testing.md`
- **A loop over a possibly-empty collection needs `assert checked >= N`.** Fourteen lacked it; several passed by iterating zero times.
- **A control can be blind to the class it exists for.** The branch control proved the WRITE landed correctly and could not see a read from the wrong place.

**The frontend**
- **Cache-bust `?v=N` on every script and CSS tag in `viz/index.html` AND `viz/deck.html` after any JS/CSS change**; `index.html`'s nine busts move together. Bump `DATA_VERSION` whenever a consumer would draw a DIFFERENT CONCLUSION from the bytes — a retrain qualifies, a content refresh does not.
- **`viz/` and `data/` must stay top-level siblings**; every fetch is `../data/<file>`. Serve from the repo root.
- **A renderer kept behind a flag is a renderer nobody is testing.** → `docs/gotchas-viz.md`

### The full record, by subsystem

`CLAUDE.md` loads into every session; these do not. They hold every measurement
this project has paid for, verbatim — read the one that covers what you are
about to touch.

| page | read before touching | size |
|---|---|---|
| `docs/gotchas-viz.md` | anything under `viz/` | 57 KB |
| `docs/gotchas-bench.md` | `src/manamap/pilot/`, `src/manamap/sim/` | 121 KB |
| `docs/gotchas-analysis.md` | `src/manamap/analysis/` — synergy, power creep, roles, regions | 8 KB |
| `docs/gotchas-evidence.md` | a validator, a citation, `engine.json` | 50 KB |
| `docs/gotchas-magazine-legacy.md` | the frozen renderer (it is not extended) | 17 KB |


### SLEEVED IS BUILT AUTOMATICALLY; ON THE BENCH IS TRIGGERED BY HAND

A deck with a **paper lock** is one the pilot plays, so the whole chain is kept
complete for it without being asked — measurements, then simulation, then the
agent artifacts, then the Pilot's Manual, then the dossier, **in that order**,
because each stage's output is the next one's input. That is what pinning MEANS.

A deck **on the bench** is malleable: it changes daily, nobody has claimed it
exists in cardboard, and its stages run when the pilot asks for them. It is
allowed to be incomplete, and the dossier says so per section rather than
pretending otherwise. Building it automatically would manufacture artifacts for
a list that will be different tomorrow, and put a freshness gate on work in
progress.

`regen.is_pinned()` is the predicate, reading `deck_versions.paper` — the one
authored claim about cardboard. It gates `regen.BOOTSTRAP`, which exists because
`targets()` used to return only places an artifact ALREADY was: a deck missing
`diagnostic.json` was skipped forever in silence, and two of the three SLEEVED
decks were in that state, so the dossier's vitals and the cover sheet's
engine-health word were absent on decks that are played.

### Data artifacts

- **The deck manifest is generated, not hand-kept**: `manamap pilot build-index` writes `data/decks/index.json` (deck list + each deck's passing stack filenames) because a browser can list neither `data/decks/` nor `stacks/`. `viz/deck.html` reads it; a test asserts it matches the artifacts. Add a deck, run `build-index`.
- **`mana_analysis.json` is tracked and staleness-tested**: a decklist edit or a change to the maths needs `manamap pilot mana-analysis <slug>`, or `tests/test_pilot_mana_analysis.py` fails. Run it AFTER `goldfish`, since it embeds goldfish figures.
- **CI runs `make test` on every push and PR** (`.github/workflows/test.yml`), plus one gate the suite cannot make for itself: `make manuals` followed by `git diff --exit-code -- manuals/ data/decks/index.json`, which is the determinism claim asserted from outside the code that asserts it. It runs the ENTRY POINT rather than a bare pytest, because otherwise the Makefile is the one thing nothing checks — and it was, until `make manuals` broke on CI's first run by assuming a `.venv` CI never creates. The **browser suite is deliberately excluded**: it needs a chromium download, takes four minutes, and asserts on real rendering under contention, which is how all five of its historical flakes were born — it stays a local pre-push gate. No cache flag is needed: the regenerate-and-compare cache lives in gitignored `.pytest_cache/`, so a fresh checkout runs everything for real by construction. Lint and format are still deliberately absent.

## Pointers

- **`docs/vision.md`** — START HERE: what the workbench is, the hypothesis loop, the evidence contract, what is live / legacy / honest, the vocabulary
- **`docs/prd.md`** — WHAT IS BEING BUILT (Sept 2026): three environments, five epics, the metrics catalog, and the four blocking decisions resolved in its **Intake notes**. `vision.md` says what the bench IS; this says where it is going. `docs/prd-2026-08.md` is the superseded one that ~27 `PRD-v1 §N` citations resolve against
- **`docs/simulation.md`** — the centre: Forge's spike and verdict, the seeded harness, the parser, the pod, the bridge, commander damage, the distribution, S0–S5
- `PLAN.md` — current state and what's next (read second when resuming work)
- **`/publish-deck`** — the deck lifecycle end to end, every phase in dependency order with its gate; `manamap pilot deck-info <slug>` is the workbench view and the thing to run first on any deck
- `docs/pilot.md` — the bench's commands and artifacts: evidence contract, citation contract, rules + strategy DBs, log/debrief/prescribe/versions, game_state v2, the resolve loop, the build loop (the magazine layer is a LEGACY section at the end)
- `docs/manual-v5-spec.md` — the compact deck page that replaces the magazine (spec, awaiting strikes)
- `docs/agent-audit-2026-08-19.md` — the audit of the agents behind the pivot
- `docs/agent-cost.md` — where LLM spend lives, per-routine token sizing, the invocation cache
- `docs/architecture.md` — models, training mining, mechanical tags, deckbuilding roles, synergy rules, power-creep criteria, region clustering
- `docs/pipeline.md` — all 15 steps: commands, inputs/outputs, runtimes, when to re-run what
- `docs/data-artifacts.md` — every `data/` file: producer, size, git status, consumers
- `docs/viz.md` — frontend structure, `window.MM` API, DATA map, the deck dossier, Pages deployment
- `docs/testing.md` — test layout, skip markers, conventions; the ONLY place test counts are stated
- ~~`STYLEv3.md`~~ — the magazine's constitution, **deleted 2026-08-25**. The renderer it governs is still frozen in `src/`, and its `STYLEv3 §N` comments now resolve through git: `git show 23e8cec:STYLEv3.md`
- `docs/history/` — the magazine-era PLAN, the deck-builder v2 and frontend v2 design records, the founder/editor feedback records
