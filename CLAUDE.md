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

**Six pages over one data layer**: the landing page (`viz/workbench.html`), the card
atlas (`viz/index.html`), the **deck page** (`viz/deck.html?deck=<slug>`), the branch
workbench (`viz/branch.html`), **Curate** (`viz/library.html`) and the embedding-space
appendix (`viz/spaces.html`) — all rendering committed artifacts, with sim figures that carry their intervals. The magazine
that used to be the product was **DELETED on 2026-09-13**; the Pilot's Operating
Handbook (`src/manamap/pilot/poh.py`) replaced it on 2026-09-02 and renders
`manuals/p/<slug>.html`. Its record is `docs/gotchas-magazine-legacy.md` and git. Runs locally on a Mac. The **pipeline and the pilot commands make zero LLM
calls**; two deliberate, opt-in exceptions do — `serve.py`'s `ask` bridge, which
shells out to `claude -p` as a polled job, and `mm ask` (Sven Botstrom), whose
SDK is an optional extra the core install does not pull.

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
                      #   engine_casts.py: did the AI CAST the engine, or hold it —
                      #   a rate on a deck it never cast is a FLOOR, and this says so;
                      #   pilot_quality.py / pod_behaviour.py / power.py / failure.py;
                      #   validate_sim.py re-proves a record against its logs;
                      #   bridge.py lifts a board into a game_state v2 scenario;
                      #   board_series.py: the bridge's board at EVERY turn's
                      #   cleanup — bodies, printed power, open lands, commander
                      #   uptime — an ESTIMATE with its floors named, top-level
                      #   in the record beside engine_casts (since 2026-09-30);
                      #   opponents.py fetches a pod seat from EDHREC.
                      #   ◆ SEEDED run records under data/decks/<slug>/sim/
                      #   (docs/simulation.md — the engine lives OUTSIDE the repo)
  analysis/           # synergy, power_creep, cluster_regions, card_roles,
                      #   eval_embeddings (step 15, the quality gate), common
  pilot/              # the bench: BUILD / PROVE+MEASURE / PAGE / DIAGNOSE / LOG+ASK —
                      # one evidence contract across all of it
                      # ---- shared ----
                      #   card_pool.py     THE ONLY reader of cards.csv; one parse,
                      #                    several views (frame/pool/flags/names/oracle)
                      #   collection.py    THE ONLY reader of COLLECTION_DIR; "do I own
                      #                    this" = a BOX, never deck membership
                      #   card_search.py   deterministic corpus mining: identity, oracle/
                      #                    name regex, role, cmc, --owned/--unowned,
                      #                    and --channel / --modelled / --unmodelled,
                      #                    which answer "can the model even see this"
                      #   model_coverage.py WHAT THE MODEL CANNOT SEE: seen / DARK (feeds
                      #                    a channel that is OFF) / invisible, plus
                      #                    never_cast and silent_losses — the predicate
                      #                    behind "a card the model cannot read looks
                      #                    exactly like a card that does not help"
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
                      #   goldfish.py      seeded Monte Carlo, split across goldfish_
                      #                    {profiles,library,turn}.py. EVERY channel is
                      #                    opt-in per deck — draw, combat, Treasure,
                      #                    sacrifice/deaths/drain, discard, the SPELL
                      #                    COUNT (storm, magecraft, per-cast damage) and
                      #                    four commander abilities. CREATURES tap when
                      #                    they attack; LANDS still never do.
                      #                    docs/simulation.md has the channel table
                      #   mana_analysis.py colour sources / castability — deterministic
                      #   game_state.py    the game_state v2 vocabulary + form check
                      #   merge_prose.py   pilot-notes' five keys in; frozen legacy keys untouched
                      #   agent_cache.py / impact.py / card_refs.py  incremental regen
                      #   validate_tutor_guide / validate_strategy
                      # ---- PAGE ----
                      #   poh.py / poh_spec.py / poh_design.py / validate_poh.py
                      #                    THE PILOT'S OPERATING HANDBOOK — LIVE.
                      #                    Owns manuals/p/<slug>.html
                      #   deck_manifest.py writes data/decks/index.json, the
                      #                    manifest the whole frontend fetches;
                      #                    three modules import its `line_cards`.
                      #                    Was `build_index.py`, whose other half
                      #                    rendered the magazine rack
                      #   THE MAGAZINE RENDERER IS DELETED (2026-09-13): eleven
                      #   modules, seven subcommands, nine pages and nine test
                      #   files. Its record is docs/gotchas-magazine-legacy.md
                      #   and git; the handbook replaced it 2026-09-02.
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
                      # conftest.py: requires_data/rules/deck/strategy/roles/
                      # rulings/branch;
                      # `-m browser` needs playwright + chromium
data/                 # artifacts; mostly gitignored, viz-served files tracked
  pods/               # THE NAMED TABLES `--pod <name>` resolves against (pods.py);
                      #   nine tracked JSON files, each a set of seats and its
                      #   calibration. standard-v3 is the default
  opponents/          # THE SEATS a pod is built from, for `simulate --vs`, from
                      #   EDHREC's average deck (`fetch-opponent`) or authored; tracked
  collection/         # a PHYSICAL card collection (COLLECTION_DIR); the only
                      #   ownership question left, and it is about cardboard.
                      #   MANAMAP_COLLECTION_DIR overrides it
viz/                  # static frontend. SIX pages, one data layer:
                      #   workbench.html  THE LANDING PAGE — every deck, racked by
                      #                   SLEEVED / waiting on cardboard / on the bench /
                      #                   history, or one fleet table sorted by played /
                      #                   needs-logs / needs-analysis / optimisations /
                      #                   waiting-on-cardboard. Reads every info.json.
                      #   deck.html       one deck's dossier, in NINE sections from
                      #                   `page_spec.DOSSIER_SECTIONS` (a test locks the
                      #                   JS to it): cover sheet / rap sheet / known
                      #                   associates / vitals / priors / captain's logs /
                      #                   exhibits / open leads / analyst's assessment.
                      #                   The captain's-log section leads with THE READ —
                      #                   what the games taught, current version first —
                      #                   then each night plain, then the raw note.
                      #                   Everything else is appended below, branches LAST.
                      #   branch.html     one candidate 99: the PROPOSAL, the verdict,
                      #                   the measured table with each row's definition,
                      #                   reward/risk/cost, the bill
                      #   library.html    CURATE — the library across piles, at a
                      #                   size you can read. The drawer keeps a
                      #                   card; this cuts forty. Pile rail (a
                      #                   LOCAL filter, never `setActive`), a grid
                      #                   with multi-select and bulk move/remove,
                      #                   and a pinned pane at `normal` size.
                      #                   Sort + colour/type/role filters over
                      #                   `viz_index.json` (0.56 MB gz), fetched
                      #                   AFTER the first render so a failed
                      #                   fetch degrades to the piles, the name
                      #                   box and the three factless sorts. A
                      #                   card the index cannot resolve gets its
                      #                   own chip and sorts LAST, never as 0.
                      #                   The grid MOVES its tiles rather than
                      #                   rebuilding them, or a re-sort would
                      #                   re-queue 190 images. It cannot ADD a
                      #                   card: keeping stays the Atlas's gesture
                      #   index.html      the atlas + the graph
                      #   spaces.html     the embedding-space appendix — what each
                      #                   space is for and how they differ. Linked
                      #                   from `shell.js`'s SURFACES nav
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
manamap <step>                # single step; `manamap --help` lists all 28 top-level subcommands
manamap synergy && manamap power-creep && manamap cluster-regions && manamap card-roles
                              # fast analysis-only refresh (no retrain)
manamap pilot <cmd>           # the bench (121 pilot subcommands); `manamap pilot --help`

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
manamap pilot simulate <slug> --pod standard-v3 --games N
                              # Forge, seeded, against THE STANDARD TABLE — three
                              # bracket-3 decks with ZERO combos between them,
                              # chosen by a ROUND ROBIN with none of our decks
                              # seated, then CALIBRATED with five of them (185
                              # decided games): sythis 1.45x, subject null 0.292,
                              # jarad a 0.48x floor. `standard` (giada 2.15x) and
                              # `vito-era` (13 two-card infinites, 0.447) are kept so
                              # old records resolve; naming either is deliberate.
                              # The table is `vito-era`; `vito` alone is an
                              # opponent SEAT under data/opponents/, not a pod.
                              # A POD'S NULL IS A PROPERTY OF THE TABLE WITH THE
                              # SUBJECT IN IT: sythis reads 0.25 against heliod
                              # and 0.66 against zur. Read `pods <name> --calibration`.
                              # A clock-out is `truncated`, has NO winner, and is
                              # excluded from the rate — it used to be awarded to
                              # the last seat, which our deck can never be.
                              # THE PREFLIGHT PRINTS FIRST: the null, the MDE at N,
                              # and what +0.05..+0.20 would need. `--detect X`
                              # REFUSES a run that cannot see X at 80% power;
                              # `--anyway` runs it as a screen (same on `experiment`).
                              # `--list` labels every run with the VERSION it
                              # played and `NOT the current list` where it is not.
manamap pilot fetch-opponent "<commander>" --as <slug>  # a pod seat under data/opponents/
manamap pilot validate-forge-hints <slug>               # forge_hints.json: per-card AILogic / AIPreference
                              # hints derived onto the shipped scripts (the shape Forge's own
                              # aristocrat cards use), plus a `forge` rule in pilot_policy.json
                              # for the AiProps knobs. `forge-install --generate` installs
                              # both; the record's card_overrides / ai_profile shas say so.
                              # Measured 2026-09-30: an unflagged outlet is CAST and then sits
                              # idle — `idle_on_battlefield` names it — until it is hinted.
manamap pilot forge-telemetry [--build]                 # THE PATCHED LOG FORMATTER. Forge's shipped log
                              # keeps two zone transitions; one patched method logs EVERY
                              # zone change by name and owner (draws, tutors, mills, wheels,
                              # arrivals). Measured purely observational 2026-09-30: pristine
                              # twice and patched once on a quiet machine differ on the
                              # millisecond line only. The jar is a COPY beside the pristine
                              # one; `simulate`/`experiment` use it when it is there and
                              # stamp `-tl<sha8>` + a `telemetry` block; a class the manifest
                              # does not register REFUSES the run. THE PATCH SET HAS KINDS:
                              # `log` (the formatter, observational) and `ai` (MillAi's
                              # `AILogic$ SacOutlet`, 2026-09-30 — a sacrifice-cost mill
                              # ability fires when a creature of ours is about to die anyway);
                              # a set with an `ai` class changes play and `net-change`
                              # buckets on it like a card override.
manamap pilot sim-scenario <slug> <run> --game G --turn T --stack   # lift a board -> /resolve-stack
manamap pilot sim-findings <slug> --write   # THE SIM DEBRIEF'S SKELETON (sim_findings.json):
                              # per run, findings with ids, intervals and sources. Prose is
                              # the sim-debrief agent's and may cite only finding ids; the
                              # merge recomputes the skeleton. The captain's log is NEVER
                              # written from a Forge game — it has no pilot.
manamap pilot sim-boards <slug> <run> --criterion held --lift --stack   # WHICH board: a criterion names
                              # a cut and a SHAPE, the shortlist is ranked by how many games
                              # RECUR to it (one game is an anecdote), and `extras.finder`
                              # carries the provenance a handbook proposal cites.
                              # `validate-lift` re-lifts every committed board from its own
                              # cut: FAIL while it has no verdict yet, NOTE once finished — the
                              # gate CLAUDE.md:400 said was missing.
manamap pilot prescribe <slug> "<question>"             # open a question to the doctor (then /prescribe)
manamap pilot experiment <slug> --a V1 --b working --pod <name> --games N [--looks K]
                              # THE CONTROLLED A/B. `--looks K` (<=4, O'Brien-Fleming):
                              # each look is WHOLE ROTATED JOBS from both arms at its own
                              # boundary z, never a partial job; the record is rewritten
                              # after every look and `--resume` continues it; `--until-mde X`
                              # is non-binding futility. `--aa` is one list twice (the noise
                              # floor); `--profile-b P` is policy-on vs policy-off on one list.
                              # Seats rotate per global job like `simulate`; the id carries
                              # clock, overrides and AI-profile shas, empty at their defaults.
manamap pilot campaign <name> plan|run|status   # THE OVERNIGHT QUEUE. data/campaigns/<name>.json
                              # is a TRACKED pre-registration of A/Bs; `plan` pins refs to
                              # shas, preflights, prepends an A/A per harness; `run` skips
                              # DONE/STALE, resumes RUNNING, NEVER merges; state is derived
                              # from the records, never stored.
manamap pilot net-change <slug> --branch <name> --write  # what a branch costs and buys.
                              # ONE PRIMARY (the objective), TWELVE EXPLORATORY rows,
                              # Holm-corrected. THE REAL TABLE IS IN THE RULE: a Forge
                              # win-rate loss whose interval excludes zero at the same
                              # pod blocks a merge whatever the goldfish said; one that
                              # spans zero changes nothing. The block stores the pod's
                              # null and every Forge endpoint with its interval.
manamap pilot deck-branch <slug> new <name> --objective "forge.win_rate >= 0.25 @standard-v3"
                              # a Forge objective NAMES ITS TABLE or is refused; graded on
                              # the branch's pooled rate there, with the interval on the
                              # difference and the null in the grade. THE AXES INCLUDE THE
                              # DECK'S IDENTITY (2026-09-30): forge.drain_dealt (life loss
                              # that was not damage), forge.life_gained, forge.biggest_hit,
                              # forge.evasive_damage_share, forge.kills_by_ability — each
                              # with its floor named in analysis.limits
manamap pilot deck-branch <slug> propose <name> --as v1.0.2   # accept it; wait for cards
manamap pilot deck-branch <slug> withdraw|reject <name> --reason "…"  # the reason goes in the LEDGER
manamap pilot decisions <slug> [outcome|backfill]   # THE DECISION LEDGER (decisions.jsonl,
                              # append-only): every propose/withdraw/reject/merge with the
                              # report's prediction frozen; `outcome` joins the merged list's
                              # own runs at the same pod+harness back to the merge —
                              # predicted beside realised, inside the interval or not.
                              # `deck-info` says when a merge can be closed.
manamap pilot card-search --deck <slug> --oracle REGEX [--owned]         # mine the corpus
manamap pilot scan-candidates <slug> [--dimension drain|gain|threat|outlet|sweeper|draw] [--against-branch B] --write
                              # ONE PASS along the deck's DIMENSIONS: every row names the
                              # predicate that admitted it (oracle id / role / tag / printed
                              # keyword); a converter or a two-card infinite with the (staged)
                              # 99 is FLAGGED and sorted last, never ranked or dropped; death-
                              # draw splits on `nontoken`. Retrieval, not judgement — sorted by
                              # EDHREC rank. Writes the dated candidate_scan.json (validated)
manamap pilot forge-cast-check <slug> --card "Toxic Deluge" [--games 8 --copies 4 --vs giada-angels]
                              # PROVE THE AI PLAYS A CARD BEFORE A NIGHT IS SPENT ON IT: a
                              # two-seat shell (the deck's commander, N copies, its own cheap
                              # spells as filler, basics), counted from the telemetry hand
                              # facts — drawn / cast / activated / HELD while castable.
                              # Unflagged is NOT castable: Toxic Deluge was drawn 28 times
                              # and cast 0 in a 200-game branch arm (2026-10-01) because its
                              # script lacks IsCurse$ and X is priced before the life is paid.
                              # Every add that must be cast or activated for a branch's
                              # objective runs this first; a HELD card is a piloting item.
                              # `--branch B --adds --write` proves EVERY add and writes the
                              # branch's cast_proofs.json (validated, stamped with the
                              # harness); `simulate <slug>@<branch>` REFUSES an add that is
                              # not PLAYED under the current harness (--anyway runs it with
                              # the slots recorded as FLOORS), net-change prints CAST PROOFS
                              # and marks the primary FLOOR, deck-info NEXT names the check
manamap pilot fetch-edhrec <slug> [--theme aristocrats] # EDHREC's commander page(s) as dated per-card
                              # synergy / inclusion (edhrec_cards.json, ★ evidence, validated);
                              # cards newer than the corpus are listed apart, not failed
manamap pilot model-coverage <slug>                     # WHAT THE MODEL CANNOT SEE, before the games:
                              # seen / DARK (feeds a channel that is OFF) / invisible.
                              # 236 DARK cards across the fleet when it shipped; goldfish
                              # and net-change now print the headline as a PREFLIGHT.
manamap pilot regen [--only STAGE] [--slug S] [--jobs N] [--dry-run]
                              # REBUILD THE FLEET after a model change, in dependency
                              # order (goldfish -> mana-analysis -> net-change ->
                              # diagnose -> benchmark -> deck-info), parallel across
                              # TARGETS. MEASURED 2026-09-04 at 72 targets: 109s at
                              # --jobs 8, the goldfish stage alone 83.6s -> 23.7s. The
                              # fleet is 96 targets now, so that is a ratio, not a
                              # runtime. BIT-IDENTICAL: games inside
                              # one run are never split, only decks are.
                              # A MISSING artifact is CREATED, not skipped -- but only
                              # on a SLEEVED deck (`regen.BOOTSTRAP` + `is_pinned`).
                              # REFRESH IS EVERY LIVE DECK; BOOTSTRAP IS SLEEVED ONLY.
                              # Two questions, and one gate used to answer both: the
                              # sweep was sleeved-only while the freshness tests check
                              # every deck that is not RETIRED, so a model change left
                              # emiel-blink and meren-recursion stale and the board red
                              # (2026-09-26). An artifact that EXISTS is tracked and
                              # already gated, so it is rebuilt wherever it lives; an
                              # artifact that is MISSING is still only created on a
                              # SLEEVED deck, because minting a tracked figure for a
                              # list that changes daily is the pilot's call.
                              # `manamap pilot regen --jobs 8 && make manuals` is now
                              # the WHOLE recipe after a model change.
manamap pilot deck-info <slug> --write                  # write info.json for the deck page
manamap pilot build-poh <slug> && manamap pilot build-index    # the HANDBOOK + the manifest
# agents (Claude Code skills): /publish-deck sequences the lifecycle; then
# /build-deck /analyze-engine /resolve-stack /write-manual /poh-procedures
# /debrief /sim-debrief /captains-log /prescribe /diagnose-deck /research-strategy /refresh-corpus.
# 22 skills in .claude/skills/, 18 charters in .claude/agents/

make test                     # THE INNER LOOP — non-browser, -n auto, cached.
make test-fresh               # same with nothing cached; trust this one.
                              # RUNTIMES LIVE IN docs/testing.md, not here. This
                              # line said ~22s/~29s for weeks while the real
                              # figure was 772s — a number nobody re-measured
                              # after the suite tripled.
make test-browser             # the playwright suite
.venv/bin/pytest -n0 -k NAME  # one test, no worker startup
.venv/bin/pytest -m forge     # ONE real Forge game; needs ~/.mana-map/forge
.venv/bin/pytest -m ""        # literally everything, browser included

# .mcp.json registers an MCP SERVER (`manamap.mcp_server`) exposing seven read-only
# tools to Claude Code: deck_state, fleet, search_docs, search_code, stats,
# run_command, command_help. Structured data from the warm daemon instead of parsed prose —
# `deck-status heliod` is 2.9s cold, 0.003s warm, byte-identical. No SDK: MCP is
# JSON-RPC over stdio and the subset a tool server needs is ~150 lines, the same
# reasoning that keeps scipy out of sim/stats.py. It CANNOT write; the gate is
# `serve._cli`, imported rather than restated.

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
# http://localhost:8000/manuals/p/heliod.html       its Pilot's Operating Handbook (printable, no JS)
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
- **Every rate carries its interval, and a comparison carries the interval on the DIFFERENCE.** Two marginal intervals overlapping implies nothing at all. **And a FAMILY of comparisons carries its correction**: `net-change`'s twelve rows are exploratory and Holm-corrected; the objective is the one pre-registered primary, as `win_rate` is in `experiment`. → `docs/gotchas-bench.md`, `docs/simulation.md`
- **Never `cache-record` to make a board green**, and never hand-patch an agent's prose to make a gate pass. Editing prose to satisfy a check puts a fresh claim under an old byline. → `docs/gotchas-bench.md`
- **THE HARNESS PRODUCED A SIGNIFICANT RESULT FROM NO CHANGE AT ALL, and the bench had never run the control that says so.** `experiment --aa` is one list against itself and no A/A record existed for ANY deck. The first one, 2026-10-02, edgar-vampires' champion at standard-v3: arm A **7/42 = 0.167**, arm B **15/41 = 0.366** at 50 games/arm, a gap of **0.199 on an identical list**, and the plain Newcombe interval on the difference **[+0.009, +0.373] EXCLUDES ZERO**. The treasury-v1 arm's whole claimed effect was 0.093. The O'Brien-Fleming boundary (z 4.049 at look 1) correctly refuses it — the protection exists and works; what fails is the PLAIN interval, which is what `net_change` prints and what every branch objective is graded on. So: **no branch is graded on a Forge win rate or a per-game mean without an A/A at the same N and harness beneath it.** Mechanism endpoints are still readable, because a cast, a trigger and an activation are COUNTS rather than estimated rates. → `docs/gotchas-bench.md`
- **A mean is not a result.** Carry median, min and max: a mean of 17.42 against 2.25 read as a sevenfold win when the median was 0 in both arms and two games were the whole difference. → `docs/gotchas-bench.md`
- **THE GOLDFISH HAS NO BLOCKERS, so its verdict on board QUALITY is not evidence.** With eminence, the token doublers, the sacrifice engine and four draw channels all finally modelled, it still preferred a go-wide Edgar refactor on damage, kill rate and card advantage — and Forge, 400 games per arm against the pilot's own pod, gave the refactor **31/400 against the champion's 50/400**, a difference whose interval EXCLUDES ZERO. The mechanism is one number: combat damage dealt to players fell **29.07 → 18.20**, because 1/1 tokens do not connect and the refactor had cut every lord. That one missing assumption outweighed every other gap closed the same day. Judge a go-wide or token strategy in FORGE from the start. → `docs/gotchas-bench.md`
- **THE FORGE AI WILL NOT SACRIFICE FOR A BENEFIT ITS EVALUATOR CANNOT PRICE**, so a Forge result on a sacrifice deck is a FLOOR. `Indulgent Aristocrat` puts +1/+1 COUNTERS on the board and activates 0.41/cast under `--profile Experimental` against 0.07 under Default; `Ashnod's Altar` makes colourless MANA and is **0 for 59 castings under BOTH**, while `Viscera Seer` (scry) was cast 0 times in 500 games. Cost is not the discriminator -- the Aristocrat costs {2} and the Altar is free. Prefer an outlet with a visible BOARD payoff, and a TRIGGER over an ACTIVATION. Check `activated <card>` against `cast <card>` before trusting any result that rests on one. -> `docs/gotchas-bench.md`
- **`AI:RemoveDeck:All` MEANS THE AI NEVER CASTS THE CARD, and this file said the opposite for a month.** `AiController.getSpellAbilityToPlay` filters every non-land ability whose host `isCardRemAIDeck` (2.0.14 bytecode, read 2026-09-30). Measured first: in a 26-land shell with four copies against a slow opponent, Vish Kal was castable in **7 of 12 games and cast in none**; the same script with only that line removed was cast **10 times**; a plain seven-drop control was cast 4 of 5. The fleet sweep agreed — **every `All` permanent in a live deck with games on record was cast 0 times, 14 of 14** (Viscera Seer, Goblin Bombardment, Altar of Dementia, Isochron Scepter, Magus of the Wheel, Vish Kal in 860 games…), while every `Random` one was cast normally. The earlier belief rested on Swan Song, 28 casts in 160 games: a COUNTERSPELL, cast through the reactive path, which does not filter. The sharknado Windfall finding below was THIS. `data/forge_overrides/unflag.txt` lists the 23 cards whose override is the shipped script minus that line; `forge-install --generate` prints any a live deck has since acquired, and a fleet test refuses one. Every run since carries `-ov8bf04cfc`, and THE NULL MUST BE RE-MEASURED under it — `pods.calibration` counts plain-harness runs only, so until then no fresh run feeds a null. → `docs/gotchas-bench.md`
- **A FORGE RESULT ON A DECK WHOSE ENGINE THE AI NEVER CAST IS A FLOOR, and the record now says so.** sharknado's seat cast Wheel of Fortune once and Windfall never in 60 games while DISCARDING Windfall three times and Faithless Looting six -- the log's own statement that the card was held and passed over. `record["engine_casts"]` carries per-card cast / activated / discarded for our seat (MEASURED, top-level, validated where present); `sim/engine_casts.py` reads it at print time against the deck's declaration and prints "held and never cast" at the `simulate` tail, in `deck-info` and live in `sim-progress`. Under the telemetry jar it is no longer an inference: every card carries `castable_uncast` (own turns it ended in hand with the lands to cast it) and `held_while_castable` names the cards held on two or more such turns — HELD WHILE CASTABLE (MEASURED). Check it before reading any rate, the way the piloting gate is checked. → `docs/gotchas-bench.md`
- **THE AI WILL NOT TARGET ITS OWN COMMANDER TO SWITCH ON A COPY ABILITY, and that makes a run measure a DIFFERENT DECK rather than a floor.** Zada, Hedron Grinder's whole deck is "an instant or sorcery that targets only Zada, copy it for each other creature you control". Over 60 games she was **cast 100 times and her trigger fired 21** — 0.35 per game against ~11.9 spells cast. Forge implements the card correctly; its evaluator prices `Brute Force` on Zada as +3/+3 on one creature, identical to any other target, so the targeting heuristic cannot see that this one multiplies. `engine_casts` said the plan was played at 134-160% of expected natural draws and was right: every card was cast, just never at Zada. So goblin-storm's 0.031 baseline and the branch's 0.040 / 0.065 are all void as deck measurements, and the A/B is not rescued by comparing them. `--profile Experimental` does not help — `tokens_observed` moved 1.98 → 1.95. THE PREFLIGHT is one grep: `grep -rc "triggered <Commander>" .../sim/logs/<run>/*.log`. The fix for evidence is `sim-scenario --stack` plus `/resolve-stack` — one of the 21 real boards, proven by CITATION (✓) instead of a rate. → `docs/gotchas-bench.md`
- **UNFLAGGED IS NOT CASTABLE. MEASURE TWICE: NO ADD ENTERS A FORGE ARM UNPROVEN.** Three cards cleared by the scan were found NOT PLAYED only after a night of games — Vish Kal's −X/−X (0 activations, six passes), Toxic Deluge (drawn 28, cast 0 across a 200-game arm) and Bastion of Remembrance (cast in 10 of 30 games it was drawn, castable on 48 turns). `AI:RemoveDeck:All` is one filter; each API's AI class refuses for its own reasons — X priced before its cost is paid (`Count$xPaid` + `PayLife<X>`, `SVar$CostCountersRemoved`), a −X/−X with no `IsCurse$`, no `AILogic$` for the shape, a cheap do-nothing-now permanent cast behind every creature. `forge-cast-check <slug> --branch B --adds --write` proves every add in a two-seat shell and names the class and the remedy; `simulate` on a branch seat REFUSES an add without a PLAYED proof under the current harness. → `docs/gotchas-bench.md`
- **A LIFTED BOARD IS AN INFERENCE, AND IT WAS OMITTING PERMANENTS IN FIVE WAYS.** `sim-scenario` reconstructs a board from an event stream, and every gap in that reconstruction reads as a fact about the game. Found in ONE session (2026-09-28) by pointing `rules-checker` at two real boards: a creature with a characteristic-defining power (`Creature * / *`) never reached the board because `_CREATURE` demanded digits — ~107 resolutions in one run, **all opponents', so it flattered every lethal claim**; every AURA was discarded as an instant (`Rancor (203) - Attach to X` matches `_SPELL`), which made Sphere of Safety's tax read {2} instead of {3}; a TOKEN existed only from the moment it ACTED, so **the lift listed 187 of 1843 live tokens — 10%**, and a published board-width figure was wrong by 18x (">= 6 other bodies" 2.0% against ~21%); `_WORDS` stopped at "seven" so Krenko's "creates eight" was unreadable, losing the BIGGEST boards; and **705 of 857 token deaths (82%) had no owner**, so a death was charged to whichever seat held a match — which can delete an opponent's blocker. All five fixed and tested. A sixth is DOCUMENTED, not fixed: the graveyard is battlefield-deaths-only (mills and discards are not zone events), so a `*/*` power cannot be computed from it — `graveyard_is_a_floor` says so. The repo's only PASSING lifted scenario is radagast/008, which predates all of this — re-lifting gives our seat 11 creatures against the 7 its ✓ was resolved with — but radagast is BROKEN DOWN FOR PARTS, so scoped to live decks the exposure is **zero**. The gate exists since 2026-09-30: `validate-lift` re-lifts every committed board from its own cut and `validate-stack` carries the result as a NOTE — FAIL while it has no verdict yet, NOTE once the loop finished. → `docs/gotchas-bench.md`
- **A FORGE RECORD DESCRIBES THE LIST IT PLAYED, NOT THE LIST ON DISK.** Every record stamps `seats[].decklist_sha256` and nothing read it, so `net-change` reported 120 games as goblin-storm/zada-v1's rate that were played on its FOURTH commit against its seventh on disk — eight cards in and nine out since, including the branch's largest claimed gain. Fleet sweep: **29 of 41 branches, 10,120 games**, mostly champion arms, because a merged branch rewrites the champion's list. It FLAGS rather than suppresses (`forge.list_mismatch`, printed ABOVE the rate in the CLI and on `branch.html`) because the sha is over file BYTES: edgar's list dropped a set code from `Gifted Aetherborn (AER) 61` in the same commit as two real swaps, so a cosmetic edit trips it identically. `engine_casts` IS strict — it named five cards "held and never cast" that were not in the simulated list at all. → `docs/gotchas-bench.md`
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
- **The goldfish CANNOT rank two lands that make the same colours.** It plays the first land in hand and credits its colours the same turn — LANDS have no tapped state and there is no choice of which land to play. (CREATURES do tap when they attack, added 2026-09-26; that is a different question and does not help a land swap.) A twelve-land `candidates` sweep returned exactly two distinct readings, with always-tapped Grand Coliseum tying never-tapped Forbidden Orchard. `mana-analysis` and `mana-fit` are deterministic for exactly this reason and are the whole of the evidence for a land swap. → `docs/gotchas-bench.md`
- **A flag the model sets is a claim the model must ACT ON.** `treasure_doubler` shipped set-and-unread; fifteen candidates returned byte-identical −0.026. `tests/test_metric_hygiene.py` checks this now.
- **A ONE-TURN GRANT IS NOT A PERMANENT DOUBLER, and a channel placed where it cannot fire is not a channel.** `team_damage_multiplier` matched "creatures you control gain double strike" anywhere in an oracle, so Elesh Norn // The Argent Etchings doubled all of sharknado's damage off a SAGA BACK FACE chapter lasting one turn, behind a three-creature sacrifice — cutting it measured **−2.2 damage** and read as a reason to keep it. **75 corpus cards read as permanent doublers, 53 after the fix: 22 were phantom**, and sharknado's damage@8 fell 31.45 → 29.47. The rule is ONE-SHOT versus RE-APPLIED, not temporary versus permanent: Atarka's identical clause fires every combat and is real, which an existing test caught when the first fix dropped it. Separately, the Blood crack was placed on leftover mana and a goldfish spends its pool casting — **49 of 300 games ended with Blood uncracked**, a confident zero from a channel that never ran. → `docs/gotchas-bench.md`
- **A CARD THE MODEL CANNOT READ LOOKS EXACTLY LIKE A CARD THAT DOES NOT HELP.** Four in a row measured "no effect" from a sweep that had never priced them — 400 of 405 sacrifice-gated draws read as zero, and Blood, artifact-sacrifice payoffs and draw DOUBLERS had no channel at all. Before trusting a null on a swap, check `draw_profile`'s `unmodelled` and `model-coverage`. When a card's value is contingent on something the model omits, measure the CEILING with the omission reversed and LABEL it: Jaws came back floor −0.43, ceiling +0.09, which settles the card instead of leaving a standing doubt about the instrument. → `docs/gotchas-bench.md`
- **A model change makes every derived artifact stale.** `meta.model_version` (a sha over `goldfish.py`, `goldfish_profiles.py`, `goldfish_library.py` and `goldfish_turn.py` — `_MODEL_FILES`, so splitting the module did not blind the stamp) makes that decidable; the three prose validators REPORT it and never fail on it. Regenerate the fleet after any model change. The 39 figures already stale predate stamping and report as unknown, not stale. → `docs/gotchas-bench.md`
- **Adding a metric requires re-running the independence check.** Three magnitude axes shipped that were one axis at r = 0.92–0.98. → `tests/test_metric_hygiene.py`
- **THE COMMANDER'S OWN TEXT IS NOT MODELLED UNTIL SOMEBODY MODELS IT.** zur-enchantress was rebuilt around Zur, Eternal Schemer and the goldfish read NEITHER of his abilities — the static grant of deathtouch/lifelink/hexproof to every enchantment creature, nor the `{1}{W}` that animates an enchantment into a body whose power is its mana value. Modelling them took kill-by-t8 from 0.153 to 0.327 on an unchanged 99. A commander ability that only one card in the corpus has is DECLARED per deck (`model_commander_animate`, `model_commander_attack_tutor`); one a handful share is parsed after a sweep. → `docs/gotchas-bench.md`
- **A CARD CAN BE READ CORRECTLY AND NEVER PLAYED.** Every casting loop in the goldfish selects on a CHANNEL — draws, ramps, makes Treasure, has a body — and a card matching none of them sits in hand for ten turns while its profile says exactly what it would have done. Found FIVE times in one session and only caught as a class on the fourth: the Shrines measured as exactly nothing, a SLEEVED deck ran its sacrifice engine on 2 of its 4 outlets, and four of six attack enablers were uncastable, which made the model unable to start its own engine. `model_coverage.never_cast` / `silent_losses` are the predicate and a fleet test asserts no deck computes an effect it never applies. **Teach the casting predicate in the SAME commit as the ability.** → `docs/gotchas-bench.md`
- **A RATE DRIVING A FIGURE MUST NAME WHERE IT WAS MEASURED.** `model_commander_attack_tutor` fired every turn and reported 5.70 fires a game; Forge resolved the search 1.22 times. Correcting it took kill-by-t8 from 0.501 to 0.173 and undid more than half of one day's measured gains. `fires_per_turn`, `model_deaths` and their `source` keys are REQUIRED for exactly this reason, and a CEILING (1.0 when the attack is free) is labelled as one in the record rather than read as a forecast. → `docs/gotchas-bench.md`
- **A land whose only coloured mode costs extra mana is not a coloured source on curve.** `land_colors` counts `{1}, {T}: Add one mana of any color` at full value; six such lands made every colour in zur-enchantress read at or above target when all three were short. Reported (`sources.gated`, `on_curve_probability.lands_only_ungated`) rather than discounted, because a fraction to divide by would be an authored number driving a headline. The obvious fix is BACKWARDS — cutting them for basics makes every colour worse. → `docs/gotchas-bench.md`

**Branches, paths and artifacts**
- **A branched write needs a branched READ.** Three instances now, the third committed inside the commit fixing the class: `goldfish.main` measured the champion and filed it under the branch, understating turn-10 hoard by 4×. Every branch measurement must record the branch's own `decklist_sha256`. → `docs/gotchas-bench.md`
- **A BRANCH MAY NOT DECLARE A MODEL CHANNEL ITS DECK DOES NOT — that is two simulators, not an A/B.** `common.deck_file` prefers a branch's copy of an authored file, and its docstring asserted nobody writes a second `goldfish_targets.json`. goblin-storm/zada-v1 did: it declared `model_draw`, `model_combat` and `model_commander_copy` while the deck declared NOTHING, so Zada's whole ability was modelled on one arm only. `net_change` then dropped **7 of its 12 rows** (no champion `output` block at all) and the two surviving rows that moved were the two the asymmetry flattered — missed-drop-by-T5 −0.083 "better" → noise, interaction-affordable-@T6 −0.037 "worse" → **+0.095 better**. Declared identically, DARK went 7 → 0 and the branch won ten of twelve rows: damage @T10 **16.01 → 27.73**. A branch MAY name its own new cards in a target's `any_of` (meren-recursion does, legitimately — a target names cards); it may never change the instrument. → `docs/gotchas-bench.md`
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
| `docs/gotchas-viz.md` | anything under `viz/` | 63 KB |
| `docs/gotchas-bench.md` | `src/manamap/pilot/`, `src/manamap/sim/` | 304 KB |
| `docs/gotchas-analysis.md` | `src/manamap/analysis/` — synergy, power creep, roles, regions | 8 KB |
| `docs/gotchas-evidence.md` | a validator, a citation, `engine.json` | 51 KB |
| `docs/gotchas-magazine-legacy.md` | the DELETED renderer (it is not extended) | 18 KB |


### SLEEVED IS BUILT AUTOMATICALLY; ON THE BENCH IS TRIGGERED BY HAND

A deck with a **paper lock** is one the pilot plays, so the whole chain is kept
complete for it without being asked — measurements, then simulation, then the
agent artifacts, then the Pilot's Operating Handbook, then the dossier, **in that order**,
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
- **`docs/simulation.md`** — BOTH ENGINES. Forge: the spike and verdict, the seeded harness, the parser, the pod, the bridge, commander damage, the distribution, S0–S5. And **the goldfish channel by channel** — which flag switches on what, what a copied spell does per effect, storm/magecraft/per-cast damage, why creatures tap and lands do not, the named gaps. READ THE CHANNEL TABLE before trusting a goldfish figure
- `PLAN.md` — current state and what's next (read second when resuming work)
- **`/publish-deck`** — the deck lifecycle end to end, every phase in dependency order with its gate; `manamap pilot deck-info <slug>` is the workbench view and the thing to run first on any deck
- `docs/pilot.md` — the bench's commands and artifacts: evidence contract, citation contract, rules + strategy DBs, log/debrief/prescribe/versions, game_state v2, the resolve loop, the build loop (the magazine layer is a LEGACY section at the end)
- `docs/history/manual-v5-spec.md` — the compact deck page that replaces the magazine (spec, awaiting strikes)
- `docs/history/agent-audit-2026-08-19.md` — the audit of the agents behind the pivot
- `docs/agent-cost.md` — where LLM spend lives, per-routine token sizing, the invocation cache
- `docs/architecture.md` — models, training mining, mechanical tags, deckbuilding roles, synergy rules, power-creep criteria, region clustering
- `docs/pipeline.md` — all 15 steps: commands, inputs/outputs, runtimes, when to re-run what
- `docs/data-artifacts.md` — every `data/` file: producer, size, git status, consumers
- `docs/viz.md` — frontend structure, `window.MM` API, DATA map, the deck dossier, Pages deployment
- `docs/testing.md` — test layout, skip markers, conventions; the ONLY place test counts are stated
- ~~`STYLEv3.md`~~ — the magazine's constitution, **deleted 2026-08-25**. The renderer it governed was **deleted too, 2026-09-13** (`443cf6b7`), so both halves now read out of git: `git show 23e8cec:STYLEv3.md`
- `docs/known-issues.md` — THE INVENTORY OF WHAT NO TEST FAILS ON. The suite is green, so this is the list of things that are wrong and that nothing will tell you about
- `docs/paydown-plan.md` — the six-phase paydown and its tracker; every task has an id, a gate, a proof and a status
- `docs/agent-inventory.md` — every agent and skill with its path, what it owns, and which skill spawns it. Read before touching a charter
- `docs/README.md` — indexes and sorts all of the above, current reference above history, with line counts a test asserts
- `docs/history/` — the deck-builder v2 and frontend v2 design records and the three Ur-Dragon memos (none applied). The magazine-era PLAN and the founder/editor feedback records were **deleted 2026-08-25**
