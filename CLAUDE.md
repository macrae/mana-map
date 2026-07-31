# CLAUDE.md — Mana Map

MTG card embedding pipeline: downloads ~34,300 oracle cards from Scryfall, trains two small neural nets to embed them (128-dim), projects to 2D, and serves an interactive map + deck builder from `viz/`. Runs locally on a Mac.

## Layout

```
src/manamap/          # the Python package (pip install -e ".[dev]")
  config.py           # ALL constants: paths, hyperparams, tag patterns, synergy rules
  mechanical_tags.py  # regex tag extraction (shared by ingest + analysis)
  pipeline.py         # ordered STEPS registry + runner
  cli.py              # `manamap` console script
  ingest/             # download, extract, preprocess, download_combos, process_combos
  training/           # model, train, train_ability, embed, common
  export/             # reduce (PaCMAP), export_embeddings (.bin for JS)
  analysis/           # synergy, power_creep, cluster_regions, card_roles, common
  pilot/              # two subsystems sharing one contract:
                      # BUILD — brief -> a legal, tier-conditioned 99
                      #   build_deck.py    pool -> score -> fill -> enforce_bracket
                      #   manabase.py      hypergeometric colour-source math
                      #   bracket.py       computed bracket floor + evidence
                      #   validate_build.py
                      # PUBLISH — a deck -> a magazine issue
                      #   rules DB + strategy KB (RAG), deck ingestion, citation
                      #   contract, goldfish simulator, agent cache, artist credits
                      #   issue_spec.py  SECTION system (single source of truth)
                      #   design.py      tokens, stylesheet, component library
                      #   mana_analysis.py  Sources Say — deterministic, no agent
                      #   build_manual.py / build_index.py  issue + newsstand
                      #   validate_issue.py / agent_cache.py / artist_credits.py
                      #   validate_considering.py / validate_tutor_guide.py
                      #   impact.py / card_refs.py  incremental regeneration
tests/                # pytest suite (1,097: 1,034 fast + 63 browser). Markers in
                      # conftest.py: requires_data/rules/deck/strategy/roles;
                      # `-m browser` needs playwright + chromium
data/                 # artifacts; mostly gitignored, viz-served files tracked
viz/                  # static frontend: index.html (the map: explore / deck lens /
                      # build / the walk, plus drill) + deck.html (the dossier).
                      # Plotly + d3 CDN, IIFE, window.MM. force.js is canvas+d3,
                      # the first step off Plotly — see docs/viz.md
docs/                 # reference docs (see Pointers below)
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
manamap pilot <cmd>           # build + publish subsystem (33 subcommands) — see docs/pilot.md

.venv/bin/python -m pytest    # 1,097; data-dependent ones skip if artifacts missing
.venv/bin/python -m pytest -m "not browser"   # 1,034, skips the browser suite (~70s)

python -m http.server 8000    # serve viz FROM REPO ROOT
# http://localhost:8000/viz/index.html          the card map
# http://localhost:8000/viz/deck.html?deck=heliod   a deck's dossier
```

## Gotchas

- **Frozen config**: changing `MECHANICAL_TAGS` (or any model-facing dim in `config.py`) invalidates `model_ability.pt` — retrain steps 3–5. Don't touch config values in refactors. `ROLE_PATTERNS` is a **separate** dict for exactly this reason: roles change often, tags must not. Editing roles needs only `manamap card-roles` (step 13) — then `manamap viz-index` (14), which bakes role tags into `viz_index.json`.
- **Roles ≠ mechanical tags**: `MECHANICAL_TAGS` is a retrieval vocabulary ("what is this card like"); `ROLE_PATTERNS` answers "what job does it do in a 99". One `ramp` tag versus five `ramp:rock|dork|land|ritual|cost-reduction` roles is the canonical difference — a curve model that conflates a Signet with a Dark Ritual is wrong.
- **Index alignment**: `projection[i]` == `cards.csv[i]` == `embeddings[i]`. Never partially regenerate after the card count changes; re-run from the changed step onward.
- **No Git LFS on `data/`**: GitHub Pages serves LFS pointers, which would break the deployed viz. Large tracked JSON/bin files are intentional.
- **Viz serving root**: all fetches are `../data/<file>` relative to `viz/index.html` — `viz/` and `data/` must stay top-level siblings; serve from repo root.
- **Cache busting**: bump `?v=N` on the script/CSS tags in `viz/index.html` **and `viz/deck.html`** after any JS/CSS change. `index.html`'s four script busts must move together — a test asserts it, since a mismatched pair is how `deck-map.js` ends up calling a stale `mana-map.js`. `manuals/magazine.css` is content-addressed instead (`?v=<sha8>`), so a CSS edit there obligates rebuilding every manual page.
- **Synergy ≠ Similar**: synergy is complementary (blink→ETB, rule-based); Find Similar is embedding neighbors. Different algorithms.
- **Plotly**: `Plotly.relayout` fires `plotly_relayout` — guard against event loops. `Plotly.react` replaces layout wholesale, so `render()` must write the live axis range back or it silently resets the camera; `Plotly.restyle` preserves it and is the only fast path (~32ms on a 1,200-point `scattergl`).
- **Data cache-busting**: `MM.DATA` URLs carry `?v=DATA_VERSION`. Bump it whenever a consumer would draw a **different conclusion** from the bytes — not only when the parser would. Schema changes (new key, renamed field) obviously qualify; so does a retrain, which keeps every shape identical and changes every value. Caught in a browser after the embedding rebuild: the page happily served the pre-retrain neighbours for Doubling Season out of cache while a cache-busted fetch of the same URL returned the new ones. A pure content *refresh* (same model, new Scryfall dump) still does not need it. Adding `membership` to `regions_*.json` broke drill-by-region for every browser that had already cached the old shape — politely, which is what made it expensive to find.
- **Frontend source-assertion tests catch nothing**: `test_viz_{camera,drill,deck_lens,viewer}.py` grep JS as text. A `ReferenceError` that killed drill mode outright passed all 13 drill tests. Real coverage lives in `tests/test_viz_behaviour.py` (playwright, `-m browser`) — add there when you change rendering or interaction. See `docs/testing.md`.
- **Two coordinate systems**: drill mode re-lays-out a subset from the embeddings, so a drilled position is **local** and is not the world map's. Anything anchored to world coords (region labels, search highlight, selection ring) must be suppressed or re-anchored via `Drill.localPosition()` while drilling. See `docs/viz.md`.
- Paths in `config.py` are `__file__`-anchored (CWD-independent); override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR`.
- **Two embedding spaces, two different jobs.** `embeddings.npy` is the **layout** space (colour/type) and feeds `projection_2d.json` *only* — its near-zero triplet loss is fine for a task whose whole content is "same colour, same type". `embeddings_ability.npy` is the **function** space and is the sole source of similarity: Find Similar, the walk and drill all read it regardless of which map is displayed (`SIMILARITY_EMBEDDINGS` in `viz/js/mana-map.js`). They used to follow the displayed map, which is why Doubling Season's neighbours were arbitrary green enchantments.
- **The function model was rebuilt after measuring a collapse**: it had been using 5.97 of its 128 dimensions and losing to the frozen MiniLM text it was built from. Now 27.87 effective dims, recall@10 0.093 → 0.245, median rank 995 → 78. Three changes: in-batch InfoNCE instead of a triplet margin (a margin stops teaching once satisfied), positives from `card_roles.json` rarest-role-first (the old ≥2-shared-tags rule fell back to random for most of the corpus, and `ROLE_BODY_FALLBACK` is excluded because it labels all 19,050 creatures), and a fixed-weight text passthrough making similarity exactly `0.7·cos_learned + 0.3·cos_text` so the model **cannot** discard the text as it did before.
- **Two decklist parsers, one contract, enforced by test.** `pilot/fetch_deck.py:parse_decklist` (CLI) and `viz/js/decklist.js` (browser paste-import) are checked against the same **hand-authored** fixtures in `tests/fixtures/decklists/` — hand-authored because generating them from Python would make Python the oracle and both sides would agree with its bugs. The contract is a **projection** onto `{name, quantity, is_commander, is_sideboard}`: Python also resolves printings and foils, the viz strips-and-discards them, and that is where the one real hazard lives (`_PRINTING_RE` is `$`-anchored, so `*F*`/`*CMDR*` must come off *first*). Imports resolve against `viz_index.json`, never `data/decks/index.json` — Deck Lens refuses unknown slugs and an imported deck has none.
- **`d3.drag` swallows clicks by default.** `clickDistance` is 0, so one pixel of movement between mousedown and mouseup suppresses the `click` event entirely — measured here: 0px delivered, 1px and 3px swallowed. Any canvas that has both a drag and a click handler needs an explicit tap tolerance (`force.js` uses 6) or interactions will feel intermittently dead in a way that looks like latency and isn't.
- **Discovery is the front door; `?mode=explore` gets the atlas.** The landing is one random card (`viz/js/discovery.js`), not the 34,322-point scatter. It is **the same force engine as The Walk with different chrome** — `Force.enter(rows, label, {chrome:'discovery'})` — because a second simulation would be the duplicate-k-NN mistake this repo has undone twice. Boot costs 2.26 MB (viz_index + neighbours) against the 18.4 MB it used to take to reach a first branch, and **branching is synchronous**: nothing in discovery reads the embeddings, so they are never fetched on that path. A *seeded* walk (deck/region) still awaits them — `linkWithinFromTable` only links cards whose top-12 are also in the set, which on a 97-card deck is 38 links instead of ~290.
- **Anything called during mana-map.js's boot runs INSIDE the IIFE, before `window.MM` exists.** Touching `MM.*` there throws, which aborts the IIFE, so `MM` is never exported and every later module fails at its own top level too — one ordering mistake, four broken files. Discovery takes its URLs by injection (`Discovery.configure`) and the boot mode is applied in a `queueMicrotask` for exactly this reason.
- **`neighbours.bin` is pre-sorted and must never be re-sorted client-side.** It carries 12 similar + 10 synergy + 5 obsoleted-by row ids per card so the browser can branch *synchronously* — no await mid-gesture, and without the 16.8 MB embedding matrix (2.4 MB gzipped for the whole discovery boot, against 18.4 MB before). Its similarities are uint8-quantised **for edge length only**; ordering is the array order. Re-sorting by a lossy value changes the top-10 for ~two thirds of cards, because the space is a narrow cone (median pairwise cosine 0.714) — it would read as a model regression rather than a precision bug. The header carries a sha256 of the embeddings it was built from and `tests/test_viz_index.py` fails if they diverge, because a stale table parses fine and answers confidently.
- **`manamap eval-embeddings` (step 15) is how you know.** It scores every embedding artifact against `data/eval/similarity_golden.json` — hand-authored, and it must **stay** hand-authored: training mines positives from tags/roles, so an eval derived from those would only measure whether training memorised its own supervision. Quote the **test** split; `dev` was used while diagnosing. **Do not tune hyperparameters on it** — sweeping the text weight looked like a win (0.258 recall@10) until selecting on `dev` picked a different value and the splits disagreed; at ~50 dev / ~160 test queries those differences are noise. `tests/test_embedding_quality.py` holds regression floors plus one still-failing `xfail(strict=True)` gate (neighbour spread 0.0315 vs a 0.05 target) whose threshold was deliberately not lowered to match the result.
- **⚠ 23 cache routines are MISSed right now, on purpose.** The embedding rebuild regenerated `synergy_graph.json` + `obsolescence_index.json`, so `writer-prose`, `the-ten` and `issue-plan` are stale on all seven decks (hapatra also has `candidate-pool` and `deck-build`). Re-spawning all of them is ~2.46M tokens, so it was left as a deliberate human decision — **do not `cache-record` these to make the board green**, and do not treat a MISS here as a bug. The record is a claim that someone read the artifact and agreed it still holds. Full breakdown and the re-bless-vs-re-spawn call: `PLAN.md`.
- **Agent cache**: subagent spawns are the only LLM cost (there are no LLM calls in Python). Always `cache-status` before spawning, `cache-record` **after** validating. Editing a `.claude/agents/*.md` prompt invalidates that agent's routines by design; `build-manual` is deliberately uncached. Costs and per-routine sizing: `docs/agent-cost.md`.
- **Strategy DB staleness**: any edit to `data/strategy/strategy.md` requires `manamap pilot build-strategy-db` — `load_strategy_db` hard-errors on a sha256 mismatch. Doc + CHANGELOG are tracked; the derived index/embeddings are gitignored.
- **The combo data is two files**: `combo_graph.json` is `{"partners": {...}}` **only** — it's what the viz fetches on the main thread, so nothing else belongs in it. The per-combo records live in `combo_details.json` (`{combos, by_card, meta}`, Python/agents only) with `bracket`, `mana_value_needed`, `popularity`. If you remember a `combos` key on the graph, that moved. Sizes in `docs/data-artifacts.md`.
- **Combo data is format-agnostic**: Commander Spellbook combos may assume a card is your commander ("Infinite commander casts" in `produces` is the tell) — verify lines with a resolve-stack run before presenting them as fact (stack 004 refuted one this way; `bracket.py` now excludes such lines automatically). Their per-combo `bracket` tag is also not gospel: it tags a real Hapatra two-card infinite as bracket 1, which is why the engine runs its own infinite test.
- **`bracket-check` needs a pipeline run**: the Game Changers signal is the `game_changer` column in `cards.csv`, which is gitignored. A fresh clone can render manuals but cannot compute a bracket floor until `manamap extract` has run.
- **`deck-facts` first, always**: `manamap pilot deck-facts <slug>` is the deterministic brief — DFC-correct colours, curve, pip load, role coverage and holes, contained combos, and a `notes[]` block naming the traps. It is computed on demand and never committed (same rule as `artist-credits`). Every deck agent is told to run it before deriving anything; re-deriving by hand costs tokens and has produced wrong answers.
- **One rules domain per scenario.** The checker's verdict is atomic over the whole artifact, so citation count predicts iterations — small scenarios pass, big ones fail and take correct answers down with them. `RESOLVE_SCOPE_BUDGET` warns; `validate-stack --scenario-only` preflights for free before any spawn. The measured evidence is in `docs/pilot.md`.
- **Agents hand off by path, not inline JSON**: deck agents write `data/decks/<slug>/.agent-out/<agent>.json` (gitignored) and return the path plus a summary; the orchestrator validates and merges. Returning a large artifact inline burns context for nothing.
- **Count copies, not decklist entries**: `cards.json` stores basics as one entry with `quantity: N`, so anything the shuffler would see (land totals, colour sources, hypergeometric draws) must go through `common.expand_copies()`. Counting entries once published "18 lands" for a 33-land deck and understated every colour fleet-wide. `mana_analysis` reports `lands.total` (copies) beside `lands.entries` (distinct cards); `validate-issue` lints prose that quotes the entry count as a land count. `artist_credits` counts entries **on purpose** — authorship is per card.
- **Never transcribe the section list or its count into a prompt**: read `issue_spec.DEPARTMENTS`. The magazine-editor's charter once enumerated the old 15 ids, in the old order, three lines after telling itself to read the spec. `tests/test_docs_section_count.py` fails on any stale count or hardcoded id list.
- **The deck manifest is generated, not hand-kept**: `manamap pilot build-index` writes `data/decks/index.json` (deck list + each deck's passing stack filenames) because a browser can list neither `data/decks/` nor `stacks/`. `viz/deck.html` reads it; a test asserts it matches the artifacts. Add a deck, run `build-index`.
- **`mana_analysis.json` is tracked and staleness-tested**: a decklist edit or a change to the maths needs `manamap pilot mana-analysis <slug>`, or `tests/test_pilot_mana_analysis.py` fails. Run it AFTER `goldfish`, since it embeds goldfish figures.
- Lint/format/CI intentionally not set up; revisit if the project grows.

## Pointers

- `docs/architecture.md` — models, training mining, mechanical tags, deckbuilding roles, synergy rules, power-creep criteria, region clustering
- `docs/pipeline.md` — all 15 steps: commands, inputs/outputs, runtimes, when to re-run what
- `docs/deck-builder-v2.md` — the deck builder's design record: bracket engine, role taxonomy, the architect ⇄ critic loop, and where the implementation departed from the design
- `docs/data-artifacts.md` — every `data/` file: producer, size, git status, consumers
- `docs/viz.md` — frontend structure, `window.MM` API, DATA map, the deck dossier, Pages deployment
- `docs/testing.md` — test layout, skip markers, conventions
- `docs/agent-cost.md` — where LLM spend lives, per-routine token sizing, the invocation cache
- `docs/pilot.md` — pilot subsystem: three-tier evidence contract, citation contract, rules DB, strategy DB + strategy-researcher agent, resolve loop, build loop, goldfish, manual generation
- `PLAN.md` — ACTIVE plan: current state, what's done, what's next (read this first when resuming work)
- `STYLEv3.md` — the magazine's editorial + design constitution (the 17-section five-act system, the Commander Mandate, the three columnists, L10, voice, component library); read before touching `build_manual.py`, `design.py`, or `issue_spec.py`
- `docs/frontend-v2.md` — the deck-building surface: what shipped (the dossier), what's next (`viz_index.json`, the Worker port), and the audit header saying which of its premises expired
- `docs/history/PLAN.md` — historical deck-builder planning doc (outdated, unmaintained)
