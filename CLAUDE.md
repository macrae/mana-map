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
tests/                # pytest suite (966 tests: 365 card-pipeline + 601 pilot),
                      # conftest markers: requires_data/rules/deck/strategy/roles
data/                 # artifacts; mostly gitignored, viz-served files tracked
viz/                  # static frontend: index.html (the map: explore / deck lens /
                      # build, plus drill — a re-layout from the embeddings) +
                      # deck.html (the dossier); Plotly CDN, IIFE, window.MM
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
manamap run                   # full 13-step pipeline (steps 1 & 7 need internet)
manamap run --from STEP       # resume from a step
manamap <step>                # single step; see `manamap --help` for all 16 subcommands
manamap synergy && manamap power-creep && manamap cluster-regions && manamap card-roles
                              # fast analysis-only refresh (no retrain)
manamap pilot <cmd>           # build + publish subsystem (33 subcommands) — see docs/pilot.md

.venv/bin/python -m pytest    # 966 tests; data-dependent ones skip if artifacts missing

python -m http.server 8000    # serve viz FROM REPO ROOT
# http://localhost:8000/viz/index.html          the card map
# http://localhost:8000/viz/deck.html?deck=heliod   a deck's dossier
```

## Gotchas

- **Frozen config**: changing `MECHANICAL_TAGS` (or any model-facing dim in `config.py`) invalidates `model_ability.pt` — retrain steps 3–5. Don't touch config values in refactors. `ROLE_PATTERNS` is a **separate** dict for exactly this reason: roles change often, tags must not. Editing roles needs only `manamap card-roles` (step 13).
- **Roles ≠ mechanical tags**: `MECHANICAL_TAGS` is a retrieval vocabulary ("what is this card like"); `ROLE_PATTERNS` answers "what job does it do in a 99". One `ramp` tag versus five `ramp:rock|dork|land|ritual|cost-reduction` roles is the canonical difference — a curve model that conflates a Signet with a Dark Ritual is wrong.
- **Index alignment**: `projection[i]` == `cards.csv[i]` == `embeddings[i]`. Never partially regenerate after the card count changes; re-run from the changed step onward.
- **No Git LFS on `data/`**: GitHub Pages serves LFS pointers, which would break the deployed viz. Large tracked JSON/bin files are intentional.
- **Viz serving root**: all fetches are `../data/<file>` relative to `viz/index.html` — `viz/` and `data/` must stay top-level siblings; serve from repo root.
- **Cache busting**: bump `?v=N` on the script/CSS tags in `viz/index.html` **and `viz/deck.html`** after any JS/CSS change. `index.html`'s four script busts must move together — a test asserts it, since a mismatched pair is how `deck-map.js` ends up calling a stale `mana-map.js`. `manuals/magazine.css` is content-addressed instead (`?v=<sha8>`), so a CSS edit there obligates rebuilding every manual page.
- **Synergy ≠ Similar**: synergy is complementary (blink→ETB, rule-based); Find Similar is embedding neighbors. Different algorithms.
- **Plotly**: `Plotly.relayout` fires `plotly_relayout` — guard against event loops. `Plotly.react` replaces layout wholesale, so `render()` must write the live axis range back or it silently resets the camera; `Plotly.restyle` preserves it and is the only fast path (~32ms on a 1,200-point `scattergl`).
- **Data cache-busting**: `MM.DATA` URLs carry `?v=DATA_VERSION`. Bump it when a data artifact's **schema** changes (new key, renamed field), not for content refreshes. Adding `membership` to `regions_*.json` broke drill-by-region for every browser that had already cached the old shape — politely, which is what made it expensive to find.
- **Two coordinate systems**: drill mode re-lays-out a subset from the embeddings, so a drilled position is **local** and is not the world map's. Anything anchored to world coords (region labels, search highlight, selection ring) must be suppressed or re-anchored via `Drill.localPosition()` while drilling. See `docs/viz.md`.
- Paths in `config.py` are `__file__`-anchored (CWD-independent); override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR`.
- Color+Type model hitting near-zero triplet loss by epoch ~3 is expected, not a bug.
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
- `docs/pipeline.md` — all 13 steps: commands, inputs/outputs, runtimes, when to re-run what
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
