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
  analysis/           # synergy, power_creep, cluster_regions, common
  pilot/              # pilot's-manual subsystem: rules DB + strategy KB (RAG),
                      # deck ingestion, citation-contract enforcement, goldfish
                      # simulator, manual renderer
tests/                # pytest suite (364 tests: 261 card-pipeline + 103 pilot),
                      # conftest markers: requires_data/rules/deck/strategy
data/                 # artifacts; mostly gitignored, viz-served files tracked
viz/                  # static frontend (Plotly CDN, two IIFE scripts, window.MM / window.DeckBuilder)
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
manamap run                   # full 12-step pipeline (steps 1 & 7 need internet)
manamap run --from STEP       # resume from a step
manamap <step>                # single step; see `manamap --help` for the 13 subcommands
manamap synergy && manamap power-creep && manamap cluster-regions   # fast analysis-only refresh
manamap pilot <cmd>           # pilot's-manual subsystem (14 subcommands) — see docs/pilot.md

.venv/bin/python -m pytest    # 364 tests; data-dependent ones skip if artifacts missing

python -m http.server 8000    # serve viz FROM REPO ROOT
# http://localhost:8000/viz/index.html
```

## Gotchas

- **Frozen config**: changing `MECHANICAL_TAGS` (or any model-facing dim in `config.py`) invalidates `model_ability.pt` — retrain steps 3–5. Don't touch config values in refactors.
- **Index alignment**: `projection[i]` == `cards.csv[i]` == `embeddings[i]`. Never partially regenerate after the card count changes; re-run from the changed step onward.
- **No Git LFS on `data/`**: GitHub Pages serves LFS pointers, which would break the deployed viz. Large tracked JSON/bin files are intentional.
- **Viz serving root**: all fetches are `../data/<file>` relative to `viz/index.html` — `viz/` and `data/` must stay top-level siblings; serve from repo root.
- **Cache busting**: bump `?v=N` on the script/CSS tags in `viz/index.html` after any JS/CSS change.
- **Synergy ≠ Similar**: synergy is complementary (blink→ETB, rule-based); Find Similar is embedding neighbors. Different algorithms.
- **Plotly**: `Plotly.relayout` fires `plotly_relayout` — guard against event loops.
- Paths in `config.py` are `__file__`-anchored (CWD-independent); override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR`.
- Color+Type model hitting near-zero triplet loss by epoch ~3 is expected, not a bug.
- **Strategy DB staleness**: any edit to `data/strategy/strategy.md` requires `manamap pilot build-strategy-db` — `load_strategy_db` hard-errors on a sha256 mismatch. Doc + CHANGELOG are tracked; the derived index/embeddings are gitignored.
- **Combo graph is format-agnostic**: Commander Spellbook combos may assume a card is your commander ("Infinite commander casts" in `produces` is the tell) — verify lines with a resolve-stack run before presenting them as fact (stack 004 refuted one this way).
- Lint/format/CI intentionally not set up; revisit if the project grows.

## Pointers

- `docs/architecture.md` — models, training mining, mechanical tags, synergy rules, power-creep criteria, region clustering
- `docs/pipeline.md` — all 12 steps: commands, inputs/outputs, runtimes, when to re-run what
- `docs/data-artifacts.md` — every `data/` file: producer, size, git status, consumers
- `docs/viz.md` — frontend structure, `window.MM` API, DATA map, Pages deployment
- `docs/testing.md` — test layout, skip markers, conventions
- `docs/pilot.md` — pilot subsystem: three-tier evidence contract, citation contract, rules DB, strategy DB + strategy-researcher agent, resolve loop, goldfish, manual generation
- `PLAN.md` — ACTIVE plan: current state, what's done, what's next (read this first when resuming work)
- `docs/history/PLAN.md` — historical deck-builder planning doc (outdated, unmaintained)
