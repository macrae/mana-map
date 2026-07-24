# Mana Map

An interactive 2D map of every Magic: The Gathering card, built by embedding ~34,300 oracle cards into a shared vector space and projecting them with PaCMAP.

Cards that play alike land near each other — red burn spells cluster together, green fatties form their own continent, and multicolor bombs float between their guilds.

## Quickstart

```bash
# Requires Python 3.10 (PyTorch has no 3.14 wheels)
python -m venv .venv
source .venv/bin/activate

# macOS: install prebuilt numba wheels first (pacmap dependency)
pip install llvmlite==0.41.1 numba==0.58.1
pip install -e ".[dev]"

# Run the full pipeline (12 steps; steps 1 and 7 need internet)
manamap run

# Launch the visualization (from the repo root)
python -m http.server 8000
# Open http://localhost:8000/viz/index.html
```

## Features

### Two Maps
- **Color+Type Map** — cards cluster by color identity and supertype
- **Abilities Map** — cards cluster by what they *do* (all blink cards together regardless of color)

### Explore Mode
- **Color by** Primary Color, Supertype, or Rarity; **filter** by supertype
- **Search** by card name or oracle text (4-tier fallback)
- **Multi-select** up to 8 cards (Shift+click or Shift+drag), full keyboard navigation
- **Find Similar** — 20 nearest neighbors in 128D embedding space
- **Find Synergies** — complementary cards (blink finds ETB, sac finds death triggers)
- **Obsolescence** — strictly-better replacements per card (power creep detection)
- **Named regions** — HDBSCAN-clustered map regions with zoom-dependent labels, optional density contours

### Deck Builder
- 8 formats (Standard, Modern, Legacy, Vintage, Commander, Pioneer, Pauper, Historic)
- Click cards as seeds, get recommendations (6-factor scoring: similarity, combos, synergies, popularity, curve fit, keywords)
- Auto-generate an optimized mana base (greedy set cover)
- Analytics (mana curve, color distribution), upgrade warnings, clipboard export

## How It Works

Twelve pipeline steps, orchestrated by the `manamap` CLI (`manamap run`, or `manamap <step>` individually):

| Step | What it does |
|------|-------------|
| 1 `download` | Fetch Scryfall oracle card bulk data |
| 2 `extract` | Parse JSON into flat CSV with derived columns and mechanical tags |
| 3 `preprocess` | Sentence embeddings (all-MiniLM-L6-v2), categorical + keyword/tag encoding |
| 4a `train` | Triplet loss — positives by (supertype, color) groups |
| 4b `train-ability` | Triplet loss — positives by mechanical tag overlap |
| 5 `embed` | Run all cards through both models → 128-dim embeddings |
| 6 `reduce` | PaCMAP 128D → 2D for both maps |
| 7 `download-combos` | Fetch Commander Spellbook combo data |
| 8 `process-combos` | Build combo partner graph |
| 9 `export` | Embeddings → raw Float32 binary for the browser |
| 10 `synergy` | Synergy graph from 24 complementary tag rules |
| 11 `power-creep` | Strictly-better replacement detection (tiered similarity gate) |
| 12 `cluster-regions` | HDBSCAN named map regions at two zoom levels |

Two lightweight fusion MLPs (~180K params each) produce 128-dim L2-normalized embeddings; the text encoder (all-MiniLM-L6-v2) stays frozen. The ability model deliberately shrinks its color inputs so cards cluster by function instead of color.

## Project Structure

```
src/manamap/           # Python package
  config.py            # all constants: paths, hyperparams, tag patterns, synergy rules
  pipeline.py, cli.py  # step registry + `manamap` CLI
  ingest/              # download, extract, preprocess, combos
  training/            # model, both trainers, embed
  export/              # PaCMAP reduce, binary export
  analysis/            # synergy, power_creep, cluster_regions
tests/                 # 261 pytest tests
data/                  # artifacts (viz-served files git-tracked; the rest regenerable)
viz/                   # static frontend: Plotly map + deck builder
docs/                  # architecture, pipeline, data, viz, and testing reference
```

Deep dives: [architecture](docs/architecture.md) · [pipeline](docs/pipeline.md) · [data artifacts](docs/data-artifacts.md) · [visualization](docs/viz.md) · [testing](docs/testing.md)

## Tests

```bash
python -m pytest       # 261 tests; the 41 data-dependent ones skip until you've run the pipeline
```
