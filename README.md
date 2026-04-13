# Mana Map

An interactive 2D map of every Magic: The Gathering card, built by embedding ~33,700 oracle cards into a shared vector space and projecting them with PaCMAP.

Cards that play alike land near each other — red burn spells cluster together, green fatties form their own continent, and multicolor bombs float between their guilds.

## Quickstart

```bash
# Requires Python 3.10 (PyTorch has no 3.14 wheels)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (11 steps)
python pipeline.py

# Launch the visualization
python -m http.server 8000
# Open http://localhost:8000/viz/index.html
```

## Features

### Two Maps
- **Color+Type Map** — cards cluster by color identity and supertype
- **Abilities Map** — cards cluster by what they *do* (all blink cards together regardless of color)

### Explore Mode
- **Color by** Primary Color, Supertype, or Rarity
- **Filter** by supertype using toggle buttons
- **Search** by card name or oracle text (4-tier fallback)
- **Multi-select** up to 8 cards (Shift+click or Shift+drag)
- **Find Similar** — 20 nearest neighbors in 128D embedding space
- **Find Synergies** — complementary cards (blink finds ETB, sac finds death triggers)
- **Obsolescence** — see strictly-better replacements for each card (power creep detection)

### Deck Builder
- 8 formats (Standard, Modern, Legacy, Vintage, Commander, Pioneer, Pauper, Historic)
- Click cards as seeds, get AI-powered recommendations (5-factor scoring)
- Auto-generate optimized mana base (greedy set cover algorithm)
- Analytics: mana curve + color distribution
- Export to clipboard (compatible with MTGO, Arena, Moxfield, Archidekt)

## How It Works

| Step | File | What it does |
|------|------|-------------|
| 1 | `download.py` | Fetch Scryfall oracle card bulk data |
| 2 | `extract.py` | Parse JSON into flat CSV with derived columns and mechanical tags |
| 3 | `preprocess.py` | Sentence embeddings (all-MiniLM-L6-v2), categorical encoding, keyword/tag multi-hot |
| 4a | `train.py` | Triplet margin loss — positives by (supertype, color) groups |
| 4b | `train_ability.py` | Triplet margin loss — positives by mechanical tag overlap |
| 5 | `embed.py` | Run all cards through both models → 128-dim embeddings |
| 6 | `reduce.py` | PaCMAP 128D → 2D for visualization |
| 7 | `download_combos.py` | Fetch Commander Spellbook combo data |
| 8 | `process_combos.py` | Build combo partner graph |
| 9 | `export_embeddings.py` | Convert embeddings to binary format for JS |
| 10 | `synergy.py` | Build synergy graph from complementary mechanical tags |
| 11 | `power_creep.py` | Detect strictly-better card replacements |

### Models

Two lightweight fusion MLPs (~180K params each), both producing 128-dim L2-normalized embeddings:

- **Color+Type model** — groups cards by color identity and supertype
- **Ability model** — groups cards by function (mechanical tags), with color deliberately de-emphasized

Text encoder (all-MiniLM-L6-v2) is frozen — only the MLP trains. Triplet margin loss with early stopping.

### Mechanical Tags

33 regex-based tags extracted from oracle text, covering triggers (ETB, death, attack), effects (removal, draw, counterspell), generators (tokens, ramp, lifegain), modifiers (anthem, evasion), and more. ~80% coverage of non-land cards.

### Synergy Detection

24 complementary rules covering 27/33 tags (82%) — finds cards that *complete* each other (blink + ETB, tokens + anthem, sacrifice + death trigger, evasion + damage triggers, and more). Up to 10 partners per card, ranked by rule matches + ability embedding similarity. Different from "Find Similar" which finds embedding neighbors.

### Power Creep Detection

Finds strictly-better replacements using a tiered similarity gate:
- **2+ tag cards**: cosine similarity >= 0.75 in ability embedding space
- **1-tag cards**: cosine similarity >= 0.98 (stricter, since single tags are coarse)
- Plus: tag superset check, CMC/color/stats comparison, release date ordering

## Project Structure

```
pipeline.py            # Orchestrates all 11 steps
config.py              # All paths, hyperparams, vocab sizes, tag patterns, synergy rules
download.py            # Step 1: Scryfall bulk data
extract.py             # Step 2: JSON → CSV with derived columns
mechanical_tags.py     # Regex-based tag extraction (33 tags)
preprocess.py          # Step 3: Feature extraction
model.py               # CardEmbeddingModel (fusion MLP, ~180K params)
train.py               # Step 4a: Color+Type triplet training
train_ability.py       # Step 4b: Ability triplet training
embed.py               # Step 5: Generate embeddings
reduce.py              # Step 6: PaCMAP 2D reduction
download_combos.py     # Step 7: Commander Spellbook API
process_combos.py      # Step 8: Combo partner graph
export_embeddings.py   # Step 9: Embeddings → binary
synergy.py             # Step 10: Synergy graph
power_creep.py         # Step 11: Obsolescence detection
viz/
  index.html           # HTML structure
  css/mana-map.css     # Dark theme, responsive styles
  js/mana-map.js       # Explore mode (~1159 lines)
  js/deck-builder.js   # Deck builder (~1299 lines)
```

## Tests

```bash
# Unit tests (no data files required)
pytest test_extract.py test_preprocess.py test_combos.py test_mechanical_tags.py test_synergy.py test_power_creep.py -v

# Integration tests (requires data/ from pipeline run)
pytest test_pipeline_integration.py -v

# Embedding quality tests (requires data/)
pytest test_find_similar.py -v
```

225 tests total across 8 test files.
