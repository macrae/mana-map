# CLAUDE.md — Mana Map

## What This Project Is

Mana Map is an MTG card embedding pipeline that downloads all 33,504 oracle cards from Scryfall, trains a lightweight neural network to embed them into a 128-dim vector space, then projects them to 2D for interactive visualization. The entire pipeline runs locally on a Mac.

## Environment

- **Python 3.10** via conda `py310` env — PyTorch does NOT have wheels for Python 3.14
- **Venv**: `.venv` in project root (created from the conda py310 base)
- **Device**: MPS (Apple Silicon) for training; falls back to CUDA then CPU
- **Key version constraints**: `sentence-transformers<4`, `numpy<2` (required for PyTorch 2.2.2 compat)
- Install deps: `pip install -r requirements.txt`
- Run everything: `python pipeline.py`

## Pipeline Steps

| Step | File | What it does | Output |
|------|------|-------------|--------|
| 1 | `download.py` | Fetches Scryfall oracle_cards bulk JSON | `data/oracle-cards.json` |
| 2 | `extract.py` | Parses JSON → flat CSV with derived columns (supertype, primary_color, embedding_text) | `data/cards.csv` (33,504 rows) |
| 3 | `preprocess.py` | Sentence embeddings (all-MiniLM-L6-v2, frozen), categorical encoding, keyword multi-hot vectors | `data/text_embeddings.npy`, `data/card_features.npz`, `data/color_vectors.npy` |
| 4 | `train.py` | Triplet margin loss training with attribute-based mining on (supertype, primary_color) groups | `data/model.pt` |
| 5 | `embed.py` | Runs all cards through trained model, builds metadata CSV | `data/embeddings.npy`, `data/card_metadata.csv` |
| 6 | `reduce.py` | PaCMAP dimensionality reduction (128D → 2D), exports compact JSON | `data/projection_2d.json` |

`pipeline.py` orchestrates steps 1–6 in order.

## Architecture

### Model (`model.py`)
- **CardEmbeddingModel**: fusion MLP, ~181K trainable parameters, 128-dim L2-normalized output
- Inputs: text embedding (384) + supertype (16) + rarity (8) + color_identity (32) + layout (16) + continuous (2) + keywords (50) = 508-dim concatenation
- 3-layer MLP: 508 → 256 → 128 → 128, with ReLU and dropout
- Text encoder (all-MiniLM-L6-v2) is **frozen** — only the MLP trains

### Training (`train.py`)
- Triplet margin loss (margin=0.3) with online mining
- Positives: same (supertype, primary_color) group; negatives: different supertype AND color
- Converges very fast (~2 epochs to near-zero loss); 40 epochs configured but early convergence is normal
- Batch size 256, Adam optimizer, lr=1e-3, 90/10 train/val split

### Visualization (`viz/index.html`)
- Single-file HTML with inline CSS/JS, Plotly.js from CDN (`scattergl` for WebGL)
- Loads `data/projection_2d.json` via fetch (serve from project root)
- Features: color-by dropdown (Primary Color / Supertype / Rarity), supertype toggle filters, debounced card search (300ms, exact→prefix→substring), hover tooltips
- Dark theme (#1a1a2e background, #c4a747 gold accents)
- Serves via `python -m http.server 8000` → `http://localhost:8000/viz/index.html`

## Configuration (`config.py`)

All constants live here — paths, hyperparameters, vocab sizes, embedding dims. No magic numbers in other files.

Key paths:
- `DATA_DIR = Path("data")` — all data artifacts
- `VIZ_DIR = Path("viz")` — visualization files
- `PROJECTION_PATH = DATA_DIR / "projection_2d.json"` — 2D projection for viz

## Data Artifacts (all in `data/`, gitignored)

| File | Shape / Size | Description |
|------|-------------|-------------|
| `oracle-cards.json` | ~200MB | Raw Scryfall bulk data |
| `cards.csv` | 33,504 rows | Extracted flat CSV |
| `text_embeddings.npy` | (33504, 384) ~49MB | Frozen sentence embeddings |
| `card_features.npz` | varies | Categorical indices, continuous features, keywords |
| `color_vectors.npy` | (33504, 5) | WUBRG binary vectors |
| `model.pt` | ~711KB | Trained model checkpoint |
| `embeddings.npy` | (33504, 128) ~16MB | Final card embeddings |
| `card_metadata.csv` | 33,504 rows | oracle_id, name, supertype, colors, cmc, rarity |
| `projection_2d.json` | ~3MB | 2D PaCMAP projection for viz |

## Tests

```bash
pytest test_extract.py test_preprocess.py -v
```

- `test_extract.py` — tests for multi-face card handling, derived column logic
- `test_preprocess.py` — 23 tests for vocab building, categorical encoding, normalization, keyword multi-hot, color vector parsing

## Common Tasks

**Re-run just the visualization step:**
```bash
python reduce.py
```

**Serve the visualization:**
```bash
python -m http.server 8000
# http://localhost:8000/viz/index.html
```

**Run individual pipeline steps:**
```bash
python download.py   # Step 1
python extract.py    # Step 2
python preprocess.py # Step 3 (slow — computes sentence embeddings)
python train.py      # Step 4
python embed.py      # Step 5
python reduce.py     # Step 6
```

## Gotchas

- `pacmap` requires `numba` + `llvmlite`. On macOS, install specific versions with pre-built wheels: `pip install llvmlite==0.41.1 numba==0.58.1` before `pip install pacmap`
- The `data/` directory is gitignored. Run `python pipeline.py` to regenerate everything from scratch (requires internet for Step 1)
- The viz fetches `../data/projection_2d.json` relative to `viz/index.html` — must serve from project root
- Training converges to near-zero triplet loss very quickly; this is expected since the margin (0.3) is easily satisfied
