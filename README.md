# Mana Map

An interactive 2D map of every Magic: The Gathering card, built by embedding 33,504 oracle cards into a shared vector space and projecting them with PaCMAP.

Cards that play alike land near each other — red burn spells cluster together, green fatties form their own continent, and multicolor bombs float between their guilds.

## Quickstart

```bash
# Requires Python 3.10 (PyTorch has no 3.14 wheels)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (download → extract → preprocess → train → embed → reduce)
python pipeline.py

# Launch the visualization
python -m http.server 8000
# Open http://localhost:8000/viz/index.html
```

## Visualization

The interactive scatter plot renders all 33k cards as WebGL points via Plotly.js:

- **Color by** Primary Color, Supertype, or Rarity
- **Filter** by supertype using toggle buttons
- **Search** for any card by name — matches appear as white diamonds
- **Pan & zoom** to explore clusters

## How It Works

1. **Download** — Fetch Scryfall oracle card bulk data
2. **Extract** — Parse JSON into a flat CSV with derived columns (supertype, primary color, embedding text)
3. **Preprocess** — Compute sentence embeddings (all-MiniLM-L6-v2), encode categoricals, build keyword vectors
4. **Train** — Triplet margin loss on a fusion MLP that combines text, type, color, rarity, and keywords into 128-dim embeddings
5. **Embed** — Run all cards through the trained model
6. **Reduce** — PaCMAP projects 128 dims down to 2D for visualization

## Project Structure

```
pipeline.py          # Orchestrates all 6 steps
config.py            # All paths, hyperparams, vocab sizes
download.py          # Step 1: Scryfall bulk data download
extract.py           # Step 2: JSON → CSV with derived columns
preprocess.py        # Step 3: Feature extraction (text, categorical, keywords)
model.py             # CardEmbeddingModel (fusion MLP, ~181K params)
train.py             # Step 4: Triplet margin loss training
embed.py             # Step 5: Generate 128-dim embeddings
reduce.py            # Step 6: PaCMAP 2D reduction → JSON export
viz/index.html       # Interactive Plotly.js visualization
test_extract.py      # Tests for extraction logic
test_preprocess.py   # Tests for feature encoding
```
