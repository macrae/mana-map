# Pipeline

Twelve steps orchestrated by `src/manamap/pipeline.py`. Run everything with `manamap run`, resume with `manamap run --from STEP`, or run any single step with `manamap <step>` (see `manamap --help`).

| # | CLI step | Module | What it does | Output |
|---|----------|--------|--------------|--------|
| 1 | `download` | `ingest/download.py` | Fetches Scryfall oracle_cards bulk JSON (idempotent via `.download-meta.json` sidecar) | `data/oracle-cards.json` |
| 2 | `extract` | `ingest/extract.py` | Parses JSON → flat CSV with derived columns (supertype, primary_color, mechanical_tags, embedding_text) | `data/cards.csv` |
| 3 | `preprocess` | `ingest/preprocess.py` | Sentence embeddings (all-MiniLM-L6-v2, frozen), categorical encoding, keyword + tag multi-hot | `data/text_embeddings.npy`, `data/card_features.npz`, `data/color_vectors.npy`, `data/mechanical_tags.npy` |
| 4a | `train` | `training/train.py` | Triplet training — positives by (supertype, primary_color) | `data/model.pt` |
| 4b | `train-ability` | `training/train_ability.py` | Triplet training — positives by tag overlap (>= 2 shared) | `data/model_ability.pt` |
| 5 | `embed` | `training/embed.py` | Runs all cards through both models, builds metadata CSV | `data/embeddings.npy`, `data/embeddings_ability.npy`, `data/card_metadata.csv` |
| 6 | `reduce` | `export/reduce.py` | PaCMAP 128D → 2D, both projections | `data/projection_2d.json`, `data/projection_2d_ability.json` |
| 7 | `download-combos` | `ingest/download_combos.py` | Paginates Commander Spellbook API (~2.5 min, internet) | `data/combos_raw.json` |
| 8 | `process-combos` | `ingest/process_combos.py` | Builds combo partner graph | `data/combo_graph.json` |
| 9 | `export` | `export/export_embeddings.py` | Both embeddings → raw Float32 binary for JS | `data/embeddings.bin`, `data/embeddings_ability.bin` |
| 10 | `synergy` | `analysis/synergy.py` | Synergy partner graph from complementary tag rules | `data/synergy_graph.json` |
| 11 | `power-creep` | `analysis/power_creep.py` | Strictly-better replacement index | `data/obsolescence_index.json` |
| 12 | `cluster-regions` | `analysis/cluster_regions.py` | HDBSCAN named regions, both maps, 2 zoom levels | `data/regions_default.json`, `data/regions_ability.json` |

Steps 1 and 7 need internet. Every module also keeps a main-guard, so `python -m manamap.ingest.download` etc. works too.

## When to re-run what

- **New Scryfall data** (new sets): full `manamap run`. Card count changes invalidate *everything* downstream — the index-alignment invariant (`projection[i]` == `cards.csv[i]`) means partial re-runs on changed data produce inconsistent artifacts.
- **Changed `MECHANICAL_TAGS`**: retrain required — `manamap run --from preprocess`. Tag dim changes make `model_ability.pt` incompatible.
- **Changed `SYNERGY_RULES` / obsolescence thresholds / region params**: only steps 10–12 — `manamap synergy && manamap power-creep && manamap cluster-regions`. Fast (no retraining).
- **Changed viz only**: nothing to re-run; bump the cache-bust `?v=` in `viz/index.html`.

## Approximate runtimes (Apple Silicon, MPS)

- Steps 1–2: ~1 min (download size ~200MB)
- Step 3: ~5–10 min (sentence embeddings for ~34K cards)
- Steps 4a/4b: a few minutes each (early stopping: Color+Type ~7 epochs, ability ~16)
- Step 5: ~1 min · Step 6: ~5 min (PaCMAP) · Steps 7–8: ~3 min · Steps 9–12: ~2 min

Training uses MPS on Apple Silicon, falling back to CUDA then CPU (`training/common.py:get_device`).
