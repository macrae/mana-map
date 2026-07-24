# Data Artifacts

Everything lives in `data/`. Most files are gitignored (regenerable via `manamap run`); the ones the deployed viz fetches are **git-tracked** via explicit `.gitignore` exceptions.

**Do NOT move these to Git LFS.** The viz is served by GitHub Pages straight from the repo, and Pages serves LFS pointer files, not content — LFS would silently break every data fetch on the deployed site.

| File | Producer (step) | Shape / size | Git | Consumed by |
|------|-----------------|--------------|-----|-------------|
| `oracle-cards.json` | download (1) | ~200MB | ignored | extract |
| `.download-meta.json` | download (1) | tiny | ignored | download (idempotency) |
| `cards.csv` | extract (2) | ~34,300 rows | ignored | preprocess, train×2, embed, synergy, power_creep, cluster_regions |
| `text_embeddings.npy` | preprocess (3) | (N, 384) ~50MB | ignored | train×2, embed |
| `card_features.npz` | preprocess (3) | ~13MB | ignored | train×2, embed |
| `color_vectors.npy` | preprocess (3) | (N, 5) | ignored | viz metadata build |
| `mechanical_tags.npy` | preprocess (3) | (N, 33) | ignored | train_ability |
| `model.pt` | train (4a) | ~711KB | ignored | embed |
| `model_ability.pt` | train_ability (4b) | ~711KB | ignored | embed |
| `embeddings.npy` | embed (5) | (N, 128) ~17MB | ignored | reduce, export, synergy (fallback) |
| `embeddings_ability.npy` | embed (5) | (N, 128) ~17MB | ignored | reduce, export, synergy, power_creep |
| `card_metadata.csv` | embed (5) | ~3MB | ignored | reduce |
| `projection_2d.json` | reduce (6) | ~13MB | **tracked** | viz (Color+Type map) |
| `projection_2d_ability.json` | reduce (6) | ~13MB | **tracked** | viz (Abilities map) |
| `combos_raw.json` | download-combos (7) | ~50–100MB | ignored | process-combos |
| `combo_graph.json` | process-combos (8) | ~24MB | **tracked** | viz deck builder, synergy (exclusions) |
| `embeddings.bin` | export (9) | ~17MB | **tracked** | viz (Find Similar, deck builder) |
| `embeddings_ability.bin` | export (9) | ~17MB | **tracked** | viz (Abilities map similarity) |
| `synergy_graph.json` | synergy (10) | ~8–27MB | **tracked** | viz (Find Synergies, deck builder) |
| `obsolescence_index.json` | power-creep (11) | ~5–8MB | **tracked** | viz (obsolescence panels) |
| `regions_default.json` | cluster-regions (12) | ~27KB | **tracked** | viz (region labels) |
| `regions_ability.json` | cluster-regions (12) | ~16KB | **tracked** | viz (region labels) |

N = card count, ~34,300 as of July 2026; grows as Scryfall adds sets.

## Consistency invariant

`projection[i]`, `embeddings[i]`, and `cards.csv[i]` all refer to the same card by **position**. Never partially regenerate after the card count changes — re-run the pipeline from the changed step onward (see `docs/pipeline.md`). The integration tests (`tests/test_pipeline_integration.py`) assert cross-artifact count consistency.

## Paths

All paths are defined in `src/manamap/config.py`, anchored to the repo root via `__file__` (CWD-independent). Override with `MANAMAP_DATA_DIR` / `MANAMAP_VIZ_DIR` env vars for sandboxed runs.
