# Pipeline

Thirteen steps orchestrated by `src/manamap/pipeline.py`. Run everything with `manamap run`, resume with `manamap run --from STEP`, or run any single step with `manamap <step>` (see `manamap --help`).

| # | CLI step | Module | What it does | Output |
|---|----------|--------|--------------|--------|
| 1 | `download` | `ingest/download.py` | Fetches Scryfall oracle_cards bulk JSON (idempotent via `.download-meta.json` sidecar) | `data/oracle-cards.json.gz` |
| 2 | `extract` | `ingest/extract.py` | Parses JSON → flat CSV (35 cols) with derived columns (supertype, primary_color, mechanical_tags, embedding_text) and Scryfall's `game_changer` flag | `data/cards.csv` |
| 3 | `preprocess` | `ingest/preprocess.py` | Sentence embeddings (all-MiniLM-L6-v2, frozen), categorical encoding, keyword + tag multi-hot | `data/text_embeddings.npy`, `data/card_features.npz`, `data/color_vectors.npy`, `data/mechanical_tags.npy` |
| 4a | `train` | `training/train.py` | Triplet training — positives by (supertype, primary_color) | `data/model.pt` |
| 4b | `train-ability` | `training/train_ability.py` | Triplet training — positives by tag overlap (>= 2 shared) | `data/model_ability.pt` |
| 5 | `embed` | `training/embed.py` | Runs all cards through both models, builds metadata CSV | `data/embeddings.npy`, `data/embeddings_ability.npy`, `data/card_metadata.csv` |
| 6 | `reduce` | `export/reduce.py` | PaCMAP 128D → 2D, both projections | `data/projection_2d.json`, `data/projection_2d_ability.json` |
| 7 | `download-combos` | `ingest/download_combos.py` | Paginates Commander Spellbook API (~2.5 min, internet) | `data/combos_raw.json.gz` |
| 8 | `process-combos` | `ingest/process_combos.py` | Builds the partner adjacency map **and** the per-combo detail index (Spellbook bracket tag, mana value needed, popularity, `by_card`) | `data/combo_graph.json`, `data/combo_details.json` |
| 9 | `export` | `export/export_embeddings.py` | Both embeddings → raw Float32 binary for JS | `data/embeddings.bin`, `data/embeddings_ability.bin` |
| 10 | `synergy` | `analysis/synergy.py` | Synergy partner graph from complementary tag rules | `data/synergy_graph.json` |
| 11 | `power-creep` | `analysis/power_creep.py` | Strictly-better replacement index | `data/obsolescence_index.json` |
| 12 | `cluster-regions` | `analysis/cluster_regions.py` | HDBSCAN named regions, both maps, 2 zoom levels | `data/regions_default.json`, `data/regions_ability.json` |
| 13 | `card-roles` | `analysis/card_roles.py` | Deckbuilding role taxonomy over type line + oracle text; reports its own coverage | `data/card_roles.json` |
| 14 | `viz-index` | `export/viz_index.py` | Discovery index + precomputed neighbour tables (all three relations) | `data/viz_index.json`, `data/neighbours.bin` |
| 15 | `eval-embeddings` | `analysis/eval_embeddings.py` | Reports embedding quality against a hand-authored golden set | *(none — the only reporting step)* |

Steps 1 and 7 need internet. Every module also keeps a main-guard, so `python -m manamap.ingest.download` etc. works too.

## When to re-run what

- **New Scryfall data** (new sets): full `manamap run`. Card count changes invalidate *everything* downstream — the index-alignment invariant (`projection[i]` == `cards.csv[i]`) means partial re-runs on changed data produce inconsistent artifacts.
- **Changed `MECHANICAL_TAGS`**: retrain required — `manamap run --from preprocess`. Tag dim changes make `model_ability.pt` incompatible.
> **⚠ Currently outstanding.** The embedding rebuild ran exactly this refresh, so
> `writer-prose`, `the-ten` and `issue-plan` are MISSed on all seven decks (hapatra also
> `candidate-pool` and `deck-build`) — 23 routines, ~2.46M tokens to re-spawn. Left MISSed
> deliberately; the re-bless-vs-re-spawn decision is a human's and is recorded in `PLAN.md`.

- **Changed `SYNERGY_RULES` / obsolescence thresholds / region params**: only steps 10–12 — `manamap synergy && manamap power-creep && manamap cluster-regions`. Fast (no retraining), but it invalidates **five** agent-cache routines that hash those graphs — `writer-prose`, `the-ten`, `issue-plan`, `candidate-pool`, `deck-build`. Verified prose is usually still correct after a graph refresh, so re-bless (`cache-record`) rather than re-spawn; make it a stated decision.
- **Changed `ROLE_PATTERNS`**: only step 13 — `manamap card-roles` (~10 s, no retraining). Roles are deliberately *not* model-facing, so unlike `MECHANICAL_TAGS` they never force a retrain. Note this invalidates the `candidate-pool` and `deck-build` agent-cache routines, which hash `card_roles.json`.
- **Changed the embedding text, features, or training objective**: `manamap run --from preprocess` — *not* from `download`. Two reasons, and the agent cache is not one of them: card digests are computed over each deck's `cards.json`, not over `data/cards.csv`, so regenerating the corpus does not touch them. The reasons are (a) a new Scryfall dump changes the card *count*, which invalidates the index-alignment invariant for every downstream artifact at once, and (b) it confounds the before/after quality comparison — you can no longer tell an embedding improvement from a corpus change. The real cache cost of a retrain comes later in the run: regenerating `synergy_graph.json` and `obsolescence_index.json` MISSes the five routines listed above, which is the *previous* bullet's problem, and the same advice applies (re-bless rather than re-spawn when the prose is still correct).
- **Changed viz only**: nothing to re-run; bump the cache-bust `?v=` on the page you touched — `viz/index.html` (map) or `viz/deck.html` (dossier). `manuals/magazine.css` is content-addressed instead, so a magazine stylesheet change means rebuilding every manual page.

## Approximate runtimes (Apple Silicon, MPS)

- Steps 1–2: ~1 min (download size ~200MB)
- Step 3: ~5–10 min (sentence embeddings for ~34K cards)
- Steps 4a/4b: a few minutes each (early stopping: Color+Type ~7 epochs, ability ~16)
- Step 5: ~1 min · Step 6: ~5 min (PaCMAP) · Steps 7–8: ~3 min
- Step 9: seconds · Step 10 (synergy): ~30 s · Step 11 (power-creep): ~4 min · Step 12 (regions): ~10 s · Step 13 (card-roles): ~10 s · Step 14 (viz-index): ~30 s · Step 15 (eval-embeddings): ~40 s

Training uses MPS on Apple Silicon, falling back to CUDA then CPU (`training/common.py:get_device`).
