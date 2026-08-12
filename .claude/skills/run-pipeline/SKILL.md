---
name: run-pipeline
description: Run the Mana Map data pipeline — full 15-step run, resume from a step, or single steps. Use when the user wants to (re)generate data artifacts, refresh Scryfall data, or rebuild any data/ file.
---

# Run the pipeline

All commands use the project venv. Full step reference: `docs/pipeline.md`.

```bash
.venv/bin/manamap run                # full pipeline (steps 1 & 7 need internet)
.venv/bin/manamap run --from STEP    # resume from a step
.venv/bin/manamap <step>             # one step; `manamap --help` lists all 16
```

Steps in order: `download`, `extract`, `preprocess`, `train`, `train-ability`, `embed`, `reduce`, `download-combos`, `process-combos`, `export`, `synergy`, `power-creep`, `cluster-regions`, `card-roles`, `viz-index`, `eval-embeddings`. (For a Scryfall refresh specifically, use the `refresh-corpus` skill — it wraps this run in the gates, doc sweep, cache pass and deploy checklist a refresh needs.)

## Rules

- **Never run `train`/`train-ability` casually** — retraining replaces `model.pt`/`model_ability.pt` (gitignored, unrecoverable) and every downstream artifact. Confirm with the user first unless they explicitly asked for a retrain.
- **Index alignment**: if `download`/`extract` ran (card count may change), everything downstream is stale — you must continue through step 14 (`viz-index`; it owns the `neighbours.bin` sha256 gate that `tests/test_viz_index.py` enforces, and `card_roles.json` is a build-routine cache input). Never leave `data/` partially regenerated.
- Long steps (preprocess, train×2, reduce) → run in background and verify artifacts by timestamp/shape afterward.
- Expected artifact per step is listed in `docs/pipeline.md`; spot-check with `ls -la data/`.

## Post-run verification

```bash
.venv/bin/python -m pytest tests/test_pipeline_integration.py tests/test_find_similar.py
```
42 tests validate cross-artifact consistency. All green = pipeline output is coherent.
