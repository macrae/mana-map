# Testing

```bash
.venv/bin/python -m pytest          # full suite (discovery via testpaths = tests/)
```

261 tests in `tests/`. Two categories:

**Unit tests (220) — no data files needed, run anywhere:**

| File | Tests | Covers |
|------|-------|--------|
| `test_extract.py` | 51 | Multi-face cards, derived columns, supertype classification |
| `test_preprocess.py` | 23 | Vocab building, encoding, normalization, multi-hot |
| `test_mechanical_tags.py` | 45 | All 33 tag regexes, removal edge cases, multi-hot |
| `test_synergy.py` | 19 | Rule matching, bidirectionality, combo exclusion, ranking |
| `test_power_creep.py` | 36 | Strictly-better detection, tiered similarity gate, stat parsing |
| `test_combos.py` | 15 | Combo extraction, graph building, dedup |
| `test_cluster_regions.py` | 31 | Region naming (color/type/guild/TF-IDF), geometry, dedup |

**Data-dependent tests (41) — need artifacts from a pipeline run:**

| File | Tests | Covers |
|------|-------|--------|
| `test_pipeline_integration.py` | 29 | Cross-artifact count consistency, output quality checks |
| `test_find_similar.py` | 12 | Binary format fidelity, L2 normalization, 128D vs 2D ranking |

Both are skip-guarded: `test_pipeline_integration.py` skips per-file via `requires_file(...)`; `test_find_similar.py` uses the module-level `requires_data` marker from `tests/conftest.py` (gates on `embeddings.npy` existing).

## conftest.py

- `requires_data` — skipif marker for artifact-dependent tests
- `data_dir` — session fixture returning the resolved `config.DATA_DIR`

Paths come from `manamap.config` (never hardcode `Path("data")`), so the suite runs from any CWD and honors `MANAMAP_DATA_DIR`:

```bash
MANAMAP_DATA_DIR=/nonexistent .venv/bin/python -m pytest   # data tests skip cleanly
```

## Notes for writing tests

- Unit tests build inline DataFrames/dicts — keep it that way (no fixture files)
- `test_synergy.py` patches `manamap.analysis.synergy.load_combo_partners` to stub combo I/O
- Integration count assertions enforce the index-alignment invariant — if you change the card count (new Scryfall data), re-run the full pipeline before expecting green
