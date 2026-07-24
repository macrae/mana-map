---
name: run-tests
description: Run the Mana Map test suite and interpret results. Use before commits, after refactors, and after pipeline runs.
---

# Run the tests

```bash
.venv/bin/python -m pytest                # full suite: 261 tests, discovery via tests/
.venv/bin/python -m pytest tests/test_synergy.py -v    # single file
```

## Interpreting results

- **220 unit tests** need no data files and must always pass.
- **41 data-dependent tests** (`test_pipeline_integration.py`, `test_find_similar.py`) auto-skip when `data/` artifacts are missing (the `requires_data` marker in `tests/conftest.py` gates on `embeddings.npy`). Skips are normal on a fresh clone; failures on an existing `data/` usually mean artifacts are mutually inconsistent (partial pipeline run) — fix by re-running the pipeline from the changed step, not by editing tests.
- The suite is CWD-independent and honors `MANAMAP_DATA_DIR` (point it elsewhere to sandbox or force skips).

Conventions for writing new tests: `docs/testing.md`. Keep unit tests self-contained (inline DataFrames, no fixture files); put shared fixtures in `tests/conftest.py`.
