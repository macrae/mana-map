---
name: run-tests
description: Run the Mana Map test suite and interpret results. Use before commits, after refactors, and after pipeline runs.
---

# Run the tests

```bash
make test                 # THE INNER LOOP: non-browser, non-forge, -n auto, cached (~22-40 s)
make test-fresh           # same with nothing served from the regenerate-and-compare cache
make test-browser         # the playwright suite (-n 4, plus the one serial_only test)
pytest -m forge           # ONE real Forge game (~10 s; needs ~/.mana-map/forge; opt-in)
.venv/bin/pytest -n0 -k NAME   # one test, no worker startup
.venv/bin/pytest -m ""    # literally everything
```

A bare `pytest` is `make test` — `addopts` carries `-m 'not browser and not forge' -n auto`.

## Interpreting results

- **Counts live in one place**: `docs/testing.md` (the only file allowed to state them);
  re-derive with `.venv/bin/python -m pytest -m "not browser and not forge" --collect-only -q | tail -1`.
- **Data-gated tests auto-skip** on a fresh clone via the markers in `tests/conftest.py`
  (`requires_data` / `requires_rules` / `requires_deck` / `requires_strategy` / `requires_roles`).
  Skips are normal there; a *failure* on a developed checkout usually means `data/`
  artifacts are mutually inconsistent (a partial pipeline run) — fix by re-running the
  pipeline from the changed step, never by editing tests.
- **The regenerate-and-compare cache is real and never silent**: `make test` prints
  `N test(s) served from the cache`; `make test-fresh` runs them; a failing test never
  records; a corrupted artifact re-runs and fails rather than hiding.
- **The doc guards** (`tests/test_docs_counts.py`, `tests/test_docs_section_count.py`)
  fail when a doc states a stale count (agents, skills, subcommands, routines), omits a
  pilot subcommand from `docs/pilot.md`'s command block, or leaves a tracked per-deck
  file undocumented — they are the first thing to read when a docs-only change goes red.
- **One `xfail(strict=True)`** is a deliberately unmet ship gate (`test_embedding_quality.py`),
  not a broken test.
- The suite is CWD-independent and honors `MANAMAP_DATA_DIR` (point it elsewhere to
  sandbox or force skips). The browser suite needs `playwright install chromium`.

Conventions for writing new tests: `docs/testing.md`. Keep unit tests self-contained
(inline data, no fixture files unless the fixture is a real artifact worth pinning —
`tests/fixtures/forge/` holds two real Forge logs for exactly that reason); shared
fixtures in `tests/conftest.py`.
