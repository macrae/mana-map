# Testing

```bash
make test            # THE FAST TIER — what you run all day                    ~2.8 min
make test-fleet      # the slow tier: every producer re-run, fleet constants   ~11 min
make prepush         # both tiers — before every push (CI runs both too)
make test-fresh      # both tiers, nothing served from the cache — trust this one
make test-browser    # playwright, -n 4, then the serial_only tests            ~7 min
make test-all        # test-fresh + test-browser
pytest -n0 -k NAME   # one test; worker startup outweighs the split
pytest -m forge      # ONE real Forge game (needs ~/.mana-map/forge)
pytest -m ""         # literally everything, browser included
pytest --lf          # only what failed last time
```

**A bare `pytest` is `make test`.** `addopts` in `pyproject.toml` carries
`-m 'not browser and not forge and not fleet and not slow' -n auto`.

**This file is the only place that states test counts and runtimes.** They move on
almost every commit; README and CLAUDE.md point here instead. The dated record of
how the suite got here — every earlier measurement, every lesson in full, the
2026-09-21 adversarial audit — is [`history/testing-log.md`](history/testing-log.md).

## The tiers, measured 2026-10-05 (8-core Mac, `-n auto`)

| tier | selects | tests | wall |
|---|---|---:|---:|
| fast — `make test`, warm cache | everything not marked below | 4,023 | **2:46** |
| fast — `--no-test-cache` | | | **5:17** |
| fleet — `make test-fleet` | `slow or fleet` | 142 | **11:07** |
| browser — `make test-browser` | `browser` | 264 | ~7 min (2026-09-12) |
| forge | `forge` | 1 | ~10 s |
| **total collected** | | **4,430** | |

To print today's numbers rather than trust these:

```bash
.venv/bin/pytest --co -n0 -m "" | tail -1                     # everything
.venv/bin/pytest --co -n0 | tail -1                           # the fast tier
.venv/bin/pytest --durations=30                               # where the time goes
```

**What decides the tier is time, not importance.** The fleet tier is not optional
— `make prepush` and CI run it — it is the part that re-runs 10,000-game producers
per deck and branch (`slow`) or re-derives a calibrated constant across the whole
fleet (`fleet`). A test that takes over ~10 s is `slow`; keep the decision loop
(`try`, the paired intervals) in the fast tier even when it costs a few seconds,
because a regression there should surface the same day.

## Markers

Registered in `pyproject.toml`:

| marker | means | default run |
|---|---|---|
| `slow` | re-runs a 10,000-game producer, or thousands of goldfish games | excluded — fleet tier |
| `fleet` | re-derives a calibrated constant across every tracked deck | excluded — fleet tier |
| `browser` | drives a real Chromium via playwright | excluded — `make test-browser` |
| `serial_only` | asserts a wall-clock budget; run alone, `-n0` | excluded with `browser` |
| `forge` | runs the Forge engine headless | excluded — opt in |

## Skip conditions (`tests/conftest.py`)

A test that needs an artifact SKIPS when it is missing, with the command that
builds it in the reason. Each gates on the LAST artifact of its stage, so a
half-built `data/` skips cleanly. CI prints every reason (`-rs`): a skip nobody
sees is not a signal.

| marker | gates on | build it with |
|---|---|---|
| `requires_data` | `embeddings.npy` | `manamap run` |
| `requires_corpus` | `cards.csv` alone (CI's weekly job has it, the push job does not) | `manamap download && manamap extract` |
| `requires_rules` | the rules index | `manamap pilot download-rules && manamap pilot build-rules-db` |
| `requires_rulings` | the rulings dump | `manamap pilot download-rulings` |
| `requires_deck` | `data/decks/goblin-storm/cards.json` | `manamap pilot fetch-deck goblin-storm` |
| `requires_strategy` | the strategy index | `manamap pilot build-strategy-db` |
| `requires_roles` | `card_roles.json` | `manamap card-roles` |
| `requires_branch` | any branch on the branch deck | `manamap pilot deck-branch <slug> new …` |

Paths come from `manamap.config`, so the suite runs from any directory and honours
`MANAMAP_DATA_DIR` — `MANAMAP_DATA_DIR=/nonexistent pytest` skips every data test.

## The cache: `unchanged`

Every producer is seeded and deterministic, so a test that regenerates an artifact
and compares it to the tracked one gets the same answer from the same inputs. The
`unchanged` fixture skips such a test when every input it names is byte-identical
to the last passing run; `--no-test-cache` turns it off. The cache lives in the
gitignored `.pytest_cache/`, so a fresh checkout — and CI — runs everything.

**The key is only as good as its inputs, and a missing input is a cached PASS the
code never earned.** That has happened twice: freshness tests keyed on a
hand-picked file list (fixed by `module_closure`), and the validator gate keyed on
12 of 25 validator modules (fixed 2026-10-05 by deriving the list from
`deck_status.VALIDATED`, with a test holding it there).

| helper | does |
|---|---|
| `unchanged(*paths)` | call FIRST, with every file the answer depends on; skips if none moved |
| `module_closure(*modules)` | every `src/manamap/` file those modules reach, transitively — the code half of a key |
| `simulator_source()` | the goldfish's four files concatenated, for tests that read the simulator's text |
| `patch_model(monkeypatch, name, value)` | patch a goldfish name everywhere it is bound, and assert it landed |
| `declares_nothing(monkeypatch, …)` | make a deck's declaration read as empty, without needing a blind deck |
| `is_retired(deck)` | `regen.is_retired` — a retired deck is out of every fleet gate |
| `assert_corpus_count(got, expected, what)` | a corpus sweep held to a band of max(2, 3%) |

## Writing a test

Each rule cost something; the full story of each is in the log.

- **Drive the production function. A test that re-derives the rule is testing
  itself.** Then prove the test by RE-INTRODUCING the bug it was written for —
  blind the channel, revert the line — and watch it fail. A check never seen to
  fail proves nothing.
- **Assert behaviour, not a figure.** A pinned simulated number moves whenever
  anything else does: the Archivist floor was re-baselined nine times in three
  weeks by changes that never touched the wheels it guarded. Assert the delta the
  test exists for.
- **A corpus count is a band, not an exact number** (`assert_corpus_count`). A set
  release adds a handful; a moved pattern moves dozens. The card-by-card reading of
  what a pattern change matched belongs in the commit that changes it.
- **A loop over a possibly-empty collection needs `assert checked >= N`.** Fourteen
  tests passed by iterating zero times.
- **Wait for the condition, never for a timer** (browser). All five historical
  browser flakes were a fixed `wait_for_timeout` measuring the machine.
- **A test that inherits a default is testing the default.** Pass the value.
- **Do not assert on things outside the code's control** — the clock, the network,
  another deck staying half-finished.
- **Do not test a fact twice.** Five tests re-ran a tracked artifact against a fresh
  one that `test_pilot_artifact_freshness.py` already compares for every deck; they
  were deleted 2026-10-05. Before writing a "matches the tracked file" test, check
  the freshness file.
- **A goal is a strict xfail.** `xfail(strict=True)` with the reason naming where
  the issue is tracked: it fails loudly the day it starts passing.
  `test_no_two_axes_measure_the_same_thing` is one (known-issues 9b).
- **Unit tests build inline dicts and DataFrames.** No fixture files.
- **Import `conftest`, never `tests.conftest`.** `python -m pytest` puts the repo
  root on `sys.path` and the console script `pytest` does not, so `tests.conftest`
  collects under one and dies at collection under the other.
- **Never leave the repository changed.** Write to `tmp_path` where you can. A test
  that must write a tracked path to prove what it proves (the branch write path does)
  owns putting it back — a snapshot-and-restore fixture, as
  `branch_artifacts_restored` does. Check with `git status --short -- data/` after a
  run: a regenerated file that happens to match the committed bytes hides the write.

## Where things are

`tests/` holds 198 `test_*.py` files, one per module or concern, named after what
they test (`test_pilot_net_change.py` tests `pilot/net_change.py`). The ones that
are about the fleet rather than a module:

| file | holds |
|---|---|
| `test_pilot_artifact_freshness.py` | every tracked derived artifact equals a fresh run, per deck and branch |
| `test_pilot_tracked_artifacts_validate.py` | every tracked authored artifact passes its validator |
| `test_metric_hygiene.py` | no two axes are one measurement; every flag the model sets is read |
| `test_docs_counts.py`, `test_docs_section_count.py` | the doc gates: stated counts, the index, every command and file named |
| `test_conftest_cache.py` | the cache keys cover what the producers import |
| `test_pilot_imports.py` | no module uses a name it never binds (the AST scan) |
