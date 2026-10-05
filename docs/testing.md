# Testing

```bash
make test                 # UNIT: no tracked data — the inner loop          ~1 min
make test-unit-isolated   # the unit tier against an EMPTY data dir — proves it
make regression           # REGRESSION: the tracked fleet + corpus           ~15 min
make integration          # INTEGRATION: browser, Forge, pages byte-identical
make prepush              # unit + isolation + regression — before every push (CI runs it)
make test-fresh           # unit + regression, nothing served from the cache
make test-report          # measure it: counts, time, coverage -> data/test_reports/
make test-browser         # playwright, -n 4, then the serial_only tests     ~7 min
pytest -n0 -k NAME        # one test; worker startup outweighs the split
pytest -m forge           # ONE real Forge game (needs ~/.mana-map/forge)
pytest -m ""              # literally everything, browser included
pytest --lf               # only what failed last time
```

**A bare `pytest` is `make test`, the unit tier.** `addopts` in `pyproject.toml`
carries `-m unit -n auto`.

**This file is the only place that states test counts and runtimes.** They move on
almost every commit; README and CLAUDE.md point here instead. The dated record of
how the suite got here — every earlier measurement, every lesson in full, the
2026-09-21 adversarial audit — is [`history/testing-log.md`](history/testing-log.md).

## The three tiers (2026-10-05, 8-core Mac, `-n auto`)

| tier | what it touches | tests | wall |
|---|---|---:|---:|
| **unit** — `make test` | inline data only; no tracked artifact, no outside system | 2,514 | **0:58** uncached |
| unit, isolated — `make test-unit-isolated` | the same, against an empty `MANAMAP_DATA_DIR` | 2,514 | **0:52** |
| **regression** — `make regression` | the tracked fleet and corpus: every artifact validator, every producer re-run per deck and branch, the constants calibrated across the fleet | 1,664 | **15:06** uncached |
| **integration** — `make integration` | a real browser (264), the Forge engine (1), the network; then `make manuals` + `git diff --exit-code` | 265 | ~7 min + |
| **total collected** | | **4,443** | |

**What decides the tier is what a test TOUCHES, never its runtime** — runtime flaps,
and a slow pure function is still a unit test. `tests/conftest.py` (`tier_of`)
assigns every test exactly one, before `-m` deselects anything:

- **integration** — marked `browser`, `forge` or `network`.
- **regression** — marked `slow` or `fleet`; or carrying any `skipif` (every one in
  this suite is a data gate: `requires_*`, `needs_*`); or taking a fixture whose value
  is tracked data (`conftest.DATA_FIXTURES`: `unchanged`, `deck`, `slug`, `corpus`, …).
- **unit** — everything else.
- An explicit `@pytest.mark.unit|regression|integration` (or a module `pytestmark`)
  wins. 181 tests carry one: the ones `make test-unit-isolated` caught reading the
  fleet with no gate (134 failed against an empty data dir, 47 skipped only there).

**The unit tier is proven, not assumed.** `make test-unit-isolated` runs it against an
empty data dir; a failure or a skip there is a test that reads tracked data and must
be marked `regression`. `prepush` and CI run it, and `tests/test_tiers.py` holds the
classifier to its rules through a real inner pytest. What the isolation run cannot
see is a test that reads `ROOT / "data"` directly rather than through `config` —
it still passes against an empty `MANAMAP_DATA_DIR`. Route data reads through
`config`.

To print today's numbers rather than trust these:

```bash
.venv/bin/pytest --co -n0 -qq | awk -F': ' '{s+=$2} END {print s}'               # unit
.venv/bin/pytest --co -n0 -qq -m regression | awk -F': ' '{s+=$2} END {print s}' # regression
.venv/bin/pytest --durations=30                                                   # where the time goes
```

Keep the decision loop (`try`, the paired intervals) testable in the unit tier
with inline decks where it can be, because a regression there should surface in
the inner loop, not before a push.

## The measured report (`make test-report`)

The table above is written by hand; this one is generated. `make test-report` runs
the unit and regression tiers with nothing from the cache and under coverage, recording every test
(`tests/report_plugin.py`, `--record-run`), and `python -m manamap.suite_report
record` folds the runs into a dated, tracked report in `data/test_reports/`. `diff`
recomputes the findings between the latest two — failures, counts by file, wall
time, files that slowed, skip reasons that grew, coverage by module, and whether the
harness itself changed in between — and `render --write-docs` rewrites the block
below, which `tests/test_suite_report.py` holds to the latest report. The
`/test-report` skill runs it end to end with a pre-test (`test-preflight`) and a
post-test (`test-debrief`) agent; the agents read the figures and compute none.
Coverage costs time, so these walls compare only with each other.

<!-- suite-report:begin (generated by `python -m manamap.suite_report render --write-docs`; do not edit) -->

Measured 2026-10-05 at `2156ec90` (uncommitted changes in src/ or tests/), 8 CPUs, Python 3.10.0, by `make test-report` — the tracked record is `data/test_reports/2026-10-05-2156ec90.json`.

| tier | collected | passed | skipped | xfailed | failed | wall | in tests |
|---|---:|---:|---:|---:|---:|---:|---:|
| fast (uncached, under coverage) | 4,036 | 4,024 | 9 | 1 | 2 | **9:57** | 34:10 |
| fleet (uncached, under coverage) | 142 | 137 | 3 | 2 | 0 | **23:21** | 178:07 |

Line coverage: the fast tier alone **73.5%**; both measured tiers **76.0%** of 30,797 statements in `src/manamap/` (7,394 never run).

The five slowest tests: `test_net_change_matches_a_fresh_run[edgar-vampires@drain-v1]` 304s; `test_net_change_matches_a_fresh_run[goblin-storm@copy-burst-v1]` 283s; `test_every_tracked_simulation_run_passes_its_validator[edgar-vampires]` 218s; `test_net_change_matches_a_fresh_run[sharknado@momentum-v1]` 211s; `test_net_change_matches_a_fresh_run[edgar-vampires@entry-v1]` 188s.

<!-- suite-report:end -->

## Markers

Registered in `pyproject.toml`:

| marker | means | tier |
|---|---|---|
| `unit` / `regression` / `integration` | the tier — assigned by `conftest.tier_of`; set by hand only to override it | itself |
| `slow` | re-runs a 10,000-game producer, or thousands of goldfish games | regression |
| `fleet` | re-derives a calibrated constant across every tracked deck | regression |
| `browser` | drives a real Chromium via playwright | integration — `make test-browser` |
| `serial_only` | asserts a wall-clock budget; run alone, `-n0` | integration, with `browser` |
| `forge` | runs the Forge engine headless | integration — opt in |
| `network` | reaches an outside service (Scryfall, EDHREC); none yet — every such test mocks it, and the weekly `corpus-gates` CI job is the real network leg | integration |

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
