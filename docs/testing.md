# Testing

```bash
make test                 # UNIT: no tracked data — the inner loop          ~1 min
make test-unit-isolated   # the unit tier against an EMPTY data dir — proves it
make regression           # REGRESSION: the tracked fleet + corpus, then the fleet regen alone
make integration          # INTEGRATION: browser, Forge, pages byte-identical
make prepush              # unit + isolation + regression — before every push (CI runs it)
make test-fresh           # unit + regression, nothing served from the cache
make test-report          # measure the unit tier: counts, time, coverage   ~2 min
make test-report FULL=1   # + regression — the baseline, run in the background
make test-browser         # playwright, -n 4, then the serial_only tests     ~7 min
pytest -n0 -k NAME        # one test; worker startup outweighs the split
pytest -m forge           # ONE real Forge game (needs ~/.mana-map/forge)
pytest -m ""              # literally everything, browser included
pytest --lf               # only what failed last time
```

**A bare `pytest` is `make test`, the unit tier.** `addopts` in `pyproject.toml`
carries `-m unit -n auto --dist worksteal`: an idle worker takes queued tests off
a busy one, so a file of heavy tests no longer strands two workers with the tail.

**This file is the only place that states test counts and runtimes.** They move on
almost every commit; README and CLAUDE.md point here instead. The dated record of
how the suite got here — every earlier measurement, every lesson in full, the
2026-09-21 adversarial audit — is [`history/testing-log.md`](history/testing-log.md).

## The three tiers (2026-10-05, 8-core Mac, `-n auto`)

| tier | what it touches | tests | wall |
|---|---|---:|---:|
| **unit** — `make test` | inline data only; no tracked artifact, no outside system | 2,515 | **0:42** uncached |
| unit, isolated — `make test-unit-isolated` | the same, against an empty `MANAMAP_DATA_DIR` | 2,515 | **0:38** |
| **regression** — `make regression` | the tracked fleet and corpus: every artifact validator, the constants calibrated across the fleet — in parallel; then, ALONE, one `regen` of the whole fleet compared byte for byte (`test_the_fleet_regenerates_byte_identically`, marker `regen`; it replaced 85 per-artifact re-derivations that were 74% of the tier's CPU) | 1,582 | **13:10** uncached (6:25 parallel + 6:43 regen) |
| **integration** — `make integration` | a real browser (264), the Forge engine (1), the network; then `make manuals` + `git diff --exit-code` | 265 | ~7 min + |
| **total collected** | | **4,362** | |

**What decides the tier is what a test TOUCHES, never its runtime** — runtime flaps,
and a slow pure function is still a unit test. `tests/conftest.py` (`tier_of`)
assigns every test exactly one, before `-m` deselects anything:

- **integration** — marked `browser`, `forge` or `network`.
- **regression** — marked `slow` or `fleet`; or carrying any `skipif` (every one in
  this suite is a data gate: `requires_*`, `needs_*`); or taking a fixture whose value
  is tracked data (`conftest.DATA_FIXTURES`: `unchanged`, `deck`, `slug`, `corpus`, …).
- **unit** — everything else.
- An explicit `@pytest.mark.unit|regression|integration` (or a module `pytestmark`)
  wins. 183 tests carry one: the ones `make test-unit-isolated` caught reading the
  fleet with no gate (134 failed against an empty data dir, 47 skipped only there, and 2 more
  whose runtime skip the first test report caught).

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
the unit tier (~2 min) — and with `FULL=1` the regression tier too — with nothing from
the cache, under coverage and under `caffeinate` (a laptop that idles to sleep
mid-run measured 1h50m of nothing on 2026-10-05), recording every test
(`tests/report_plugin.py`, `--record-run`), and `python -m manamap.suite_report
record` folds the runs into a dated, tracked report in `data/test_reports/`
(`<date>-<sha8>.json` for a full report, `….unit.json` for a unit one). `diff`
recomputes the findings between the latest two OF THE SAME SCOPE — failures, counts by file, wall
time, files that slowed, skip reasons that grew, coverage by module, and whether the
harness itself changed in between — and `render --write-docs` rewrites the block
below from the latest FULL report, which `tests/test_suite_report.py` holds it to. The
`/test-report` skill runs it end to end with a pre-test (`test-preflight`) and a
post-test (`test-debrief`) agent; the agents read the figures and compute none.
Coverage costs time, so these walls compare only with each other.

**Watching a run.** Every pytest run rooted in this repo writes
`.progress/pytest-<pid>.json` (done/total, failures, a 5-second heartbeat), which the
`job-band` Claude Code mod draws above the prompt: a bar, elapsed, an ETA, and a
yellow NO HEARTBEAT when the run died or the machine slept. `MANAMAP_NO_PROGRESS=1`
turns the writer off.

<!-- suite-report:begin (generated by `python -m manamap.suite_report render --write-docs`; do not edit) -->

Measured 2026-10-05 at `71014751`, 8 CPUs, Python 3.10.0, by `make test-report` — the tracked record is `data/test_reports/2026-10-05-71014751.json`.

| tier | collected | passed | skipped | xfailed | failed | wall | in tests |
|---|---:|---:|---:|---:|---:|---:|---:|
| unit (uncached, under coverage) | 2,515 | 2,513 | 2 | 0 | 0 | **1:06** | 4:36 |
| regression (uncached, under coverage) | 1,664 | 1,651 | 10 | 3 | 0 | **38:19** | 173:25 |

Line coverage: the unit tier alone **56.6%**; both measured tiers **76.0%** of 30,805 statements in `src/manamap/` (7,387 never run).

The five slowest tests: `test_every_tracked_simulation_run_passes_its_validator[edgar-vampires]` 356s; `test_net_change_matches_a_fresh_run[goblin-storm@copy-burst-v1]` 268s; `test_net_change_matches_a_fresh_run[edgar-vampires@drain-v1]` 204s; `test_a_procedure_page_names_only_cards_the_deck_runs` 174s; `test_futility_stops_only_when_the_asked_for_effect_is_excluded` 163s.

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
