---
name: test-report
description: Measure the test suite and report it — a pre-test check of the harness (test-preflight), the unit tier (~2 min; FULL=1 adds regression) run uncached under coverage with every test recorded (`make test-report`), a dated tracked report in data/test_reports/, the computed diff against the last one, and a post-test reading (test-debrief) that cites only those findings. Updates docs/testing.md's generated block and proposes the rest. Use when the user wants test statistics, test time, coverage, or a before/after on the suite — and after any change to the harness.
---

# Measure the suite

Code computes, agents read. `tests/report_plugin.py` records each run,
`src/manamap/suite_report.py` builds the report and every finding, and the two agents
only judge and write — the same split as `sim-findings` and `sim-debrief`.

1. **Preflight.** Spawn `test-preflight`. On `WAIT`, tell the user what is running
   and stop — do not start a measured run beside a regen, a Forge run or another
   suite. On `FIX`, say what to install and stop. On `GO`, carry its caveats forward.
2. **Run.** `make test-report` — the unit tier, ~2 min, the default and the usual
   answer to "how are the tests". `make test-report FULL=1` adds regression (~25 min
   under coverage): run it in the BACKGROUND, only when the baseline itself is the
   question, and tell the user the ETA up front; the `job-band` mod shows it live
   (`.progress/`). Both run `--no-test-cache`, under coverage and `caffeinate`. Do not edit
   anything under `src/` or `tests/` while it runs. A failing tier still records —
   `record` writes the report and exits non-zero, so a red run is a report too.
3. **Render the docs block** — after a FULL report only: `.venv/bin/python -m
   manamap.suite_report render --write-docs` rewrites the generated block in
   docs/testing.md from the latest full report, and `tests/test_suite_report.py`
   fails until it is done. A unit report (`….unit.json`) leaves the block alone.
4. **Debrief.** Spawn `test-debrief`. It returns a reading citing finding ids and, if
   the hand-written tiers table is now wrong, a diff for it. Check every number it
   quotes against `suite_report diff` before passing it on; apply its docs diff only
   if the figures match. Then `make docs-sizes` if a doc's length moved
   (docs/README.md's line counts are tested).
5. **Report to the user**: the headline, failures first, then time, coverage and skips
   — with the report's path. The new report, docs/testing.md and anything the debrief
   changed are left in the working tree; commit them only when the user asks (or as
   part of a push they already asked for), after `make test` passes on them.

## Reading the report directly

```bash
.venv/bin/python -m manamap.suite_report preflight --collect   # harness facts
.venv/bin/python -m manamap.suite_report diff                  # latest vs previous
.venv/bin/python -m manamap.suite_report diff A.json B.json    # any two
.venv/bin/python -m manamap.suite_report render                # the docs block
```

## Gotchas

- **Coverage costs time.** A `test-report` wall compares only with another
  `test-report` wall; `diff` refuses the rest as `wall-not-comparable`.
- **Scope.** `diff` compares a report with the previous one of the SAME scope (unit
  with unit, full with full); a unit report's coverage is the inner loop's, not a drop.
- **The cache skips work.** A cached run under-counts time and coverage; the report
  flags it (`<tier>:cached`), and `make test-report` never serves from the cache.
- **The unit tier's coverage is the inner loop's.** Much of the goldfish is covered
  only by the regression tier's producers; read `coverage.unit` against `coverage.combined`
  before calling a module untested.
- **A harness edit moves the instrument.** `harness:changed` names the files; the
  delta beside it may not be the suite's.
