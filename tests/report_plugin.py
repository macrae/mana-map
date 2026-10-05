"""Record a test run for `manamap.suite_report` (`--record-run PATH`), and show
it live (`.progress/`, read by the `job-band` Claude Code mod; see `Progress`).

Registered by `conftest.pytest_configure`; it does nothing unless the option is
given, so an ordinary `make test` pays nothing for it. `make test-report` passes it
once per tier and `python -m manamap.suite_report record` folds the files into a
dated report (docs/testing.md, "The measured report").

Runs on the CONTROLLER only: under xdist every worker's reports are relayed to the
controller's `pytest_runtest_logreport`, so one process sees every test exactly
once. A worker (`config.workerinput`) registers nothing.

What it writes is RAW — one row per test, keyed by nodeid — and stays out of git
(`.pytest_cache/`). The tracked report is the compact roll-up the builder makes.
"""
import json
import os
import time
from pathlib import Path

import pytest

#: Markers that decide a test's tier (pyproject's `markers`). Read off the
#: report's keywords, which xdist relays with it.
TIER_MARKERS = ("slow", "fleet", "browser", "forge", "serial_only")


def pytest_addoption(parser):
    parser.addoption(
        "--record-run", default=None, metavar="PATH",
        help="write this run's per-test outcomes and durations to PATH (JSON) "
             "for `python -m manamap.suite_report record`")


class Recorder:
    def __init__(self, config, path):
        self.config = config
        self.path = Path(path)
        self.tests = {}
        self.started = None
        self.collected = 0

    def _row(self, nodeid):
        return self.tests.setdefault(
            nodeid, {"outcome": None, "duration": 0.0, "markers": [], "reason": None})

    def pytest_sessionstart(self, session):
        self.started = time.time()

    def pytest_collection_finish(self, session):
        # Serial runs only: under xdist the controller collects nothing and
        # this reads 0; the workers' count arrives through the hook below.
        self.collected = max(self.collected, len(session.items))

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.collected = max(self.collected, len(ids))

    def pytest_runtest_logreport(self, report):
        row = self._row(report.nodeid)
        row["duration"] += report.duration
        if not row["markers"]:
            row["markers"] = sorted(m for m in TIER_MARKERS if m in report.keywords)
        # The first non-pass phase decides the outcome: an error in setup is an
        # ERROR even though no call ran; a skip in setup is a skip.
        if report.when == "call" or report.outcome != "passed":
            if row["outcome"] in (None, "passed"):
                if hasattr(report, "wasxfail"):
                    row["outcome"] = "xfailed" if report.skipped else "xpassed"
                elif report.failed:
                    row["outcome"] = "failed" if report.when == "call" else "error"
                else:
                    row["outcome"] = report.outcome
                if report.skipped and isinstance(report.longrepr, tuple):
                    row["reason"] = str(report.longrepr[2])

    def pytest_sessionfinish(self, session, exitstatus):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "markexpr": self.config.getoption("markexpr") or "",
            "no_test_cache": bool(self.config.getoption("--no-test-cache", False)),
            "instrumented": bool(self.config.pluginmanager.hasplugin("_cov")
                                 and self.config.getoption("cov_source", None)),
            "exitstatus": int(exitstatus),
            "collected": self.collected,
            "wall_seconds": round(time.time() - self.started, 2),
            "tests": {k: {**v, "duration": round(v["duration"], 4)}
                      for k, v in sorted(self.tests.items())},
        }
        self.path.write_text(json.dumps(payload, indent=1) + "\n")


# ── Live progress, for the job band (on for every run; MANAMAP_NO_PROGRESS=1 opts out)
#
# `manamap.progress` writes `.progress/pytest-<pid>.json` with a heartbeat, which
# the Claude Code `job-band` mod draws above the prompt — a run that died, or a
# laptop that slept through one (the first three-tier report crawled 1h50m that
# way, unseen), shows as a stopped heartbeat rather than a quiet terminal. Only for
# a run rooted in this repo: the inner pytests some tests launch in a tmp dir stay
# off the band.
from manamap.progress import REPO, Progress as _Writer  # noqa: E402


class Progress:
    def __init__(self, config):
        mark = config.getoption("markexpr") or "everything"
        self.writer = _Writer(f"pytest {mark}" if len(mark) <= 24 else "pytest",
                              unit="tests", name="pytest")
        self.seen = set()

    def pytest_sessionstart(self, session):
        self.writer.start()

    def pytest_collection_finish(self, session):
        if session.items:
            self.writer.set(total=len(session.items))

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.writer.set(total=max(self.writer.total or 0, len(ids)))

    def pytest_runtest_logreport(self, report):
        if (report.when == "call" or report.outcome != "passed") \
                and report.nodeid not in self.seen:
            self.seen.add(report.nodeid)
            self.writer.advance(failed=int(report.failed and not hasattr(report, "wasxfail")))

    def pytest_sessionfinish(self, session, exitstatus):
        self.writer.finish(ok=exitstatus in (0, 5))


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if hasattr(config, "workerinput"):
        return
    path = config.getoption("--record-run", None)
    if path:
        config.pluginmanager.register(Recorder(config, path), "manamap-recorder")
    if (not os.environ.get("MANAMAP_NO_PROGRESS")
            and Path(config.rootpath).resolve() == REPO
            and not config.getoption("collectonly")):
        config.pluginmanager.register(Progress(config), "manamap-progress")
