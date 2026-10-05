"""The suite's own report: the recorder, the builder, the findings, the docs block.

Every test drives the production code — the recorder by running a real inner
pytest with it loaded, the builder and `findings` on rows shaped exactly like the
recorder's output. The last two hold the TRACKED history to the code: every report
in `data/test_reports/` is well-formed and named for its own contents, and the
generated block in docs/testing.md is the render of the latest one.
"""
import json
import pathlib
import subprocess
import sys

import pytest

from manamap import suite_report as sr

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _raw(tests, **over):
    base = {"markexpr": "", "no_test_cache": True, "instrumented": True,
            "exitstatus": 0, "collected": len(tests), "wall_seconds": 100.0,
            "tests": tests}
    return {**base, **over}


def _t(outcome="passed", duration=1.0, reason=None):
    return {"outcome": outcome, "duration": duration, "markers": [], "reason": reason}


def _report(tmp_path, tests, coverage=None, sha="a" * 40, **over):
    raw = tmp_path / sha[:4]
    raw.mkdir()
    (raw / "unit.json").write_text(json.dumps(_raw(tests, **over)))
    if coverage is not None:
        (raw / "coverage.json").write_text(json.dumps(coverage))
    return sr.build(raw, sha=sha, dirty=False)


def _cov(files):
    return {"totals": {"percent_covered": sum(files.values()) / len(files),
                       "num_statements": 100, "missing_lines": 10},
            "files": {f: {"summary": {"percent_covered": p}} for f, p in files.items()}}


def test_the_recorder_sees_every_outcome_through_a_real_run(tmp_path):
    """An inner pytest with the plugin loaded: pass, fail, setup error, skip with
    its reason, xfail. If any phase's outcome were dropped, a count here moves."""
    (tmp_path / "test_inner.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def broken():\n    raise RuntimeError('setup')\n"
        "def test_ok(): pass\n"
        "def test_bad(): assert False\n"
        "def test_err(broken): pass\n"
        "def test_skip(): pytest.skip('needs data')\n"
        "@pytest.mark.xfail(strict=True)\n"
        "def test_xf(): assert False\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    out = tmp_path / "raw.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n0", "-p", "no:cacheprovider",
         "-p", "report_plugin", f"--record-run={out}", "test_inner.py"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "tests")})
    raw = json.loads(out.read_text())
    got = {k.split("::")[-1]: v["outcome"] for k, v in raw["tests"].items()}
    assert got == {"test_ok": "passed", "test_bad": "failed", "test_err": "error",
                   "test_skip": "skipped", "test_xf": "xfailed"}
    assert raw["tests"]["test_inner.py::test_skip"]["reason"].endswith("needs data")
    assert raw["collected"] == 5 and raw["exitstatus"] == 1


def test_build_rolls_up_outcomes_files_and_the_cache(tmp_path):
    rep = _report(tmp_path, {
        "tests/test_a.py::x": _t(duration=3.0),
        "tests/test_a.py::y": _t("failed", 1.0),
        "tests/test_b.py::z": _t("skipped", 0.0, reason=sr.CACHED_PREFIX + " since"),
    }, coverage=_cov({"src/manamap/a.py": 50.0}))
    t = rep["tiers"]["unit"]
    assert t["outcomes"] == {"failed": 1, "passed": 1, "skipped": 1}
    assert t["cached"] == 1
    assert t["not_passed"] == [["tests/test_a.py::y", "failed"]]
    assert t["by_file"]["tests/test_a.py"] == {"tests": 2, "seconds": 4.0}
    assert t["slowest"][0] == ["tests/test_a.py::x", 3.0]
    assert rep["coverage"]["combined"]["by_module"] == {"src/manamap/a.py": 50.0}
    assert sr.report_name(rep) == f"{rep['date']}-aaaaaaaa.unit.json"  # unit tier only: unit scope


def test_findings_name_count_wall_skips_and_coverage_moves(tmp_path):
    prev = _report(tmp_path, {"tests/test_a.py::x": _t(duration=10.0)},
                   coverage=_cov({"src/manamap/a.py": 80.0}), sha="b" * 40)
    cur = _report(tmp_path, {
        "tests/test_a.py::x": _t(duration=30.0),
        "tests/test_a.py::new": _t("skipped", 0, reason="needs data"),
    }, coverage=_cov({"src/manamap/a.py": 60.0}), sha="c" * 40, wall_seconds=130.0)
    by_id = {f["id"]: f for f in sr.findings(cur, prev, harness_changed=["Makefile"])}
    assert by_id["unit:count"]["figures"]["files"] == [["tests/test_a.py", 1, 2]]
    assert by_id["unit:wall"]["figures"]["delta"] == 30.0
    assert by_id["unit:slower-files"]["figures"]["files"] == [["tests/test_a.py", 10.0, 30.0]]
    assert by_id["unit:skips"]["figures"]["reasons"] == [["needs data", 0, 1]]
    assert by_id["coverage:total"]["figures"]["delta"] == -20.0
    assert by_id["coverage:modules"]["figures"]["modules"] == [["src/manamap/a.py", 80.0, 60.0]]
    assert by_id["harness:changed"]["figures"]["files"] == ["Makefile"]


def test_a_cached_or_differently_instrumented_wall_is_not_compared(tmp_path):
    """Coverage costs time and the cache skips work: a delta between two such runs
    would be the instrument, so the finding says so instead of printing one."""
    prev = _report(tmp_path, {"tests/t.py::x": _t()}, sha="d" * 40, instrumented=False)
    cur = _report(tmp_path, {"tests/t.py::x": _t()}, sha="e" * 40)
    kinds = {f["id"]: f["kind"] for f in sr.findings(cur, prev)}
    assert kinds["unit:wall"] == "wall-not-comparable"


def test_failures_are_findings_even_with_no_previous_report(tmp_path):
    cur = _report(tmp_path, {"tests/t.py::x": _t("error")}, exitstatus=1)
    ids = {f["id"] for f in sr.findings(cur)}
    assert {"unit:error:tests/t.py::x", "unit:exit"} <= ids


def test_a_tier_measured_in_two_runs_is_one_tier(tmp_path):
    """`make regression` is the parallel run, then the fleet regen alone: two
    raw files, one tier — walls add, tests union, a failure in either stands."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "unit.json").write_text(json.dumps(_raw({"tests/u.py::a": _t()})))
    (raw / "regression.json").write_text(json.dumps(_raw(
        {"tests/r.py::a": _t(), "tests/r.py::b": _t()}, wall_seconds=60.0)))
    (raw / "regression.regen.json").write_text(json.dumps(_raw(
        {"tests/f.py::regen": _t("failed", 300.0)}, wall_seconds=300.0, exitstatus=1)))
    rep = sr.build(raw, sha="f" * 40, dirty=False)
    r = rep["tiers"]["regression"]
    assert rep["scope"] == "full"
    assert r["collected"] == 3 and r["wall_seconds"] == 360.0 and r["exitstatus"] == 1
    assert r["not_passed"] == [["tests/f.py::regen", "failed"]]


def test_a_unit_report_never_reads_as_a_coverage_drop_against_a_full_one(tmp_path):
    """The default `make test-report` measures the unit tier alone. Its coverage
    against a full report's would read as a 20-point fall that nothing caused."""
    full = _report(tmp_path, {"tests/t.py::x": _t()}, sha="1" * 40,
                   coverage=_cov({"src/manamap/a.py": 80.0}))
    full["scope"] = "full"
    raw = tmp_path / "unitonly"
    raw.mkdir()
    (raw / "unit.json").write_text(json.dumps(_raw({"tests/t.py::x": _t()})))
    (raw / "coverage-unit.json").write_text(json.dumps(_cov({"src/manamap/a.py": 55.0})))
    unit = sr.build(raw, sha="2" * 40, dirty=False)
    assert unit["scope"] == "unit"
    assert sr.report_name(unit).endswith("-22222222.unit.json")
    kinds = {f["id"]: f["kind"] for f in sr.findings(unit, full)}
    assert kinds["scope:differs"] == "scope-not-comparable"
    assert "coverage:total" not in kinds and "coverage:modules" not in kinds


def test_write_docs_replaces_only_the_block(tmp_path):
    rep = _report(tmp_path, {"tests/t.py::x": _t()})
    doc = tmp_path / "testing.md"
    doc.write_text(f"before\n{sr.BLOCK_BEGIN}\nold\n{sr.BLOCK_END}\nafter\n")
    sr.write_docs(rep, doc)
    text = doc.read_text()
    assert text.startswith("before\n") and text.endswith("\nafter\n")
    assert sr.render(rep) in text and "\nold\n" not in text


# ── the tracked history ──────────────────────────────────────────────────


def test_every_tracked_report_is_well_formed_and_named_for_itself():
    checked = 0
    reports = sr.history()
    for i, path in enumerate(reports):
        rep = json.loads(path.read_text())
        assert rep["schema"] == sr.SCHEMA, path.name
        assert path.name == sr.report_name(rep), (path.name, sr.report_name(rep))
        assert set(rep["tiers"]) <= set(sr.TIERS) | set(sr.LEGACY_TIERS), path.name
        prev = json.loads(reports[i - 1].read_text()) if i else None
        sr.findings(rep, prev, harness_changed=[])
        checked += 1
    assert checked >= 1, "no report in data/test_reports/ — run `make test-report`"


def test_the_docs_block_is_the_render_of_the_latest_report():
    """docs/testing.md is the only page that states counts; its measured block is
    generated from the latest FULL report (a unit-scope one is a quick reading,
    not the baseline). A new full report without `render --write-docs` fails here."""
    latest = json.loads(sr.history(scope="full")[-1].read_text())
    text = sr.TESTING_MD.read_text()
    assert sr.BLOCK_BEGIN in text
    a = text.index(sr.BLOCK_BEGIN)
    b = text.index(sr.BLOCK_END, a) + len(sr.BLOCK_END)
    assert text[a:b] == sr.render(latest), (
        "docs/testing.md's measured block is stale: "
        "`python -m manamap.suite_report render --write-docs`")
