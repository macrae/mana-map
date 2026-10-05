"""The three tiers: `conftest.tier_of` sorts every test into exactly one.

Driven through a REAL inner pytest that loads the repo's own conftest, so the
hook order (the tier must be on the item before `-m` deselects) is under test
too — a classifier that ran after deselection would select nothing, and a
unit-level call to `tier_of` could never see that.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INNER = '''
import pytest

gate = pytest.mark.skipif(False, reason="requires a fetched deck")

def test_plain(): pass

@gate
def test_gated(): pass

def test_reads_the_cache(unchanged): pass

@pytest.mark.slow
def test_slow(): pass

@pytest.mark.browser
def test_browser(): pass

@pytest.mark.forge
def test_forge(): pass

@pytest.mark.regression
def test_marked_by_hand(): pass

@pytest.mark.unit
@gate
def test_explicit_wins(): pass
'''

WANT = {
    "unit": {"test_plain", "test_explicit_wins"},
    "regression": {"test_gated", "test_reads_the_cache", "test_slow",
                   "test_marked_by_hand"},
    "integration": {"test_browser", "test_forge"},
}


def _selected(tmp_path, markexpr):
    out = tmp_path / f"{markexpr}.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n0", "-p", "no:cacheprovider",
         "-m", markexpr, f"--record-run={out}", "tests/test_inner.py"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "tests")})
    return {k.split("::")[-1] for k in json.loads(out.read_text())["tests"]}


def test_every_test_lands_in_exactly_the_tier_its_rule_names(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", tests / "conftest.py")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tests / "test_inner.py").write_text(INNER)
    got = {tier: _selected(tmp_path, tier) for tier in WANT}
    assert got == WANT
    every = set().union(*WANT.values())
    assert sum(len(v) for v in got.values()) == len(every), "a test is in two tiers"
