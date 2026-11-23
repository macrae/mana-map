"""The core set holds its own shape: every entry is a real, collected, UNIT-tier test,
so `make smoke` never reads tracked data, never needs Forge, and never silently
shrinks when a test is renamed. Run in the unit tier on every commit."""
import subprocess
import sys
from pathlib import Path

from core_contracts import CORE, PENDING

ROOT = Path(__file__).resolve().parents[1]


def test_every_core_entry_is_a_collected_unit_test():
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-m", "core and unit",
         "-p", "no:xdist", "--no-header", *sorted({"tests/" + k.split("::")[0] for k in CORE})],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    ids = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    seen = {f"{Path(i.split('::')[0]).name}::{i.split('::')[-1].split('[')[0]}" for i in ids}
    missing = sorted(set(CORE) - seen)
    assert not missing, ("core entries that are not collected as UNIT tests (renamed, deleted, "
                         "or reading tracked data): " + ", ".join(missing))
    assert len(ids) >= len(CORE)


def test_every_core_entry_names_its_rule():
    assert all(why.strip() for why in CORE.values())
    assert all("::" in k and k.split("::")[0].startswith("test_") for k in CORE)


def test_every_pending_rule_still_names_a_real_test():
    if not PENDING:
        return                                        # nothing owed; never collect the whole suite for it
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
         "-p", "no:xdist", "--no-header", *sorted({"tests/" + k.split("::")[0] for k in PENDING})],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    seen = {f"{Path(i.split('::')[0]).name}::{i.split('::')[-1].split('[')[0]}"
            for i in out.stdout.splitlines() if "::" in i}
    assert not sorted(set(PENDING) - seen)
    assert not set(PENDING) & set(CORE), "a rule is either core or pending, never both"
