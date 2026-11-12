"""A warm worker refuses once the source tree moves (2026-11-12).

`manamap serve` imported `scenario_facts` at boot, the module was then fixed, and the
worker served the OLD code to a stack resolver through `/api/cli`, which told it two
branch cards were outside the deck. `_cli` now compares the tree's stamp with the one
it booted on and refuses, and the terminal client fails open to a local run. Proved by
re-introducing the bug: drop the stamp check and the first test runs the command.
"""
import pytest

from manamap import serve


def test_a_worker_on_older_code_refuses(monkeypatch):
    monkeypatch.setattr(serve, "_BOOT_CODE", (0, 0))
    with pytest.raises(ValueError, match="older than the source"):
        serve._cli(["pilot", "deck-info", "heliod"])


def test_a_current_worker_does_not_refuse_on_the_stamp(monkeypatch):
    monkeypatch.setattr(serve, "_BOOT_CODE", serve._code_stamp())
    with pytest.raises(ValueError) as e:
        serve._cli([])             # refused for having no argv, not for its code
    assert "older than the source" not in str(e.value)
