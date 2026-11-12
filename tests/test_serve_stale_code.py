"""A warm worker refuses once a module it has loaded changes (2026-11-12).

`manamap serve` imported `scenario_facts` at boot, the module was then fixed, and the
worker served the OLD code to a stack resolver through `/api/cli`. `_cli` now hashes
each loaded module's source when first seen and refuses once any changes CONTENT; the
terminal client fails open to a local run. Content of loaded modules only: an mtime
stamp over every `.py` file tripped mid-suite. Proved by re-introducing the bug: drop
the stale check and the first test runs the command.
"""
import pytest

from manamap import serve


def test_a_worker_whose_loaded_code_changed_refuses(monkeypatch):
    monkeypatch.setitem(serve._LOADED, serve.__file__, "not-the-current-hash")
    with pytest.raises(ValueError, match="older than the source"):
        serve._cli(["pilot", "deck-info", "heliod"])


def test_an_unchanged_worker_does_not_refuse_on_its_code():
    serve._stale_modules()                     # record what is loaded now
    with pytest.raises(ValueError) as e:
        serve._cli([])                         # refused for having no argv, not for its code
    assert "older than the source" not in str(e.value)
