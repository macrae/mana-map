"""A branch's objective is OPTIONAL (the pilot, 2026-10-28).

A branch is judged on its whole paired profile — twelve Holm-corrected rows — and an
objective, when one is given, is a GUARD: one row written down before the run that the
branch must not fail, shown among the others. `deck-branch new` and serve's
`branch/new` both refused a branch without one; Build's branch form could not open a
branch without picking an axis it did not mean. When a guard IS given it is still read
by `parse_objective` alone, so neither door can name an axis the bench does not compute.

Proven by putting the old refusal back: with `if not raw: raise SystemExit(...)` in
`deck_branch._dispatch`, the CLI tests fail; with serve's `if not objective: raise
ValueError(...)`, the serve test fails; with the validator's "v2+ and no objective"
error, the end-to-end test fails on the branch it just opened.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

from conftest import requires_deck
from manamap import config, serve
from manamap.pilot import deck_branch, net_change, validate_branch, validate_net_change


def _new_args(objective=None):
    return SimpleNamespace(slug="any", action="new", name="b1", source=None,
                           objective=objective, why="w")


@pytest.fixture
def stub_new(tmp_path, monkeypatch):
    """`deck_branch.new` recorded rather than run; the CLI's own logic is what is tested."""
    (tmp_path / "decklist.txt").write_text("1 Sol Ring\n")
    got = {}
    monkeypatch.setattr(deck_branch, "deck_dir", lambda slug, branch=None: tmp_path)
    monkeypatch.setattr(deck_branch, "new", lambda slug, branch, text, **kw:
                        got.update(kw) or {"path": "x", "size": 1, "warnings": []})
    return got


def test_the_cli_opens_a_branch_with_no_objective(stub_new, monkeypatch, capsys):
    def no_reading(slug, objective):
        raise AssertionError("anchored a guard nobody named")
    monkeypatch.setattr(deck_branch, "champion_reading", no_reading)
    deck_branch._dispatch(_new_args(None))
    assert stub_new["objective"] is None
    out = capsys.readouterr().out
    assert "guard: none" in out and "whole profile" in out, out


def test_a_guard_that_is_given_is_still_parsed(stub_new, monkeypatch, capsys):
    monkeypatch.setattr(deck_branch, "champion_reading", lambda slug, objective: [])
    deck_branch._dispatch(_new_args("kill_by_8 >= 0.30"))
    assert stub_new["objective"] == {"axis": "kill_by_8", "op": ">=", "value": 0.30, "why": "w"}
    assert "guard: kill_by_8 >= 0.3" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        deck_branch._dispatch(_new_args("vibes >= 9"))


def test_serve_opens_a_branch_with_no_objective(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    (decks / "zur").mkdir(parents=True)
    (decks / "zur" / "decklist.txt").write_text("1 Zur the Enchanter *CMDR*\n")
    monkeypatch.setattr(config, "DECKS_DIR", decks)
    got = {}
    monkeypatch.setattr(deck_branch, "new", lambda slug, branch, text, **kw:
                        got.update(kw) or {"size": 1, "warnings": []})
    monkeypatch.setattr(deck_branch, "refresh_dossier", lambda slug: None)
    for payload in ({"slug": "zur", "name": "t"}, {"slug": "zur", "name": "t", "objective": "  "}):
        got.clear()
        res = serve.call("branch/new", payload)
        assert res["objective"] is None and got["objective"] is None, (payload, res, got)
    res = serve.call("branch/new", {"slug": "zur", "name": "t", "objective": "kill_by_8 >= 0.3"})
    assert res["objective"]["axis"] == "kill_by_8" and got["objective"]["value"] == 0.3


def test_the_validator_accepts_no_objective_and_still_checks_one_that_is_given():
    doc = {"slug": "s", "branch": "b", "v": 2, "opened": "2026-10-28", "why": "w",
           "objective": None, "commits": []}
    assert not [e for e in validate_branch.validate(doc) if "objective" in e]
    bad = dict(doc, objective={"axis": "vibes", "op": ">=", "value": 9})
    assert any("not something the bench measures" in e for e in validate_branch.validate(bad))


SLUG = "ur-dragon"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of a real deck in a temp data dir, as `test_pilot_branch_lifecycle` does:
    `new` writes a branch directory, and that must never be the tracked deck's."""
    from manamap.pilot import deck_versions
    real = config.DECKS_DIR / SLUG
    if not (real / "decklist.txt").exists():
        pytest.skip(f"no {SLUG} fixture")
    decks = tmp_path / "decks"
    shutil.copytree(real, decks / SLUG,
                    ignore=shutil.ignore_patterns("branches", "sim", "*.bak", "edits.jsonl*",
                                                  ".edits.lock", ".rebuild.lock"))
    monkeypatch.setattr(config, "DECKS_DIR", decks)
    monkeypatch.setattr(deck_versions, "report", lambda slug: {"current_version": None})
    return decks


@requires_deck
def test_a_branch_opened_with_no_objective_is_valid_and_net_change_reads_its_profile(
        sandbox, monkeypatch, capsys):
    """END TO END: the CLI opens a real branch with no objective; the file it wrote
    passes `validate-branch`; and `net-change` on it reports the whole table with no
    primary, the "no objective" state, and a document `validate-net-change` accepts.
    The goldfish is a fixture (two fake readings) — this is about the branch's
    contract, not the simulator."""
    from test_pilot_net_change import _fake_diag

    from manamap.pilot import diagnostic, goldfish
    monkeypatch.setattr(deck_branch, "champion_reading", lambda slug, objective: [])
    deck_branch._dispatch(SimpleNamespace(slug=SLUG, action="new", name="free-v1", source=None,
                                          objective=None, why="no guard"))
    doc = json.loads((sandbox / SLUG / "branches" / "free-v1" / "branch.json").read_text())
    assert doc["objective"] is None
    assert validate_branch.validate(doc, slug=SLUG, branch="free-v1") == []

    n = 10000
    a = _fake_diag([{"rate": 5.0, "sd": 1.0, "n": n} for _ in net_change.ROWS])
    b = _fake_diag([{"rate": 5.2, "sd": 1.0, "n": n} for _ in net_change.ROWS])
    # The branch arm describes the list on disk, as a real goldfish run would.
    from manamap.pilot.common import decklist_sha256
    b["decklist_sha256"] = decklist_sha256(SLUG, "free-v1")
    monkeypatch.setattr(diagnostic, "run",
                        lambda slug, branch=None, iterations=None, seed=None, quiet=True, **kw:
                        b if branch else a)
    monkeypatch.setattr(net_change, "_refuse_a_stale_measurement", lambda s, br: None)
    monkeypatch.setattr(net_change, "blind_spots", lambda s, br, c: [])
    monkeypatch.setattr(net_change, "mana", lambda s, br: {"available": False, "why": "test"})
    monkeypatch.setattr(net_change, "forge", lambda s, br, pod=None: {"available": False, "why": "test"})
    monkeypatch.setattr(net_change, "_death_limit", lambda s, br: [])
    monkeypatch.setattr(goldfish, "model_version", lambda: "test")

    got = net_change.build(SLUG, "free-v1")
    assert got["objective"] is None
    assert got["design"]["primary"] is None
    assert got["design"]["exploratory_rows"] == len(net_change.ROWS)
    assert len(got["table"]) == len(net_change.ROWS)
    rec = got["recommendation"]
    assert rec["state"] == "no objective" and "whole profile" in rec["because"], rec
    assert rec["rose"], "the profile is still read when no guard was named"
    assert not validate_net_change.validate(got), validate_net_change.validate(got)
