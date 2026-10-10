"""A branch write refreshes the deck's dossier — on the BENCH too, and from the page.

`refresh_dossier` gated on `regen.is_pinned`, so a bench deck's tracked,
freshness-gated `info.json` went stale on every branch opened or swap staged; and
`serve`'s `branch/new` / `branch/stage` never called it at all. The rule now is
`regen.targets`' since 2026-09-26: refresh wherever `info.json` exists and the
deck is not retired; never bootstrap one.
"""
import json

import pytest

from manamap import config, serve
from manamap.pilot import deck_branch, deck_info


@pytest.fixture
def decks(tmp_path, monkeypatch):
    d = tmp_path / "decks"
    d.mkdir()
    monkeypatch.setattr(config, "DECKS_DIR", d)
    return d


@pytest.fixture
def writes(monkeypatch):
    """`deck_info.main` recorded rather than run: the gate is what is tested."""
    calls = []
    monkeypatch.setattr(deck_info, "main", lambda args: calls.append(
        (args.slug, args.write, args.branch)))
    return calls


def _deck(decks, slug, info=True, versions=None):
    base = decks / slug
    base.mkdir()
    (base / "decklist.txt").write_text("1 Zur the Enchanter *CMDR*\n")
    (base / "cards.json").write_text("{}")
    if info:
        (base / "info.json").write_text("{}")
    if versions is not None:
        (base / "deck_versions.json").write_text(json.dumps(versions))
    return base


def test_a_bench_deck_with_a_dossier_is_refreshed(decks, writes):
    _deck(decks, "zur")
    deck_branch.refresh_dossier("zur")
    assert writes == [("zur", True, None)]


def test_a_missing_dossier_is_not_bootstrapped(decks, writes):
    _deck(decks, "zur", info=False)
    deck_branch.refresh_dossier("zur")
    assert writes == []


def test_a_retired_deck_is_frozen(decks, writes):
    _deck(decks, "zur", versions={"lifecycle": {"status": "retired"}})
    deck_branch.refresh_dossier("zur")
    assert writes == []


def test_the_old_private_name_still_works(decks, writes):
    _deck(decks, "zur")
    deck_branch._refresh_dossier("zur")
    assert writes == [("zur", True, None)]


@pytest.fixture
def stub_verbs(monkeypatch):
    monkeypatch.setattr(deck_branch, "new", lambda *a, **k: {"size": 100, "warnings": []})
    monkeypatch.setattr(deck_branch, "stage", lambda *a, **k: {"staged": True})
    monkeypatch.setattr(deck_branch, "unstage", lambda *a, **k: {"unstaged": True})


def test_branch_new_from_the_page_refreshes_a_bench_dossier(decks, writes, stub_verbs):
    _deck(decks, "zur")
    got = serve.call("branch/new", {"slug": "zur", "name": "t",
                                    "objective": "hoard_8 >= 6.0"})
    assert writes == [("zur", True, None)]
    assert got["dossier"] == "refreshed"


@pytest.mark.parametrize("payload", [
    {"out": "Sol Ring", "card": "Arcane Signet"},
    {"out": "Sol Ring", "card": "Arcane Signet", "undo": True},
])
def test_branch_stage_and_undo_from_the_page_refresh_it(decks, writes, stub_verbs,
                                                         payload):
    _deck(decks, "zur")
    got = serve.call("branch/stage", dict(payload, slug="zur", branch="t"))
    assert writes == [("zur", True, None)]
    assert got["dossier"] == "refreshed"


def test_a_refresh_failure_is_reported_not_raised(decks, monkeypatch, stub_verbs):
    """The branch write already happened; a 500 would call it a failure."""
    _deck(decks, "zur")

    def boom(args):
        raise RuntimeError("compose failed")
    monkeypatch.setattr(deck_info, "main", boom)
    got = serve.call("branch/stage", {"slug": "zur", "branch": "t",
                                      "out": "A", "card": "B"})
    assert got["staged"] and "compose failed" in got["dossier"]
