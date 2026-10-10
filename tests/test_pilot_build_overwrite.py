"""A NEW-deck path never overwrites a finished deck.

`build/save` + `build/run` on an existing slug used to rewrite its brief and then
its `decklist.txt` wholesale — a sleeved list included. A deck is one with
`cards.json`; a decklist alone is an unfinished draft and stays writable.
`build_deck.main` refuses unless `--overwrite`, and refuses a sleeved or archived
deck even then; an overwrite on the bench keeps `decklist.txt.bak`.
"""
import json
from types import SimpleNamespace

import pytest

from manamap import config, serve
from manamap.pilot import build_deck

OLD_LIST = "1 Zur the Enchanter *CMDR*\n1 Sol Ring\n"


@pytest.fixture
def decks(tmp_path, monkeypatch):
    d = tmp_path / "decks"
    d.mkdir()
    monkeypatch.setattr(config, "DECKS_DIR", d)
    return d


def _deck(decks, slug, versions=None, cards=True):
    base = decks / slug
    base.mkdir()
    (base / "decklist.txt").write_text(OLD_LIST, encoding="utf-8")
    if cards:
        (base / "cards.json").write_text("{}", encoding="utf-8")
    if versions is not None:
        (base / "deck_versions.json").write_text(json.dumps(versions), encoding="utf-8")
    return base


@pytest.fixture
def fake_builder(monkeypatch):
    """`main`'s writes without the corpus: a plan and a list, both canned."""
    calls = []

    def build(slug):
        calls.append(slug)
        return {"commander": "Zur the Enchanter", "color_identity": ["W", "U", "B"],
                "slots": [{"name": "Arcane Signet"}], "land_counts": {"Island": 1},
                "bracket": {"target": 3, "target_name": "Upgraded", "computed_floor": 2},
                "cut_for_bracket": [], "manabase": {"shortfalls": []}}

    monkeypatch.setattr(build_deck, "build", build)
    monkeypatch.setattr(build_deck, "load_frame", lambda: {"name": [], "layout": []})
    monkeypatch.setattr(build_deck, "decklist_text", lambda plan, layouts: "NEW LIST\n")
    return calls


def _main(slug, overwrite=False):
    build_deck.main(SimpleNamespace(slug=slug, write_decklist=True, overwrite=overwrite))


SLEEVED = {"paper": {"version": "V1", "decklist_sha256": "x"}}
RETIRED = {"lifecycle": {"status": "retired"}}


def test_build_save_refuses_an_existing_deck(decks):
    _deck(decks, "zur")
    with pytest.raises(ValueError) as exc:
        serve.call("build/save", {"slug": "zur", "theme": "x"})
    assert "`zur` is already a deck (bench)" in str(exc.value)
    assert "another name" in str(exc.value)
    assert not (decks / "zur" / "brief.json").exists()


def test_build_save_names_the_rung(decks):
    _deck(decks, "zur", versions=SLEEVED)
    with pytest.raises(ValueError, match=r"already a deck \(sleeved\)"):
        serve.call("build/save", {"slug": "Zur", "theme": "x"})


def test_an_unfinished_draft_stays_writable(decks):
    """A decklist without cards.json is a draft the page is still working on."""
    _deck(decks, "zur", cards=False)
    got = serve.call("build/save", {"slug": "zur", "theme": "x", "fmt": "modern"})
    assert got["draft"] and (decks / "zur" / "brief.json").exists()


def test_build_run_refuses_an_existing_deck(decks, fake_builder):
    base = _deck(decks, "zur")
    (base / "brief.json").write_text(json.dumps({"commander": "Zur the Enchanter"}))
    with pytest.raises(ValueError, match="already a deck"):
        serve.call("build/run", {"slug": "zur"})
    assert fake_builder == []
    assert (base / "decklist.txt").read_text() == OLD_LIST


def test_main_refuses_without_overwrite_and_writes_nothing(decks, fake_builder):
    base = _deck(decks, "zur")
    with pytest.raises(SystemExit, match="--overwrite"):
        _main("zur")
    assert fake_builder == []
    assert not (base / "build_plan.json").exists()
    assert (base / "decklist.txt").read_text() == OLD_LIST


@pytest.mark.parametrize("versions,expect", [(SLEEVED, "SLEEVED"),
                                             (RETIRED, "revive it first")])
def test_overwrite_never_reaches_a_sleeved_or_archived_deck(decks, fake_builder,
                                                            versions, expect):
    base = _deck(decks, "zur", versions=versions)
    with pytest.raises(SystemExit, match=expect):
        _main("zur", overwrite=True)
    assert fake_builder == []
    assert (base / "decklist.txt").read_text() == OLD_LIST


def test_overwrite_on_the_bench_keeps_a_backup(decks, fake_builder):
    base = _deck(decks, "zur")
    _main("zur", overwrite=True)
    assert (base / "decklist.txt").read_text() == "NEW LIST\n"
    assert (base / "decklist.txt.bak").read_text() == OLD_LIST


def test_a_new_deck_needs_no_flag(decks, fake_builder):
    (decks / "fresh").mkdir()
    _main("fresh")
    assert (decks / "fresh" / "decklist.txt").read_text() == "NEW LIST\n"
    assert not (decks / "fresh" / "decklist.txt.bak").exists()


def test_the_cli_carries_overwrite_on_build_and_build_deck():
    from manamap.cli import build_parser
    p = build_parser()
    for argv in (["pilot", "build-deck", "zur", "--write-decklist", "--overwrite"],
                 ["pilot", "build", "zur", "--overwrite"]):
        assert p.parse_args(argv).overwrite is True
