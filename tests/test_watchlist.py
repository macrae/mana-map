"""Candidate watch lists (2026-10-07): watchlist.json, its gate and serve's watch/mark.

Unit tests run on a temporary deck with the corpus and identity faked; the tracked
files' gate is the last test (regression tier).
"""
import json

import pytest

from manamap.pilot import validate_watchlist as vw
from manamap.pilot import watchlist as wl

POOL = {"Fateful Showdown": {"legal": True, "color_identity": {"R"}},
        "Rites of Refusal": {"legal": True, "color_identity": {"U"}},
        "Sol Ring": {"legal": True, "color_identity": set()},
        "Llanowar Elves": {"legal": True, "color_identity": {"G"}},
        "Shahrazad": {"legal": False, "color_identity": {"W"}}}


@pytest.fixture
def tmp_deck(tmp_path, monkeypatch):
    from manamap.pilot import card_pool, card_search, queue
    (tmp_path / "sharknado").mkdir()
    monkeypatch.setattr(wl, "deck_dir", lambda slug: tmp_path / slug)
    monkeypatch.setattr(card_pool, "load_pool", lambda: POOL)
    monkeypatch.setattr(card_search, "commander_identity", lambda slug: {"W", "U", "R"})
    monkeypatch.setattr(card_search, "deck_names", lambda slug: {"Sol Ring"})
    monkeypatch.setattr(queue, "read", lambda path=None: [
        {"id": "Q002", "kind": "hypothesis", "deck": "sharknado", "at": "2026-10-07"}])
    return tmp_path


CARDS = [{"name": "Fateful Showdown", "pays": "both", "axis": "interaction", "why": "damage, then a wheel"},
         {"name": "Rites of Refusal", "pays": "brallin", "axis": "interaction", "why": "discard is the counter"}]


def test_a_set_is_added_and_a_mark_round_trips_with_a_stamp(tmp_deck):
    wl.add_set("sharknado", "wheel-interaction", "Wheels that also interact", CARDS, source="Q002")
    row = wl.mark("sharknado", "wheel-interaction", "fateful showdown", verdict="watching", note="  try first ")
    assert row["verdict"] == "watching" and row["note"] == "try first" and row["at"]
    doc = json.loads((tmp_deck / "sharknado" / wl.ARTIFACT).read_text())
    assert doc["sets"][0]["cards"][0]["verdict"] == "watching"
    assert doc["sets"][0]["cards"][1]["verdict"] == "unreviewed"
    assert wl.summary("sharknado", "Q002") == "watching 1 of 2, 0 passed"


def test_the_gate_refuses_bad_vocabulary_identity_legality_and_duplicates(tmp_deck):
    good = {"slug": "sharknado", "sets": [{"id": "a", "title": "A", "source": "Q002",
                                            "cards": [dict(c, verdict="unreviewed") for c in CARDS]}]}
    assert vw.validate("sharknado", good) == []
    bad = json.loads(json.dumps(good))
    bad["sets"][0]["cards"][0]["verdict"] = "maybe"
    bad["sets"][0]["cards"].append({"name": "Llanowar Elves", "pays": "none", "axis": "ramp",
                                    "why": "x", "verdict": "unreviewed"})
    bad["sets"][0]["cards"].append({"name": "Shahrazad", "pays": "none", "axis": "other",
                                    "why": "x", "verdict": "unreviewed"})
    bad["sets"].append(dict(bad["sets"][0], source="Q404"))
    errors = vw.validate("sharknado", bad)
    joined = "\n".join(errors)
    assert "verdict is one of" in joined
    assert "outside the commander's colour identity" in joined
    assert "not Commander-legal" in joined
    assert "appears twice" in joined
    assert "not a queue item" in joined


def test_a_card_now_in_the_99_is_a_warning_not_an_error(tmp_deck):
    doc = {"slug": "sharknado", "sets": [{"id": "a", "title": "A", "cards": [
        {"name": "Sol Ring", "pays": "none", "axis": "ramp", "why": "x", "verdict": "watching"}]}]}
    warnings = []
    assert vw.validate("sharknado", doc, warnings) == []
    assert any("now in the 99" in w for w in warnings)


def test_a_bad_mark_writes_nothing(tmp_deck):
    wl.add_set("sharknado", "s", "S", CARDS)
    before = (tmp_deck / "sharknado" / wl.ARTIFACT).read_text()
    for bad in (dict(card="Fateful Showdown", verdict="maybe"), dict(card="Not A Card", verdict="pass")):
        with pytest.raises(SystemExit):
            wl.mark("sharknado", "s", **bad)
    with pytest.raises(SystemExit, match="no set"):
        wl.mark("sharknado", "nope", "Fateful Showdown", verdict="pass")
    assert (tmp_deck / "sharknado" / wl.ARTIFACT).read_text() == before


def test_serve_watch_mark_writes_through_the_gate_and_refuses_cleanly(tmp_deck):
    from manamap import serve

    wl.add_set("sharknado", "s", "S", CARDS)
    out = serve.call("watch/mark", {"slug": "sharknado", "set": "s", "card": "Rites of Refusal",
                                    "verdict": "pass"})
    assert out["card"]["verdict"] == "pass"
    assert "watch/mark" not in serve.GETTABLE
    with pytest.raises(ValueError, match="no set"):
        serve.call("watch/mark", {"slug": "sharknado", "set": "nope", "card": "x", "verdict": "pass"})
    with pytest.raises(ValueError, match="verdict or a note"):
        serve.call("watch/mark", {"slug": "sharknado", "set": "s", "card": "Rites of Refusal"})


@pytest.mark.regression
def test_every_tracked_watchlist_passes_its_gate():
    from manamap import config

    checked = 0
    for p in sorted(config.DECKS_DIR.glob(f"*/{wl.ARTIFACT}")):
        slug = p.parent.name
        assert vw.validate(slug, json.loads(p.read_text())) == [], slug
        checked += 1
    assert checked >= 1
