"""deck-info: the workbench view composes what other commands own and computes
nothing new — so it must (a) never crash on a deck that has almost nothing, and
(b) derive every `next` suggestion from a condition that is true right now."""

import json

import pytest

from manamap.pilot import deck_info
from manamap.pilot.deck_notes import append_entry

from conftest import requires_deck

SLUG = "infodeck"


@pytest.fixture
def bare_deck(tmp_path, monkeypatch):
    """decklist + cards.json and nothing else — a deck someone just made."""
    decks = tmp_path / "decks"
    base = decks / SLUG
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    (base / "cards.json").write_text(json.dumps({
        "deck": SLUG, "decklist_sha256": "x",
        "cards": [{"name": "Radagast of Rhosgobel", "is_commander": True, "type_line": "Legendary Creature",
                   "mana_cost": "{3}{G}", "cmc": 4, "colors": ["G"], "color_identity": ["G"],
                   "oracle_text": "", "layout": "normal"},
                  {"name": "Forest", "quantity": 1, "type_line": "Basic Land — Forest",
                   "mana_cost": "", "cmc": 0, "colors": [], "color_identity": [],
                   "oracle_text": "({T}: Add {G}.)", "layout": "normal"}]}))
    return base


def test_a_bare_deck_composes_and_is_told_to_play(bare_deck):
    info = deck_info.compose(SLUG)
    assert info["commander"] == ["Radagast of Rhosgobel"]
    assert info["record"]["games"] == 0 and info["engine"] is None and info["diagnosis"] is None
    assert info["version"]["of"] == 0, "no git history is an empty list, not an error"
    assert any("captain's log" in n for n in info["next"])
    assert not any("un-debriefed" in n for n in info["next"])


def test_next_derives_from_what_is_true(bare_deck):
    append_entry(SLUG, "game one", result="loss")
    info = deck_info.compose(SLUG)
    assert info["record"] == {"games": 1, "win": 0, "loss": 1, "draw": 0,
                              "last_played": info["record"]["last_played"], "undebriefed": ["001"]}
    assert any("not yet debriefed" in n and "/debrief" in n for n in info["next"])
    (bare_deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": {"summary": "s", "takeaways": [],
                                           "open_questions": [{"question": "?", "settled_by": "goldfish",
                                                               "why_it_matters": "?"}]}}}))
    info = deck_info.compose(SLUG)
    assert not any("debriefed" in n for n in info["next"])
    assert info["open_questions"][0]["from"] == "log:001"
    assert any("goldfish ×1" in n for n in info["next"])


def test_json_and_print_agree_on_the_same_dict(bare_deck, capsys):
    deck_info.main(type("A", (), {"slug": SLUG, "as_json": True})())
    out = json.loads(capsys.readouterr().out)
    assert set(out) >= {"slug", "version", "status", "record", "next"}
    deck_info.main(type("A", (), {"slug": SLUG, "as_json": False})())
    text = capsys.readouterr().out
    assert "WORKBENCH — infodeck" in text and "NEXT" in text


@requires_deck
def test_a_real_deck_composes_every_panel():
    info = deck_info.compose("radagast")
    assert info["engine"]["critic"] == "pass" and info["engine"]["verified_lines"] >= 1
    assert info["goldfish"]["commander_mean_cast_turn"] is not None
    assert info["bracket"]["floor"] is not None and info["audit"]["archetype"]
    assert info["version"]["of"] >= 1 and not info["status"]["invalid"]
