"""`validate-engine`'s `verified_by` check: a stack proves a line only if the
scenario is ABOUT the line's cards — and a permanent on YOUR board whose
triggered ability is the line is named by the scenario. sharknado's stack 001
(Windfall resolving with Brallin on the battlefield) is the case that found the
gap: the hand named seven cards, so `line_cards` skipped the board, and the one
proved line in the model could not cite the one passing stack.

Driven through `validate_engine.validate(slug, doc=...)` on a throwaway deck, so
the test exercises the production check rather than re-deriving it.
"""

import json

import pytest

from manamap.pilot import validate_engine as ve

SLUG = "engine-test-deck"
CARDS = ["Windfall", "Brallin, Skyshark Rider", "Shabraz, the Skyshark", "Sol Ring",
         "Reforge the Soul", "Rielle, the Everwise", "Command Tower"]


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


@pytest.fixture
def deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    base = decks / SLUG
    _write(base / "cards.json", {"deck": SLUG, "cards": [{"name": n, "quantity": 1} for n in CARDS]})
    _write(base / "stacks" / "001-base.json", {
        "id": "001", "slug": "base", "deck": SLUG, "title": "T",
        "scenario": {"board": {"you": ["Brallin, Skyshark Rider (3/3)", "Command Tower",
                                       "Shark token (3/3)"],
                               "opponents": [{"life": 40, "board": ["Sol Ring"]}]},
                     "hand": ["Reforge the Soul", "Rielle, the Everwise"],
                     "mana_available": "{0}",
                     "stack": [{"pos": 0, "object": "Windfall", "controller": "you"}],
                     "question": "?"},
        "resolution": {"steps": [{"n": 1, "action": "a", "effect": "e",
                                  "citations": [{"rule": "117.5", "quote": "q"}]}],
                       "final_state": {"summary": "s"}},
        "checker": {"verdict": "pass", "findings": [], "iterations": 1}})
    return base


def _model(via, verified_by="001"):
    stages = {"mana": ["Sol Ring", "Command Tower"],
              "ignition": ["Windfall", "Reforge the Soul"],
              "fuel": ["Rielle, the Everwise"],
              "output": ["Brallin, Skyshark Rider", "Shabraz, the Skyshark"]}
    return {"slug": SLUG, "thesis": "t",
            "stages": [{"stage": k, "label": k, "what_it_does": "w", "cards": v,
                        "single_point_of_failure": None, "evidence": []}
                       for k, v in stages.items()],
            "lines": [{"from": "ignition", "to": "output", "via": via,
                       "verified_by": verified_by, "note": "n"}],
            "map_disagreements": [], "unassigned": [], "open_questions": [],
            "proposed_goldfish_edits": []}


def _errors(result):
    """`validate` returns (errors, ...); only the errors matter here."""
    return result[0] if isinstance(result, tuple) else result


def _naming_errors(errors):
    return [e for e in errors if "does not name" in e]


def test_a_permanent_on_your_board_is_named_by_the_scenario(deck):
    errors = _errors(ve.validate(SLUG, doc=_model(["Windfall", "Brallin, Skyshark Rider"])))
    assert not _naming_errors(errors), errors


def test_a_card_on_no_zone_of_the_scenario_is_still_refused(deck):
    """The negative that keeps the widening honest: Shabraz is in the 99 and in
    the output stage but appears nowhere in the scenario, so the stack cannot
    verify a line through it."""
    errors = _errors(ve.validate(SLUG, doc=_model(["Windfall", "Shabraz, the Skyshark"])))
    assert any("Shabraz, the Skyshark" in e for e in _naming_errors(errors)), errors


def test_an_opponents_board_and_tokens_do_not_count(deck):
    """Sol Ring sits on an OPPONENT's board in the scenario; the Shark is a
    token. Neither is what the scenario is about for a line of ours."""
    errors = _errors(ve.validate(SLUG, doc=_model(["Windfall", "Sol Ring"])))
    assert any("Sol Ring" in e for e in _naming_errors(errors)), errors
    assert "Shark token" not in " ".join(ve._your_board_cards(
        json.loads((deck / "stacks" / "001-base.json").read_text())["scenario"], set(CARDS)))


def test_the_widening_is_a_union_not_a_replacement(deck):
    """Reforge the Soul is in HAND — `line_cards` territory — and must still count."""
    errors = _errors(ve.validate(SLUG, doc=_model(["Reforge the Soul", "Brallin, Skyshark Rider"])))
    assert not _naming_errors(errors), errors
