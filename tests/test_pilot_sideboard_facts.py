"""Tests for sideboard-facts — what a sideboard card would do if you ran it."""

import json

import pytest

from conftest import requires_deck
from manamap.pilot import sideboard_facts as sf


def card(name, **overrides):
    base = {
        "name": name, "quantity": 1, "is_commander": False, "is_sideboard": False,
        "mana_cost": "{1}{R}", "cmc": 2.0, "type_line": "Instant",
        "oracle_text": "", "colors": ["R"], "color_identity": ["R"], "layout": "normal",
    }
    base.update(overrides)
    return base


ACCESSORY = card("Storm Counter", is_sideboard=True, type_line="Card",
                 mana_cost="", cmc=0.0, colors=[], color_identity=[])


# ── The accessory split ──────────────────────────────────────────────────
#
# goblin-storm's sideboard is 3 entries and 1 card: the other two are Secret Lair
# table aids with type_line "Card". "What would this do if I ran it" is a
# meaningless question about a counter token.

def test_accessories_are_separated_from_real_cards():
    doc = {"cards": [card("Sol Ring"), card("Sazacap's Brew", is_sideboard=True), ACCESSORY]}
    main, side, accessories = sf.split_deck(doc)
    assert [c["name"] for c in main] == ["Sol Ring"]
    assert [c["name"] for c in side] == ["Sazacap's Brew"]
    assert [c["name"] for c in accessories] == ["Storm Counter"]


def test_a_sideboard_of_only_accessories_is_not_analysable(monkeypatch):
    doc = {"cards": [card("Sol Ring"), ACCESSORY]}
    monkeypatch.setattr(sf, "load_deck_cards", lambda slug: doc)
    facts = sf.analyze("toy")
    assert facts["available"] is False
    assert "Storm Counter" in facts["accessories"]
    assert "0 real cards" in facts["reason"]


def test_no_sideboard_at_all_says_so(monkeypatch):
    monkeypatch.setattr(sf, "load_deck_cards", lambda slug: {"cards": [card("Sol Ring")]})
    facts = sf.analyze("toy")
    assert facts["available"] is False and facts["reason"] == "no sideboard"


# ── Lines opened ─────────────────────────────────────────────────────────

DETAILS = {
    "combos": [
        {"cards": ["A", "B"], "produces": ["Infinite mana"], "bracket": 4, "popularity": 10},
        {"cards": ["A", "C"], "produces": ["Value"], "bracket": 1, "popularity": 5},
    ],
    "by_card": {"A": [0, 1], "B": [0], "C": [1]},
}


def test_lines_opened_is_a_set_difference_not_a_rescan():
    """A line already complete without the card is not something the sideboard opens."""
    # A+B is already in the deck; adding C opens A+C and nothing else.
    opened = sf.lines_opened(["A", "B"], "C", DETAILS)
    assert [line["cards"] for line in opened] == [["A", "C"]]


def test_lines_opened_is_empty_when_the_card_completes_nothing():
    assert sf.lines_opened(["A", "B"], "Z", DETAILS) == []


def test_every_opened_line_is_flagged_as_unverified():
    """A combo the sideboard completes is a candidate, never a fact."""
    for line in sf.lines_opened(["A", "B"], "C", DETAILS):
        assert line["status"] == "needs a stack scenario"


# ── Colour identity ──────────────────────────────────────────────────────

def test_off_identity_sideboard_card_is_flagged(monkeypatch):
    """validate-deck exempts the sideboard, so nothing else reports this."""
    doc = {"cards": [
        card("Zada, Hedron Grinder", is_commander=True, color_identity=["R"]),
        card("Counterspell", is_sideboard=True, color_identity=["U"], colors=["U"]),
    ]}
    monkeypatch.setattr(sf, "load_deck_cards", lambda slug: doc)
    monkeypatch.setattr(sf, "load_reference", lambda: (_ for _ in ()).throw(SystemExit("no data")))
    facts = sf.analyze("toy")
    entry = facts["sideboard"][0]
    assert entry["in_color_identity"] is False
    assert any("COLOUR IDENTITY" in n for n in facts["notes"])


def test_in_identity_sideboard_card_is_not_flagged(monkeypatch):
    doc = {"cards": [
        card("Zada, Hedron Grinder", is_commander=True, color_identity=["R"]),
        card("Sazacap's Brew", is_sideboard=True, color_identity=["R"]),
    ]}
    monkeypatch.setattr(sf, "load_deck_cards", lambda slug: doc)
    monkeypatch.setattr(sf, "load_reference", lambda: (_ for _ in ()).throw(SystemExit("no data")))
    facts = sf.analyze("toy")
    assert facts["sideboard"][0]["in_color_identity"] is True


# ── Notes ────────────────────────────────────────────────────────────────

def test_notes_name_the_accessories_that_were_skipped():
    facts = {"accessories": ["Red Mana", "Storm Counter"], "sideboard": []}
    notes = " ".join(sf.build_notes(facts))
    assert "Red Mana" in notes and "cannot be swapped in" in notes


def test_notes_report_a_bracket_rise_as_computed_not_judged():
    facts = {"accessories": [], "sideboard": [
        {"name": "Thassa's Oracle", "in_color_identity": True, "edhrec_rank": 1,
         "bracket_if_added": {"before": 2, "after": 4, "delta": 2}},
    ]}
    notes = " ".join(sf.build_notes(facts))
    assert "2->4" in notes and "computed floor, not a judgment" in notes


def test_notes_flag_a_missing_edhrec_rank():
    """Without a popularity signal, 'stronger long-term default' has to argue from the deck."""
    facts = {"accessories": [], "sideboard": [
        {"name": "Sazacap's Brew", "in_color_identity": True, "edhrec_rank": None},
    ]}
    assert "Sazacap's Brew" in " ".join(sf.build_notes(facts))


# ── Real deck ────────────────────────────────────────────────────────────

@requires_deck
def test_goblin_storm_has_one_analysable_card_and_two_accessories():
    facts = sf.analyze("goblin-storm")
    assert facts["available"] is True
    assert [e["name"] for e in facts["sideboard"]] == ["Sazacap's Brew"]
    assert facts["accessories"] == ["Red Mana", "Storm Counter"]


@requires_deck
def test_empty_sideboards_report_unavailable():
    for slug in ("hapatra", "sisay"):
        assert sf.analyze(slug)["available"] is False


@requires_deck
def test_real_deck_is_deterministic():
    assert sf.analyze("goblin-storm") == sf.analyze("goblin-storm")
