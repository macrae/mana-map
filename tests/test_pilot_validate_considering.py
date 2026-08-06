"""The Short List contract: exactly ten, none already in the deck, claims verified.

Ownership is not part of the contract. The list used to carry
`source: "sideboard" | "pool"` and rank bench picks first; a card is now on the
list because it is worth knowing about, and whether the pilot owns it is not a
question the artifact asks.
"""

import copy

from manamap.pilot.validate_considering import validate


def deck_doc():
    cards = [
        {"name": "Test Commander", "is_commander": True,
         "type_line": "Legendary Creature"},
    ]
    for i in range(5):
        cards.append({"name": f"Main {i}", "is_commander": False,
                      "type_line": "Creature"})
    return {"cards": cards}


def good_doc():
    ten = [{"card": f"Pool Pick {i}", "why": f"Specific reason {i}."}
           for i in range(10)]
    return {"slug": "test", "assessment": "Ten cards, honestly ranked.",
            "ten": ten, "gaps": []}


def test_good_doc_passes():
    assert validate(good_doc(), deck_doc()) == []


def test_count_must_be_exactly_ten():
    doc = good_doc()
    doc["ten"] = doc["ten"][:9]
    assert any("exactly 10" in e for e in validate(doc, deck_doc()))


def test_pool_source_must_not_be_in_the_deck():
    doc = good_doc()
    doc["ten"][7]["card"] = "Main 2"
    assert any("already in the deck" in e for e in validate(doc, deck_doc()))


def test_duplicates_rejected():
    doc = good_doc()
    doc["ten"][1]["card"] = doc["ten"][0]["card"]
    assert any("duplicate" in e for e in validate(doc, deck_doc()))


def test_natural_cut_must_be_maindeck_and_unique():
    doc = good_doc()
    doc["ten"][0]["natural_cut"] = "Not In The Deck"
    errors = validate(doc, deck_doc())
    assert any("not in the maindeck" in e for e in errors)
    doc = good_doc()
    doc["ten"][0]["natural_cut"] = "Test Commander"
    assert any("may not be the commander" in e for e in validate(doc, deck_doc()))
    doc = good_doc()
    doc["ten"][0]["natural_cut"] = "Main 1"
    doc["ten"][1]["natural_cut"] = "Main 1"
    assert any("already claimed" in e for e in validate(doc, deck_doc()))


def test_empty_why_rejected():
    doc = good_doc()
    doc["ten"][3]["why"] = "  "
    assert any("`why` is empty" in e for e in validate(doc, deck_doc()))


def test_unverified_line_status_vocabulary():
    doc = good_doc()
    doc["ten"][0]["evidence"] = {
        "combo_lines_opened": [{"cards": ["A", "B"], "status": "probably fine"}]}
    assert any("status must be" in e for e in validate(doc, deck_doc()))
