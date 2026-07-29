"""The Short List contract: exactly ten, bench-first, claims verified."""

import copy

from manamap.pilot.validate_considering import validate


def deck_doc():
    cards = [
        {"name": "Test Commander", "is_commander": True, "is_sideboard": False,
         "type_line": "Legendary Creature"},
    ]
    for i in range(5):
        cards.append({"name": f"Main {i}", "is_commander": False,
                      "is_sideboard": False, "type_line": "Creature"})
    for i in range(12):
        cards.append({"name": f"Bench {i}", "is_commander": False,
                      "is_sideboard": True, "type_line": "Instant"})
    cards.append({"name": "Storm Counter", "is_commander": False,
                  "is_sideboard": True, "type_line": "Card"})
    return {"cards": cards}


def good_doc():
    ten = [{"card": f"Bench {i}", "source": "sideboard",
            "why": f"Specific reason {i}."} for i in range(7)]
    ten += [{"card": f"Pool Pick {i}", "source": "pool",
             "why": f"Specific reason {i}."} for i in range(3)]
    return {"slug": "test", "assessment": "Ten cards, honestly ranked.",
            "ten": ten, "gaps": []}


def test_good_doc_passes():
    assert validate(good_doc(), deck_doc()) == []


def test_count_must_be_exactly_ten():
    doc = good_doc()
    doc["ten"] = doc["ten"][:9]
    assert any("exactly 10" in e for e in validate(doc, deck_doc()))


def test_sideboard_source_must_be_on_the_bench():
    doc = good_doc()
    doc["ten"][0]["card"] = "Not A Bench Card"
    assert any("bench holds no such card" in e for e in validate(doc, deck_doc()))


def test_accessories_are_not_bench_cards():
    doc = good_doc()
    doc["ten"][0]["card"] = "Storm Counter"
    assert any("bench holds no such card" in e for e in validate(doc, deck_doc()))


def test_pool_source_must_not_be_in_the_deck():
    doc = good_doc()
    doc["ten"][7]["card"] = "Main 2"
    assert any("already in the deck" in e for e in validate(doc, deck_doc()))


def test_duplicates_rejected():
    doc = good_doc()
    doc["ten"][1]["card"] = "Bench 0"
    assert any("duplicate" in e for e in validate(doc, deck_doc()))


def test_natural_cut_must_be_maindeck_and_unique():
    doc = good_doc()
    doc["ten"][0]["natural_cut"] = "Bench 5"
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
