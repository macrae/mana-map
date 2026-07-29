"""Fetch Quests contract: one wish per tutor, real fetches, legal targets."""

from manamap.pilot.validate_tutor_guide import deck_tutors, validate


def deck_doc():
    return {"cards": [
        {"name": "Test Commander", "is_commander": True, "is_sideboard": False,
         "type_line": "Legendary Creature — Dinosaur", "oracle_text": ""},
        {"name": "Worldly Tutor", "is_commander": False, "is_sideboard": False,
         "type_line": "Instant",
         "oracle_text": "Search your library for a creature card, reveal it..."},
        {"name": "Nature's Lore", "is_commander": False, "is_sideboard": False,
         "type_line": "Sorcery",
         "oracle_text": "Search your library for a Forest card..."},
        {"name": "Big Dino", "is_commander": False, "is_sideboard": False,
         "type_line": "Creature — Dinosaur", "oracle_text": ""},
        {"name": "Nice Rock", "is_commander": False, "is_sideboard": False,
         "type_line": "Artifact", "oracle_text": ""},
        {"name": "Windswept Heath", "is_commander": False, "is_sideboard": False,
         "type_line": "Land",
         "oracle_text": "Search your library for a Forest or Plains card..."},
    ]}


def good_doc():
    return {"slug": "test", "assessment": "One tutor, honest wishes.",
            "tutors": [{"card": "Worldly Tutor", "targets": [
                {"scenario": "Turn three, board empty.",
                 "fetch": "Big Dino", "why": "The body the curve wants."}]}],
            "gaps": []}


def test_land_only_tutors_and_fetch_lands_are_excluded():
    assert deck_tutors(deck_doc()["cards"]) == ["Worldly Tutor"]


def test_good_doc_passes():
    assert validate(good_doc(), deck_doc()) == []


def test_missing_tutor_entry_is_a_hole():
    doc = good_doc()
    doc["tutors"] = []
    assert any("has no entry" in e for e in validate(doc, deck_doc()))


def test_non_tutor_entry_rejected():
    doc = good_doc()
    doc["tutors"].append({"card": "Nature's Lore", "targets": [
        {"scenario": "x", "fetch": "Big Dino", "why": "y"}]})
    assert any("not a maindeck library-search tutor" in e
               for e in validate(doc, deck_doc()))


def test_fetch_must_be_in_the_deck():
    doc = good_doc()
    doc["tutors"][0]["targets"][0]["fetch"] = "Black Lotus"
    assert any("not in the deck" in e for e in validate(doc, deck_doc()))


def test_fetch_must_match_the_search_constraint():
    doc = good_doc()
    doc["tutors"][0]["targets"][0]["fetch"] = "Nice Rock"
    assert any("searches for a" in e for e in validate(doc, deck_doc()))


def test_commander_is_a_legal_creature_fetch():
    doc = good_doc()
    doc["tutors"][0]["targets"][0]["fetch"] = "Test Commander"
    assert validate(doc, deck_doc()) == []
