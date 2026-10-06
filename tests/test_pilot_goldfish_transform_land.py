"""A transforming card is its front face: Legion's Landing is not a land drop.

The land test read the WHOLE type line, so "Legendary Enchantment // Legendary
Land" was played as a land: Edgar's champion had a phantom 37th land, and a
branch that cut Legion's Landing read +2.5 points of missed drops by T5 that
were the phantom leaving (found 2026-10-05). A modal DFC's land face really is a
land drop, so it still counts.
"""
from manamap.pilot import goldfish_library


def _card(name, type_line, layout, text=""):
    return {"name": name, "type_line": type_line, "layout": layout,
            "oracle_text": text, "mana_cost": "{W}", "cmc": 1}


def test_a_transforming_card_with_a_land_back_is_not_a_land():
    c = _card("Legion's Landing // Adanto, the First Fort",
              "Legendary Enchantment // Legendary Land", "transform",
              "When Legion's Landing enters, create a 1/1 white Vampire creature token with lifelink.")
    assert not goldfish_library.classify(c)["is_land"]


def test_a_modal_dfc_land_face_is_still_a_land_drop():
    c = _card("Fell the Profane // Fell Mire", "Instant // Land", "modal_dfc")
    assert goldfish_library.classify(c)["is_land"]


def test_a_plain_land_is_a_land():
    assert goldfish_library.classify(_card("Swamp", "Basic Land — Swamp", "normal"))["is_land"]
