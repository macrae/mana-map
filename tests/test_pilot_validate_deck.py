"""validate-deck holds `cards.json` to the DECK'S format — the sideboard included.

The Commander cases live beside the fetcher in `test_pilot_fetch_deck.py` and
the spec-parameter cases in `test_pilot_formats.py`; this file is the sideboard
(Area C3, 2026-10-09). `cards.json` carries `sideboard` only when there is one,
so every rule here is invisible to the fourteen tracked Commander files — the
one that is not (copies counted by name across both boards) collapses to the
old per-entry check when the sideboard is empty, and a test here says so.

Real, format-legal names throughout: `illegal_cards` reads the corpus where
there is one, and a made-up name would be reported as "not in the corpus"
rather than exercising the rule under test.
"""

from manamap.pilot import formats
from manamap.pilot.validate_deck import validate


def _card(name, quantity, type_line="Instant", identity=("R",), commander=False):
    return {"name": name, "quantity": quantity, "is_commander": commander,
            "type_line": type_line, "color_identity": list(identity)}


def _sixty(bolts=4, mountains=56):
    return [_card("Lightning Bolt", bolts),
            _card("Mountain", mountains, "Basic Land — Mountain")]


def _side(n=15):
    rows = [_card("Blood Moon", 4, "Enchantment"), _card("Smash to Smithereens", 4),
            _card("Rending Volley", 4), _card("Relic of Progenitus", 3, "Artifact", ())]
    out, left = [], n
    for r in rows:
        if left <= 0:
            break
        out.append(dict(r, quantity=min(r["quantity"], left)))
        left -= out[-1]["quantity"]
    return out


def _structural(errors):
    """Everything but a legality line, which needs a corpus to produce."""
    return [e for e in errors if "legality" not in e]


def test_a_fifteen_card_sideboard_passes_and_sixteen_fails():
    assert _structural(validate({"cards": _sixty(), "sideboard": _side(15)},
                                formats.MODERN)) == []
    errors = validate({"cards": _sixty(), "sideboard": _side(15) + [_card("Alpine Moon", 1, "Enchantment")]},
                      formats.MODERN)
    assert any(e == "Sideboard has 16 cards, Modern allows at most 15" for e in errors), errors


def test_commander_has_no_sideboard_at_all():
    doc = {"cards": [_card("Sol Ring", 1, "Artifact", ())],
           "sideboard": [_card("Alpine Moon", 1, "Enchantment")]}
    errors = validate(doc, formats.COMMANDER)
    assert any(e == "Sideboard has 1 cards, Commander has no sideboard" for e in errors), errors


def test_a_commander_in_the_sideboard_is_an_error():
    doc = {"cards": _sixty(),
           "sideboard": [_card("Zurgo Bellstriker", 1, "Legendary Creature", commander=True)]}
    errors = validate(doc, formats.MODERN)
    [e] = [e for e in errors if "Commander in the sideboard" in e]
    assert "Zurgo Bellstriker" in e


def test_copies_are_counted_across_main_and_sideboard():
    """CR 100.4a: four in the sixty and one more in the fifteen is five."""
    doc = {"cards": _sixty(), "sideboard": _side(14) + [_card("Lightning Bolt", 1)]}
    errors = validate(doc, formats.MODERN)
    [e] = [e for e in errors if "Copies violation" in e]
    assert e.startswith("Copies violation: Lightning Bolt x5 across main and sideboard")
    assert "at most 4" in e and "CR 100.4a" in e
    # Two and two across the boards is four, which is the limit, not over it.
    doc = {"cards": _sixty(bolts=2, mountains=58),
           "sideboard": _side(13) + [_card("Lightning Bolt", 2)]}
    assert not any("Copies" in e for e in validate(doc, formats.MODERN))


def test_basics_are_exempt_across_boards_too():
    doc = {"cards": _sixty(), "sideboard": [_card("Mountain", 15, "Basic Land — Mountain")]}
    assert not any("Copies" in e or "Singleton" in e for e in validate(doc, formats.MODERN))


def test_the_commander_singleton_message_is_unchanged_without_a_sideboard():
    """The by-name count IS the old per-entry check for a file with no
    `sideboard`: same name, same quantity, same words."""
    doc = {"cards": [_card("Sol Ring", 2, "Artifact", ())]}
    errors = validate(doc, formats.COMMANDER)
    assert "Singleton violation: Sol Ring x2" in errors
    assert not any("Copies violation" in e for e in errors)
    assert not any("Sideboard" in e for e in errors), "no sideboard key, no sideboard rule"


def test_the_sideboard_is_outside_the_size_and_inside_the_pool():
    """Sixty main and fifteen side is a 60-card deck, not a 75-card one."""
    doc = {"cards": _sixty(), "sideboard": _side(15)}
    assert not any("cards, expected" in e for e in validate(doc, formats.MODERN))
    doc = {"cards": _sixty(mountains=55), "sideboard": _side(15)}
    assert any("Deck has 59 cards, expected at least 60" == e
               for e in validate(doc, formats.MODERN))
