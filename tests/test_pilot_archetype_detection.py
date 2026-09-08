"""The archetype comes from what a frame DECLARES, not from a word it denies.

`detect_archetype` iterated `_ARCHETYPE_HINTS` in LIST ORDER and took the first
archetype whose substring appeared anywhere in the field. heliod's frame opens

    "Azorius draw-go CONTROL that gifts the whole table cards…"

and says, two hundred characters later,

    "…There is NO TWO-CARD COMBO — deck-facts finds one contained line…"

`combo` is tested before `control`, so the audit called a control deck `combo`
off the back of a sentence denying it.

NOT COSMETIC. Under `combo` the sweepers axis reads AT target; under `control` it
reads UNDER, 2 against 5-7. The single axis that changes verdict is the one the
deck had just been rebuilt around.

Same class as the `enters_tapped_unconditionally` bug this repo already paid for:
a word read outside the clause it attached to.
"""

import pytest

from manamap.pilot.deck_audit import _first_archetype_hint


def test_the_earliest_hint_wins_not_the_first_in_the_list():
    """THE REAL FRAME. `control` at position 13, `combo` at 200-odd, and list
    order put combo first."""
    text = ("azorius draw-go control that gifts the whole table cards, taxes and "
            "drains them for taking them, forts against combat, and wins with the "
            "second cast of approach of the second sun. there is no two-card combo "
            "— deck-facts finds one contained line.")
    name, word, at = _first_archetype_hint(text)
    assert name == "control" and word == "control"
    assert at < text.find("combo")


@pytest.mark.parametrize("phrase", [
    "there is no two-card combo in this list",
    "wins without a combo of any kind",
    "this deck is not a combo deck",
    "never a combo; it grinds",
    "zero combo lines survive the cut",
])
def test_a_hint_inside_a_negation_is_not_a_declaration(phrase):
    """A frame saying what it is NOT must not be read as saying what it is."""
    assert _first_archetype_hint(phrase) is None, phrase


def test_a_genuine_declaration_still_matches_even_late_in_the_text():
    """The fix must not stop a frame declaring itself in a later clause."""
    name, word, _at = _first_archetype_hint(
        "mono-black aristocrats-combo: a sacrifice engine that closes with one line")
    assert name == "combo"


def test_a_negated_mention_does_not_hide_a_real_one_elsewhere():
    """"There is no two-card combo, but this is a combo deck" is contrived — what
    is not contrived is a frame that denies one combo and declares another axis.
    The negated mention is skipped and the search CONTINUES rather than giving up
    on that word."""
    name, _w, _at = _first_archetype_hint(
        "there is no two-card combo here. the deck is a combo deck built on "
        "three-card lines.")
    assert name == "combo"


def test_no_hint_at_all_is_absent_rather_than_a_default():
    """A frame that names no archetype gets the BASE budget and the audit says
    so — inventing one would attribute a budget to the wrong deck silently."""
    assert _first_archetype_hint("a pile of cards that does things") is None
    assert _first_archetype_hint("") is None
