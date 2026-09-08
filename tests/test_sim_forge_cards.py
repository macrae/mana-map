"""Forge's own AI flags, read from the engine rather than remembered.

`AI:RemoveDeck` marks a card Forge's AI has no logic for. It governs deck
GENERATION and NOT whether a supplied deck may contain the card — a claim to the
contrary was made in this repo, built into an argument, and killed by measuring
it: Swan Song carries `RemoveDeck:All` and was cast 28 times in 160 games.

So the flag is reported as a fact about the INSTRUMENT. A flagged card may be
excellent in paper and impossible to price in a Forge experiment.
"""

import pytest

from manamap.sim import forge_cards

pytestmark = pytest.mark.skipif(not forge_cards.installed(),
                                reason="Forge is not installed at FORGE_HOME")


def test_the_flags_come_out_of_the_installed_engine():
    """Read from cardsfolder.zip under FORGE_HOME, so it cannot drift from the
    engine actually playing the games."""
    assert forge_cards.ai_flag("Swan Song") == "All"
    assert forge_cards.ai_flag("Vexilus Praetor") == "Random"
    assert forge_cards.ai_flag("Sol Ring") is None


def test_a_double_faced_card_matches_on_either_face():
    """Forge names the file by both faces and a decklist names one. The same
    seam `_is_commander` and `deck_branch._canonical` already close."""
    assert forge_cards.ai_flag("Nonexistent // Swan Song") == "All"
    assert forge_cards.ai_flag("Swan Song // Nonexistent") == "All"


def test_an_unknown_card_is_absent_rather_than_flagged():
    assert forge_cards.ai_flag("Not A Real Card At All") is None
