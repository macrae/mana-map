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


def test_a_script_filed_under_a_misspelled_or_two_faced_name_is_still_found():
    """THE LOOKUP, NOT THE CARD (2026-10-21). Forge files Fallaji Archaeologist as
    `fallaji_archeologist.txt` and Ojer Axonil under both faces; the filename
    guess read both as "no script" and a seat audit would have failed Oliver's
    table for our own bug. The `Name:` index catches what the guess misses.
    Proven by dropping the index fallback in `script`."""
    for name in ("Fallaji Archaeologist", "Ojer Axonil, Deepest Might"):
        text = forge_cards.script(name)
        assert text and "Name:" in text, name
    assert forge_cards.script("Not A Real Card At All") is None
    assert forge_cards.script("Sol Ring"), "the plain path still works"


def test_an_opponent_seat_gets_mana_values_from_the_corpus():
    """A seat with no cards.json used to read `{}`, which switched off
    held-while-castable for every opponent; the corpus answers by name now."""
    from conftest import requires_corpus
    from manamap.config import OUTPUT_CSV_PATH
    if not OUTPUT_CSV_PATH.exists():
        pytest.skip("corpus not on disk")
    from pathlib import Path
    from manamap.sim import forge
    if not Path("data/opponents/teval-graveyard/decklist.txt").exists():
        pytest.skip("no opponent seat on disk")
    cmc = forge.cmc_map("teval-graveyard")
    assert cmc and "Teval, the Balanced Scale" in cmc and cmc["Teval, the Balanced Scale"] == 4
    assert all("Land" not in str(k) for k in ()) and not any(
        n in cmc for n in ("Forest", "Swamp", "Island")), "lands are played, not cast"
    kw = forge.keyword_map("teval-graveyard")
    assert isinstance(kw, dict) and kw
