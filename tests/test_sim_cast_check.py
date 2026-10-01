"""`forge-cast-check`: prove the Forge AI plays a card before a branch depends on it.

The counting is driven through the production parser on a synthetic log in Forge's own
line shapes (the telemetry patch's owner-bearing zone lines), and the shell is built from
the real edgar-vampires deck. The bug this exists for: Toxic Deluge, drawn 28 times and
cast 0 across a 200-game branch arm, read as "castable" because its script was unflagged.
"""
import pytest

from manamap import config
from manamap.sim import cast_check as cc

from conftest import requires_deck

OURS = "Ai(1)-mm-castcheck-edgar-vampires"
OPP = "Ai(2)-mm-giada-angels"
HEAD = [f"Mulligan: {OURS} has kept a hand of 7 cards", f"Mulligan: {OPP} has kept a hand of 7 cards"]
TAIL = ["Game Outcome: Turn 9", f"Game Outcome: {OPP} has won because all opponents have lost",
        "Game Result: Game 1 ended in 100 ms."]


def _game(lines):
    return "\n".join(HEAD + lines + TAIL) + "\n"


def _held_game():
    """Deluge drawn on turn 1, three lands by our turn 5, never cast, held at the end."""
    return _game([
        f"Turn: Turn 1 ({OURS})", f"Zone Change: Toxic Deluge (211) was put into Hand from Library. owner {OURS}",
        f"Land: {OURS} played Swamp (1)", f"Turn: Turn 2 ({OPP})",
        f"Turn: Turn 3 ({OURS})", f"Land: {OURS} played Swamp (2)", f"Turn: Turn 4 ({OPP})",
        f"Turn: Turn 5 ({OURS})", f"Land: {OURS} played Swamp (3)", f"Turn: Turn 6 ({OPP})",
        f"Turn: Turn 7 ({OURS})", f"Land: {OURS} played Swamp (4)", f"Turn: Turn 8 ({OPP})"])


def _cast_game():
    return _game([
        f"Turn: Turn 1 ({OURS})", f"Zone Change: Toxic Deluge (211) was put into Hand from Library. owner {OURS}",
        f"Land: {OURS} played Swamp (1)", f"Turn: Turn 2 ({OPP})",
        f"Turn: Turn 3 ({OURS})", f"Land: {OURS} played Swamp (2)", f"Turn: Turn 4 ({OPP})",
        f"Turn: Turn 5 ({OURS})", f"Land: {OURS} played Swamp (3)",
        f"Add To Stack: {OURS} cast Toxic Deluge (211)",
        f"Zone Change: Toxic Deluge (211) was put into Stack from Hand. owner {OURS}",
        f"Resolve Stack: Toxic Deluge (211) - All creatures get -X/-X until end of turn.",
        f"Zone Change: Toxic Deluge (211) was put into Graveyard from Stack. owner {OURS}",
        f"Turn: Turn 6 ({OPP})"])


def test_a_held_card_is_counted_as_castable_and_uncast_and_the_verdict_says_held():
    c = cc.count(_held_game(), "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["games"] == 1 and c["drawn_games"] == 1 and c["drawn"] == 1
    assert c["cast"] == 0 and c["activated"] == 0 and c["discarded"] == 0
    assert c["castable_uncast_turns"] == 2 and c["held_castable_games"] == 1 and c["held_at_end_games"] == 1
    assert cc.verdict(c).startswith("HELD:")


def test_a_cast_card_is_counted_and_the_verdict_says_played():
    c = cc.count(_cast_game() + _held_game(), "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["games"] == 2 and c["cast"] == 1 and c["drawn_games"] == 2
    assert c["castable_uncast_turns"] == 2, "the held game's turns, not the cast game's turn of casting"
    assert cc.verdict(c).startswith("PLAYED: cast 1")


def test_the_opponents_copy_is_not_ours():
    """The bug: counting every `cast Toxic Deluge` line — jarad runs it too, and the branch
    arm's logs carried the opponent's casts beside our zero."""
    text = _game([f"Turn: Turn 1 ({OPP})", f"Zone Change: Toxic Deluge (99) was put into Hand from Library. owner {OPP}",
                  f"Add To Stack: {OPP} cast Toxic Deluge (99)", f"Turn: Turn 2 ({OURS})"])
    c = cc.count(text, "Toxic Deluge", "mm-castcheck-edgar-vampires", 3.0)
    assert c["cast"] == 0 and c["drawn_games"] == 0
    assert cc.verdict(c).startswith("NOT DRAWN")


@requires_deck
@pytest.mark.skipif(not (config.DECKS_DIR / "edgar-vampires" / "cards.json").exists(), reason="requires edgar-vampires")
def test_the_shell_is_the_decks_commander_copies_filler_and_basics_of_the_cards_colours():
    text, facts = cc.shell_decklist("edgar-vampires", "Toxic Deluge", copies=4)
    lines = text.splitlines()
    assert lines[:2] == ["Commander:", "1 Edgar Markov"] and "4 Toxic Deluge" in lines
    assert facts["lands"] == 36 and facts["basics"] == {"Swamp": 36}, "Deluge is mono-black; its basics are Swamps"
    assert facts["copies"] + facts["filler"] + facts["lands"] == 99 and facts["cmc"] == 3.0
    with pytest.raises(SystemExit):
        cc.shell_decklist("edgar-vampires", "Counterspell")          # outside the identity
    with pytest.raises(SystemExit):
        cc.shell_decklist("edgar-vampires", "Not A Card")
