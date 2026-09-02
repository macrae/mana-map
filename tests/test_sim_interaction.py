"""THE SIM COULD SEE WHAT DIED AND NOT WHAT KILLED IT.

`creatures_lost` counts a creature leaving the battlefield. It cannot separate a
Swords to Plowshares from a chump block from a sacrifice outlet — which is the
whole distinction on Edgar and Yawgmoth, where feeding your own creatures to your
own engine IS the deck working. A record where both decks read "1.9 creatures
lost" says nothing about whether either was dismantled.

The targets were in the log the entire time. `parse.RX["stack"]` captured group 4
(`targeting [...]`) and `_event` put it in the event dict, and then nothing ever
read it: 260 of 1566 stack objects in one log named a target and every one was
discarded before it reached a fact.

WHAT THIS IS NOT. It is not a removal count, and the tests below pin the two
places that distinction bites: a permanent the seat controls itself (the equip
that prompted the check was Lightning Greaves on the caster's own commander), and
a spell whose EFFECT the log never states.
"""

import pytest

from manamap.sim import parse

SEATS = ["Ai(1)-me", "Ai(2)-them"]


def _log(*lines):
    # A GAME BEGINS AT THE MULLIGAN LINE. `parse_games` opens a game and learns
    # its seats there and nowhere else, so a fixture without these produces zero
    # games and every assertion below fails on a missing seat rather than on the
    # thing it is testing.
    head = [f"Mulligan: {s} has kept a hand of 7 cards" for s in SEATS]
    head.append("Turn: Turn 1 (Ai(1)-me)")
    return "\n".join(head + list(lines) + [
        "Game Outcome: Turn 1",
        "Game Outcome: Ai(1)-me has won because all opponents have lost",
        "Game Result: Game 1 ended in 100 ms",
    ])


def _facts(*lines):
    games = parse.parse_games(_log(*lines))
    assert len(games) == 1, games
    return parse.game_facts(games[0])["per_seat"]


def test_a_spell_aimed_at_an_opponents_permanent_is_counted_on_both_sides():
    """The base case, and the one the whole channel exists for."""
    per = _facts(
        # OWNERSHIP COMES FROM AN ATTACK, A BLOCK OR A LAND — never from the
        # cast. `Add To Stack: SEAT cast Bloodghast` names the seat and the card
        # and carries NO id, while the target group is `Bloodghast (11)`; there
        # is nothing to join them on. See the coverage test below.
        "Combat: Ai(2)-them assigned Bloodghast (11) to attack Ai(1)-me.",
        "Add To Stack: Ai(1)-me cast Swords to Plowshares targeting [Bloodghast (11)]",
    )
    assert per["Ai(1)-me"]["interaction_cast"] == 1
    assert per["Ai(2)-them"]["interaction_received"] == 1
    # ...and never the other way round.
    assert per["Ai(2)-them"]["interaction_cast"] == 0
    assert per["Ai(1)-me"]["interaction_received"] == 0


def test_a_seat_targeted_directly_is_counted_without_the_owner_map():
    """`targeting [Ai(2)-them]` names the seat itself — a drain or a burn spell
    at the face — and must not need a permanent lookup to be attributed."""
    per = _facts("Add To Stack: Ai(1)-me cast Lava Spike targeting [Ai(2)-them]")
    assert per["Ai(1)-me"]["interaction_cast"] == 1
    assert per["Ai(2)-them"]["interaction_received"] == 1


def test_aiming_at_your_own_board_is_not_interaction():
    """THE BUG THIS GUARDS. Most targeting traffic in a real log is a seat
    aiming at itself — the equip that prompted the check was Lightning Greaves
    on the caster's own commander, and an unfiltered count would have reported a
    deck that never interacts with anybody as the most interactive at the table.

    Re-introduce the bug by deleting `hit.discard(src)` in `_aim` and this fails.
    """
    per = _facts(
        "Land: Ai(1)-me played Plains (3)",
        "Add To Stack: Ai(1)-me cast Giada (5)",
        "Add To Stack: Ai(1)-me activated Lightning Greaves targeting [Giada (5)]",
    )
    assert per["Ai(1)-me"]["interaction_cast"] == 0, (
        "equipping your own creature was counted as interaction")
    assert per["Ai(1)-me"]["interaction_received"] == 0


def test_one_stack_object_counts_once_per_opposing_seat_not_once_per_target():
    """A wrath naming four of one player's creatures is ONE act of interaction
    against that player. Counting per target would make a board-sweeping deck
    read four times as interactive as a spot-removal one, which is a difference
    in what the cards do, not in how much the seat interacts.

    Re-introduce the bug by incrementing inside the target loop; this fails 3:1.
    """
    per = _facts(
        "Combat: Ai(2)-them assigned A (11), B (12) and C (13) to attack Ai(1)-me.",
        "Add To Stack: Ai(1)-me cast Wrath targeting [A (11), B (12), C (13)]",
    )
    assert per["Ai(1)-me"]["interaction_cast"] == 1
    assert per["Ai(2)-them"]["interaction_received"] == 1


def test_a_permanent_nobody_was_seen_playing_is_attributed_to_nobody():
    """ABSENT MEANS ABSENT. An unattributable target must not be guessed at —
    and the guess that would feel natural, "assume the active seat", is exactly
    the one that would inflate whichever seat is taking its turn."""
    per = _facts("Add To Stack: Ai(1)-me cast Swords to Plowshares targeting [Ghost (99)]")
    assert per["Ai(1)-me"]["interaction_cast"] == 0
    assert per["Ai(2)-them"]["interaction_received"] == 0


def test_a_trigger_is_not_an_act_of_interaction():
    """`_aim` is wired to `cast` and `activated` only.

    Vito's drain TRIGGERS at a seat 3x a game. They are the consequence of a
    choice, not a choice — counting them would have read the pod's least
    interactive deck as its most, and that misreading survived until this was
    checked against the actual card names behind the count.
    """
    per = _facts("Add To Stack: Ai(1)-me triggered Vito targeting [Ai(2)-them]")
    assert per["Ai(1)-me"]["interaction_cast"] == 0
    assert per["Ai(2)-them"]["interaction_received"] == 0


@pytest.mark.parametrize("key", ["interaction_cast", "interaction_received"])
def test_the_channel_reaches_the_aggregate(key):
    """A metric the model sets and nothing reads is the `treasure_doubler`
    failure: set-and-unread, fifteen candidates returning byte-identical
    figures. The fact carrying it is not enough — the aggregate must too."""
    facts = [parse.game_facts(g) for g in parse.parse_games(_log(
        "Combat: Ai(2)-them assigned Bloodghast (11) to attack Ai(1)-me.",
        "Add To Stack: Ai(1)-me cast Swords to Plowshares targeting [Bloodghast (11)]",
    ))]
    agg = parse.aggregate(facts, "me", {"Ai(1)-me": "me", "Ai(2)-them": "them"})
    assert key in agg["seats"]["me"], f"{key} never reaches the aggregate"
    assert agg["seats"]["me"][key]["mean"] == (1.0 if key == "interaction_cast" else 0.0)


def test_the_record_states_what_this_measure_is_not():
    """Every figure carries its definition IN THE REPORT THAT PRINTS IT. This one
    is named `interaction`, not `removal`, precisely because the log says a spell
    targeted something and never says what it did — and a reader who assumes
    otherwise will read a pump spell aimed at an opponent's creature as removal.
    """
    facts = [parse.game_facts(g) for g in parse.parse_games(_log(
        "Add To Stack: Ai(1)-me cast Lava Spike targeting [Ai(2)-them]"))]
    agg = parse.aggregate(facts, "me", {"Ai(1)-me": "me", "Ai(2)-them": "them"})
    limits = " ".join(agg["limits"])
    assert "interaction_cast" in limits
    assert "not a removal count" in limits
    # The Forge CEILING, recorded where the figures are, not only in a doc.
    assert "Card advantage" in limits and "from Library" in limits


def test_a_cast_does_not_establish_ownership_and_the_limit_says_so():
    """THE COVERAGE CEILING, pinned so nobody assumes the channel sees everything.

    `Add To Stack: SEAT cast Bloodghast` gives the seat and the card name and no
    id; the target group gives `Bloodghast (11)`. There is no key to join them
    on, so a permanent that was cast and then targeted before it ever attacked,
    blocked or was a land is UNATTRIBUTABLE.

    Measured on one 15-game pod log: 105 targets, of which 14 named a seat
    directly and 48 resolved through the owner map — 59% coverage, with 43
    permanents lost. That is a floor on interaction, not a count of it, and the
    figure is stated in `limits` for the reader who would otherwise assume
    otherwise.

    Attributing by NAME instead would raise coverage and is deliberately not
    done: Sol Ring, Swords to Plowshares and every basic land appear in several
    seats at once, so a name join would attribute interaction to whichever seat
    the parser happened to see first.
    """
    per = _facts(
        "Add To Stack: Ai(2)-them cast Bloodghast",
        "Add To Stack: Ai(1)-me cast Swords to Plowshares targeting [Bloodghast (11)]",
    )
    assert per["Ai(1)-me"]["interaction_cast"] == 0, (
        "a cast established ownership — the log carries no id to join on, so "
        "this can only have come from guessing")

    facts = [parse.game_facts(g) for g in parse.parse_games(_log(
        "Add To Stack: Ai(1)-me cast Lava Spike targeting [Ai(2)-them]"))]
    agg = parse.aggregate(facts, "me", {"Ai(1)-me": "me", "Ai(2)-them": "them"})
    limits = " ".join(agg["limits"])
    assert "unattributed" in limits.lower(), (
        "the coverage ceiling is not stated where the figure is printed")
