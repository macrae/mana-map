"""A game the clock stopped has no winner, and used to have the wrong one.

THE BUG. Forge's `-c` clock does not end a game — it abandons one. Decompiled,
Forge catches its own timeout, prints "Stopping slow match as draw" and calls
`setGameOver(GameEndReason.Draw)`. It then prints

    Game Outcome: Ai(1)-mm-ours  has won because all opponents have lost
    Game Outcome: Ai(2)-mm-opp-a has won because all opponents have lost
    Game Outcome: Ai(3)-mm-opp-b has won because all opponents have lost

for EVERY SEAT STILL ALIVE. Both parsers assigned `winner` on each match, so the
LAST line won — the highest-numbered survivor.

WHAT IT COST. Our deck is always `Ai(1)`, so it is structurally last in that
contest and won it never: across 121 truncated games it was credited with ZERO
while surviving to the clock in 93 of them. `baylen-tokens`, always the final
seat, took 73 of its 85 recorded wins that way — its true rate is 0.015, not the
0.094 every "the pod is uneven" argument was built on. It is also the entire
"win rate falls as N grows" signature: the clock-hit share runs 0% at n=20, 9%
at n=100, 18% at n=400.

WHY NOTHING CAUGHT IT. Three reasons, each worth its own test below.
"""

import json

import pytest

from manamap.sim.forge import parse_outcomes, tally_wins

WON = "Game Outcome: {seat} has won because all opponents have lost"
LOST = "Game Outcome: {seat} has lost because life total reached 0"


def _log(*games):
    out = []
    for i, (turn, lines, ms) in enumerate(games, 1):
        out.append("Turn: Turn %d (Ai(1)-mm-ours)" % turn)
        out.append("Game Outcome: Turn %d" % turn)
        out += lines
        out.append(f"Game Result: Game {i} ended in {ms} ms. Ai(1)-mm-ours has won!")
    return "\n".join(out)


def test_a_clock_stopped_game_has_no_winner():
    """THE BUG, re-introduced by hand. Three seats declared winners is Forge
    saying nobody won; taking the last one credits whoever sat last."""
    text = _log((25, [WON.format(seat="Ai(1)-mm-ours"),
                      WON.format(seat="Ai(2)-mm-opp-a"),
                      WON.format(seat="Ai(3)-mm-opp-b"),
                      LOST.format(seat="Ai(4)-mm-opp-c")], 300009))
    (g,) = parse_outcomes(text)
    assert g["truncated"] is True
    assert g["winner"] is None, (
        "a game with three declared winners was awarded to one of them")


def test_a_decided_game_still_names_its_winner():
    """The fix must not cost the ordinary case, which is 87% of games."""
    text = _log((14, [WON.format(seat="Ai(2)-mm-opp-a"),
                      LOST.format(seat="Ai(1)-mm-ours"),
                      LOST.format(seat="Ai(3)-mm-opp-b"),
                      LOST.format(seat="Ai(4)-mm-opp-c")], 91234))
    (g,) = parse_outcomes(text)
    assert g["truncated"] is False
    assert g["winner"] == "Ai(2)-mm-opp-a"


def test_a_truncated_game_is_not_counted_for_anyone():
    """OUR SEAT IS ALWAYS Ai(1) AND THEREFORE ALWAYS LAST IN THE OLD CONTEST.
    The asymmetry is the whole harm: the bug is not random noise, it points one
    way, and it points away from the deck being measured."""
    text = _log(
        (25, [WON.format(seat="Ai(1)-mm-ours"),
              WON.format(seat="Ai(4)-mm-opp-c")], 300011),
        (12, [WON.format(seat="Ai(1)-mm-ours"),
              LOST.format(seat="Ai(4)-mm-opp-c")], 60000))
    games = parse_outcomes(text)
    assert [g["truncated"] for g in games] == [True, False]
    seats = ["ours", "opp-a", "opp-b", "opp-c"]
    decided = [g for g in games if not g["truncated"]]
    # `tally_wins` matches the Forge meta name; relabel the way `run()` does.
    labelled = [{"winner": g["winner"].split("-mm-")[-1]} for g in decided]
    wins = tally_wins(labelled, seats)
    assert wins["ours"] == 1 and wins["opp-c"] == 0, wins


def test_a_real_draw_line_closes_its_game():
    """A LATENT LANDMINE THE FIRST FIX WOULD HAVE ARMED.

    Forge has two `Game Result` formats — "ended in N ms." for a decided game
    and "ended in a Draw! Took N ms." for `isDraw()`. The pattern matched only
    the first, and `parse_outcomes` closes a game ONLY on that line — so a real
    draw left the record open, the next game's outcome lines fell into it, and
    TWO GAMES MERGED INTO ONE credited to the second one's winner.

    It had never fired because `isDraw()` was false in all 901 tracked games —
    precisely because the clock-outs Forge calls draws were being handed to a
    survivor instead. Fixing that is what would have armed this.
    """
    text = ("Game Outcome: Turn 30\n"
            + WON.format(seat="Ai(2)-mm-opp-a") + "\n"
            "Game Result: Game 1 ended in a Draw! Took 300004 ms.\n"
            "Game Outcome: Turn 11\n"
            + WON.format(seat="Ai(3)-mm-opp-b") + "\n"
            "Game Result: Game 2 ended in 55000 ms. Ai(3)-mm-opp-b has won!\n")
    games = parse_outcomes(text)
    assert len(games) == 2, (
        "the draw line did not close its game — two games merged into one")
    assert games[0]["ms"] == 300004 and games[1]["winner"] == "Ai(3)-mm-opp-b"


def test_every_tracked_run_accounts_for_all_its_games():
    """WON + DRAWN + UNFINISHED == PLAYED.

    The old invariant was `wins + draws == n`, and it held perfectly THROUGH the
    bug: the parser reassigned wins rather than losing them, so the books
    balanced while the attribution was wrong. An accounting check cannot see a
    misattribution that conserves the total.
    """
    from manamap.config import DECKS_DIR

    checked = 0
    for path in sorted(DECKS_DIR.glob("*/sim/*.json")):
        rec = json.loads(path.read_text())
        s = rec.get("summary") or {}
        if "truncated" not in s:
            continue
        checked += 1
        n = rec["games_completed"]
        assert sum(s["wins"].values()) + s.get("draws", 0) + s["truncated"] == n, path.name
        assert s["decided"] == n - s["truncated"], path.name
    assert checked >= 10, "the tracked runs have not been re-derived"


def test_no_tracked_run_still_credits_a_truncated_game():
    """Held at rest over the committed records, so a future `--analyze` that
    reintroduces the bug is caught by the artifacts rather than by a reader."""
    from manamap.config import DECKS_DIR

    checked = 0
    for path in sorted(DECKS_DIR.glob("*/sim/*.json")):
        rec = json.loads(path.read_text())
        for g in rec.get("games") or []:
            if g.get("truncated"):
                checked += 1
                assert not g.get("winner"), (
                    f"{path.name}: a truncated game names {g['winner']} as winner")
    if not checked:
        pytest.skip("no truncated games among the tracked runs")
