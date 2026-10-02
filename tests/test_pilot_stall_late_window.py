"""The late-window stall split: the number that settled a theory the aggregate hid.

edgar-vampires' captain's log says, six times across six games, that the deck runs out
of cards at turns eight to ten. `stall.cause` counts from `STALL_FROM_TURN = 2` and reads
78.2% mana-short, which contradicted the pilot — and was almost entirely turns 1 and 2,
where a Commander deck stalls 0.559 and 0.247 of the time for the structural reason the
module's own comment describes. Inside turns 7-10 the same split reads **67.6% hand-empty**,
and 1,254 of the 1,329 hand-empty stalls in the whole game live there.

Four nights of branches were aimed at the output side and then at mana on the strength of
the aggregate. The window was the whole disagreement.

Every test drives `diagnostic.stall` / `diagnostic._late`, and each names the bug it was
written for so the bug can be put back to prove the test fails.
"""
import pytest

from conftest import requires_deck

from manamap.pilot import diagnostic


def _rows(stalls_by_turn, hands_by_turn, n=1):
    """n identical goldfish rows with the stall/hand panels the block reads."""
    return [{"stall_by_turn": list(stalls_by_turn),
             "hand_size_by_turn": list(hands_by_turn)} for _ in range(n)]


def test_the_window_is_what_separates_the_two_causes():
    """BUG: counting from turn 2 reported mana-short and hid an empty-hand late game.

    A deck that stalls early holding cards (no lands yet) and late holding nothing is
    the exact shape of the real deck. Read from turn 2 it is mana-short; read from
    turn 7 it is hand-empty. Both readings are correct about their own window, and
    only one of them answers the pilot.

    Re-introduce by passing `STALL_FROM_TURN` into `_late` instead of
    `LATE_FROM_TURN`, or by deleting the `late` key from `stall()`'s return.
    """
    # turns 1-10; stalls on T1, T2 (hand full) and T8, T9 (hand empty)
    stalls = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    hands = [5, 4, 3, 3, 2, 2, 2, 0, 0, 1]
    got = diagnostic.stall(_rows(stalls, hands, n=50))

    # TURN 1 IS EXCLUDED FROM THE HEADLINE and the module says why: a Commander deck
    # has one mana on turn one, so including it would make the metric a restatement
    # of that. So the aggregate sees T2, T8 and T9 — three turns, not four.
    assert got["cause"]["stall_turns"] == 3 * 50
    assert got["cause"]["hand_empty"] == 2 * 50, "T8 and T9"
    assert got["cause"]["mana_short"] == 1 * 50, "T2 alone, holding four cards"

    late = got["late"]
    assert late["from_turn"] == diagnostic.LATE_FROM_TURN == 7
    assert late["stall_turns"] == 2 * 50, "only T8 and T9 are in the window"
    assert late["hand_empty"] == 2 * 50
    assert late["mana_short"] == 0, (
        "the early mana-short stalls must NOT leak into the late window — that leak "
        "is the whole defect this block exists to fix")
    # and the two readings genuinely disagree, which is the point
    assert got["cause"]["mana_short"] > 0 and late["mana_short"] == 0


def test_the_rise_is_measured_against_a_found_floor_not_a_guessed_turn():
    """The complaint is a SHAPE — fine, then worse — so the block reports a difference.

    The floor turn is FOUND (the pre-window turn with the fewest stalls) and reported,
    rather than hard-coded to six. T6 happens to be the floor on every deck measured
    so far; asserting it would be a fleet claim this module has no standing to make.

    Re-introduce by hard-coding `floor_i = 5`, and this test's T4 floor fails.
    """
    #                T1 T2 T3 T4 T5 T6 T7 T8 T9 T10   — floor is T4, not T6
    stalls = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    hands = [4, 4, 4, 4, 4, 4, 0, 0, 0, 0]
    late = diagnostic.stall(_rows(stalls, hands, n=100))["late"]
    rise = late["vs_floor"]
    assert rise["from_turn"] == 4, f"the floor must be found, not assumed: {rise}"
    assert rise["floor_rate"] == 0.0
    assert rise["diff"] > 0, "the late window stalls more than its floor"
    assert rise["excludes_zero"] is True
    assert rise["ci95"][0] > 0


def test_hand_size_is_reported_per_turn_with_its_interval():
    """The pilot's literal words are "not enough cards in hand", so the series ships.

    It has been collected per game since the stall block was written and aggregated
    nowhere; its only consumer was the empty-hand count.

    Re-introduce by reading `hand_size_by_turn[i + 1]` — the off-by-one is the
    obvious bug here, because the panel is 0-indexed and the report is 1-indexed.
    """
    hands = [7, 6, 5, 4, 3, 3, 2, 1, 1, 0]
    late = diagnostic.stall(_rows([0] * 10, hands, n=40))["late"]
    hs = late["hand_size"]
    assert sorted(hs, key=int) == ["7", "8", "9", "10"], "the window, and only it"
    assert hs["7"]["rate"] == 2.0, f"turn 7 is the SEVENTH entry, index 6: {hs['7']}"
    assert hs["8"]["rate"] == 1.0
    assert hs["10"]["rate"] == 0.0
    for turn, cell in hs.items():
        assert "ci95" in cell and "sd" in cell and cell["n"] == 40, (turn, cell)


def test_absent_rather_than_zero_when_the_model_never_reaches_the_window():
    """A 0.0 would read as "it never stalls late". A short model has nothing to say."""
    assert diagnostic._late(_rows([0] * 5, [3] * 5, n=10), 5) is None
    assert diagnostic._late([], 10) is None
    # and a model that reaches it exactly produces a one-turn window rather than None
    got = diagnostic._late(_rows([1] * 7, [0] * 7, n=10), 7)
    assert got is not None and got["stall_turns"] == 10


def test_the_pooled_rate_says_its_own_denominator():
    """`stall_rate`'s denominator is turn-opportunities, `by_turn`'s is games.

    Two rates that look alike and divide by different things is how a reader
    compares the wrong pair, so the block states it in `basis`.
    """
    late = diagnostic._late(_rows([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [1] * 10, n=25), 10)
    assert late["stall_rate"]["n"] == 25 * 4, "25 games x turns 7..10"
    assert late["stall_rate"]["rate"] == pytest.approx(2 / 4, abs=1e-4)
    assert "turn-OPPORTUNITIES" in late["basis"]
    assert "FLOOR" in late["basis"], "the hand is read after the model spends its mana"


@requires_deck
def test_the_real_deck_contradicts_the_aggregate_and_agrees_with_the_pilot():
    """The finding itself, asserted against the tracked artifact.

    If a model change ever flips this, the direction of four branches' worth of work
    flips with it and somebody must notice.
    """
    import json
    from conftest import ROOT

    path = ROOT / "data" / "decks" / "edgar-vampires" / "diagnostic.json"
    if not path.exists():
        pytest.skip("edgar-vampires has no diagnostic.json")
    stall = json.loads(path.read_text(encoding="utf-8"))["stall"]
    late = stall.get("late")
    if late is None:
        pytest.skip("diagnostic.json predates the late window — regenerate")

    whole = stall["cause"]
    assert whole["mana_short"] > whole["hand_empty"], (
        "the all-game aggregate reads mana-short; that is the reading that misled four "
        "branches and the test exists to keep both halves visible")
    assert late["hand_empty"] > late["mana_short"], (
        "inside turns 7-10 the deck stalls on an EMPTY HAND, which is what the pilot's "
        f"log has said six times: {late}")
    # and the early window is where the mana-short mass lives
    assert whole["mana_short"] - late["mana_short"] > late["mana_short"] * 3, (
        "most mana-short stalls must be outside the late window, or the aggregate was "
        "not actually swamped by the opening and this whole block is unnecessary")
    assert late["vs_floor"]["excludes_zero"] is True, (
        "the rise after the floor is the pilot's complaint and it must be real")
