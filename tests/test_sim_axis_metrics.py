"""The drain axis and the threat axis, read from the log: drain dealt, life gained by
source, combat damage by keyword, the biggest hit, kills by ability.

Driven through `parse.parse_games` + `game_facts` on a synthetic log in Forge's own line
shapes, so every attribution rule is exercised by the production parser — and each test
names the bug it was written against.
"""
from manamap.sim import parse

A, B = "Ai(1)-a", "Ai(2)-b"
HEAD = [f"Mulligan: {A} has kept a hand of 7 cards", f"Mulligan: {B} has kept a hand of 7 cards",
        f"Turn: Turn 1 ({A})", f"Land: {A} played Swamp (1)", f"Turn: Turn 2 ({B})", f"Land: {B} played Plains (2)",
        f"Combat: {B} assigned Serra Angel (7) to attack {A}.", f"{A} didn't block Serra Angel (7)",
        f"Damage: Serra Angel (7) deals 4 combat damage to {A}.", f"Life: Life: {A} 40 > 36"]
TAIL = ["Game Outcome: Turn 3", f"Game Outcome: {A} has won because all opponents have lost",
        "Game Result: Game 1 ended in 100 ms."]
KW = {"Vish Kal, Blood Arbiter": ["Flying", "Lifelink"], "Vampire Token": []}


def _facts(lines, keywords=KW):
    text = "\n".join(HEAD + [f"Turn: Turn 3 ({A})"] + lines + TAIL) + "\n"
    games = parse.parse_games(text)
    assert len(games) == 1
    return parse.game_facts(games[0], keywords=keywords)["per_seat"]


def test_a_drain_is_life_loss_after_our_life_loss_ability_resolved():
    per = _facts([f"Resolve Stack: Blood Artist (9) - Target player loses 1 life. You gain 1 life.",
                  f"Life: Life: {B} 40 > 39", f"Life: Life: {A} 36 > 37"])
    assert per[A]["drain_dealt"] == 1
    assert per[A]["life_gained_by_source"] == {"lifelink": 0, "trigger": 1, "other": 0}
    assert per[B]["drain_dealt"] == 0


def test_a_life_loss_that_was_damage_is_not_a_drain():
    """The bug: crediting every opposing life drop after a drain resolve to the drain,
    including the combat damage that happened to follow it."""
    per = _facts([f"Resolve Stack: Blood Artist (9) - Target player loses 1 life.",
                  f"Damage: Vampire Token (30) deals 2 combat damage to {B}.", f"Life: Life: {B} 40 > 38"])
    assert per[A]["drain_dealt"] == 0 and per[A]["combat_damage_dealt_to_players"] == 0, \
        "the token's owner is unknown here, and unknown is credited to nobody"


def test_lifelink_is_the_gain_right_after_our_combat_damage_and_a_resolve_in_between_makes_it_a_trigger():
    per = _facts([f"Combat: {A} assigned Vish Kal, Blood Arbiter (12) to attack {B}.",
                  f"Damage: Vish Kal, Blood Arbiter (12) deals 12 combat damage to {B}.",
                  f"Life: Life: {B} 40 > 28", f"Life: Life: {A} 36 > 48"])
    assert per[A]["life_gained_by_source"] == {"lifelink": 12, "trigger": 0, "other": 0}
    assert per[A]["biggest_hit"] == {"amount": 12, "source": "Vish Kal, Blood Arbiter"}
    assert per[A]["combat_damage_by_keyword"] == {"evasive": 12, "flying": 12, "evasive_share": 1.0}
    per2 = _facts([f"Combat: {A} assigned Vish Kal, Blood Arbiter (12) to attack {B}.",
                   f"Damage: Vish Kal, Blood Arbiter (12) deals 12 combat damage to {B}.",
                   f"Life: Life: {B} 40 > 28",
                   f"Resolve Stack: Sanguine Bond (5) - Target opponent loses that much life.",
                   f"Life: Life: {A} 36 > 48"])
    assert per2[A]["life_gained_by_source"]["lifelink"] == 0 and per2[A]["life_gained_by_source"]["trigger"] == 12


def test_the_keyword_split_is_ours_only_and_ground_when_the_source_has_no_evasion():
    per = _facts([f"Combat: {A} assigned Vampire Token (30) to attack {B}.",
                  f"Damage: Vampire Token (30) deals 2 combat damage to {B}.", f"Life: Life: {B} 40 > 38"])
    assert per[A]["combat_damage_by_keyword"] == {"ground": 2, "evasive_share": 0.0}
    assert per[B]["combat_damage_by_keyword"] is None, "no keywords for the opponent: absent, not zero"
    assert per[B]["biggest_hit"] == {"amount": 4, "source": "Serra Angel"}


def test_a_kill_is_an_opposing_death_directly_after_our_activation_resolved():
    per = _facts([f"Add To Stack: {A} activated Vish Kal, Blood Arbiter targeting [Serra Angel (7)]",
                  f"Resolve Stack: Vish Kal, Blood Arbiter (12) - Target creature gets -1/-1 until end of turn.",
                  f"Zone Change: Serra Angel (7) was put into Graveyard from Battlefield."])
    assert per[A]["kills_by_ability"] == {"Vish Kal, Blood Arbiter": 1}
    assert per[B]["creatures_lost"] == 1


def test_an_intervening_event_breaks_the_kill_attribution():
    """The bug: crediting a death to the last activation of the turn however much
    happened in between — a wrath resolving after our pump would read as our kill."""
    per = _facts([f"Add To Stack: {A} activated Vish Kal, Blood Arbiter targeting [Serra Angel (7)]",
                  f"Resolve Stack: Vish Kal, Blood Arbiter (12) - Target creature gets -1/-1 until end of turn.",
                  f"Resolve Stack: Wrath of God - Destroy all creatures.",
                  f"Zone Change: Serra Angel (7) was put into Graveyard from Battlefield."])
    assert per[A]["kills_by_ability"] == {}


def test_the_aggregate_carries_every_axis_with_an_interval():
    game = "\n".join(HEAD + [f"Turn: Turn 3 ({A})",
                             f"Resolve Stack: Blood Artist (9) - Target player loses 1 life.", f"Life: Life: {B} 40 > 39"] + TAIL) + "\n"
    facts, agg = parse.analyze_logs([game + game], {A: "a", B: "b"}, keywords=KW)   # two games, so intervals exist
    a = agg["seats"]["a"]
    assert a["drain_dealt"]["mean"] == 1 and "ci95" in a["drain_dealt"]
    assert set(a["life_gained_by_source"]) == {"lifelink", "trigger", "other"}
    assert "mean" in a["biggest_hit"] and "sources" in a["biggest_hit"]
    assert "mean" in a["kills_by_ability"] and "by_card" in a["kills_by_ability"]
    assert "combat_damage_by_keyword" not in agg["seats"]["b"], "absent for the seat with no keywords"
    assert any(l.startswith("drain_dealt is an opponent's life LOSS") for l in agg["limits"])


def test_a_trigger_stack_line_is_a_drain_or_a_gain_source_too():
    """Forge logs a trigger's life change straight after its `Add To Stack … triggered`
    line, without a resolve line first: 83 opposing losses and 26 of our gains on one
    run. The bug: reading only resolve lines, which credited almost none of them."""
    per = _facts([f"Add To Stack: {A} triggered Blood Artist", f"Life: Life: {B} 40 > 39",
                  f"Add To Stack: {A} triggered Bloodthirsty Conqueror", f"Life: Life: {A} 36 > 37"])
    assert per[A]["drain_dealt"] == 1
    assert per[A]["life_gained_by_source"]["trigger"] == 1


def test_a_pain_lands_self_loss_after_our_drain_is_not_our_drain():
    """The bug: a drain credit that survived until the end of the turn, so an opponent's
    pain-land or fetch loss minutes later was ours."""
    per = _facts([f"Resolve Stack: Blood Artist (9) - Target player loses 1 life.", f"Life: Life: {B} 40 > 39",
                  f"Mana: Sulfurous Springs (55) - {{T}}: Add {{B}}.", f"Life: Life: {B} 39 > 38"])
    assert per[A]["drain_dealt"] == 1, "the first loss is the drain, the second is the land"


# ── the draw axis (2026-09-30): two readers over the telemetry hand facts ─────────────

def _hand(lib, sizes, empty=0):
    return {"hand": {"library_to_hand": lib, "end_of_turn_size": sizes, "empty_own_turns": empty}}


def test_extra_draw_is_read_per_own_turn_beyond_the_natural_draw():
    from manamap.sim.experiment import PER_GAME
    read = PER_GAME["extra_draw_per_turn"]
    # on the draw: ten own turns, ten natural draws, fourteen moves -> 0.4 a turn
    assert read(_hand(14, {str(t): 2 for t in range(2, 21, 2)})) == 0.4
    # on the play (turn 1 is ours and has no draw): nine natural draws over ten turns
    assert read(_hand(14, {str(t): 2 for t in range(1, 20, 2)})) == 0.5
    # the control game's two seats: 6 over 7 on the play, 7 over 7 on the draw — ZERO extra
    assert read(_hand(6, {t: 3 for t in (1, 3, 5, 7, 9, 11, 13)})) == 0.0
    assert read(_hand(7, {t: 3 for t in (2, 4, 6, 8, 10, 12, 14)})) == 0.0


def test_extra_draw_counts_own_turns_not_the_games_turns():
    """The bug: dividing by the game's last turn number (a four-seat game's turn 40 is
    our tenth), which would read a quarter of the real rate."""
    from manamap.sim.experiment import PER_GAME
    sizes = {str(t): 1 for t in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38)}   # seat 2 of 4, turn 38 last
    assert PER_GAME["extra_draw_per_turn"](_hand(15, sizes)) == 0.5


def test_the_draw_axis_is_absent_without_hand_facts_never_zero():
    from manamap.sim.experiment import PER_GAME
    for p in ({}, {"hand": None}, {"hand": {"library_to_hand": 3, "end_of_turn_size": {}}}):
        assert PER_GAME["extra_draw_per_turn"](p) is None
        assert PER_GAME["empty_hand_turns"](p) is None
    assert PER_GAME["empty_hand_turns"](_hand(9, {"1": 0, "3": 0}, empty=2)) == 2


def test_the_draw_axes_are_objective_axes_with_per_game_readers():
    from manamap.pilot import candidates, net_change
    from manamap.sim import experiment
    for axis in ("forge.extra_draw_per_turn", "forge.empty_hand_turns"):
        spec = candidates.FORGE_OBJECTIVE_AXES[axis]
        assert spec["kind"] == "mean" and spec["per_game"] in experiment.PER_GAME
        assert spec["per_game"] in net_change._PER_GAME and spec["per_game"] in experiment.SKEWED
    assert candidates.FORGE_OBJECTIVE_AXES["forge.empty_hand_turns"]["lower_is_better"] is True


# ── total output (2026-10-01): the axis the two failed branches needed ───────────────────

def test_life_removed_total_is_the_sum_of_the_three_ways_life_leaves_an_opponent():
    """THE BUG THIS EXISTS FOR, measured twice: drain-v1 and boss-v1 each raised
    `drain_dealt` by about 4.5 and took the deck's TOTAL output from 59.5 to 47.3, and the
    objective named only the component that rose. One row makes that trade visible."""
    from manamap.sim.experiment import PER_GAME
    read = PER_GAME["life_removed_total"]
    assert read({"combat_damage_dealt_to_players": 31, "noncombat_damage_dealt_to_players": 5, "drain_dealt": 11}) == 47
    # an absent component is zero in the SUM but an absent combat figure is absent overall:
    # a record with no combat reading has not measured output at all.
    assert read({"combat_damage_dealt_to_players": 31}) == 31
    assert read({"noncombat_damage_dealt_to_players": 5, "drain_dealt": 11}) is None
    assert read({}) is None
    # the trade the two branches made, as the row reads it
    champ = read({"combat_damage_dealt_to_players": 48.6, "noncombat_damage_dealt_to_players": 4.3, "drain_dealt": 6.6})
    branch = read({"combat_damage_dealt_to_players": 31.0, "noncombat_damage_dealt_to_players": 5.1, "drain_dealt": 11.0})
    assert champ > branch, "the branch raised drain and lowered total output; the row must say so"


def test_the_two_output_axes_are_objective_axes_with_per_game_readers():
    from manamap.pilot import candidates, net_change
    from manamap.sim import experiment
    for axis in ("forge.life_removed_total", "forge.noncombat_damage_dealt_to_players"):
        spec = candidates.FORGE_OBJECTIVE_AXES[axis]
        assert spec["kind"] == "mean" and spec["lower_is_better"] is False
        assert spec["per_game"] in experiment.PER_GAME and spec["per_game"] in net_change._PER_GAME
        assert spec["per_game"] in experiment.SKEWED, "both are long-tailed; the median rides along"
