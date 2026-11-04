"""Rush of Knowledge, Well of Lost Dreams, and the clock on 21 commander damage.

Both cards read as nothing until 2026-11-04, on the branch that added them because the
deck had no pure draw and nothing to turn spare mana into cards: Rush because "draw cards
equal to" missed `_DRAW_RE` (a silent zero, not even named), Well because it had no
channel. The rules read of the chain was checked (CR 603.5, 118.12, 119.9, 605.3a).
"""
import random

import pytest

from manamap.config import LIFE_CHAIN_LIBRARY_FLOOR
from manamap.pilot import goldfish, model_coverage, net_change

RUSH = "Draw cards equal to the greatest mana value among permanents you control."
WELL = ("Whenever you gain life, you may pay {X}, where X is less than or equal to the amount "
        "of life you gained. If you do, draw X cards.")
LIFE_PER_DRAW = "Whenever you draw a card, put a +1/+1 counter on this creature and you gain 1 life."


def _c(name, text, cmc, type_line, quantity=1, **extra):
    return dict({"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
                 "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
                 "color_identity": [], "power": None, "toughness": None,
                 "is_commander": False, "layout": "normal"}, **extra)


def _doc(*cards, lands=40):
    return {"cards": [_c("Forest", "({T}: Add {G}.)", 0, "Basic Land — Forest", quantity=lands),
                      *cards,
                      _c("Filler", "", 1, "Sorcery", quantity=99 - lands - sum(c["quantity"] for c in cards))]}


def _run(doc, iterations=200, **kw):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(7)
    return [goldfish.simulate_once(rng, library, 3, [], 10, model_draw=True, model_discard=True,
                                   model_combat=True, **kw)
            for _ in range(iterations)]


def _mean_extra(results, turn=10):
    return sum(r["drawn_extra_by_turn"][turn - 1] for r in results) / len(results)


def _profile(text, type_line):
    return goldfish.draw_profile({"name": "X", "oracle_text": text, "type_line": type_line,
                                  "mana_cost": "", "cmc": 1})


def test_the_two_cards_are_read():
    assert _profile(RUSH, "Sorcery")["spell_draw_greatest_mv"] is True
    assert _profile(RUSH, "Sorcery")["unmodelled"] is None
    assert _profile(WELL, "Artifact")["life_gain_paid_draw"] is True
    assert _profile(WELL, "Artifact")["unmodelled"] is None


def test_a_draw_no_channel_reads_is_named_not_a_silent_zero():
    p = _profile("Draw cards equal to the number of Goblins you control.", "Sorcery")
    assert p["unmodelled"] == "X"


def test_a_card_that_does_not_draw_is_not_named():
    assert _profile("Destroy target creature.", "Instant")["unmodelled"] is None


def test_the_coverage_mirrors_know_both_channels():
    for text, tl in ((RUSH, "Sorcery"), (WELL, "Artifact")):
        p = goldfish.classify(_c("X", text, 5, tl))
        assert "draw" in model_coverage.channels_for(p)
        assert not model_coverage.never_cast(p, {"model_draw": True, "model_discard": True})


def test_rush_draws_off_the_board_and_is_cast():
    body = _c("Body", "", 5, "Creature — Beast", quantity=12, power="1", toughness="1")
    with_rush = _mean_extra(_run(_doc(body, _c("Rush", RUSH, 5, "Sorcery", quantity=12))))
    without = _mean_extra(_run(_doc(body, _c("Blank", "", 5, "Sorcery", quantity=12))))
    assert with_rush > without + 3


def _well_doc(well=True, source=True):
    cards = [_c("Source", LIFE_PER_DRAW if source else "", 3, "Creature — Shark", quantity=12,
                power="1", toughness="1")]
    if well:
        cards.append(_c("Well", WELL, 4, "Artifact", quantity=12))
    return _doc(*cards)


def test_well_turns_spare_mana_into_cards_with_a_life_per_draw_source():
    assert _mean_extra(_run(_well_doc())) > _mean_extra(_run(_well_doc(well=False))) + 3


def test_well_does_nothing_without_a_source_of_life():
    assert _mean_extra(_run(_well_doc(source=False))) == pytest.approx(
        _mean_extra(_run(_well_doc(well=False, source=False))), abs=0.01)


def test_the_chain_stops_short_of_decking_him():
    rock = _c("Rock", "{T}: Add {C}{C}{C}{C}{C}{C}{C}{C}{C}{C}.", 1, "Artifact", quantity=8)
    source = _c("Source", LIFE_PER_DRAW, 3, "Creature — Shark", quantity=12, power="1", toughness="1")
    well = _c("Well", WELL, 4, "Artifact", quantity=12)
    results = _run(_doc(rock, source, well), iterations=100)
    assert max(r["drawn_extra_by_turn"][-1] for r in results) > 40, "the chain never ran down"
    # The library is 99 less the opener and the ten draw steps; the chain stops at the
    # floor whatever the mana, so it never takes the last cards.
    assert all(r["drawn_extra_by_turn"][-1] <= 99 - 7 - 1 - LIFE_CHAIN_LIBRARY_FLOOR + 1 for r in results)


def test_the_chain_waits_a_turn_after_the_pieces_land():
    import inspect
    from manamap.pilot import goldfish_turn
    body = inspect.getsource(goldfish_turn.simulate_once)
    assert body.index("if model_draw and life_chain_armed:") < body.index(
        "life_chain_armed = bool(life_paid_draws"), "armed before it is read would fire a turn early"


NO_MAX = "You have no maximum hand size."


@pytest.mark.parametrize("text,type_line,want", [
    (NO_MAX, "Land", True),
    (NO_MAX, "Artifact", True),
    ("Players have no maximum hand size. {T}: Add {C}.", "Artifact", True),
    ("You have no maximum hand size for the rest of the game.", "Sorcery", True),
    (NO_MAX, "Sorcery", False),
    ("{T}: You have no maximum hand size until end of turn.", "Artifact", False),
    ("You have no maximum hand size until end of turn.", "Enchantment", False),
    ("Your maximum hand size is ten.", "Artifact", False),
    ("As long as you control a Wizard, you have no maximum hand size.", "Artifact", False),
    ("Draw two cards.", "Sorcery", False),
])
def test_only_a_standing_no_maximum_switches_the_limit_off(text, type_line, want):
    from manamap.pilot.goldfish_profiles import no_max_hand
    assert no_max_hand({"oracle_text": text, "type_line": type_line}) is want


def _big_draw(*extra):
    # Enough draw to run a hand well over seven every game.
    return _doc(_c("Draw", "Draw three cards.", 1, "Sorcery", quantity=30), *extra)


def test_the_hand_limit_holds_a_hand_at_seven():
    free = _run(_big_draw(), iterations=60)
    held = _run(_big_draw(), iterations=60, model_hand_limit=True)
    assert max(max(r["hand_size_by_turn"]) for r in free) > 7, "the test never ran a big hand"
    assert all(max(r["hand_size_by_turn"]) <= 7 for r in held)


def test_cleanup_is_a_discard_and_it_counts():
    free = _run(_big_draw(), iterations=60)
    held = _run(_big_draw(), iterations=60, model_hand_limit=True)
    assert sum(r["discarded_by_turn"][-1] for r in held) > sum(r["discarded_by_turn"][-1] for r in free)


@pytest.mark.parametrize("type_line,cost", [("Land", 0), ("Artifact", 1)])
def test_a_no_maximum_card_in_play_lifts_the_limit(type_line, cost):
    tower = _c("Tower", NO_MAX, cost, type_line, quantity=25)
    held = _run(_big_draw(tower), iterations=60, model_hand_limit=True)
    assert max(max(r["hand_size_by_turn"]) for r in held) > 7


RIELLE = ("Rielle gets +1/+0 for each instant and sorcery card in your graveyard.\n"
          "Whenever you discard one or more cards for the first time each turn, draw that many cards.")


def test_rielle_is_read_not_named():
    p = goldfish.classify(_c("Rielle", RIELLE, 3, "Legendary Creature — Human Wizard", power="0", toughness="3"))
    assert p["event"]["first_discard_draw"] is True
    assert p["draw"]["unmodelled"] is None


def test_rielle_draws_the_first_batch_each_turn():
    wheel = _c("Wheel", "Each player discards their hand, then draws seven cards.", 3, "Sorcery", quantity=12)
    body = dict(power="0", toughness="3")
    withr = _run(_doc(wheel, _c("Rielle", RIELLE, 3, "Legendary Creature — Human Wizard", quantity=12, **body)), iterations=150)
    blank = _run(_doc(wheel, _c("Plain", "", 3, "Creature — Human", quantity=12, **body)), iterations=150)
    assert _mean_extra(withr) > _mean_extra(blank) + 5


def test_the_first_batch_counts_a_loot_not_only_a_wheel():
    loot = _c("Loot", "Draw two cards, then discard two cards.", 1, "Sorcery", quantity=20)
    body = dict(power="0", toughness="3")
    withr = _run(_doc(loot, _c("Rielle", RIELLE, 2, "Legendary Creature — Human Wizard", quantity=12, **body)), iterations=150)
    blank = _run(_doc(loot, _c("Plain", "", 2, "Creature — Human", quantity=12, **body)), iterations=150)
    assert _mean_extra(withr) > _mean_extra(blank) + 2


def test_the_results_carry_the_clock_and_the_hand():
    r = _run(_well_doc(), iterations=5)[0]
    assert "commander_lethal_turn" in r and len(r["hand_size_by_turn"]) == 10


def test_a_commander_reaches_21_in_its_own_damage():
    cmd = goldfish.classify(_c("Big", "", 3, "Legendary Creature — Beast", power="7", toughness="7"))
    combat = cmd["combat"]
    combat = dict(combat, power=7, is_creature=True, haste=False, double_strike=False)
    results = _run(_doc(_c("Blank", "", 9, "Sorcery", quantity=5)), iterations=50,
                   commander_combat=combat)
    # Cast by turn 3 or so and swinging from the next: three hits of seven reach 21.
    arrived = [r for r in results if r["commander_turn"] is not None and r["commander_turn"] <= 6]
    assert arrived and all(r["commander_lethal_turn"] == r["commander_turn"] + 3 for r in arrived)


def test_blind_spots_flags_a_draw_with_no_role(monkeypatch):
    monkeypatch.setattr("manamap.pilot.common.load_card_roles", lambda: {"Odd": ["threat:body"]})
    monkeypatch.setattr(net_change, "_draw_is_blind", lambda slug, branch, name: name == "Odd")
    got = net_change.blind_spots("x", "b", {"spells": [{"out": "Plain", "in": "Odd"}], "lands": []})
    assert [b["cards"] for b in got if b["class"] == "draw"] == [["Odd"]]
