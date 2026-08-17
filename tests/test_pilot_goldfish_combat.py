"""The combat model: a goldfish that actually goldfishes.

For most of this simulator's life it modelled resource development and called
itself a goldfish. It had no combat and no opponents, so an entire class of card
was priced at exactly zero — attack triggers, additional combat phases, and the
combat-gated Treasure sources `treasure_profile` returns `unmodelled` for. On
ur-dragon that was nine of fourteen Treasure sources and BOTH halves of the
deck's only checker-verified win line, which is how a rebuild could measure as an
improvement on every axis while moving the deck's actual kill rate by nothing.

These tests pin the three properties that keep the model honest:

1. **It is genuinely opt-in.** Turning combat on changes `mean_bodies_by_turn`
   for every deck that makes non-creature tokens — all nine of them — and those
   figures are quoted in published prose on five decks and in one `engine.json`
   carrying a critic verdict. A deck that has not opted in must be byte-identical.
2. **A Treasure is not a body.** `body_count` counts "create a Treasure token" as
   a creature; measured on ur-dragon that was 37% of the reported turn-six board.
   The corrected count rides with the flag, never ahead of it.
3. **What it cannot read, it NAMES.** Same contract as the Treasure model. A
   trigger whose effect the parser cannot price scores zero, and a zero nobody is
   told about reads as a fact about the deck rather than a gap in the tool.
"""

import json
import random

import pytest

from manamap.pilot import goldfish

from conftest import requires_deck


def _card(name, text, cmc=3, type_line="Creature — Dragon", power="4",
          toughness="4"):
    return {"name": name, "type_line": type_line, "cmc": cmc,
            "oracle_text": text, "quantity": 1, "is_commander": False,
            "power": power, "toughness": toughness}


# ── 1. A Treasure is not a body ───────────────────────────────────────────

def test_body_count_and_creature_body_count_disagree_about_a_treasure():
    """The bug and its fix, side by side, on the exact shape that caused it."""
    smaug = _card("Smaug", "At the beginning of your upkeep, create a Treasure token.")
    # The creature itself is one body under both counts. The Treasure is a body
    # only under the old one.
    assert goldfish.body_count(smaug) == 2
    assert goldfish.creature_body_count(smaug) == 1


def test_a_real_creature_token_still_counts_as_a_body():
    """The fix must not throw out creature tokens with the Treasures."""
    lathliss = _card("Lathliss", "create a 5/5 red Dragon creature token")
    assert goldfish.creature_body_count(lathliss) == 2


@pytest.mark.parametrize("token", ["Treasure", "Clue", "Food", "Blood", "Powerstone"])
def test_no_artifact_token_is_ever_a_blocker(token):
    card = _card("Maker", f"create a {token} token", type_line="Enchantment",
                 power=None, toughness=None)
    assert goldfish.creature_body_count(card) == 0


# ── 2. Attack triggers fire, and only under the flag ──────────────────────

@pytest.mark.parametrize("text,field,expected", [
    ("Whenever this creature attacks, add {R}{R}{R}{G}{G}{G}.", "attack_mana", 6),
    ("Whenever this creature deals combat damage to a player, create three "
     "Treasure tokens.", "attack_treasure", 3),
    ("Whenever this creature attacks, it deals 4 damage to any target.",
     "attack_damage", 4),
    ("Whenever a Dragon you control attacks, create a 6/6 red Dragon creature "
     "token.", "attack_token_bodies", 1),
])
def test_the_parser_reads_the_attack_trigger(text, field, expected):
    assert goldfish.combat_profile(_card("X", text))[field] == expected


def test_an_activated_extra_combat_carries_its_price():
    """Aggravated Assault is a repeat button you buy every turn, not a free one."""
    assault = goldfish.combat_profile(_card(
        "Aggravated Assault",
        "{3}{R}{R}: Untap all creatures you control. After this main phase, "
        "there is an additional combat phase followed by an additional main phase. "
        "Activate only as a sorcery.",
        type_line="Enchantment", power=None, toughness=None))
    assert assault["extra_combat_cost"] == 5
    assert assault["extra_combat_free"] is False


def test_a_triggered_extra_combat_is_free_and_is_not_called_unreadable():
    """Scourge of the Throne's reminder text pushes the clause past the trigger
    window. Flagging a card whose effect IS modelled makes the not-modelled list
    a liar, which is the one thing that list exists not to be."""
    scourge = goldfish.combat_profile(_card(
        "Scourge of the Throne",
        "Flying Dethrone (Whenever this creature attacks the player with the most "
        "life or tied for most life, put a +1/+1 counter on it.) Whenever this "
        "creature attacks for the first time each turn, if it's attacking the "
        "player with the most life or tied for most life, untap all attacking "
        "creatures. After this phase, there is an additional combat phase."))
    assert scourge["extra_combat_free"] is True
    assert scourge["unreadable"] is None


def test_an_unpriceable_trigger_is_named_rather_than_scored_zero():
    """Ancient Gold Dragon rolls a d20. That is not modellable and the model must
    say so instead of quietly reporting no value."""
    profile = goldfish.combat_profile(_card(
        "Ancient Gold Dragon",
        "Whenever this creature deals combat damage to a player, roll a d20. You "
        "create a number of 1/1 blue Faerie Dragon creature tokens with flying "
        "equal to the result."))
    assert profile["unreadable"] == "Ancient Gold Dragon"


# ── 3. The flag is genuinely opt-in ───────────────────────────────────────

def _tiny_deck():
    return {"cards": [
        {"name": "Cmd", "type_line": "Legendary Creature — Dragon", "cmc": 3,
         "oracle_text": "", "quantity": 1, "is_commander": True,
         "power": "3", "toughness": "3"},
        {"name": "Mountain", "type_line": "Basic Land — Mountain", "cmc": 0,
         "oracle_text": "", "quantity": 30, "power": None, "toughness": None},
        dict(_card("Hoarder", "At the beginning of your upkeep, create a Treasure "
                   "token.", cmc=2), quantity=10),
        dict(_card("Beater", "", cmc=2), quantity=10),
    ]}


def _run(model_combat, model_treasures=False, iterations=60):
    library, commanders = goldfish.build_library(_tiny_deck())
    rng = random.Random(7)
    results = [goldfish.simulate_once(rng, library, 3, [], 10,
                                      model_treasures=model_treasures,
                                      model_combat=model_combat)
               for _ in range(iterations)]
    return goldfish.aggregate(results, [], 10, model_treasures, model_combat)


def test_the_combat_block_is_absent_rather_than_zeroed_when_off():
    """Absent keys are what keep a non-opted deck's artifact byte-identical."""
    assert "combat" not in _run(model_combat=False)
    assert "combat" in _run(model_combat=True)


def test_the_flag_off_leaves_bodies_counting_treasures_as_before():
    """The body fix must ride WITH the flag. A deck that has not opted in keeps
    the number it published, wrong though that number is."""
    off = _run(model_combat=False)["mean_bodies_by_turn"]["10"]
    on = _run(model_combat=True)["mean_bodies_by_turn"]["10"]
    # Hoarder is a creature that makes a Treasure: 2 bodies off, 1 on.
    assert off > on


def test_the_clock_exists_and_is_a_turn_not_a_win_rate():
    combat = _run(model_combat=True)["combat"]
    assert set(combat) >= {"mean_kill_turn", "median_kill_turn",
                           "kill_by_turn_rate", "no_kill_by_max_turn_rate",
                           "mean_board_power_by_turn", "mean_damage_by_turn"}
    rates = combat["kill_by_turn_rate"]
    # Monotone: you cannot un-kill an opponent on a later turn.
    values = [rates[str(t)] for t in range(1, 11)]
    assert values == sorted(values)
    assert 0.0 <= combat["no_kill_by_max_turn_rate"] <= 1.0


def test_board_power_is_not_body_count():
    """The instrument the columnists deadlocked over: a 2/2 and a 6/6 are one
    body each, and that is exactly why body count could not settle the argument."""
    metrics = _run(model_combat=True)
    assert metrics["combat"]["mean_board_power_by_turn"]["10"] > \
        metrics["mean_bodies_by_turn"]["10"]


@requires_deck
def test_every_tracked_deck_is_byte_identical_with_the_flag_absent():
    """The gate. Nine decks, none opted in, none may move.

    This is the check that lets the model ship at all: `model_treasures` set the
    precedent, and the reason is the same. A figure quoted in a critic-verdicted
    artifact cannot be changed by a tool upgrade nobody asked for.
    """
    from manamap.config import DATA_DIR
    decks = sorted(p.parent.name for p in (DATA_DIR / "decks").glob("*/goldfish_metrics.json"))
    assert decks, "no tracked goldfish metrics found"
    for slug in decks:
        targets = DATA_DIR / "decks" / slug / "goldfish_targets.json"
        if targets.exists():
            assert "model_combat" not in json.loads(targets.read_text()), (
                f"{slug} has opted in; update this test's premise deliberately")
        on_disk = json.loads((DATA_DIR / "decks" / slug / "goldfish_metrics.json").read_text())
        fresh = goldfish.run(slug)
        assert json.dumps(fresh, sort_keys=True) == json.dumps(on_disk, sort_keys=True), (
            f"{slug} moved without opting in")
