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

from conftest import requires_data, requires_deck


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
    """The instrument the legacy magazine's panel deadlocked over: a 2/2 and a 6/6 are one
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
    # A RETIRED DECK IS OUT OF SCOPE AND OUT OF THE DENOMINATOR. Its metrics are
    # history: nothing plays the list, so a model correction leaves them behind
    # for good and regenerating them measures a deck nobody will shuffle. The
    # pilot's rule, 2026-08-27. Counting them in `decks` would then make the
    # coverage guard below unsatisfiable.
    def _retired(slug):
        info = DATA_DIR / "decks" / slug / "info.json"
        if not info.exists():
            return False
        try:
            return bool((json.loads(info.read_text()) or {}).get("lifecycle"))
        except Exception:                        # pragma: no cover - defensive
            return False

    decks = [d for d in decks if not _retired(d)]
    assert decks, "every tracked deck is retired"
    checked = 0
    for slug in decks:
        targets = DATA_DIR / "decks" / slug / "goldfish_targets.json"
        if targets.exists() and json.loads(targets.read_text()).get("model_combat"):
            # ur-dragon opted in when its two-engine rebuild was applied, which is
            # exactly the "re-baselined deliberately" condition the flag exists
            # for. An opted-in deck is not evidence against the invariant — the
            # invariant is about decks that did NOT ask.
            continue
        on_disk = json.loads((DATA_DIR / "decks" / slug / "goldfish_metrics.json").read_text())
        fresh = goldfish.run(slug)
        assert json.dumps(fresh, sort_keys=True) == json.dumps(on_disk, sort_keys=True), (
            f"{slug} moved without opting in")
        checked += 1
    assert checked >= len(decks) - 1, (
        f"only {checked} of {len(decks)} decks are un-opted — if the fleet is "
        "being re-baselined, retire this invariant deliberately rather than "
        "letting it quietly stop checking anything")


def _profile(name, text, type_line="Creature"):
    return goldfish.combat_profile({"name": name, "oracle_text": text,
                                    "type_line": type_line, "cmc": 4})


def test_whenever_you_attack_is_an_attack_trigger():
    """`_ATTACKS_RE` matched "whenever <NAME> attacks" and not "whenever YOU
    attack" — a phrasing 180 corpus cards use, including Karlach. It gates the
    whole combat profile, so a card written that way had its attack mana,
    treasures, draw and damage all read as nothing.

    Fleet impact when it was fixed: ZERO — no tracked artifact moved, asserted
    across every deck and the branch. Worth having anyway, and worth saying:
    the corpus number is not the fleet number, and quoting one for the other is
    how "34 of 50 invisible" became a finding about 1 card.
    """
    assert goldfish._ATTACKS_RE.search("Whenever you attack, create a Treasure.")
    assert goldfish._ATTACKS_RE.search("Whenever Karlach attacks, untap.")
    assert not goldfish._ATTACKS_RE.search("Destroy target creature.")


def test_an_extra_combat_the_model_cannot_place_is_named():
    """IT FELL THROUGH BOTH BRANCHES AND SET NOTHING.

    Not activated (no mana cost binds) and not triggered on an attack: a
    one-shot spell, or a permanent keyed on being blocked, on exert, on
    landfall, on a loyalty ability. The model has no channel for those — a
    boundary, not a bug — but the card contributed nothing to the clock AND
    appeared in no not-modelled list, so a low kill figure was illegible.
    """
    spell = _profile("Seize the Day",
                     "Untap target creature. After this main phase, there is "
                     "an additional combat phase.", "Sorcery")
    assert spell["extra_combat_cost"] is None
    assert not spell["extra_combat_free"]
    assert spell["unreadable"] == "Seize the Day"

    # The two the model CAN place must not be swept into the list with it.
    bought = _profile("Aggravated Assault",
                      "{3}{R}{R}: Untap all creatures you control. After this "
                      "main phase, there is an additional combat phase.")
    assert bought["extra_combat_cost"] == 5 and not bought["unreadable"]
    free = _profile("Scourge of the Throne",
                    "Whenever Scourge of the Throne attacks, untap it. After "
                    "this combat phase, there is an additional combat phase.")
    assert free["extra_combat_free"] and not free["unreadable"]


# ── damage multiplication ────────────────────────────────────────────────

def _prof(name, text, type_line="Creature — Dragon", power=4):
    return goldfish.combat_profile(
        {"name": name, "oracle_text": text, "type_line": type_line,
         "power": power, "toughness": power})


def test_three_wordings_one_effect_and_the_model_read_none_of_them():
    """THE DEFECT THIS EXISTS AGAINST. `combat_effects_not_modelled` named
    Atarka, Thrakkus and Hellkite Tyrant on ur-dragon — the model could see NO
    form of damage multiplication, so a damage doubler measured as a vanilla
    body and a card that triples the deck's output looked like a downgrade.

    Three different rules, one measured effect:
      replacement on damage dealt  — Twinflame Tyrant
      granted double strike        — Atarka, World Render
      power doubling               — Thrakkus the Butcher
    """
    assert _prof("Twinflame Tyrant",
                 "Flying If a source you control would deal damage to an "
                 "opponent or a permanent an opponent controls, it deals double "
                 "that damage instead.")["team_damage_multiplier"] == 2
    assert _prof("Atarka, World Render",
                 "Flying, trample Whenever a Dragon you control attacks, it "
                 "gains double strike until end of turn."
                 )["team_damage_multiplier"] == 2
    assert _prof("Thrakkus the Butcher",
                 "Trample Whenever Thrakkus attacks, double the power of each "
                 "Dragon you control until end of turn."
                 )["team_damage_multiplier"] == 2


def test_a_card_with_no_multiplier_is_untouched():
    """The widening rule: a deck without one of these must measure exactly as
    it did before."""
    for name, text in (("Terror of the Peaks", "Flying Whenever another creature "
                        "you control enters, this creature deals damage equal to "
                        "that creature's power to any target."),
                       ("Glorybringer", "Flying, haste You may exert this "
                        "creature as it attacks.")):
        got = _prof(name, text)
        assert got["team_damage_multiplier"] == 1
        assert got["double_strike"] is False


def test_own_double_strike_is_per_creature_and_never_the_team():
    """Different scope, kept apart. A creature that merely HAS double strike
    multiplies itself; one that GRANTS it multiplies everyone, and treating the
    first as the second would double a whole board off one keyword."""
    self_only = _prof("Boros Swiftblade", "Double strike")
    assert self_only["double_strike"] is True
    assert self_only["team_damage_multiplier"] == 1
    granted = _prof("Atarka, World Render",
                    "Whenever a Dragon you control attacks, it gains double "
                    "strike until end of turn.")
    assert granted["team_damage_multiplier"] == 2
    # A card that grants AND has it is not counted twice for its own body.
    assert granted["double_strike"] is False


def test_a_multiplier_is_no_longer_reported_as_unreadable():
    """`combat_effects_not_modelled` is a PROMISE about what the figures leave
    out. A card whose effect is now priced must leave that list, or the list is
    a liar in the more dangerous direction."""
    got = _prof("Thrakkus the Butcher",
                "Trample Whenever Thrakkus attacks, double the power of each "
                "Dragon you control until end of turn.")
    assert got["unreadable"] is None


@requires_deck
def test_multipliers_stack_multiplicatively_because_the_rules_do():
    """Double the power, swing twice, then double the damage dealt is EIGHT
    times, not four. Driven through the simulator rather than asserted: a deck
    holding all three must out-damage the same deck holding one."""
    import copy
    from manamap.pilot import card_pool
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("ur-dragon")
    pool, oracle = card_pool.load_pool(), card_pool.corpus_oracle()

    def with_only(keep):
        d = copy.deepcopy(doc)
        drop = {"Atarka, World Render", "Thrakkus the Butcher"} - set(keep)
        d["cards"] = [c for c in d["cards"] if c["name"] not in drop]
        return goldfish.run("ur-dragon", doc=d, quiet=True, iterations=1500,
                            seed=99)["metrics"]["combat"]["mean_damage_by_turn"]["10"]

    both = with_only({"Atarka, World Render", "Thrakkus the Butcher"})
    one = with_only({"Atarka, World Render"})
    assert both > one, (
        f"two multipliers ({both}) must out-damage one ({one}); if they are "
        f"equal the multiplier is being added or overwritten, not multiplied")


# ── the enters-the-battlefield payoff ────────────────────────────────────

def test_the_model_had_no_etb_damage_channel_at_all():
    """THE DEFECT THIS EXISTS AGAINST, and it is the largest of the session.

    ETB was read for Treasure and nothing else. ur-dragon's stated win condition
    is "ETB and attack-trigger burn (Terror of the Peaks, Scourge of Valkas,
    Dragon Tempest)" and all three were priced at ZERO — the first two as vanilla
    bodies, the third as nothing whatever, because an enchantment with no body
    and no mana falls through every cast loop.
    """
    terror = _prof("Terror of the Peaks",
                   "Flying Whenever another creature you control enters, this "
                   "creature deals damage equal to that creature's power to any "
                   "target.")
    assert terror["etb_damage_self_power"] is True

    scourge = _prof("Scourge of Valkas",
                    "Flying Whenever this creature or another Dragon you "
                    "control enters, it deals X damage to any target, where X "
                    "is the number of Dragons you control.")
    assert scourge["etb_damage_count"] is True

    tempest = _prof("Dragon Tempest",
                    "Whenever a creature you control with flying enters, it "
                    "gains haste until end of turn. Whenever a Dragon you "
                    "control enters, it deals X damage to any target, where X "
                    "is the number of Dragons you control.",
                    type_line="Enchantment", power=None)
    assert tempest["etb_damage_count"] is True


def test_a_token_maker_and_a_copier_are_read_apart():
    lathliss = _prof("Lathliss, Dragon Queen",
                     "Flying Whenever another nontoken Dragon you control "
                     "enters, create a 5/5 red Dragon creature token with flying.")
    assert lathliss["etb_token_bodies"] == 1 and lathliss["etb_token_power"] == 5
    assert lathliss["etb_copy"] is False

    miirym = _prof("Miirym, Sentinel Wyrm",
                   "Flying, ward {2} Whenever another nontoken Dragon you "
                   "control enters, create a token that's a copy of it, except "
                   "the token isn't legendary.")
    assert miirym["etb_copy"] is True
    assert miirym["etb_token_bodies"] == 0, "a copy is not a fixed-size token"


def test_nontoken_is_the_brake_the_rules_already_had():
    """WITHOUT THIS THE BOARD EXPLODES. The first cut produced 67,000 damage by
    turn six, because Miirym's copy re-triggered Miirym. Both Lathliss and
    Miirym say "another NONTOKEN Dragon", so their own tokens do not re-trigger
    them — the rules stop it, and the model simply had to read the word."""
    miirym = _prof("Miirym, Sentinel Wyrm",
                   "Whenever another nontoken Dragon you control enters, create "
                   "a token that's a copy of it.")
    assert miirym["etb_nontoken_only"] is True
    # Terror of the Peaks has no such clause, so tokens DO trigger it.
    terror = _prof("Terror of the Peaks",
                   "Whenever another creature you control enters, this creature "
                   "deals damage equal to that creature's power to any target.")
    assert terror["etb_nontoken_only"] is False


def test_a_landfall_payoff_is_not_a_creature_payoff():
    """THE CORPUS SWEEP BOUGHT THIS. The lazy noun run swallowed "land ", so
    Omnath, Rampaging Baloths, Titania and Zektar Shrine Expedition all read as
    creature-entering payoffs and would have fired on every creature cast.
    71 matches became 44 once the lookahead went in."""
    omnath = _prof("Omnath, Locus of Rage",
                   "Landfall — Whenever a land you control enters, create a 5/5 "
                   "red Elemental creature token.")
    assert omnath["etb_token_bodies"] == 0
    assert omnath["etb_damage_count"] is False


@requires_data
def test_the_etb_sweep_is_scoped():
    """A PATTERN SHIPS WITH ITS SWEEP. If this moves, something else matches and
    the tail needs reading card by card before the figures are believed."""
    from manamap.pilot import card_pool
    o, pool = card_pool.corpus_oracle(), card_pool.load_pool()
    n = 0
    for name, info in pool.items():
        got = goldfish.combat_profile(
            dict(info, name=name, oracle_text=o.get(name, "")))
        if any((got["etb_damage_self_power"], got["etb_damage_count"],
                got["etb_token_bodies"], got["etb_copy"])):
            n += 1
    assert 30 <= n <= 60, (
        f"{n} cards carry an ETB payoff; measured at 44 after the landfall "
        f"lookahead (71 before it).")


@requires_deck
def test_the_payoffs_move_the_damage_they_are_played_for():
    """Driven through the simulator. Removing the three declared payoffs must
    cost real damage — if it does not, the channel is not wired to the loop."""
    import copy
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("ur-dragon")
    without = copy.deepcopy(doc)
    without["cards"] = [c for c in without["cards"]
                        if c["name"] not in ("Terror of the Peaks",
                                             "Scourge of Valkas", "Dragon Tempest")]
    a = goldfish.run("ur-dragon", quiet=True, iterations=1500, seed=7
                     )["metrics"]["combat"]["mean_damage_by_turn"]["10"]
    b = goldfish.run("ur-dragon", doc=without, quiet=True, iterations=1500, seed=7
                     )["metrics"]["combat"]["mean_damage_by_turn"]["10"]
    assert a > b, f"cutting all three payoffs did not lower damage ({a} vs {b})"


@requires_deck
def test_the_commander_is_on_the_battlefield_and_swings():
    """It used to be cast and dropped — a flag and a mana sink. A 10/10 flier
    that never attacked, which is an entire stated win condition measured as
    zero."""
    import copy
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("ur-dragon")
    real = goldfish.combat_profile
    try:
        # Re-introduce the bug: a commander with no combat profile cannot join
        # the battlefield, which is exactly the old behaviour.
        goldfish.combat_profile = lambda c: dict(real(c), is_creature=False)
        without = goldfish.run("ur-dragon", doc=copy.deepcopy(doc), quiet=True,
                               iterations=1200, seed=11
                               )["metrics"]["combat"]["mean_damage_by_turn"]["10"]
    finally:
        goldfish.combat_profile = real
    with_cmd = goldfish.run("ur-dragon", quiet=True, iterations=1200, seed=11
                            )["metrics"]["combat"]["mean_damage_by_turn"]["10"]
    assert with_cmd > without, (
        f"the commander contributes nothing to damage ({with_cmd} vs {without})")
