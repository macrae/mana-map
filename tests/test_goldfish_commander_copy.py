"""The commander-copy channel — Zada, Hedron Grinder.

Zada reads "whenever you cast an instant or sorcery spell that targets only
Zada, copy that spell for each other creature you control that the spell could
target." It is the ONLY card in the corpus with that ability, so the channel is
declared per deck; what the copies multiply is the spell's DRAW, and which
spells qualify is mechanical and lives in `copy_fodder`.

Unlike `model_commander_attack_tutor` and `model_commander_combat_reveal`, this
channel needs no authored rate: the copy count is the number of other creatures
you control, which the model already measures. There is no number to write down
and therefore none driving the headline.
"""

import copy
import json

import pytest

from manamap.pilot.goldfish_profiles import (
    commander_copies_spells,
    copy_fodder,
    draw_profile,
)

from conftest import requires_data, requires_deck


def _spell(name, text, type_line="Instant"):
    return {"name": name, "oracle_text": text, "type_line": type_line}


# ── the ability ──


ZADA_TEXT = ("Whenever you cast an instant or sorcery spell that targets only "
             "Zada, Hedron Grinder, copy that spell for each other creature you "
             "control that the spell could target. Each copy targets a "
             "different one of those creatures.")
#: THE NEAR-MISS, and it must not match. Agrus Kos copies only for creatures
#: that are Warriors or Soldiers — a typed subset this model does not track, so
#: reading it as an untyped board copy would overstate every Agrus deck.
AGRUS_TEXT = ("Whenever you cast an instant or sorcery spell that targets only "
              "Agrus Kos, Eternal Soldier, copy that spell for each other "
              "creature you control that's a Warrior or a Soldier and that the "
              "spell could target.")


def test_the_ability_matches_zada_and_not_agrus_kos():
    assert commander_copies_spells(
        {"name": "Zada, Hedron Grinder", "oracle_text": ZADA_TEXT})
    assert not commander_copies_spells(
        {"name": "Agrus Kos, Eternal Soldier", "oracle_text": AGRUS_TEXT})
    assert not commander_copies_spells({"name": "Krenko", "oracle_text": "Tap: make Goblins."})


# ── the fodder ──


def test_copy_fodder_accepts_a_single_target_cantrip():
    assert copy_fodder(_spell(
        "Expedite", "Target creature gains haste until end of turn. Draw a card."))
    assert copy_fodder(_spell(
        "Crimson Wisps",
        "Target creature becomes red and gains haste until end of turn. "
        "(It can attack and {T} this turn.)\nDraw a card."))


def test_copy_fodder_rejects_everything_that_cannot_trigger_the_ability():
    """The gate is the word ONLY: one more target of any kind and Zada never
    fires. Each of these is a shape the ability cannot see."""
    # MANDATORY plurals: you must pick two or more, so never "only Zada".
    assert not copy_fodder(_spell(
        "Plural", "Target creatures you control get +1/+1. Draw a card."))
    assert not copy_fodder(_spell(
        "Two targets", "Two target creatures each get +2/+2 until end of turn."))
    assert not copy_fodder(_spell(
        "Each", "Each creature you control gains haste. Draw a card."))
    assert not copy_fodder(_spell(
        "Divided", "Deal 4 damage divided as you choose among any number of "
                   "target creatures."))
    # A CHOOSABLE count is fodder, and asserting otherwise was the bug: cast on
    # Zada alone the spell targets only Zada and copies for the whole board.
    assert copy_fodder(_spell(
        "Up to two", "Up to two target creatures gain haste. Draw a card."))
    assert copy_fodder(_spell(
        "Strive", "Up to two target creatures each get +1/+0 until end of turn."))
    # A second target of ANOTHER type — the spell no longer targets only her.
    assert not copy_fodder(_spell(
        "Second target", "Target creature gains haste. Target player draws a card."))
    # Aimed at somebody else's board: it cannot name your own commander.
    assert not copy_fodder(_spell(
        "Opponent's", "Target creature an opponent controls gets -2/-0. Draw a card."))
    # A permanent is not an instant or a sorcery.
    assert not copy_fodder(_spell(
        "Aura", "Enchanted creature gets +2/+2.", type_line="Enchantment — Aura"))


def test_copy_fodder_rejects_a_spell_you_would_not_want_copied():
    """Copying "destroy target creature" across your own board destroys it. The
    predicate is about spells whose effect you want on THIRTY bodies."""
    assert not copy_fodder(_spell("Murder", "Destroy target creature."))
    assert not copy_fodder(_spell("Exile", "Exile target creature."))


# ── the fleet-wide draw fix this channel needed first ──


def test_spell_draw_reads_a_cantrip_whose_rider_carries_reminder_text():
    """`_SPELL_DRAW_RE` is sentence-anchored and `draw_profile` matches raw
    oracle text, so reminder text between the period and the draw clause left
    the anchor landing on ".)" and failing.

    OPT AND CONSIDER WERE IN THAT SET — two of the most-played cantrips in the
    game, read as drawing nothing, on every deck in the fleet. Re-introducing
    either half of the fix (the `\\s+` or the reminder strip) fails this test.
    """
    assert draw_profile(_spell(
        "Opt", "Scry 1. (Look at the top card of your library. You may put that "
               "card on the bottom of your library.)\nDraw a card.",
        type_line="Instant"))["spell_draw"] == 1
    assert draw_profile(_spell(
        "Crimson Wisps",
        "Target creature becomes red and gains haste until end of turn. "
        "(It can attack and {T} this turn.)\nDraw a card."))["spell_draw"] == 1
    # And the plain case still reads, so the widening did not break the anchor.
    assert draw_profile(_spell(
        "Expedite", "Target creature gains haste until end of turn. Draw a card."
    ))["spell_draw"] == 1


def test_a_draw_bought_with_a_discard_is_still_not_credited():
    """DELIBERATELY NOT WIDENED. "You may discard a card. If you do, draw two
    cards" is a draw you BUY, and crediting it without pricing the cost reads
    Witch's Mark at +2 when it is net +1. 21 corpus cards share the shape; they
    need the paired discard read in the same commit, which is a modelling
    change rather than a regex fix."""
    assert draw_profile(_spell(
        "Witch's Mark",
        "You may discard a card. If you do, draw two cards. Create a Wicked "
        "Role token attached to up to one target creature you control."
    ))["spell_draw"] == 0


# ── the declaration ──


@requires_data
@requires_deck
def test_the_flag_is_refused_on_a_commander_without_the_ability():
    """A declaration cannot lie about the ability. The flag is an opt-in; the
    model reads the commander's text to confirm it, or a deck gets a silent
    multiplier on every cantrip it casts."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    try:
        doc = load_deck_cards("goblin-storm")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover - deck absent
        pytest.skip("goblin-storm not built")
    # Zada really has it, so this must NOT raise.
    tt = dict(targets, model_draw=True, model_commander_copy=True)
    goldfish.run("goblin-storm", doc=copy.deepcopy(doc), _targets_doc=tt,
                 iterations=30, seed=1, quiet=True, _band=False)
    # Swap the commander's text out and the same declaration must be refused.
    blind = copy.deepcopy(doc)
    checked = 0
    for card in blind["cards"]:
        if card.get("is_commander") or "Zada" in card["name"]:
            card["oracle_text"] = "Flying"
            checked += 1
    assert checked >= 1, "no commander found to blind"
    with pytest.raises(goldfish.DeclarationError, match="model_commander_copy"):
        goldfish.run("goblin-storm", doc=blind, _targets_doc=tt,
                     iterations=30, seed=1, quiet=True, _band=False)


@requires_data
@requires_deck
def test_the_copies_multiply_the_draw_and_the_flag_is_what_does_it():
    """THE CHANNEL, MEASURED. Same 99, same seed; the only difference is the
    flag. Zada's copies must move the draw figure and nothing else may."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    try:
        doc = load_deck_cards("goblin-storm")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover - deck absent
        pytest.skip("goblin-storm not built")

    def cards_at_ten(copy_on):
        tt = dict(targets, model_draw=True, model_commander_copy=copy_on)
        m = goldfish.run("goblin-storm", doc=copy.deepcopy(doc), _targets_doc=tt,
                         iterations=2000, seed=7, quiet=True, _band=False)["metrics"]
        return m["mean_extra_cards_drawn_by_turn"]["10"]

    off, on = cards_at_ten(False), cards_at_ten(True)
    assert on > off, f"the copy channel drew nothing: {off} -> {on}"
    # The deck holds four drawing fodder spells; the lift is not a rounding
    # artefact. Re-introducing the bug (ignoring `copy_fodder`, or dropping the
    # `commander_turn is not None` guard) moves this in a way the assert sees.
    assert on >= off * 1.5, f"lift too small to be the channel: {off} -> {on}"


@requires_data
@requires_deck
def test_a_lone_commander_multiplies_by_nothing():
    """`battlefield` includes the commander, so the copy count is
    len(battlefield) - 1. With no other creature there are no copies, and an
    off-by-one here would hand every deck a free extra card per cantrip."""
    from manamap.pilot import goldfish_turn
    import inspect
    src = inspect.getsource(goldfish_turn.simulate_once)
    assert "max(len(battlefield) - 1, 0)" in src, (
        "the copy count must exclude the commander herself")


@requires_data
@requires_deck
def test_only_fodder_is_copied_and_nothing_else_is():
    """THE CONTROL FOR OVER-COPYING, and the first version of this file did not
    have it: dropping the `copy_fodder` guard entirely failed no test, because
    goblin-storm's other draw spells would simply be multiplied too and the lift
    assert above only checks the lift is LARGE enough.

    Faithless Looting draws two and is not fodder — it targets no creature, so
    Zada never sees it. Blind every real fodder spell and the channel must add
    EXACTLY nothing; if it still moves, something not-fodder is being copied.
    """
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.goldfish_profiles import copy_fodder as _fodder
    try:
        doc = load_deck_cards("goblin-storm")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover - deck absent
        pytest.skip("goblin-storm not built")

    blind = copy.deepcopy(doc)
    checked = 0
    for card in blind["cards"]:
        if _fodder(card):
            # Keep the draw, remove the single-creature target that Zada needs.
            card["oracle_text"] = "Draw a card."
            checked += 1
    assert checked >= 4, f"goblin-storm should hold at least four fodder spells, not {checked}"

    def cards_at_ten(copy_on):
        tt = dict(targets, model_draw=True, model_commander_copy=copy_on)
        m = goldfish.run("goblin-storm", doc=copy.deepcopy(blind), _targets_doc=tt,
                         iterations=2000, seed=7, quiet=True, _band=False)["metrics"]
        return m["mean_extra_cards_drawn_by_turn"]["10"]

    assert cards_at_ten(True) == cards_at_ten(False), (
        "with every fodder spell blinded the copy channel must add nothing — "
        "a non-fodder draw spell is being multiplied")


# ── the pump arm ──
#
# A single-target pump copied across the board is how a go-wide Zada deck kills.
# It was invisible: `combat_profile` had 44 fields and not one read a pump, so
# Haze of Rage and Goblin Bushwhacker — both sleeved in goblin-storm since it was
# built — contributed nothing to any damage figure.


def test_spell_pump_reads_power_and_distinguishes_single_from_team():
    from manamap.pilot.goldfish_profiles import spell_pump
    assert spell_pump(_spell("Seething Anger",
        "Target creature gets +3/+0 until end of turn.", "Sorcery")) == (3, 0)
    assert spell_pump(_spell("Haze of Rage",
        "Target creature gets +1/+0 until end of turn.")) == (1, 0)
    # A team pump already hits everything and must NOT be multiplied again.
    assert spell_pump(_spell("Team",
        "Creatures you control get +2/+2 until end of turn.")) == (0, 2)
    # Power only — this model has no blockers, so toughness changes nothing.
    assert spell_pump(_spell("Might of Oaks",
        "Target creature gets +7/+7 until end of turn.")) == (7, 0)
    # A permanent is not a spell pump; its anthem is a different channel.
    assert spell_pump({"name": "Lord", "type_line": "Creature — Goblin",
                       "oracle_text": "Other Goblins you control get +1/+1."}) == (0, 0)


def test_an_X_based_pump_is_deliberately_not_read():
    """X is a count this function has no board to resolve — the same reason
    `land_colors` refuses a fetchland without a pool, and the same safe
    direction. 46 corpus cards; understating a pump is recoverable."""
    from manamap.pilot.goldfish_profiles import spell_pump
    assert spell_pump(_spell("Fists of Flame",
        "Until end of turn, target creature gains trample and gets +1/+0 for "
        "each card you've drawn this turn.")) == (0, 0)


def test_a_second_target_of_the_same_type_is_not_fodder():
    """Monstrous Step pumps +7/+7 and then names ANOTHER target creature, so it
    does not target only Zada. It matched every other clause in the plural
    pattern and needed its own; the tightening dropped copy_fodder 774 -> 758."""
    assert not copy_fodder(_spell("Monstrous Step",
        "Target creature gets +7/+7 until end of turn. Up to one other target "
        "creature gets +1/+1 until end of turn."))


def test_the_pump_lasts_one_turn_and_never_joins_the_permanent_anthem():
    """THE DOCUMENTED TRAP, asserted structurally. `team_anthem` is permanent;
    a pump is not. A saga back face lasting one turn read as a permanent damage
    doubler on sharknado and cutting it measured as a LOSS.

    `turn_pump` and `flat_pump` must be reset inside the turn loop, and the
    pump must never be added to `team_anthem`.
    """
    import inspect
    from manamap.pilot import goldfish_turn
    src = inspect.getsource(goldfish_turn.simulate_once)
    assert "turn_pump = 0" in src and "flat_pump = 0" in src
    # Reset INSIDE the loop, not once before it.
    after_loop = src.split("for turn in range(1, max_turn + 1):", 1)[1]
    assert "turn_pump = 0" in after_loop.split("\n\n", 1)[0] or \
           "turn_pump = 0" in after_loop[:600], "turn_pump must reset every turn"
    assert "team_anthem += single" not in src and "team_anthem += team" not in src, \
        "a one-turn pump must never join the permanent anthem"


@requires_data
@requires_deck
def test_the_pump_moves_damage_and_the_copy_flag_is_what_multiplies_it():
    """MEASURED. A single-target pump with the copy ability on becomes a
    board-wide bonus; with it off it pumps one attacker. The damage figure must
    separate the two."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    try:
        doc = load_deck_cards("goblin-storm")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover - deck absent
        pytest.skip("goblin-storm not built")

    def damage_at_ten(copy_on):
        tt = dict(targets, model_draw=True, model_combat=True,
                  model_commander_copy=copy_on)
        m = goldfish.run("goblin-storm", doc=copy.deepcopy(doc), _targets_doc=tt,
                         iterations=3000, seed=5, quiet=True, _band=False)["metrics"]
        return m["combat"]["mean_damage_by_turn"]["10"]

    off, on = damage_at_ten(False), damage_at_ten(True)
    assert on > off, (
        f"copying a pump across the board must beat pumping one creature: "
        f"{off} -> {on}")


@requires_data
@requires_deck
def test_a_pump_spell_is_actually_cast():
    """THE CASTING PREDICATE, which this file has had to learn six times. A pump
    spell has no body, no draw and makes no mana, so every other loop skips it.
    Blind every pump in the deck and the damage figure must MOVE — if it does
    not, nothing was being cast and the channel is decorative."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.goldfish_profiles import spell_pump as _sp
    try:
        doc = load_deck_cards("goblin-storm")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover - deck absent
        pytest.skip("goblin-storm not built")
    tt = dict(targets, model_draw=True, model_combat=True, model_commander_copy=True)

    def damage_at_ten(d):
        m = goldfish.run("goblin-storm", doc=copy.deepcopy(d), _targets_doc=tt,
                         iterations=3000, seed=5, quiet=True, _band=False)["metrics"]
        return m["combat"]["mean_damage_by_turn"]["10"]

    blind = copy.deepcopy(doc)
    checked = 0
    for card in blind["cards"]:
        if _sp(card) != (0, 0):
            card["oracle_text"] = "Flying"
            checked += 1
    assert checked >= 1, "goblin-storm should hold at least one pump spell"
    assert damage_at_ten(doc) > damage_at_ten(blind), (
        "blinding every pump changed nothing — no pump spell is being cast")


# ── the spell count, and the three channels that ride on it ──
#
# There was no spells-cast-this-turn variable anywhere in the simulator, so
# STORM — a mechanic defined entirely by that count — was unreadable, and
# goblin-storm's two payoffs (Grapeshot, Empty the Warrens) scored as a 1-damage
# ping and two 1/1s however many spells preceded them.


def test_every_cast_site_goes_through_the_counter():
    """THE STRUCTURAL GUARD, and it is the only thing that keeps this correct.

    There are twelve cast sites in `simulate_once` and storm's whole value is the
    count, so a site that removes a card from hand without routing through
    `_note_cast` silently undercounts it. The two legitimate exceptions are the
    land drop and the discard, neither of which is a cast.
    """
    import inspect
    from manamap.pilot import goldfish_turn
    src = inspect.getsource(goldfish_turn.simulate_once)
    bare = [ln.strip() for ln in src.splitlines() if "hand.remove(" in ln]
    allowed = {"hand.remove(lands[-1])", "hand.remove(pick)", "hand.remove(card)"}
    assert all(b in allowed for b in bare), f"unexpected removal site: {bare}"
    # Exactly one bare `hand.remove(card)` survives: the helper's own body.
    assert sum(1 for b in bare if b == "hand.remove(card)") == 1, (
        "a cast site is bypassing _note_cast — storm will undercount")
    assert src.count("_note_cast(") >= 12, (
        f"only {src.count('_note_cast(')} cast sites route through the counter")


def test_spell_count_profile_reads_the_three_channels():
    from manamap.pilot.goldfish_profiles import spell_count_profile as f
    g = f(_spell("Grapeshot", "Grapeshot deals 1 damage to any target. Storm "
                              "(When you cast this spell, copy it for each spell "
                              "cast before it this turn.)"))
    assert g["storm"] and g["storm_damage"] == 1
    e = f(_spell("Empty the Warrens", "Create two 1/1 red Goblin creature tokens. "
                 "Storm (When you cast this spell, copy it for each spell cast "
                 "before it this turn.)", "Sorcery"))
    assert e["storm"] and e["storm_token_bodies"] == 2
    # A permanent is not a storm spell however its text reads.
    assert not f({"name": "Lord", "type_line": "Creature — Goblin",
                  "oracle_text": "Other Goblins get +1/+1."})["storm"]


def test_per_cast_damage_and_magecraft_are_separate_because_copies_differ():
    """A RULES FACT, not a modelling choice. "Whenever you cast" does not fire on
    a copy; "whenever you cast OR COPY" does. Guttersnipe says cast, so Zada's
    copies never feed it; Storm-Kiln Artist says cast or copy, so with Zada out
    and six other bodies one cantrip is seven Treasures. Collapsing the two into
    one flag would hand every Zada deck a burn kill it does not have."""
    from manamap.pilot.goldfish_profiles import spell_count_profile as f
    g = f({"name": "Guttersnipe", "type_line": "Creature — Goblin Shaman",
           "oracle_text": "Whenever you cast an instant or sorcery spell, this "
                          "creature deals 2 damage to each opponent."})
    assert g["per_cast_damage"] == 2 and not g["magecraft"]
    s = f({"name": "Storm-Kiln Artist", "type_line": "Creature — Dwarf Shaman",
           "oracle_text": "Magecraft — Whenever you cast or copy an instant or "
                          "sorcery spell, create a Treasure token."})
    assert s["magecraft"] and s["magecraft_treasure"] == 1
    assert s["per_cast_damage"] == 0
    # The gate distinguishes a noncreature trigger from an instant/sorcery one.
    fa = f({"name": "Firebrand Archer", "type_line": "Creature",
            "oracle_text": "Whenever you cast a noncreature spell, this creature "
                           "deals 1 damage to each opponent."})
    assert fa["per_cast_damage_gate"] == "a noncreature"


@requires_data
@requires_deck
def test_each_of_the_three_channels_actually_fires():
    """MEASURED, one channel at a time. A card read perfectly and never acted on
    is this file's documented failure six times over, so each channel is proven
    by blinding its cards and requiring the figure to move."""
    from manamap.pilot import goldfish
    from manamap.pilot.common import load_deck_cards
    from manamap.pilot.goldfish_profiles import spell_count_profile as sc
    try:
        doc = load_deck_cards("goblin-storm", "zada-v1")
        targets = json.load(open("data/decks/goblin-storm/goldfish_targets.json"))
    except Exception:  # pragma: no cover
        pytest.skip("goblin-storm@zada-v1 not built")
    tt = dict(targets, model_draw=True, model_combat=True,
              model_commander_copy=True, model_treasures=True)

    def damage(d):
        m = goldfish.run("goblin-storm", doc=copy.deepcopy(d), _targets_doc=tt,
                         iterations=3000, seed=5, quiet=True, _band=False)["metrics"]
        return m["combat"]["mean_damage_by_turn"]["10"]

    base = damage(doc)
    for key in ("storm", "per_cast_damage", "magecraft_treasure"):
        blind = copy.deepcopy(doc)
        checked = 0
        for card in blind["cards"]:
            if sc(card)[key]:
                card["oracle_text"] = "Flying"
                checked += 1
        assert checked >= 1, f"no card in the branch feeds {key}"
        assert damage(blind) < base, (
            f"blinding {key} ({checked} cards) changed nothing — the channel is "
            f"read and never acted on")


@requires_data
@requires_deck
def test_storm_copies_count_the_spells_cast_BEFORE_it():
    """The off-by-one that would double every storm payoff. A storm spell cast
    as the first spell of the turn copies ZERO times, not once."""
    import inspect
    from manamap.pilot import goldfish_turn
    src = inspect.getsource(goldfish_turn.simulate_once)
    assert "copies = spells_cast_this_turn" in src
    # The count is read BEFORE _note_cast increments it.
    seg = src.split("copies = spells_cast_this_turn", 1)[1][:200]
    assert "_note_cast(card)" in seg, (
        "the spell count must be read before the cast increments it")
