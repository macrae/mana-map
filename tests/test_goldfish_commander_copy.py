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
    assert not copy_fodder(_spell(
        "Plural", "Target creatures you control get +1/+1. Draw a card."))
    assert not copy_fodder(_spell(
        "Each", "Each creature you control gains haste. Draw a card."))
    assert not copy_fodder(_spell(
        "Divided", "Deal 4 damage divided as you choose among any number of "
                   "target creatures."))
    assert not copy_fodder(_spell(
        "Two targets", "Up to two target creatures gain haste. Draw a card."))
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
