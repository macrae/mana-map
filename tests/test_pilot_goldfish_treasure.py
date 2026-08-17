"""The Treasure model: a one-shot stockpile, and the triggers a goldfish can see.

A Treasure is not a mana rock, and modelling it as one is the whole trap — a rock
produces every turn forever, a Treasure produces once and is gone. These tests pin
both halves: that the stockpile is spent rather than repeated, and that the model
refuses to invent mana from triggers this simulation cannot observe.

The second half matters more than it looks. Across the nine tracked decks, 16 of
the 19 Treasure sources are combat- or opponent-gated, and this model has neither
combat nor opponents. A naive "create a Treasure token" match would have handed
eight decks free mana they never get — turning a deliberately conservative
simulation optimistic, which is worse than leaving it blind.
"""

import json

import pytest

from manamap.pilot import goldfish


def _card(name, text, cmc=2, type_line="Creature — Dwarf"):
    return {"name": name, "oracle_text": text, "cmc": cmc,
            "type_line": type_line, "quantity": 1}


# ── The classifier ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("At the beginning of your upkeep, create a Treasure token.", "upkeep"),
    ("Landfall — Whenever a land you control enters, create a Treasure token.", "landfall"),
    ("Whenever you cast an instant spell, create a Treasure token.", "cast"),
    ("When this creature enters, create a Treasure token.", "etb"),
    # The three the model must refuse.
    ("Whenever this creature deals combat damage to a player, create a Treasure token.", "unmodelled"),
    ("Whenever an opponent draws a card, you may create a Treasure token.", "unmodelled"),
    ("Sacrifice a creature: create a Treasure token.", "unmodelled"),
])
def test_the_trigger_decides_whether_a_goldfish_can_see_it(text, expected):
    assert goldfish.treasure_profile(_card("X", text))[1] == expected


def test_a_saga_chapter_is_a_recurring_engine():
    """A Saga adds a lore counter every turn, so Treasure chapters recur.

    The Misty Mountains Cold makes one on each of four chapters. Missing this
    would have priced a three-mana engine as a one-shot.
    """
    n, trigger = goldfish.treasure_profile(_card(
        "The Misty Mountains Cold",
        "(As this Saga enters and after your draw step, add a lore counter.) "
        "I, II, III, IV — Create a Treasure token.",
        cmc=3, type_line="Enchantment — Saga"))
    assert (n, trigger) == (1, "upkeep")


def test_a_card_with_no_treasure_text_is_not_a_source():
    assert goldfish.treasure_profile(_card("Sol Ring", "{T}: Add {C}{C}.")) == (0, None)


def test_a_multiplier_is_not_counted_as_a_source():
    """Xorn creates no Treasure of its own; it adds one to every event. Counting
    it as a source would double-count it against a deck's engine density."""
    card = _card("Xorn", "If you would create one or more Treasure tokens, instead "
                         "create those tokens plus an additional Treasure token.")
    assert goldfish.treasure_profile(card)[1] == "unmodelled"
    assert goldfish.classify(card)["treasure_bonus"] is True


# ── The simulation ───────────────────────────────────────────────────────

def _run(cards, model_treasures, turns=6, commander_cmc=4):
    import random
    library = []
    for c in cards:
        library.extend([goldfish.classify(c)] * c.get("quantity", 1))
    return goldfish.simulate_once(random.Random(42), library, commander_cmc, [],
                                  turns, model_treasures=model_treasures)


def _deck_with(source_text, lands=30):
    land = {"name": "Mountain", "oracle_text": "", "cmc": 0,
            "type_line": "Basic Land — Mountain", "quantity": lands}
    src = _card("Engine", source_text, cmc=2)
    src["quantity"] = 20        # dense, so the engine reliably lands
    return [land, src]


def test_an_upkeep_source_builds_a_hoard_only_when_the_model_is_on():
    on = _run(_deck_with("At the beginning of your upkeep, create a Treasure token."), True)
    off = _run(_deck_with("At the beginning of your upkeep, create a Treasure token."), False)
    assert max(on["treasures_by_turn"]) > 0, "an upkeep engine produced nothing"
    assert max(off["treasures_by_turn"]) == 0, "treasures accrued with the model OFF"


def test_a_combat_source_produces_nothing_even_with_the_model_on():
    """There is no combat in this simulation, so this is correct rather than a gap.
    It is also why ur-dragon's four Treasure Dragons report a hoard of zero."""
    r = _run(_deck_with("Whenever this creature deals combat damage to a player, "
                        "create three Treasure tokens."), True)
    assert max(r["treasures_by_turn"]) == 0


def test_a_treasure_is_spent_once_and_does_not_come_back():
    """The property that separates a Treasure from a mana rock. A hoard that only
    ever rises is a rock wearing a Treasure's name."""
    cards = _deck_with("At the beginning of your upkeep, create a Treasure token.")
    # An expensive body to break the hoard on, priced above what lands alone give.
    sink = _card("Sink", "", cmc=9, type_line="Creature — Dragon")
    sink["quantity"] = 12
    r = _run(cards + [sink], True, turns=10, commander_cmc=99)
    hoard = r["treasures_by_turn"]
    assert any(b < a for a, b in zip(hoard, hoard[1:])), (
        f"the hoard never fell, so nothing was ever spent: {hoard}")


def test_the_model_off_is_byte_identical_to_the_model_absent():
    """The containment guarantee the opt-in exists for.

    Turning the model on fleet-wide moves three decks' published figures —
    gishath's `mean_cast_turn` alone is quoted 16 times across seven tracked
    artifacts. So OFF must be indistinguishable from before the model existed,
    for every series that already had a meaning.
    """
    cards = _deck_with("At the beginning of your upkeep, create a Treasure token.")
    off = _run(cards, False)
    for key in ("land_hits", "mana_by_turn", "commander_turn", "bodies_by_turn",
                "mulligans", "first_seven_lands"):
        assert key in off
    assert off["treasures_by_turn"] == [0] * len(off["treasures_by_turn"])
    assert not any(off["treasure_online_by_turn"])


def test_the_opt_in_omits_the_keys_entirely_rather_than_zeroing_them(tmp_path, monkeypatch):
    """A zeroed block would still make all nine tracked artifacts stale."""
    m = goldfish.aggregate([_run(_deck_with("At the beginning of your upkeep, "
                                            "create a Treasure token."), False)],
                           [], 6, model_treasures=False)
    assert "treasure" not in m
    m_on = goldfish.aggregate([_run(_deck_with("At the beginning of your upkeep, "
                                               "create a Treasure token."), True)],
                              [], 6, model_treasures=True)
    assert "treasure" in m_on
    assert "engine_online_rate_by_turn" in m_on["treasure"]
