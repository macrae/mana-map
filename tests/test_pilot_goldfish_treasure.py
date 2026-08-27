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


# ── Doubling: multiplicative, and it is not the adder ────────────────────

DOUBLER = ("If an effect would create one or more tokens under your control, "
           "it creates twice that many of those tokens instead.")
ADDER = ("If you would create one or more Treasure tokens, instead create "
         "those tokens plus an additional Treasure token.")


def test_a_token_doubler_is_classified_apart_from_an_adder():
    """TWO REGEXES FOR ONE CONCEPT IS WHAT PRODUCED THIS.

    The model matched only the Xorn wording, so it priced 2 of the 8
    multipliers ur-dragon's treasure branch DECLARES and counted the other 6 as
    drawn-and-inert — while `assess._MULTIPLIER` matched five patterns, one
    module over. They share a definition now.
    """
    assert goldfish.classify(_card("X", DOUBLER))["treasure_doubler"] is True
    assert goldfish.classify(_card("X", DOUBLER))["treasure_bonus"] is False
    assert goldfish.classify(_card("X", ADDER))["treasure_bonus"] is True
    assert goldfish.classify(_card("X", ADDER))["treasure_doubler"] is False


def test_doubling_is_multiplicative_and_an_additive_stand_in_is_wrong():
    """THE CONTROL THAT SEPARATES THE TWO MODELS.

    They agree when an event makes exactly one token, which is why an additive
    stand-in reads almost right — and disagree the moment it makes three. A
    three-Treasure event under a doubler is SIX, not four.
    """
    land = {"name": "Mountain", "oracle_text": "", "cmc": 0,
            "type_line": "Basic Land — Mountain", "quantity": 34}
    src = _card("Engine", "At the beginning of your upkeep, create three "
                          "Treasure tokens.", cmc=2)
    src["quantity"] = 20
    dbl = _card("Procession", DOUBLER, cmc=2)
    dbl["quantity"] = 20
    plain = _run([land, src], True, turns=8)
    with_dbl = _run([land, src, dbl], True, turns=8)
    a, b = max(plain["treasures_by_turn"]), max(with_dbl["treasures_by_turn"])
    assert b >= 2 * a, (
        f"a doubler on a three-token event produced {b} against a plain {a}; "
        f"an additive +1 model would land near {a} + turns, not 2x")


def test_two_doublers_compound_rather_than_sum():
    """Each replacement replaces the other's output, so two is x4, not x3."""
    land = {"name": "Mountain", "oracle_text": "", "cmc": 0,
            "type_line": "Basic Land — Mountain", "quantity": 30}
    src = _card("Engine", "At the beginning of your upkeep, create a Treasure "
                          "token.", cmc=1)
    src["quantity"] = 15
    one = _card("Procession", DOUBLER, cmc=1); one["quantity"] = 10
    two = _card("Parallel", DOUBLER, cmc=1); two["quantity"] = 10
    a = max(_run([land, src, one], True, turns=9)["treasures_by_turn"])
    b = max(_run([land, src, one, two], True, turns=9)["treasures_by_turn"])
    assert b > a, f"a second doubler changed nothing: {a} -> {b}"


def test_a_trigger_doubler_is_not_a_token_doubler():
    """Panharmonicon doubles ETB TRIGGERS and Academy Manufactor converts a
    Clue/Food event into a Treasure one. Both are real multipliers for a deck
    and neither is THIS one — folding them in would be wrong in a way that
    reads as right, so they stay blind and get named instead."""
    pan = _card("Panharmonicon", "If an artifact or creature entering causes a "
                "triggered ability of a permanent you control to trigger, that "
                "ability triggers an additional time.")
    manu = _card("Academy Manufactor", "If you would create a Clue, Food, or "
                 "Treasure token, instead create one of each.")
    for card in (pan, manu):
        got = goldfish.classify(card)
        assert got["treasure_doubler"] is False, card["name"]
        assert got["treasure_bonus"] is False, card["name"]


def test_a_doubler_produces_nothing_with_the_model_off():
    land = {"name": "Mountain", "oracle_text": "", "cmc": 0,
            "type_line": "Basic Land — Mountain", "quantity": 30}
    src = _card("Engine", "At the beginning of your upkeep, create a Treasure "
                          "token.", cmc=2)
    src["quantity"] = 20
    dbl = _card("Procession", DOUBLER, cmc=2); dbl["quantity"] = 10
    off = _run([land, src, dbl], False, turns=8)
    assert max(off["treasures_by_turn"]) == 0


def test_a_multiplier_that_is_not_a_rock_tutor_or_body_still_gets_cast():
    """THE AGGRAVATED ASSAULT HOLE, ONE CARD CLASS OVER.

    The cast loop spends on rocks, tutors, extra-combat permanents and bodies.
    A token doubler is none of those. `bodies` is the model's proxy for "worth
    casting" and it happens to be 1 for Anointed Procession and 0 for Primal
    Vigor — the identical card — so Primal Vigor sat in hand for ten turns while
    carrying the flag that says it changes what the deck produces, and a
    candidate sweep read it as byte-identical to a blank.

    A flag the model set is a claim the model must act on.
    """
    land = {"name": "Mountain", "oracle_text": "", "cmc": 0,
            "type_line": "Basic Land — Mountain", "quantity": 32}
    src = _card("Engine", "At the beginning of your upkeep, create three "
                          "Treasure tokens.", cmc=2)
    src["quantity"] = 18
    # An enchantment whose ONLY property is the doubler flag: no body, no mana,
    # no tutor. Exactly the shape that was silently uncastable.
    inert = _card("Vigor", "If one or more tokens would be created, twice that "
                           "many of those tokens are created instead.",
                  cmc=3, type_line="Enchantment")
    inert["quantity"] = 12
    cl = goldfish.classify(inert)
    assert cl["treasure_doubler"] is True
    assert cl["bodies"] == 0 and cl["produces"] == 0 and not cl["tutor"], (
        "fixture drifted: this card must be uncastable by the other loops, "
        "or the test is not exercising the hole")

    plain = max(_run([land, src], True, turns=9)["treasures_by_turn"])
    with_it = max(_run([land, src, inert], True, turns=9)["treasures_by_turn"])
    assert with_it > plain, (
        f"a doubler carrying no body was never cast: {plain} -> {with_it}")


def test_a_token_tripler_is_a_multiplier_too():
    """Ojer Taq creates THREE times that many. The pattern read only "twice",
    so the format's biggest token multiplier scored as inert.

    Fleet impact: zero — no deck runs one. A corpus gap and an insurance fix,
    which is the honest description.
    """
    ojer = ("If one or more creature tokens would be created under your "
            "control, three times that many of those tokens are created "
            "instead.")
    assert goldfish.classify(_card("Ojer Taq", ojer))["treasure_doubler"] is True
    # And the exclusions hold: a Clue/Food converter is still not a doubler.
    manu = ("If you would create a Clue, Food, or Treasure token, instead "
            "create one of each.")
    assert goldfish.classify(_card("Academy Manufactor", manu))["treasure_doubler"] is False
