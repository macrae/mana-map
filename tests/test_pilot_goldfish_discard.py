"""The discard channel: wheels, loots, and the payoffs on a discard or a draw.

The six wheels in sharknado were INVISIBLE to this model, not dark — `_DRAW_RE`
wants "draw N card" and Wheel of Fortune says "draws seven cards" — and no
casting loop ever selected one. Faithless Looting read +2 with nothing netting
the discard. "Whenever you discard a card" and "whenever you draw a card" were
modelled nowhere. And the second commander of a Partner pair was never cast:
`commanders[0]` was the only one the model knew, so Brallin, Skyshark Rider
scored zero however well his trigger parsed.
"""

import csv

import pytest

from manamap.config import OUTPUT_CSV_PATH
from manamap.pilot import goldfish
from conftest import requires_data, requires_deck


def _spell(text, type_line="Sorcery", name="a spell", mana_cost="{2}{R}"):
    return goldfish.draw_profile({"name": name, "oracle_text": text, "type_line": type_line,
                                  "mana_cost": mana_cost})


def _ev(text, name="a card"):
    return goldfish.event_payoffs({"name": name, "oracle_text": text})


def test_wheels_are_read_and_the_conditional_one_is_not():
    assert _spell("Each player discards their hand, then draws seven cards.")["wheel_draws"] == 7
    assert _spell("Each player discards their hand, then draws cards equal to the greatest "
                  "number of cards a player discarded this way.")["wheel_draws"] == -1
    assert _spell("Discard your hand, then draw three cards.")["wheel_draws"] == 3
    assert _spell("Each player discards their hand, then draws seven cards.",
                  name="Wheel of Misfortune")["wheel_draws"] == 0
    # A wheel on a creature ("whenever … discards their hand") is not a spell.
    assert _spell("Each player discards their hand, then draws seven cards.",
                  type_line="Creature — Wizard")["wheel_draws"] == 0
    assert _spell("Each player discards their hand, then draws seven cards.")["unmodelled"] is None


def test_a_loot_keeps_its_gross_draw_and_records_its_discard():
    d = _spell("Draw two cards, then discard two cards.")
    assert (d["spell_draw"], d["spell_discard"]) == (2, 2)
    d = _spell("Draw three cards, then put two cards from your hand on top of your library.")
    assert (d["spell_draw"], d["spell_discard"]) == (3, 0)
    # An additional-cost discard stays unmodelled, as before.
    d = _spell("As an additional cost to cast this spell, discard a card.\nDraw two cards.")
    assert d["spell_draw"] == 0 and d["unmodelled"]


def test_the_payoffs_are_read_per_event():
    assert _ev("Whenever you discard a card, put a +1/+1 counter on Brallin and it deals 1 "
               "damage to each opponent.") == {**_ev(""), "per_discard_damage": 1, "per_discard_counter": 1}
    assert _ev("Whenever you draw a card, put a +1/+1 counter on Shabraz and you gain 1 life.")["per_draw_counter"] == 1
    assert _ev("Whenever you draw a card, each opponent loses 1 life.")["per_draw_damage"] == 1
    assert _ev("Whenever you draw a card, Niv-Mizzet deals 1 damage to any target.")["per_draw_damage"] == 1
    assert _ev("Whenever you draw your second card each turn, this creature deals 3 damage "
               "to any target.")["second_draw_damage"] == 3
    # A payoff that draws on a draw is deliberately not read (an uncapped loop).
    assert not goldfish.has_event_payoff(_ev("Whenever enchanted creature deals damage to an opponent, draw a card."))
    assert goldfish.has_event_payoff(_ev("Whenever you discard a card, draw a card.")) is True


@requires_data
def test_the_corpus_sweep_is_locked():
    wheels = loots = disc = draw = second = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            card = {"name": row["name"], "oracle_text": row["oracle_text"],
                    "type_line": row["type_line"], "mana_cost": row.get("mana_cost")}
            d = goldfish.draw_profile(card); e = goldfish.event_payoffs(card)
            wheels += bool(d["wheel_draws"]); loots += bool(d["spell_discard"])
            disc += bool(e["per_discard_damage"] or e["per_discard_counter"]
                         or e["per_discard_draw"] or e["per_discard_token_power"])
            draw += bool(e["per_draw_damage"] or e["per_draw_counter"] or e["per_draw_token_power"])
            second += bool(e["second_draw_damage"] or e["second_draw_token_power"])
    assert (wheels, loots, disc, draw, second) == (WHEELS, LOOTS, DISC, DRAW, SECOND), \
        (wheels, loots, disc, draw, second)


# Filled in by the sweep the day the channel shipped; a widened pattern moves
# these on purpose.
WHEELS, LOOTS, DISC, DRAW, SECOND = (25, 40, 15, 29, 20)


@requires_data
@requires_deck
def test_sharknado_wheels_and_the_partner_is_cast(monkeypatch):
    on = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=True)["metrics"]
    off = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=False)["metrics"]
    assert "discard" in on and "discard" not in off, "absent, not zero, when the flag is off"
    assert on["discard"]["mean_cards_discarded_by_turn"]["8"] > 1
    assert on["mean_extra_cards_drawn_by_turn"]["8"] > off["mean_extra_cards_drawn_by_turn"]["8"] * 2, \
        "the wheels are cast now and were invisible before"
    assert on["discard"]["mean_cumulative_event_damage_by_turn"]["8"] > 0
    # THE PARTNER is cast in both arms: the second commander is not a channel.
    assert on["partner"]["cast_by_turn_6_rate"] > 0.8 and off["partner"]["cast_by_turn_6_rate"] > 0.8
    assert on["commander"]["cast_by_turn_6_rate"] > 0.7, "the first commander still casts on its own curve"
    # PROVEN BY BLINDING THE PROFILE, partners on in both arms: no wheels read,
    # and the extra-cards figure falls back to the loots alone.
    real = goldfish.draw_profile
    def blind(card):
        d = real(card); d["wheel_draws"] = 0; return d
    monkeypatch.setattr(goldfish, "draw_profile", blind)
    blinded = goldfish.run("sharknado", branch="recon-v1", iterations=1200, quiet=True, model_discard=True)["metrics"]
    assert blinded["mean_extra_cards_drawn_by_turn"]["8"] < on["mean_extra_cards_drawn_by_turn"]["8"] * 0.6


@requires_data
@requires_deck
def test_a_deck_that_did_not_opt_in_is_byte_identical_beyond_the_stamp():
    """The channel is gated: a non-opted deck computes nothing new."""
    import json
    from manamap.config import DATA_DIR
    on_disk = json.loads((DATA_DIR / "decks" / "ur-dragon" / "goldfish_metrics.json").read_text())
    fresh = goldfish.run("ur-dragon", quiet=True)
    a, b = dict(on_disk["metrics"]), dict(fresh["metrics"])
    assert a == b, "ur-dragon moved without opting into model_discard"


def test_scry_then_draw_and_upkeep_reveal_are_draw():
    """Found on ingris-infect/draw-v1: Read the Bones and Dark Confidant went in
    as card advantage and the draw axis did not move, because neither says
    'draw' where the parser looks."""
    d = goldfish.draw_profile({"name": "Read the Bones", "type_line": "Sorcery",
                               "oracle_text": "Scry 2, then draw two cards. You lose 2 life."})
    assert d["spell_draw"] == 2 and d["unmodelled"] is None
    d = goldfish.draw_profile({"name": "Dark Confidant", "type_line": "Creature — Human Wizard",
                               "oracle_text": "At the beginning of your upkeep, reveal the top card of "
                                              "your library and put that card into your hand. You lose "
                                              "life equal to its mana value."})
    assert d["recurring_draw"] == 1


@requires_data
def test_the_two_draw_shapes_are_locked_to_the_corpus():
    scry = reveal = 0
    with open(OUTPUT_CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = row["oracle_text"] or ""
            if goldfish._SCRY_THEN_DRAW_RE.search(t) and ("Instant" in row["type_line"] or "Sorcery" in row["type_line"]):
                scry += 1
            if goldfish._UPKEEP_REVEAL_RE.search(t):
                reveal += 1
    assert (scry, reveal) == (34, 3), (scry, reveal)
