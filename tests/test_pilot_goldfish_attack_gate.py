"""A creature that may not attack does not attack (2026-10-04).

Until then the goldfish swung with every body: 315 Defender creatures, and Kefnet
the Mindful — "Kefnet can't attack or block unless you have seven or more cards in
hand" — as a free 5/5 flyer every turn. Swept on the Scryfall dump: 312 never, 85
held back on a condition the model cannot evaluate, and two it can (Kefnet, seven
or more cards; Hazoret the Fervent, one or fewer).
"""
import pytest

from conftest import requires_data, requires_deck

from manamap.pilot import goldfish_profiles as gp


def _c(name, text, type_line="Creature — Test"):
    return {"name": name, "type_line": type_line, "oracle_text": text}


def test_defender_never_attacks_and_a_plain_body_does():
    assert gp.attack_gate(_c("Wall of Test", "Defender\nWhen this creature enters, draw a card."))["kind"] == "never"
    assert gp.attack_gate(_c("Plain Bear", "Trample")) is None
    assert gp.attack_gate(_c("Not A Creature", "Defender", type_line="Artifact")) is None


def test_the_gods_hand_conditions_are_read_both_ways():
    k = gp.attack_gate(_c("Kefnet the Mindful", "Flying, indestructible\n"
                          "Kefnet can't attack or block unless you have seven or more cards in hand."))
    assert k == {"kind": "hand", "min_hand": 7, "why": "you have seven or more cards in hand"}
    h = gp.attack_gate(_c("Hazoret the Fervent", "Indestructible, haste\n"
                          "Hazoret can't attack or block unless you have one or fewer cards in hand."))
    assert h["kind"] == "hand" and h["max_hand"] == 1


def test_only_the_card_itself_is_gated_not_what_it_forbids_others():
    # A card that stops OTHER creatures from attacking is not itself gated.
    assert gp.attack_gate(_c("Propaganda Golem", "Creatures your opponents control can't attack you.")) is None
    # "Can't attack alone" is not a gate in a model where nothing attacks alone.
    assert gp.attack_gate(_c("Mogg Flunkies", "Mogg Flunkies can't attack alone.")) is None
    # An unevaluable condition holds the creature back and says why.
    g = gp.attack_gate(_c("Sea Monster", "This creature can't attack unless defending player controls an Island."))
    assert g["kind"] == "unless" and "Island" in g["why"]


def test_a_transformed_back_face_is_not_the_creature_that_entered():
    g = gp.attack_gate(_c("Captive // Abomination",
                          "Defender // Trample", type_line="Creature — Werewolf // Creature — Werewolf"))
    assert g["kind"] == "never"     # read on the FRONT face only


@requires_data
@requires_deck
def test_switching_the_gate_off_gives_back_kefnets_phantom_swings(monkeypatch):
    """DRIVEN THROUGH THE SIMULATOR and proved by RE-INTRODUCING THE BUG: with the gate
    blinded, sharknado's champion deals MORE damage by turn ten, because Kefnet swings
    again without seven cards in hand. Measured 2026-10-04: 60.879 gated, 61.408 not."""
    from manamap.pilot import diagnostic
    on = diagnostic.run("sharknado", iterations=1500, seed=5, quiet=True)
    monkeypatch.setattr(gp, "attack_gate", lambda card: None)
    off = diagnostic.run("sharknado", iterations=1500, seed=5, quiet=True)
    a = on["output"]["damage_by_turn"]["10"]["rate"]
    b = off["output"]["damage_by_turn"]["10"]["rate"]
    assert b > a, f"blinding the gate should restore Kefnet's free swings ({a} vs {b})"


# ── the seats at the table (2026-10-04) ───────────────────────────────────

def test_a_trigger_window_ends_at_a_face_boundary():
    """Defacing Duskmage: the front face's trigger only PREPARES the card; the
    "Draw two cards" belongs to the spell face across the " // "."""
    e = gp.event_payoffs({"name": "Defacing Duskmage // Vandal's Edit",
                          "type_line": "Creature — Dog Warlock // Instant",
                          "oracle_text": "Deathtouch\nWhenever an opponent draws their second card "
                                         "each turn, this creature becomes prepared. // "
                                         "Draw two cards. Each player loses 2 life."})
    assert e["opponent_second_draw_our_draw"] == 0
    m = gp.event_payoffs({"name": "Faerie Mastermind", "type_line": "Creature — Faerie Rogue",
                          "oracle_text": "Flash\nFlying\nWhenever an opponent draws their second "
                                         "card each turn, you draw a card."})
    assert m["opponent_second_draw_our_draw"] == 1


@requires_data
@requires_deck
def test_what_we_draw_off_opponents_scales_with_the_seats(monkeypatch):
    """Faerie Mastermind draws once per OPPONENT drawing a second card — at a
    four-player table that is three opponents. Proved by setting the seats to one:
    the swap that adds Mastermind must then be worth fewer cards. Measured
    2026-10-04: Hallcreeper -> Mastermind +0.62 extra cards by T8 at three seats,
    -0.17 at one."""
    from manamap.pilot import diagnostic, goldfish_turn, try_swap
    swap = [("Silent Hallcreeper", "Faerie Mastermind")]

    def gain():
        a = diagnostic.run("sharknado", iterations=1500, seed=5, quiet=True)
        doc, _, _, _ = try_swap.apply_swaps("sharknado", None, swap)
        b = diagnostic.run_on(doc, "sharknado", iterations=1500, seed=5, quiet=True)
        return (b["steam"]["extra_cards_by_turn"]["8"]["rate"]
                - a["steam"]["extra_cards_by_turn"]["8"]["rate"])

    three = gain()
    monkeypatch.setattr(goldfish_turn, "GOLDFISH_OPPONENTS", 1)
    one = gain()
    assert three > one, (three, one)
