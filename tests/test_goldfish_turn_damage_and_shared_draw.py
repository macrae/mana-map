"""Damage on THEIR turn from a permanent, a symmetric draw engine, and "that much"
damage at any target — the three readers the DeckCheck sweep on sharknado needed
(2026-10-18).

Before this, Spider-Man, New Champion ("whenever you discard one or more cards,
deals that much damage to any target") read as a 3/3 body: `_EVENT_THAT_MUCH_RE`
knew only Magmakin's "to each opponent". Fevered Visions read as nothing at all:
its draw is worded "each player's end step, that player draws" (the draw path
wanted "your … draw"), and no channel priced a permanent that deals damage at
the start of the opponent's step. Both were staged on torbran-v1 as INVISIBLE.
"""

import copy

from conftest import requires_data, requires_deck

from manamap.pilot import card_pool, goldfish


def _card(name, type_line="Enchantment"):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return {"name": name, "oracle_text": text, "type_line": type_line}


@requires_data
def test_that_much_damage_at_any_target_is_one_per_card_in_one_batch():
    e = goldfish.event_payoffs(_card("Spider-Man, New Champion", "Legendary Creature"))
    assert e["per_discard_damage"] == 1 and e["per_discard_batch"] is True
    # Magmakin ("to each opponent") reads exactly as before; Brallin is per card, not a batch.
    m = goldfish.event_payoffs(_card("Magmakin Artillerist", "Creature"))
    assert m["per_discard_damage"] == 1 and m["per_discard_batch"] is True
    b = goldfish.event_payoffs(_card("Brallin, Skyshark Rider", "Legendary Creature"))
    assert b["per_discard_damage"] == 1 and b["per_discard_batch"] is False


@requires_data
def test_a_symmetric_step_draw_is_our_recurring_draw_and_their_extra_draw():
    """The 2026-10-18 sweep: 11 corpus cards say "each player's <step>, that player
    draws". The unconditional ones read for both sides; the conditional and the
    loot stay unread, as they were."""
    fv = goldfish.draw_profile(_card("Fevered Visions"))
    assert fv["recurring_draw"] == 1 and fv["recurring_each_player_draw"] == 1
    assert goldfish.draw_profile(_card("Font of Mythos", "Artifact"))["recurring_draw"] == 2
    assert goldfish.draw_profile(_card("Kami of the Crescent Moon", "Legendary Creature"))["recurring_each_player_draw"] == 1
    # ours alone stays ours alone
    pa = goldfish.draw_profile(_card("Phyrexian Arena"))
    assert pa["recurring_draw"] == 1 and pa["recurring_each_player_draw"] == 0
    # "if this artifact is untapped" / "if that player has no cards in hand" — unread
    assert goldfish.draw_profile(_card("Howling Mine", "Artifact"))["recurring_draw"] == 0
    assert goldfish.draw_profile(_card("Ghirapur Orrery", "Artifact"))["recurring_draw"] == 0
    # "draws an additional card, then discards a card" — a loot this path cannot pay
    assert goldfish.draw_profile(_card("Anvil of Bogardan", "Artifact"))["recurring_draw"] == 0
    # THE SUBJECT DECIDES WHOSE DRAW IT IS (the 35-card sweep, 2026-10-18)
    sc = goldfish.draw_profile(_card("Scrawling Crawler", "Artifact Creature"))    # "your upkeep, each player draws"
    assert sc["recurring_draw"] == 1 and sc["recurring_each_player_draw"] == 1
    mf = goldfish.draw_profile(_card("Master of the Feast", "Creature"))           # "your upkeep, each opponent draws"
    assert mf["recurring_draw"] == 0 and mf["recurring_each_player_draw"] == 1
    assert goldfish.draw_profile(_card("Midnight Oil"))["recurring_draw"] == 1   # "your draw step, draw an additional card"
    # "each player MAY draw" was an overread before this (The Second Doctor, in heliod)
    assert goldfish.draw_profile(_card("The Second Doctor", "Legendary Creature"))["recurring_draw"] == 0
    # a Class level is bought, never free; a card that discards your hand at end step is refused whole
    assert goldfish.draw_profile(_card("Bandit's Talent"))["recurring_draw"] == 0
    assert goldfish.draw_profile(_card("Grafted Skullcap", "Artifact"))["recurring_draw"] == 0


@requires_data
def test_damage_at_the_start_of_their_step_reads_with_its_hand_condition():
    """19 corpus cards; only an unconditional one, or one whose only condition is
    a hand size this model can check, is read. "unless" and any other "if" stay unread."""
    fv = goldfish.event_payoffs(_card("Fevered Visions"))
    assert fv["per_opponent_turn_damage"] == 2 and fv["per_opponent_turn_damage_hand_min"] == 4
    sv = goldfish.event_payoffs(_card("Sulfuric Vortex"))
    assert sv["per_opponent_turn_damage"] == 2 and sv["per_opponent_turn_damage_hand_min"] == 0
    owl = goldfish.event_payoffs(_card("Ebony Owl Netsuke", "Artifact"))
    assert owl["per_opponent_turn_damage"] == 4 and owl["per_opponent_turn_damage_hand_min"] == 7
    for name in ("Mogis, God of Slaughter", "Lavaborn Muse", "Impatience", "Hellfire Mongrel"):
        assert goldfish.event_payoffs(_card(name))["per_opponent_turn_damage"] == 0, name


@requires_data
@requires_deck
def test_the_turn_damage_lands_on_the_clock_and_the_seven_card_condition_waits_for_a_wheel():
    """Same seeds on sharknado, the clause written onto Brallin — the one permanent
    every game casts early, so the test measures the channel and not the cast
    policy (an enchantment at three lands behind two commanders and the wheels
    on this curve, which is the model being honest, not the reader failing).
    An unconditional 2 a turn raises cumulative event damage by turn ten by
    about two per turn Brallin is out; the seven-or-more form raises it LESS,
    because the authored hand is 4 and only a wheel turn refills it to 7.
    Re-introduce the bug (zero `per_opponent_turn_damage` in the turn loop) and
    all three read the same."""
    from manamap.pilot.common import deck_file, load_deck_cards
    import json

    targets = json.loads(deck_file("sharknado", "goldfish_targets.json").read_text())
    doc = load_deck_cards("sharknado")

    def event_damage(extra):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if c["name"] == "Brallin, Skyshark Rider":
                c["oracle_text"] = (c.get("oracle_text") or "") + " " + extra
                break
        else:
            raise AssertionError("Brallin is not in sharknado")
        m = goldfish.run("sharknado", doc=d, iterations=600, seed=11, quiet=True,
                         _band=False, _targets_doc=targets)["metrics"]
        return m["discard"]["mean_cumulative_event_damage_by_turn"]["10"]

    base = event_damage("")
    always = event_damage("At the beginning of each opponent's upkeep, this permanent deals 2 damage to that player.")
    seven = event_damage("At the beginning of each opponent's upkeep, if that player has seven or more cards in hand, this permanent deals 2 damage to that player.")
    assert always > base + 6, (base, always)
    assert base < seven < always, (base, seven, always)
