"""Damage replacements that add — Torbran, Thane of Red Fell.

Before 2026-10-11 Torbran read as a vanilla 2/4 and `model-coverage` called him
SEEN. "That much damage plus 2" is per damage INSTANCE: Brallin's one-a-discard
ping becomes three a discard, so a seven-card wheel is 21 to each opponent, and
the model priced it at nothing.
"""

import copy

from conftest import requires_data, requires_deck

from manamap.pilot import card_pool, goldfish


def _card(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return {"name": name, "oracle_text": text, "type_line": ""}


@requires_data
def test_the_scopes_are_read():
    assert goldfish.damage_bonus(_card("Torbran, Thane of Red Fell")) == {"amount": 2, "scope": "red"}
    assert goldfish.damage_bonus(_card("Thor, Asgard's Avenger")) == {"amount": 1, "scope": "any"}
    assert goldfish.damage_bonus(_card("Tor Wauki the Younger")) == {"amount": 1, "scope": "noncombat"}


@requires_data
def test_what_this_model_cannot_buy_is_refused():
    # A Class's level 3 and a Room's locked door grant nothing until bought.
    talent = dict(_card("Artist's Talent"), type_line="Enchantment — Class")
    assert goldfish.damage_bonus(talent)["amount"] == 0
    # Spells only, not every red source.
    assert goldfish.damage_bonus(_card("Pyromancer's Gauntlet"))["amount"] == 0


def test_a_bonus_applies_to_damage_and_never_to_life_loss():
    red_pinger = {"colors": ["R"], "deals": True}
    red_drain = {"colors": ["R"], "deals": False}
    blue_pinger = {"colors": ["U"], "deals": True}
    torbran = [(2, "red")]
    assert goldfish.bonus_for(red_pinger, torbran) == 2
    assert goldfish.bonus_for(red_drain, torbran) == 0
    assert goldfish.bonus_for(blue_pinger, torbran) == 0
    assert goldfish.bonus_for(blue_pinger, [(1, "noncombat")]) == 1
    assert goldfish.bonus_for(blue_pinger, [(1, "noncombat")], combat=True) == 0


@requires_data
def test_magmakin_is_one_instance_per_batch():
    """It was invisible: "deals that much damage to each opponent" had no
    number for the reader to find."""
    ev = goldfish.event_payoffs(_card("Magmakin Artillerist"))
    assert ev["per_discard_damage"] == 1 and ev["per_discard_batch"] is True
    assert ev["unmodelled"] is None
    # Brallin's trigger is per card, not per batch.
    assert goldfish.event_payoffs(_card("Brallin, Skyshark Rider"))["per_discard_batch"] is False


@requires_data
@requires_deck
def test_torbran_turns_brallins_ping_into_three():
    """Same seeds, one card's text changed: a creature that is either a vanilla
    or Torbran. Discard-event damage by turn eight must rise by well over half.
    Re-introduce the bug (drop the `_b` terms in the event block) and the arms
    read identical."""
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")
    target = next(c for c in doc["cards"]
                  if "Creature" in (c.get("type_line") or "")
                  and "R" in (c.get("colors") or [])
                  and c.get("name") not in ("Shabraz, the Skyshark", "Brallin, Skyshark Rider")
                  and (c.get("cmc") or 0) <= 4)

    def event_damage(text):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if c["name"] == target["name"]:
                c["oracle_text"] = text
        m = goldfish.run("sharknado", doc=d, iterations=1500, seed=7,
                         quiet=True, _band=False)["metrics"]
        return m["discard"]["mean_cumulative_event_damage_by_turn"]["8"]

    vanilla = event_damage("Trample")
    torbran = event_damage("If a red source you control would deal damage to an opponent "
                           "or a permanent an opponent controls, it deals that much damage "
                           "plus 2 instead.")
    assert torbran > vanilla * 1.05, (vanilla, torbran)


@requires_data
@requires_deck
def test_torbran_boosts_red_attackers_too():
    """Combat damage from a red source is +2 per attacker per strike (the
    battlefield entry carries its colours since 2026-10-11). The combat block's
    own damage series excludes the discard pings, so it can only rise through
    red attackers. Re-introduce the gap (return 0 from `_combat_bonus`) and the
    arms read identical."""
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")
    # Named, not "the first red creature at four or less": on v2.0.0 that became
    # Blade Historian, whose own text is the team's double strike, so rewriting
    # it moved the base and the lift read +0.94 (2026-10-26). Glint-Horn
    # Buccaneer is a plain red attacker; its lift measured +1.74.
    target = next(c for c in doc["cards"] if c["name"] == "Glint-Horn Buccaneer")
    assert "R" in (target.get("colors") or [])

    def combat_damage(text):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if c["name"] == target["name"]:
                c["oracle_text"] = text
        m = goldfish.run("sharknado", doc=d, iterations=1500, seed=7,
                         quiet=True, _band=False)["metrics"]
        return m["combat"]["mean_damage_by_turn"]["10"]

    vanilla = combat_damage("Trample")
    torbran = combat_damage("If a red source you control would deal damage to an opponent "
                            "or a permanent an opponent controls, it deals that much damage "
                            "plus 2 instead.")
    # A paired lift on the same seeds: +1.35 damage at 1,500 games (2026-10-18, after
    # Scrawling Crawler's symmetric draw raised the vanilla base to 67.8). Asserted as
    # an absolute margin, not a share of a base that other readers move.
    assert torbran > vanilla + 1.0, (vanilla, torbran)
