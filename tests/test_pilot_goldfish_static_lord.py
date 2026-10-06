"""A static lord or anthem pumps the board it stands on (2026-10-05).

Until then `team_anthem` was fed only by +1/+1 COUNTERS, so "Other Vampires you
control get +1/+1" was a vanilla body: `try` read Legion Lieutenant -> Vampire
Interloper as +0.022 damage@T10, a GAIN for cutting the lord. Swept on the corpus
(34,955 cards): 218 read, 172 scoped to a type and 46 to every creature; refused
are colour-, token-, commander- and condition-scoped subjects, and anything inside
a trigger, an activation or "until end of turn".
"""
from conftest import requires_data, requires_deck

from manamap.pilot import goldfish_profiles as gp


def _l(text, type_line="Creature — Vampire"):
    return gp.static_lord({"name": "Test", "type_line": type_line, "oracle_text": text})


def test_the_lord_shapes_are_read():
    assert _l("Other Vampires you control get +1/+1.") == {
        "n": 1, "types": ["Vampires"], "other": True}
    assert _l("First strike\nOther Vampire creatures you control get +1/+1 and have "
              "first strike.")["types"] == ["Vampire"]
    assert _l("Creatures you control get +1/+1.", "Enchantment") == {
        "n": 1, "types": None, "other": False}
    assert _l("Trample\nOther Dinosaurs you control get +1/+1 and have hexproof.")["types"] \
        == ["Dinosaurs"]
    assert _l("Other Wolves and Werewolves you control get +1/+1.")["types"] == [
        "Wolves", "Werewolves"]


def test_what_is_not_a_static_lord_is_refused():
    for text in ("When this creature enters, creatures you control get +1/+1 until end of turn.",
                 "{3}{W}: Creatures you control get +1/+1 until end of turn.",
                 "Other green creatures you control get +1/+1.",
                 "Creature tokens you control get +1/+1.",
                 "Commander creatures you control get +2/+2.",
                 "Attacking creatures you control get +1/+0.",
                 "During your turn, creatures you control get +2/+0.",
                 "Creatures you control get +1/+0 and have vigilance as long as you "
                 "control three or more creatures.",
                 "Other Spirit creatures you control get +0/+1."):     # no power
        assert _l(text) is None, text
    assert _l("Creatures you control get +2/+2 until end of turn.", "Instant") is None


def test_a_type_matches_its_plural_and_a_phrase_matches_whole():
    vamps = _l("Other Vampires you control get +1/+1.")
    assert gp.lord_applies(vamps, "Creature — Vampire Knight")
    assert not gp.lord_applies(vamps, "Creature — Human Soldier")
    elves = _l("Other Elves you control get +1/+1.")
    assert gp.lord_applies(elves, "Creature — Elf Druid")
    spawn = _l("Eldrazi Spawn creatures you control get +2/+1.", "Creature — Eldrazi Drone")
    assert gp.lord_applies(spawn, "Creature — Eldrazi Spawn")
    assert not gp.lord_applies(spawn, "Creature — Eldrazi Drone")


@requires_data
@requires_deck
def test_blinding_the_lords_gives_back_the_phantom_gain(monkeypatch):
    """DRIVEN THROUGH `try`, PROVED BY RE-INTRODUCING THE BUG. With lords read,
    ADDING Legion Lieutenant must earn clearly more damage by turn ten than with
    the reader blinded, which credits the lord as a vanilla 2/2 (the bug).

    It used to CUT Legion Lieutenant, which tied the test to a card edgar v2.0.0
    dropped (2026-10-06). Adding it for a Swamp keeps the claim and survives list
    changes; comparing against the blinded run separates the lord's pump from
    the cost of the land."""
    from manamap.pilot import diagnostic, goldfish_library, try_swap

    swap = [("Swamp", "Legion Lieutenant")]

    def delta():
        a = diagnostic.run("edgar-vampires", iterations=1500, seed=5, quiet=True)
        doc, _, _, _ = try_swap.apply_swaps("edgar-vampires", None, swap)
        b = diagnostic.run_on(doc, "edgar-vampires", iterations=1500, seed=5, quiet=True)
        return (b["output"]["damage_by_turn"]["10"]["rate"]
                - a["output"]["damage_by_turn"]["10"]["rate"])

    read = delta()
    real = goldfish_library.combat_profile

    def blind(card):
        p = real(card)
        p["static_lord"] = None
        return p
    monkeypatch.setattr(goldfish_library, "combat_profile", blind)
    blinded = delta()
    assert read > blinded + 0.5, (
        f"adding a +1/+1 Vampire lord should earn its pump: read {read}, blinded {blinded}")
