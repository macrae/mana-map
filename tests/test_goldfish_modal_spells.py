"""Modal and escalate spells, discard costs, and damage to a player's face.

Before 2026-10-11 the sharknado package read as blanks where it mattered:
Collective Defiance's wheel was behind "Target player", its 3 to the face and
Incendiary Command's 4 had no channel at all (a Lava Spike read as nothing),
Big Score's draw was refused for its discard cost, and every wheel — even
"discard YOUR hand" — refilled the opponent.
"""

import copy

from conftest import requires_data, requires_deck

from manamap.pilot import card_pool, goldfish


def _card(name, type_line):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return {"name": name, "oracle_text": text, "type_line": type_line,
            "mana_cost": "{2}{R}"}


@requires_data
def test_modal_and_targeted_wheels_are_read_and_who_they_hit():
    cd = goldfish.draw_profile(_card("Collective Defiance", "Sorcery"))
    assert cd["wheel_draws"] == -1 and cd["wheel_symmetric"] is False
    wf = goldfish.draw_profile(_card("Wheel of Fortune", "Sorcery"))
    assert wf["wheel_symmetric"] is True
    fury = goldfish.draw_profile(_card("Khorvath's Fury", "Sorcery"))
    assert fury["wheel_plus"] == 1 and fury["wheel_symmetric"] is False
    hex_ = goldfish.draw_profile(_card("Hex Magic", "Sorcery — Arcane"))
    assert hex_["wheel_draws"] == -1 and hex_["wheel_shuffles"] is True   # exile: no discard


@requires_data
def test_a_discard_cost_is_payable_and_a_sacrifice_is_not():
    bs = goldfish.draw_profile(_card("Big Score", "Instant"))
    assert bs["spell_draw"] == 2 and bs["spell_discard"] == 1
    vr = goldfish.draw_profile(_card("Village Rites", "Instant"))
    assert vr["spell_draw"] == 0 and vr["unmodelled"] == "Village Rites"
    bi = goldfish.draw_profile(_card("Burning Inquiry", "Sorcery"))
    assert bi["spell_draw"] == 3 and bi["spell_each_player_draws"] == 3


@requires_data
def test_face_damage_and_the_choose_one_rule():
    assert goldfish.spell_face_damage(_card("Lava Spike", "Sorcery — Arcane"))["amount"] == 3
    assert goldfish.spell_face_damage(_card("Fateful Showdown", "Instant"))["kind"] == "hand"
    ic = goldfish.spell_face_damage(_card("Incendiary Command", "Sorcery"))
    assert ic["amount"] == 4 and ic["choose_one"] is False
    bc = goldfish.spell_face_damage(_card("Boros Charm", "Instant"))
    assert bc["amount"] == 4 and bc["choose_one"] is True
    # A creature-only mode is not face damage.
    assert goldfish.spell_face_damage(_card("Cathartic Pyre", "Instant"))["kind"] is None


@requires_data
@requires_deck
def test_collective_defiance_deals_and_wheels_in_a_game():
    """Same seeds on the sharknado list: one Sorcery becomes Collective
    Defiance's text or a blank. Damage by turn ten and cards discarded by turn
    eight must both rise. Re-introduce either bug (drop "target player" from
    `_WHEEL_RE`, or `_face_spell`) and one of them reads identical."""
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")
    target = next(c for c in doc["cards"] if c.get("type_line") == "Sorcery"
                  and goldfish.draw_profile(c)["wheel_draws"] == 0)
    text = card_pool.corpus_oracle()["Collective Defiance"]

    def run(t):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if c["name"] == target["name"]:
                c["oracle_text"] = t
                c["mana_cost"] = "{1}{R}{R}"
                c["cmc"] = 3
        m = goldfish.run("sharknado", doc=d, iterations=1500, seed=7, quiet=True,
                         _band=False)["metrics"]
        return (m["combat"]["mean_damage_by_turn"]["10"],
                m["discard"]["mean_cards_discarded_by_turn"]["8"])

    wheel_only = text.split("\u2022 Collective Defiance deals 3")[0]
    assert wheel_only != text, "the face mode was not found to strip"
    blank, wheel, live = run("Flash"), run(wheel_only), run(text)
    assert wheel[1] > blank[1], ("the wheel half must discard", blank, wheel)
    assert live[0] > wheel[0], ("the face half must add damage", wheel, live)
