"""Counter replacements — Lae'zel, Hardened Scales, Doubling Season.

Before 2026-10-11 Lae'zel, Vlaakith's Champion read as a vanilla 3/3 and
`model-coverage` called her SEEN. On sharknado every Shabraz draw and Brallin
discard is its own counter event, so "that many plus one" doubles the pair's
growth, and the model priced it at nothing.
"""

import copy

from conftest import requires_data, requires_deck

from manamap.pilot import card_pool, goldfish


def _mod(name):
    text = card_pool.corpus_oracle().get(name)
    assert text, f"{name} is not in the corpus"
    return goldfish.counter_mod({"name": name, "oracle_text": text})


@requires_data
def test_the_two_shapes_are_read():
    assert _mod("Lae'zel, Vlaakith's Champion") == {"plus": 1, "times": 1}
    assert _mod("Hardened Scales") == {"plus": 1, "times": 1}
    assert _mod("Doubling Season") == {"plus": 0, "times": 2}
    assert _mod("Vorinclex, Monstrous Raider") == {"plus": 0, "times": 2}


@requires_data
def test_what_is_not_ours_or_not_general_is_refused():
    # Zabaz's +1 applies to modular triggers only.
    assert _mod("Zabaz, the Glimmerwasp") == {"plus": 0, "times": 1}
    # A damage replacement is not a counter replacement.
    assert _mod("Torbran, Thane of Red Fell") == {"plus": 0, "times": 1}
    assert _mod("Solemnity") == {"plus": 0, "times": 1}


def test_add_before_multiply_and_zero_stays_zero():
    assert goldfish.apply_counter_mod(1, 1, 1) == 2
    assert goldfish.apply_counter_mod(1, 1, 2) == 4      # (1+1)x2, the pilot's order
    assert goldfish.apply_counter_mod(0, 1, 2) == 0      # no event, nothing to replace


@requires_data
@requires_deck
def test_laezel_grows_the_commanders_in_a_game():
    """Same seeds, one card's text changed: a 3-mana creature that is either a
    vanilla flyer or Lae'zel. The commanders' counters by turn eight must rise.
    Re-introduce the bug (drop `counter_mod` from `classify` or stop applying it
    at the event block) and the two arms read identical."""
    from manamap.pilot.common import load_deck_cards

    doc = load_deck_cards("sharknado")
    target = next(c for c in doc["cards"]
                  if "Creature" in (c.get("type_line") or "")
                  and c.get("name") not in ("Shabraz, the Skyshark", "Brallin, Skyshark Rider")
                  and (c.get("cmc") or 0) == 3)

    def counters(text):
        d = copy.deepcopy(doc)
        for c in d["cards"]:
            if c["name"] == target["name"]:
                c["oracle_text"] = text
        m = goldfish.run("sharknado", doc=d, iterations=1500, seed=7,
                         quiet=True, _band=False)["metrics"]
        return m["discard"]["mean_commander_counters_by_turn"]["8"]

    vanilla = counters("Flying")
    laezel = counters("If you would put one or more counters on a creature or "
                      "planeswalker you control or on yourself, put that many plus one "
                      "of each of those kinds of counters on that permanent or player "
                      "instead.")
    assert laezel > vanilla * 1.05, (vanilla, laezel)
