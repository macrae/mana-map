"""Malakir Bloodwitch drains for the Vampires you control, tokens included (2026-11-11).

"When this creature enters, each opponent loses life equal to the number of Vampires you
control. You gain life equal to the life lost this way." It read as an unmodelled drain on
Edgar's threat-v1 branch: a 4/4 flyer and nothing else. The corpus holds one card in this
shape. The count reads `creature_types` as well as the cast permanents, so Edgar's Vampire
tokens are counted, as the rules count them.

Proved by re-introducing the bug: delete the `_ETB_DRAIN_EQUAL_COUNT_RE` branch in
`drain_profile` and `test_the_entry_drain_scales_with_the_vampires` reads the same drain
for both arms.
"""
import random

from manamap.pilot import goldfish, model_coverage

from conftest import requires_data

MALAKIR = ("Flying, protection from white\nWhen this creature enters, each opponent loses life "
           "equal to the number of Vampires you control. You gain life equal to the life lost "
           "this way.")
THRONE = ("At the beginning of your end step, each opponent loses life equal to the number of "
          "tapped creatures you control.")


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def test_the_shape_reads_with_its_subject():
    d = goldfish.classify(_c("Malakir Bloodwitch", MALAKIR, 5, "Creature — Vampire Shaman",
                             power="4", toughness="4"))["drain"]
    assert (d["drain_etb"], d["gain_etb"], d["scales_with"]) == (1, 1, "Vampire")
    assert d["unmodelled"] is None
    assert "drain" in model_coverage.channels_for(
        goldfish.classify(_c("Malakir Bloodwitch", MALAKIR, 5, "Creature — Vampire Shaman",
                             power="4", toughness="4")))


def test_the_end_step_cousin_is_not_this_shape():
    d = goldfish.classify(_c("Throne", THRONE, 5, "Legendary Artifact"))["drain"]
    assert d["drain_etb"] == 0


def _drain(body_type, iterations=300):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Malakir Bloodwitch", MALAKIR, 5, "Creature — Vampire Shaman", quantity=10,
                power="4", toughness="4"),
             _c("Body", "", 1, body_type, quantity=30, power="1", toughness="1")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    library, _ = goldfish.build_library({"cards": cards})
    rng = random.Random(5)
    runs = [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_drain=True)
            for _ in range(iterations)]
    return sum(sum(r["drain_by_turn"]) for r in runs) / iterations


def test_the_entry_drain_scales_with_the_vampires():
    vampires, humans = _drain("Creature — Vampire"), _drain("Creature — Human")
    assert humans > 0
    assert vampires > humans * 1.5, (vampires, humans)


@requires_data
def test_the_corpus_sweep_is_one_card():
    from manamap.pilot import card_pool
    from manamap.pilot.goldfish_profiles import _ETB_DRAIN_EQUAL_COUNT_RE
    hits = sorted(n for n, t in card_pool.corpus_oracle().items()
                  if t and _ETB_DRAIN_EQUAL_COUNT_RE.search(t))
    assert hits == ["Malakir Bloodwitch"], hits
