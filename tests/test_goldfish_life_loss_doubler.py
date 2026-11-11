"""Bloodletter of Aclazotz doubles an opponent's life loss on our turn (2026-11-11).

"If an opponent would lose life during your turn, they lose twice that much life instead."
Damage causes life loss (CR 120.3a), so the swing, the pings and the drain all double on our
turn. It read as a 2/4 flyer whose replacement `model_coverage` named as unread ("partial").
The corpus holds one card in this shape; Selenia's doubling is a Curse token made on her
death and is not read.

Proved by re-introducing the bug: make `life_loss_mult *= 2` a no-op and
`test_bloodletter_doubles_the_drain_and_the_damage` fails.
"""
import random

from manamap.pilot import goldfish, model_coverage
from manamap.pilot.goldfish_profiles import life_loss_doubler

from conftest import requires_data

BLOODLETTER = ("Flying\nIf an opponent would lose life during your turn, they lose twice that "
               "much life instead.")
DRAINER = "At the beginning of your upkeep, each opponent loses 1 life and you gain 1 life."


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def _run(with_bloodletter, iterations=300):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Bloodletter" if with_bloodletter else "Blank", BLOODLETTER if with_bloodletter
                else "Flying", 4, "Creature — Vampire Demon", quantity=10, power="2",
                toughness="4"),
             _c("Drainer", DRAINER, 2, "Enchantment", quantity=10),
             _c("Body", "", 2, "Creature — Vampire", quantity=20, power="2", toughness="2")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    library, _ = goldfish.build_library({"cards": cards})
    rng = random.Random(3)
    runs = [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_drain=True)
            for _ in range(iterations)]
    return (sum(sum(r["drain_by_turn"]) for r in runs) / iterations,
            sum(sum(r["damage_by_turn"]) for r in runs) / iterations)


def test_the_clause_reads_and_is_marked_read():
    assert life_loss_doubler({"oracle_text": BLOODLETTER})
    p = goldfish.classify(_c("Bloodletter of Aclazotz", BLOODLETTER, 4,
                             "Creature — Vampire Demon", power="2", toughness="4"))
    assert p["life_loss_doubler"] is True
    assert model_coverage.unread_replacement(
        _c("Bloodletter of Aclazotz", BLOODLETTER, 4, "Creature — Vampire Demon"), p) is None


def test_bloodletter_doubles_the_drain_and_the_damage():
    (drain_with, dmg_with), (drain_without, dmg_without) = _run(True), _run(False)
    assert drain_without > 0 and dmg_without > 0
    assert drain_with > drain_without * 1.3, (drain_with, drain_without)
    assert dmg_with > dmg_without * 1.3, (dmg_with, dmg_without)


@requires_data
def test_the_corpus_sweep_is_one_card():
    from manamap.pilot import card_pool
    hits = sorted(n for n, t in card_pool.corpus_oracle().items()
                  if t and life_loss_doubler({"oracle_text": t}))
    assert hits == ["Bloodletter of Aclazotz"], hits
