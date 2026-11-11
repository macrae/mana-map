"""Teysa Karlov doubles our death triggers (2026-11-11).

"If a creature dying causes a triggered ability of a permanent you control to trigger,
that ability triggers an additional time." Teysa read as a vanilla 2/4 on Edgar's
threat-v1 branch, the card the strategist and the deck-doctor both ranked first for
turning one death into several drains. The corpus holds exactly two cards with the
clause (Teysa Karlov, Drivnod, Carnage Dominus); The Masamune's narrower grant is not it.

Proved by re-introducing the bug: set `death_trigger_mult` back to a constant 1 and
`test_teysa_doubles_the_death_drain` fails, the two arms reading the same drain.
"""
import random

from manamap.pilot import goldfish, model_coverage
from manamap.pilot.goldfish_profiles import death_trigger_doubler

from conftest import requires_data

TEYSA = ("If a creature dying causes a triggered ability of a permanent you control to trigger, "
         "that ability triggers an additional time.\nCreature tokens you control have vigilance "
         "and lifelink.")
ARTIST = "Whenever another creature you control dies, each opponent loses 1 life and you gain 1 life."
MASAMUNE = ('Equipped creature has "If a creature dying causes a triggered ability of this creature '
            'or an emblem you own to trigger, that ability triggers an additional time."')


def _c(name, text, cmc, type_line, quantity=1, power=None, toughness=None):
    return {"name": name, "quantity": quantity, "oracle_text": text, "cmc": float(cmc),
            "type_line": type_line, "mana_cost": "{%d}" % cmc, "keywords": [], "colors": [],
            "color_identity": [], "power": power, "toughness": toughness,
            "is_commander": False, "layout": "normal"}


def _doc(teysa):
    cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
             _c("Artist", ARTIST, 2, "Creature — Vampire", quantity=8, power="0", toughness="1"),
             _c("Teysa" if teysa else "Blank Body", TEYSA if teysa else "", 4,
                "Legendary Creature — Human Advisor", quantity=8, power="2", toughness="4"),
             _c("Body", "", 2, "Creature — Vampire", quantity=20, power="2", toughness="2")]
    cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
    return {"cards": cards}


def _drain(doc, iterations=300):
    library, _ = goldfish.build_library(doc)
    rng = random.Random(11)
    runs = [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_drain=True,
                                   model_sacrifice=True,
                                   model_deaths={"own_per_turn": 1.0, "opponent_per_turn": 0.0})
            for _ in range(iterations)]
    return sum(sum(r["drain_by_turn"]) for r in runs) / iterations


def test_the_clause_reads_on_the_two_corpus_cards_only():
    assert death_trigger_doubler({"oracle_text": TEYSA})
    assert not death_trigger_doubler({"oracle_text": MASAMUNE})
    assert not death_trigger_doubler({"oracle_text": ARTIST})


def test_the_coverage_mirror_reads_the_doubler():
    p = goldfish.classify(_c("Teysa Karlov", TEYSA, 4, "Legendary Creature — Human Advisor",
                             power="2", toughness="4"))
    assert p["death_trigger_doubler"] is True
    assert "sacrifice" in model_coverage.channels_for(p)


def test_teysa_doubles_the_death_drain():
    with_teysa, without = _drain(_doc(True)), _drain(_doc(False))
    assert without > 0, "the payoff must fire at all for the comparison to mean anything"
    assert with_teysa > without * 1.3, (with_teysa, without)


@requires_data
def test_the_corpus_sweep_is_two_cards():
    from manamap.pilot import card_pool
    hits = sorted(n for n, t in card_pool.corpus_oracle().items()
                  if t and death_trigger_doubler({"oracle_text": t}))
    assert hits == ["Drivnod, Carnage Dominus", "Teysa Karlov"], hits


# ── Teysa's second line: creature tokens have lifelink ─────────────────────

MAKER = "When this creature enters, create two 1/1 black Vampire creature tokens."
BOND = "Whenever you gain life, target opponent loses that much life."


def test_teysas_token_lifelink_reads_as_a_grant():
    from manamap.pilot.goldfish_profiles import LIFELINK_TO_TOKENS
    p = goldfish.classify(_c("Teysa Karlov", TEYSA, 4, "Legendary Creature — Human Advisor",
                             power="2", toughness="4"))
    assert p["drain"]["grants_lifelink_to"] == LIFELINK_TO_TOKENS
    # and the grant is not her OWN lifelink: she read as a 2/4 lifelinker until 2026-11-11
    assert p["drain"]["lifelink"] is False


def test_lifelink_tokens_feed_a_gain_converter():
    """Re-introduce the bug (drop the `_tok` clause in the grant loop) and the arms read alike."""
    def run(teysa):
        cards = [_c("Wastes", "({T}: Add {C}.)", 0, "Basic Land", quantity=40),
                 _c("Bond", BOND, 2, "Enchantment", quantity=10),
                 _c("Teysa" if teysa else "Blank", TEYSA if teysa else "Vigilance", 4,
                    "Legendary Creature — Human Advisor", quantity=8, power="2", toughness="4"),
                 _c("Maker", MAKER, 2, "Creature — Vampire", quantity=20, power="1", toughness="1")]
        cards.append(_c("Filler", "", 1, "Sorcery", quantity=99 - sum(c["quantity"] for c in cards)))
        library, _ = goldfish.build_library({"cards": cards})
        rng = random.Random(13)
        runs = [goldfish.simulate_once(rng, library, 3, [], 10, model_combat=True, model_drain=True)
                for _ in range(300)]
        return sum(sum(r["drain_by_turn"]) for r in runs) / 300
    with_teysa, without = run(True), run(False)
    assert with_teysa > without * 1.2, (with_teysa, without)
