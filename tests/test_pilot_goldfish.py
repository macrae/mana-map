"""Tests for the goldfish simulator (pilot, tier-2 data-derived evidence)."""

import random

import pytest

from manamap.pilot.goldfish import (
    aggregate,
    body_count,
    build_library,
    classify,
    keepable,
    produced_mana,
    simulate_once,
)

from conftest import requires_deck


def card(name, type_line="Creature — Goblin", cmc=2, oracle="", quantity=1,
         is_commander=False, is_sideboard=False):
    return {
        "name": name, "type_line": type_line, "cmc": cmc, "oracle_text": oracle,
        "quantity": quantity, "is_commander": is_commander, "is_sideboard": is_sideboard,
    }


def synthetic_deck():
    """60ish-card synthetic deck: commander + lands + rocks + bodies + spells."""
    return {"cards": [
        card("Test Commander", "Legendary Creature — Goblin", cmc=4, is_commander=True),
        card("Mountain", "Basic Land — Mountain", cmc=0, quantity=40),
        card("Sol Ring", "Artifact", cmc=1, oracle="{T}: Add {C}{C}.", quantity=1),
        card("Mana Rock", "Artifact", cmc=2, oracle="{T}: Add {R}.", quantity=4),
        card("Token Maker", "Sorcery", cmc=2, oracle="Create two 1/1 red Goblin creature tokens.", quantity=10),
        card("Goblin Grunt", "Creature — Goblin", cmc=1, quantity=20),
        card("Cantrip", "Instant", cmc=1, oracle="Draw a card.", quantity=15),
        card("Payoff", "Sorcery", cmc=2, oracle="Storm", quantity=5),
        card("Sideboard Token", "Card", cmc=0, is_sideboard=True),
    ]}


# ── unit: card classification ──


def test_produced_mana():
    assert produced_mana("{T}: Add {C}{C}.") == 2
    assert produced_mana("{T}: Add {R}.") == 1
    assert produced_mana("Sacrifice a Goblin: Add {R}.") == 0
    assert produced_mana("Draw a card.") == 0
    assert produced_mana(None) == 0


def test_body_count():
    assert body_count(card("X", "Creature — Goblin")) == 1
    assert body_count(card("X", "Sorcery", oracle="Create two 1/1 red Goblin creature tokens.")) == 2
    assert body_count(card("X", "Creature — Goblin", oracle="When this enters, create a Treasure token.")) == 2
    assert body_count(card("X", "Sorcery", oracle="Create three 1/1 red Goblin creature tokens.")) == 3
    assert body_count(card("X", "Instant", oracle="Draw a card.")) == 0


def test_classify_land_and_creature_land():
    assert classify(card("Mountain", "Basic Land — Mountain"))["is_land"] is True
    assert classify(card("Grunt", "Creature — Goblin"))["is_land"] is False


def test_build_library_excludes_commander_and_sideboard():
    library, commanders = build_library(synthetic_deck())
    names = {c["name"] for c in library}
    assert "Test Commander" not in names
    assert "Sideboard Token" not in names
    assert len(commanders) == 1
    assert len(library) == 95  # 40+1+4+10+20+15+5


def test_keepable_land_bounds():
    lands = [classify(card("Mountain", "Basic Land — Mountain"))] * 7
    spells = [classify(card("Grunt"))] * 7
    assert not keepable(spells)          # 0 lands
    assert not keepable(lands)           # 7 lands
    assert keepable(lands[:3] + spells[:4])  # 3 lands


# ── simulation behavior ──


def run_sim(seed=1, iterations=200, max_turn=8, targets=None):
    library, commanders = build_library(synthetic_deck())
    rng = random.Random(seed)
    return [
        simulate_once(rng, library, int(commanders[0]["cmc"]), targets or [], max_turn)
        for _ in range(iterations)
    ]


def test_determinism_same_seed():
    assert run_sim(seed=7) == run_sim(seed=7)


def test_different_seeds_differ():
    assert run_sim(seed=1) != run_sim(seed=2)


def test_commander_cast_turn_bounds():
    for result in run_sim():
        if result["commander_turn"] is not None:
            # A 4-drop cannot be cast before turn 2 even with Sol Ring (T1: 1 land + rock cast, produces next turn).
            assert result["commander_turn"] >= 2


def test_mana_curve_monotone():
    for result in run_sim(iterations=50):
        mana = result["mana_by_turn"]
        assert all(b >= a - 4 for a, b in zip(mana, mana[1:]))  # never collapses (commander spend can dip pool view)
        assert result["bodies_by_turn"] == sorted(result["bodies_by_turn"])  # cumulative


def test_target_assembly_uses_drawn_not_hand():
    targets = [{"label": "token maker drawn", "need": [{"any_of": ["Token Maker"]}]}]
    results = run_sim(iterations=300, targets=targets)
    aggregated = aggregate(results, targets, 8)
    # 10 copies in ~95 cards over 15 draws: should assemble in well over half of games.
    assert aggregated["targets"][0]["assembled_rate"] > 0.5


def test_aggregate_shapes():
    targets = [{"label": "t", "need": [{"any_of": ["Cantrip"]}]}]
    results = run_sim(iterations=100, targets=targets)
    metrics = aggregate(results, targets, 8)
    assert metrics["iterations"] == 100
    assert set(metrics["land_drop_hit_rate_by_turn"]) == {str(t) for t in range(1, 9)}
    assert 0 <= metrics["commander"]["cast_by_turn_6_rate"] <= 1
    assert metrics["opening_hand"]["keep_first_seven_rate"] > 0.5  # 40 lands in 95 keeps most hands


# ── data-gated: real metrics artifact ──


@requires_deck
def test_real_metrics_artifact_consistency():
    import json

    from manamap.config import DECKS_DIR
    from manamap.pilot import goldfish

    path = DECKS_DIR / "goblin-storm" / "goldfish_metrics.json"
    if not path.exists():
        pytest.skip("goldfish_metrics.json not generated yet")
    doc = json.loads(path.read_text())
    assert doc["meta"]["seed"] == 42
    assert doc["metrics"]["iterations"] == doc["meta"]["iterations"]
    # Regenerating with the same seed must reproduce the committed artifact.
    regenerated = goldfish.run("goblin-storm")
    assert regenerated == doc
