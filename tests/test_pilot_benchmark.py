"""The standard benchmark: four measures under one frozen configuration.

PRD §9.2 is the whole requirement — "the aggregate is only meaningful if the
simulations are controlled … uncontrolled sim output cannot be aggregated into a
ranking." These tests are mostly about that word CONTROLLED, and about the one
thing the fleet run established: that a single score is not yet honest.
"""

import json

import pytest
from conftest import requires_deck

from manamap.pilot import benchmark

pytestmark = requires_deck


def test_the_harness_is_frozen_and_versioned():
    """A score computed under a different configuration is not comparable to one
    that is not, so the configuration travels with the record and carries a
    version. Changing any of it without bumping `version` silently makes two
    incomparable numbers look like a ranking."""
    for key in ("version", "iterations", "seed", "max_turn",
                "model_treasures", "model_combat"):
        assert key in benchmark.HARNESS, key
    assert benchmark.HARNESS["model_treasures"] is True
    assert benchmark.HARNESS["model_combat"] is True, (
        "the flags must be UNIFORM — reading each deck's own opt-in is the "
        "uncontrolled aggregation §9.2 forbids")


def test_it_overrides_the_decks_own_declaration(tmp_path, monkeypatch):
    """OF TWELVE DECKS WITH A 99, EXACTLY ONE opts into combat.

    Ranking the fleet off their own `goldfish_metrics.json` would compare a deck
    measured with a kill clock against eleven measured without. So the benchmark
    passes its own flags, and this asserts the override reaches the simulation
    rather than being decorative.
    """
    seen = {}

    def fake_run(slug, **kw):
        seen.update(kw)
        return {"metrics": {"land_drop_hit_rate_by_turn": {str(t): 0.5 for t in range(1, 11)},
                            "mean_available_mana_by_turn": {str(t): 3.0 for t in range(1, 11)},
                            "opening_hand": {"keep_first_seven_rate": 0.7,
                                             "mean_mulligans": 0.3}},
                "_results": [{"mana_by_turn": [1] * 10, "land_hits": [True] * 10}] * 2}

    monkeypatch.setattr("manamap.pilot.goldfish.run", fake_run)
    monkeypatch.setattr(benchmark, "response", lambda slug: {"answer_cards": 0})
    monkeypatch.setattr(benchmark, "load_deck_cards", lambda slug: {"decklist_sha256": "x"})

    benchmark.measure("anything")
    assert seen["model_combat"] is True and seen["model_treasures"] is True
    assert seen["seed"] == benchmark.HARNESS["seed"]
    assert seen["iterations"] == benchmark.HARNESS["iterations"]


def test_no_aggregate_score_is_published():
    """§14.1 is OPEN, and the fleet run says leave it open.

    `kill_by_turn_8` ranges 0.001 to 0.405 across the bench — a 400x spread —
    and the bottom of it is heliod and hapatra, whose declared kills are "win
    condition access" and a two-card combo. The goldfish's combat model cannot
    see either, so a weighted sum including speed would rank a combo deck last
    for not attacking. That is not a ranking, it is an archetype filter.
    """
    from manamap.config import DECKS_DIR

    paths = sorted(DECKS_DIR.glob("*/benchmark.json"))
    if not paths:
        pytest.skip("no benchmark records on disk")
    for path in paths:
        doc = json.loads(path.read_text())
        assert doc.get("score") is None, (
            f"{path.parent.name} published an aggregate score — §14.1 is open, "
            f"and speed is not comparable across archetypes")
        assert doc["limits"], "a record with no stated limits"


def test_every_record_states_that_it_is_not_a_win_rate():
    """There is no pod, no opponent and no interaction. `simulate` measures a
    table; this measures a deck. Shown side by side with nothing said, the
    second reads as the first."""
    from manamap.config import DECKS_DIR

    paths = sorted(DECKS_DIR.glob("*/benchmark.json"))
    if not paths:
        pytest.skip("no benchmark records on disk")
    for path in paths:
        limits = " ".join(json.loads(path.read_text())["limits"]).lower()
        assert "not a win rate" in limits, path.parent.name
        assert "no pod" in limits or "no opponent" in limits, path.parent.name


@requires_deck
def test_consistency_is_not_speed_wearing_another_name():
    """MEASURED AND REPLACED. The first version took the spread of the kill-turn
    histogram and correlated with speed at r = 0.78 — an artifact of counting,
    since the spread was computed over the games that killed. A deck killing in
    0.1% of games contributed ten clustered late kills and scored as supremely
    consistent. Exactly backwards.

    It measures mana spread now: every deck, every game, nothing censored.
    """
    import statistics

    from manamap.config import DECKS_DIR

    recs = [json.loads(p.read_text()) for p in sorted(DECKS_DIR.glob("*/benchmark.json"))]
    if len(recs) < 8:
        pytest.skip("too few decks to characterise")
    speed = [r["metrics"]["speed"]["kill_by_turn_8"] for r in recs]
    cons = [r["metrics"]["consistency"]["mana_stdev_turn_five"] for r in recs]
    r = statistics.correlation(speed, cons)
    assert abs(r) < 0.5, (
        f"consistency correlates with speed at {r:+.2f} — it is measuring the "
        f"same thing again, which is what the kill-turn version did at 0.78")


def test_a_benchmark_run_leaves_the_decks_own_goldfish_untouched():
    """The benchmark runs its own configuration; a deck's tracked metrics must
    never move because a benchmark ran.

    The raw per-iteration rows it needs are OPT-IN. The first version returned
    them unconditionally and two freshness tests went red immediately — they
    compare `run()` against the tracked artifact byte for byte, which is exactly
    what they are for.
    """
    from manamap.pilot import goldfish

    plain = goldfish.run("heliod", iterations=50, seed=1)
    assert "_results" not in plain, (
        "the raw rows leaked into the default document, which is compared "
        "against the tracked artifact by two freshness tests")
    rich = goldfish.run("heliod", iterations=50, seed=1, with_results=True)
    assert rich["_results"], "with_results=True returned no rows"
    assert rich["metrics"] == plain["metrics"], "asking for rows changed a figure"
