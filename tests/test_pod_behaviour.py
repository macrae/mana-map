"""The opponent-gated blind spot, and the estimate that narrows it."""

import pytest

from manamap.sim import pod_behaviour as pb


def test_a_per_draw_trigger_beats_a_second_spell_trigger_against_this_pod():
    """The finding that reversed a recommendation.

    Reading the cards alone, a 2-mana "second card each turn" tax looks like
    better value than a 4-mana "whenever an opponent draws" one. The pod's own
    games say otherwise, and the gap is not close.
    """
    tithe = pb.rate_for("Whenever an opponent draws a card, that player may pay {2}")
    tax = pb.rate_for("Whenever an opponent casts their second spell each turn")
    assert tithe["per_round"] == pytest.approx(3.0)
    assert tax["per_round"] < 1.0
    assert tithe["per_round"] > 4 * tax["per_round"]


def test_a_second_draw_is_bounded_rather_than_estimated():
    """Forge does not log draws, so this one has no figure — and says so.

    Reporting a number here would be inventing the measurement that is missing,
    which is the failure `model_treasures`' absent-not-zeroed rule exists for.
    """
    est = pb.rate_for("Whenever an opponent draws their second card each turn")
    assert est["per_round"] is None
    assert est["bound"]


def test_an_ungated_card_gets_no_estimate():
    assert pb.rate_for("Destroy target creature.") is None
    assert pb.rate_for("") is None
    assert pb.rate_for(None) is None


def test_the_constants_cannot_outlive_their_evidence():
    """Re-derived from the logs where they exist, so a corpus of new runs that
    moved the pod's behaviour fails here rather than sitting behind a stale
    constant — the rule `BROAD_GROUP` already follows.
    """
    obs = pb.observed()
    if obs is None:
        pytest.skip("sim logs are gitignored; nothing to re-derive against")
    assert obs["turns"] > 500, obs
    for key in ("casts_per_turn", "second_spell_rate"):
        assert obs[key] == pytest.approx(pb.POD[key], abs=0.05), (key, obs, pb.POD)


def test_no_logs_means_no_observation_rather_than_a_zero():
    assert pb.observed(logs=[]) is None
