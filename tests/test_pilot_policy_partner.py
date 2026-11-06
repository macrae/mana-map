"""The `partner` channel of `pilot_policy.json`: the second commander can be HELD.

Sharknado, 2026-11-06, turning the deck Shabraz-first: "Brallin only comes in to give
sharks trample at the end (and to give us access to lovely red cards!!!)". The simulator
cast Brallin on his own curve, around turn 3.7, ahead of Shabraz at 4.8, so four mana
went into the partner before the shark in nearly every game. A held partner waits for
the commanders the rule counts and is then cast LAST in the main phase, from what the
turn left.

Every test drives the production simulator on the same seeds; each was proven by
re-introducing the bug it names.
"""

import pytest

from manamap.pilot import goldfish, pilot_policy as pp

from conftest import requires_deck

SLUG = "sharknado"      # the fleet's one Partner pair


def _hold(n):
    return {"rules": [{"id": "partner-test", "why": "the test's hand",
                       "when": {"channel": "partner"}, "hold_until": {"commanders_out": n}}]}


def _run(monkeypatch, policy, n=600):
    monkeypatch.setattr(pp, "load", lambda slug, branch=None: policy)
    return goldfish.run(SLUG, iterations=n, seed=11, quiet=True, _band=False)["metrics"]


def test_a_partner_is_held_only_by_the_commanders_it_waits_for():
    with pytest.raises(pp.PolicyError, match="cannot be held until other_creatures"):
        pp.validate({"rules": [{"id": "r", "why": "w", "when": {"channel": "partner"},
                                "hold_until": {"other_creatures": 1}}]})
    assert pp.hold_thresholds(pp.validate(_hold(1))) == {"partner": {"commanders_out": 1}}
    assert "then cast it last" in pp.render(_hold(1))[0]


@requires_deck
def test_a_threshold_of_zero_still_casts_the_partner_last(monkeypatch):
    """Zero commanders is no wait, but the hold still moves the cast to the late slot,
    so the partner never lands EARLIER than with no policy at all."""
    free = _run(monkeypatch, {})
    late = _run(monkeypatch, _hold(0))
    assert late["partner"]["mean_cast_turn"] >= free["partner"]["mean_cast_turn"]


@requires_deck
def test_a_held_partner_waits_for_the_first_commander(monkeypatch):
    """PROVED BY IGNORING THE HOLD: without it Brallin lands around turn 3.7, a full turn
    ahead of Shabraz, in most games."""
    m = _run(monkeypatch, _hold(1))
    assert m["partner"]["mean_cast_turn"] > m["commander"]["mean_cast_turn"]


@requires_deck
def test_holding_the_partner_speeds_the_first_commanders_clock(monkeypatch):
    """The point of the rule: the mana Brallin took on turns three to five goes to the
    shark and the cards around it. Same seeds, same list."""
    by6 = lambda m: m["commander_lethal"]["games_by_turn"]["6"]
    assert by6(_run(monkeypatch, _hold(1))) > by6(_run(monkeypatch, {}))
