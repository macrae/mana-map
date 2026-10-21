"""The `wheel` channel of `pilot_policy.json`: a wheel can be HELD until a commander.

sharknado log 002 (2026-10-20): the pilot's read is that a wheel is a finisher dropped
onto Brallin or Shabraz, never early tempo — and the simulator was wheeling on any
thin hand, which measured a deck flown by a hand he disagrees with. These tests hold
the three claims the design makes: a threshold of zero is the control (byte-identical
to no policy), a threshold of one commander means no wheel resolves before one, and
both wheel doors (the spell and the activated engine) obey the same gate.

Every test here drives the production simulator; none re-derives the rule. Each was
proven by re-introducing the bug it names (see the docstrings).
"""

import pytest

from manamap.pilot import goldfish, pilot_policy as pp

from conftest import patch_model, requires_deck

SLUG = "sharknado"      # the one deck that opts into `model_discard` with two commanders


def _hold(n, counter="commanders_out"):
    return {"rules": [{"id": "wheel-test", "why": "the test's hand",
                       "when": {"channel": "wheel"}, "hold_until": {counter: n}}]}


def _run(monkeypatch, policy, **kw):
    monkeypatch.setattr(pp, "load", lambda slug, branch=None: policy)
    return goldfish.run(SLUG, iterations=120, seed=11, max_turn=10,
                        with_results=True, quiet=True, _band=False, **kw)


# ------------------------------------------------------------------ the vocabulary

def test_a_channel_is_held_only_by_a_counter_its_gate_reads():
    """A wheel held "until other_creatures" would validate against the flat lists
    and then be ignored by the turn loop — a rule that changes nothing and says it
    did. Proven by deleting the `CHANNEL_COUNTERS` check: both calls then pass."""
    with pytest.raises(pp.PolicyError, match="cannot be held until other_creatures"):
        pp.validate(_hold(1, counter="other_creatures"))
    with pytest.raises(pp.PolicyError, match="cannot be held until event_payoffs"):
        pp.validate({"rules": [{"id": "r", "why": "w",
                                "when": {"channel": "spell_token_copy"},
                                "hold_until": {"event_payoffs": 1}}]})
    assert pp.validate(_hold(1))["rules"][0]["hold_until"] == {"commanders_out": 1}
    assert pp.validate(_hold(1, counter="event_payoffs"))
    assert set(pp.CHANNEL_COUNTERS) == set(pp.CHANNELS)
    assert set().union(*pp.CHANNEL_COUNTERS.values()) == set(pp.COUNTERS)


def test_a_forge_only_policy_still_validates():
    """Edgar's file has one `forge` rule and no `when`; the channel check must not
    reach it."""
    doc = {"rules": [{"id": "sac-pref", "why": "w",
                      "forge": {"SACRIFICE_DEFAULT_PREF_ENABLE": True}}],
           "forge": {"SACRIFICE_DEFAULT_PREF_ENABLE": True}}
    assert pp.validate(doc)
    assert pp.hold_thresholds(doc) == {}


def test_the_thresholds_flatten_and_render():
    doc = _hold(1)
    assert pp.hold_thresholds(doc) == {"wheel": {"commanders_out": 1}}
    assert any("hold a wheel until commanders_out >= 1" in ln for ln in pp.render(doc))


# ------------------------------------------------------------------ the control

@requires_deck
def test_a_hold_of_zero_is_byte_identical_to_no_policy(monkeypatch):
    """THE A/A. A hold is an AND-gate layered on the built-in rule, so a declared
    threshold of zero must change nothing but the policy stamp. Proven by reading
    the presence of the key as a threshold of one (`1 if "commanders_out" in
    _hold_wheel else 0`): the two runs then diverge.

    What this test CANNOT see: a bug inside the shared closure that moves both
    arms together (a `>` for a `>=` held every wheel forever in both runs and
    they still matched). That control is the fleet regen after the engine
    change: `git diff -- data/decks` showed only `meta.model_version` moving on
    every deck, and the artifact-freshness tests hold the tracked figures to a
    fresh run from then on."""
    off = _run(monkeypatch, {})
    on = _run(monkeypatch, _hold(0))
    assert "pilot_policy" not in off["meta"] and "pilot_policy" in on["meta"]
    off["meta"].pop("pilot_policy", None); on["meta"].pop("pilot_policy", None)
    assert off["_results"] == on["_results"]
    assert off["metrics"] == on["metrics"]


# ------------------------------------------------------------------ the hold

@requires_deck
def test_held_until_a_commander_no_wheel_resolves_before_one(monkeypatch):
    """With `commanders_out >= 1`, every wheel that fires has a commander out, and
    wheels STILL fire (a rule that silences the channel would read as "held" too).
    Proven by dropping the AND clause from `_wheel_castable`: thin-hand wheels then
    fire with no payoff and `wheels_with_payoff < wheels_cast` in some game."""
    on = _run(monkeypatch, _hold(1))
    rows = on["_results"]
    fired = [r for r in rows if r["wheels_cast_by_turn"][-1] > 0]
    assert len(fired) >= 20, "the hold must delay wheels, not silence them"
    for r in fired:
        assert r["wheels_with_payoff"] == r["wheels_cast_by_turn"][-1]
        first_cmd = min(t for t in (r["commander_turn"], r["partner_turn"])
                        if t is not None)
        assert r["first_wheel_turn"] >= first_cmd


@requires_deck
def test_the_activated_wheel_obeys_the_same_hold(monkeypatch):
    """Jace's Archivist is the second door. Blind the spell wheels so only the
    engine can fire, and the hold must still keep it behind a commander. Proven by
    reverting the activated site to its old inline gate."""
    from manamap.pilot import goldfish_profiles

    real = goldfish_profiles.draw_profile

    def blind(card):
        prof = real(card)
        if prof.get("wheel_draws"):
            prof = dict(prof, wheel_draws=0)
        return prof

    patch_model(monkeypatch, "draw_profile", blind)
    on = _run(monkeypatch, _hold(1))
    fired = [r for r in on["_results"] if r["wheels_cast_by_turn"][-1] > 0]
    assert fired, "the activated engine must still wheel once a commander is out"
    for r in fired:
        assert r["wheels_with_payoff"] == r["wheels_cast_by_turn"][-1]
        first_cmd = min(t for t in (r["commander_turn"], r["partner_turn"])
                        if t is not None)
        assert r["first_wheel_turn"] >= first_cmd


# ------------------------------------------------------------------ the observers

@requires_deck
def test_the_wheel_observers_are_consistent_per_game(monkeypatch):
    """`wheels_cast_by_turn` is cumulative; the first wheel is where the count
    first rises; a finisher is a wheel, so it cannot precede the first one; a wheel
    with a payoff out is a wheel."""
    for policy in ({}, _hold(1)):
        for r in _run(monkeypatch, policy)["_results"]:
            s = r["wheels_cast_by_turn"]
            assert all(b >= a for a, b in zip(s, s[1:]))
            assert len(s) == len(r["dead_wheels_in_hand_by_turn"])
            if s[-1] == 0:
                assert r["first_wheel_turn"] is None and r["finisher_wheel_turn"] is None
            else:
                assert r["first_wheel_turn"] == 1 + next(i for i, v in enumerate(s) if v > 0)
            if r["finisher_wheel_turn"] is not None:
                assert r["finisher_wheel_turn"] >= r["first_wheel_turn"]
            assert 0 <= r["wheels_with_payoff"] <= s[-1]


@requires_deck
def test_under_the_hold_the_clog_rises_and_the_payoff_share_is_total(monkeypatch):
    """The point of the policy, read off the model: holding wheels for a commander
    puts more dead wheels in hand at turn five than firing them on a thin hand, and
    every wheel fired is paid. Direction only, no figure pinned."""
    off = _run(monkeypatch, {})["_results"]
    on = _run(monkeypatch, _hold(1))["_results"]

    def dead5(rows):
        return sum(r["dead_wheels_in_hand_by_turn"][4] for r in rows) / len(rows)

    assert dead5(on) >= dead5(off)
    assert sum(r["wheels_with_payoff"] for r in on) == sum(r["wheels_cast_by_turn"][-1] for r in on)
    assert sum(r["wheels_with_payoff"] for r in off) < sum(r["wheels_cast_by_turn"][-1] for r in off), (
        "without a policy the built-in rule fires some wheels on a thin hand alone; "
        "if this fails the control no longer exercises the clause the hold removes")
