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


# ------------------------------------------------------------------ a finisher

def _wheel_deck(commander_cmc):
    """A synthetic 99 with a wheel spell and a Glint-Horn-style discard payoff,
    and a commander whose text pays nothing. `commander_cmc=99` is a commander
    that is never cast; 3 is one that lands on curve."""
    from test_pilot_goldfish import card
    return {"cards": [
        card("Quiet Commander", "Legendary Creature — Human", cmc=commander_cmc,
             is_commander=True),
        card("Mountain", "Basic Land — Mountain", cmc=0, quantity=38),
        card("Buccaneer", "Creature — Pirate", cmc=3, quantity=12,
             oracle="Whenever you discard a card, this creature deals 1 damage to each opponent."),
        # A wheel needs a MANA COST to be read as one (a free wheel is excluded).
        dict(card("Wheel", "Sorcery", cmc=3, quantity=12,
                  oracle="Each player discards their hand, then draws seven cards."),
             mana_cost="{2}{R}"),
        card("Grunt", "Creature — Human", cmc=2, quantity=20),
        card("Cantrip", "Instant", cmc=1, oracle="Draw a card.", quantity=17),
    ]}


def _sim(doc, games=150, seed=5):
    import random
    library, commanders = goldfish.build_library(doc)
    rng = random.Random(seed)
    return [goldfish.simulate_once(rng, library, int(commanders[0]["cmc"]), [], 10,
                                   model_discard=True, model_draw=True)
            for _ in range(games)]


def test_a_finisher_needs_a_commander_not_merely_a_payoff():
    """With the Buccaneer out a wheel is PAID (`wheels_with_payoff`), but with no
    commander on the battlefield it is never a FINISHER. Land the commander and
    finishers appear. Proven by counting any payoff in `_note_wheel`: the
    99-mana commander's games then report finishers."""
    never = _sim(_wheel_deck(99))
    assert sum(r["wheels_with_payoff"] for r in never) > 0, "the payoff must fire wheels"
    assert all(r["finisher_wheel_turn"] is None for r in never)
    assert all(r["commander_turn"] is None for r in never)
    on_curve = _sim(_wheel_deck(3))
    assert any(r["finisher_wheel_turn"] is not None for r in on_curve)
    for r in on_curve:
        if r["finisher_wheel_turn"] is not None:
            assert r["commander_turn"] is not None
            assert r["finisher_wheel_turn"] >= r["commander_turn"]


# ------------------------------------------------------------------ the clog

@requires_deck
def test_dead_wheels_fall_when_wheels_leave_the_list(monkeypatch):
    """Two wheel spells become a Mountain and the clog at turn five falls.
    Direction only — no figure pinned. Proven by counting every nonland in
    `_dead_wheels`: the swap then moves the figure the wrong way or not at all."""
    import copy
    from manamap.pilot.common import load_json, deck_file
    doc = load_json(deck_file(SLUG, "cards.json"))
    monkeypatch.setattr(pp, "load", lambda slug, branch=None: {})
    base = goldfish.run(SLUG, iterations=200, seed=3, max_turn=10, doc=doc,
                        with_results=True, quiet=True, _band=False)
    fewer = copy.deepcopy(doc)
    out = {"Windfall", "Time Reversal"}
    fewer["cards"] = [c for c in fewer["cards"] if c["name"] not in out]
    for c in fewer["cards"]:
        if c["name"] == "Mountain":
            c["quantity"] = int(c.get("quantity") or 1) + len(out)
    assert len([c for c in doc["cards"] if c["name"] in out]) == 2, "both must be in the list"
    less = goldfish.run(SLUG, iterations=200, seed=3, max_turn=10, doc=fewer,
                        with_results=True, quiet=True, _band=False)
    d5 = lambda got: got["metrics"]["discard"]["mean_dead_wheels_in_hand_by_turn"]["5"]
    assert d5(less) < d5(base), (d5(less), d5(base))


# ------------------------------------------------------------------ absent means absent

def test_a_deck_outside_the_discard_channel_has_no_wheel_figures():
    """`metrics.discard` is the only home of the wheel figures and it is absent
    off the channel; the diagnostic's `wheels` block follows it."""
    from manamap.pilot import diagnostic
    from test_pilot_goldfish import synthetic_deck
    import random
    library, commanders = goldfish.build_library(synthetic_deck())
    rows = [goldfish.simulate_once(random.Random(i), library, 4, [], 8) for i in range(20)]
    got = goldfish.aggregate(rows, [], 8)
    assert "discard" not in got
    assert diagnostic.wheels(rows, {"metrics": got}) is None
    assert diagnostic.per_game(rows)["wheels|finisher_by_turn|8"] == [None] * 20


# ------------------------------------------------------------------ the keep rule

def _keep_policy():
    return {"rules": [{"id": "keep-needs-a-body", "why": "log 002",
                       "keep": {"max": {"wheel": 1}, "needs_any": ["body", "ramp"]}}]}


def test_the_keep_vocabulary_is_closed_and_folds_strictest():
    from manamap.pilot import goldfish_library as gl
    assert set(gl.KEEP_CLASS_PREDICATES) == set(pp.KEEP_CLASSES)
    with pytest.raises(pp.PolicyError, match="keep.max.land"):
        pp.validate({"rules": [{"id": "k", "why": "w", "keep": {"max": {"land": 2}}}]})
    with pytest.raises(pp.PolicyError, match="passes every hand"):
        pp.validate({"rules": [{"id": "k", "why": "w", "keep": {"max": {}}}]})
    two = {"rules": [{"id": "a", "why": "w", "keep": {"max": {"wheel": 2}, "needs_any": ["body"]}},
                     {"id": "b", "why": "w", "keep": {"max": {"wheel": 1}, "needs_any": ["ramp"]}}]}
    assert pp.validate(two)
    assert pp.keep_rule(two) == {"max": {"wheel": 1}, "needs_any": ["body", "ramp"]}
    assert pp.keep_rule({}) is None and pp.keep_rule(_hold(1)) is None
    assert any("reported, never enforced" in ln for ln in pp.render(_keep_policy()))


def test_a_hand_of_one_body_and_four_wheels_fails_the_rule_but_is_still_kept():
    """The pilot's doubt in his own words, as a hand: the model keeps it (three
    lands) and the rule says ship it. Proven by routing `keepable` through the
    rule: the first assertion then flips."""
    from manamap.pilot import goldfish_library as gl
    from test_pilot_goldfish import card
    land = gl.classify(card("Mountain", "Basic Land — Mountain", cmc=0))
    body = gl.classify(card("Grunt", "Creature — Human", cmc=2))
    wheel = gl.classify(dict(card("Wheel", "Sorcery", cmc=3,
                                  oracle="Each player discards their hand, then draws seven cards."),
                             mana_cost="{2}{R}"))
    assert wheel["draw"]["wheel_draws"], "the fixture must be read as a wheel"
    rock = gl.classify(card("Rock", "Artifact", cmc=2, oracle="{T}: Add {R}."))
    rule = pp.keep_rule(_keep_policy())
    hand = [land, land, land, body, wheel, wheel, wheel]
    assert gl.keepable(hand) and not gl.keep_rule_passes(hand, rule)
    assert gl.keep_rule_passes([land, land, land, body, wheel, rock, rock], rule)
    assert not gl.keep_rule_passes([land, land, land, wheel, land, land, land], rule), "no body or ramp"
    assert gl.keep_rule_passes([land, land, land, rock, land, land, land], rule), "ramp counts"


@requires_deck
def test_the_keep_rule_is_reported_and_never_enforced(monkeypatch):
    """With a keep rule the per-game mulligans are byte-identical to without,
    the pass rate appears, and without a rule it is absent rather than 1.0.
    Proven by making `keepable` consult the rule."""
    off = _run(monkeypatch, {})
    on = _run(monkeypatch, _keep_policy())
    assert ([r["mulligans"] for r in off["_results"]]
            == [r["mulligans"] for r in on["_results"]])
    assert all(r["keep_rule_pass"] is None for r in off["_results"])
    assert all(r["keep_rule_pass"] in (True, False) for r in on["_results"])
    assert "pilot_keep_rule_pass_rate" not in off["metrics"]["opening_hand"]
    rate = on["metrics"]["opening_hand"]["pilot_keep_rule_pass_rate"]
    assert 0.0 < rate < 1.0, rate
    from manamap.pilot import diagnostic
    assert "kept_hand_passes_pilot_rule" not in diagnostic.steam(off["_results"], off)
    cell = diagnostic.steam(on["_results"], on)["kept_hand_passes_pilot_rule"]
    assert cell["n"] == len(on["_results"]) and abs(cell["rate"] - rate) < 1e-3


# ------------------------------------------------------------------ the diagnostic

@requires_deck
def test_the_diagnostic_carries_the_wheels_block_and_validates():
    from manamap.pilot import diagnostic
    from manamap.pilot.validate_diagnostic import validate
    doc = diagnostic.run(SLUG, iterations=120, seed=7, quiet=True, keep_games=True)
    w = doc["wheels"]
    for key in ("first_wheel_by_turn", "wheels_cast_by_turn", "finisher_by_turn",
                "dead_in_hand_by_turn"):
        assert set(w[key]) >= {"5", "8", "10"}, key
        for cell in w[key].values():
            assert {"rate", "ci95", "n"} <= set(cell)
    assert w["with_payoff"]["n"] > 0
    assert len(doc["_games"]["wheels|finisher_by_turn|8"]) == 120
    assert not validate(doc), validate(doc)
