"""Colour, and the two independent methods that agree about it.

`spend()` took a scalar, so a five-colour Ur-Dragon and a mono-green Radagast
with the same land count had identical curves — while `mana_analysis`, one
module over, had always computed that the BINDING colour is available on curve
only 56%-99% of the time (median 82%). The simulation was casting at 100%.
"""

import pytest

from conftest import requires_deck
from manamap.pilot import goldfish
from conftest import ROOT

F = frozenset


@pytest.mark.parametrize("cost,want", [
    ("{2}{W}{W}", ["W", "W"]),
    ("{B}{B}{B}", ["B", "B", "B"]),
    ("{3}", []),
    # A HYBRID IS ONE PIP PAYABLE TWO WAYS, not half a pip to each side.
    # `manabase.count_pips` half-charges it and is right to — that answers
    # "how big a base does this need", where a full pip to both over-builds.
    # Half a pip is not a thing you can pay, so casting needs its own reader.
    ("{W/U}{R}", ["UW", "R"]),
    # {2/W} is payable with two generic, which a deck with the mana always has,
    # so it never constrains a colour.
    ("{2/W}{G}", ["G"]),
])
def test_a_casting_pip_is_not_a_mana_base_pip(cost, want):
    got = ["".join(sorted(p)) for p in goldfish.cast_pips(cost)]
    assert got == want


@pytest.mark.parametrize("pips,sources,wild,want", [
    ("{B}", ["B", "R"], 0, True),
    ("{B}", ["R", "R"], 0, False),
    ("{B}", ["R"], 1, True),                 # a Treasure is any colour
    ("{W}{W}", ["W"], 0, False),             # one source pays one pip
    ("{W}{U}", ["WU", "W"], 0, True),
    ("{W}{U}", ["W", "W"], 0, False),
    ("{W/U}", ["U"], 0, True),               # hybrid, either side
    ("{3}", [], 0, True),                    # generic never constrains
])
def test_can_pay(pips, sources, wild, want):
    assert goldfish.can_pay(goldfish.cast_pips(pips),
                            [F(s) for s in sources], wild) is want


def test_the_most_constrained_pip_is_assigned_first():
    """A greedy that takes pips in written order spends its only dual land on
    the pip a mono source could have paid, and then reports a castable spell as
    uncastable. `{W}{U}` off a WU land and a W land is castable; taking W first
    off the WU land leaves U with nothing."""
    assert goldfish.can_pay(goldfish.cast_pips("{W}{U}"), [F("WU"), F("W")]) is True
    assert goldfish.can_pay(goldfish.cast_pips("{W}{U}"), [F("W"), F("WU")]) is True


@requires_deck
def test_a_mono_colour_deck_is_barely_affected_and_a_five_colour_one_is():
    """The property that makes this a fix rather than a change."""
    mono = [goldfish.run("radagast", iterations=800, quiet=True, model_colors=f)
            ["metrics"]["commander"]["cast_by_turn_6_rate"] for f in (False, True)]
    five = [goldfish.run("ur-dragon", iterations=800, quiet=True, model_colors=f)
            ["metrics"]["commander"]["cast_by_turn_6_rate"] for f in (False, True)]
    assert abs(mono[1] - mono[0]) < 0.02, f"mono-green deck moved: {mono}"
    assert five[1] < five[0], f"five-colour deck did not move: {five}"


@requires_deck
def test_the_simulation_agrees_with_the_closed_form_about_which_decks_are_screwed():
    """THE FIRST EXTERNAL VALIDATION IN THIS SUITE.

    `mana_analysis` computes P(the binding colour is on curve) from a
    hypergeometric over source counts. The simulation shuffles a real library
    and refuses casts it cannot colour. They share no code. Measured across 11
    decks, corr(binding-colour P, relative drop in commander-by-six) = -0.71:
    the decks the closed form says are fine do not move, and the ones it says
    are screwed move most. Two methods agreeing is worth more than either.
    """
    import glob
    import json
    binding, drop = [], []
    for p in sorted(glob.glob(str(ROOT / "data/decks/*/mana_analysis.json"))):
        slug = p.split("/")[2]
        oc = (json.load(open(p)).get("on_curve_probability") or {}).get(
            "with_rocks_and_dorks") or {}
        if not oc:
            continue
        try:
            a = goldfish.run(slug, iterations=600, quiet=True, model_colors=False)
            b = goldfish.run(slug, iterations=600, quiet=True, model_colors=True)
        except FileNotFoundError:
            # ONLY a missing fixture is skippable. A bare `except Exception`
            # made a deck that CRASHES the code under test indistinguishable
            # from one that is absent — the failure this suite exists to see.
            continue
        ra = a["metrics"]["commander"]["cast_by_turn_6_rate"]
        rb = b["metrics"]["commander"]["cast_by_turn_6_rate"]
        if not ra:
            continue
        binding.append(min(oc.values()))
        drop.append((ra - rb) / ra)
    if len(binding) < 8:
        pytest.skip("too few decks with a mana analysis to correlate")
    n = len(binding)
    ma, mb = sum(binding) / n, sum(drop) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(binding, drop))
    den = (sum((x - ma) ** 2 for x in binding)
           * sum((y - mb) ** 2 for y in drop)) ** 0.5
    r = num / den if den else 0
    assert r < -0.4, (
        f"the simulation and the closed form stopped agreeing (r={r:+.2f}). "
        f"One of them is now wrong about colour and this test cannot say which.")


@requires_deck
def test_colour_never_makes_a_deck_faster():
    """A one-directional check: refusing casts can only slow a deck down. If any
    figure improves, the colour path is letting something through it should not."""
    for slug in ("heliod", "sisay", "kianne"):
        a = goldfish.run(slug, iterations=600, quiet=True, model_colors=False)
        b = goldfish.run(slug, iterations=600, quiet=True, model_colors=True)
        assert (b["metrics"]["commander"]["cast_by_turn_6_rate"]
                <= a["metrics"]["commander"]["cast_by_turn_6_rate"] + 0.01), slug
