"""The magnitude axes, and why a membership axis needed them.

A goldfish target asks whether a card was DRAWN. So the ninth member of a
component raises assembly by the same amount whichever card it is — measured on
ur-dragon's treasure branch, all eight declared multipliers returned the
identical +0.039. True, and no answer at all to the question the pilot asked.
"""

import copy

import pytest

from conftest import requires_deck
from manamap.pilot import candidates, diagnostic

DOUBLER = ("If an effect would create one or more tokens under your control, "
           "it creates twice that many of those tokens instead.")


def _card(name, text, cmc=2, type_line="Artifact", qty=1):
    return {"name": name, "oracle_text": text, "cmc": cmc,
            "type_line": type_line, "quantity": qty}


def _doc(extra):
    cards = [{"name": "Mountain", "oracle_text": "", "cmc": 0, "quantity": 32,
              "type_line": "Basic Land — Mountain"},
             {"name": "Cmdr", "oracle_text": "", "cmc": 4, "is_commander": True,
              "type_line": "Legendary Creature — Dragon"},
             _card("Engine", "At the beginning of your upkeep, create three "
                             "Treasure tokens.", qty=12)]
    return {"cards": cards + extra, "decklist_sha256": "x" * 64}


TARGETS = [{"label": "MULTIPLIER", "required": True,
            "need": [{"any_of": ["Doubler", "Blank"]}]}]


def _read(doc, iterations=600):
    return diagnostic.run_on(doc, "ur-dragon", iterations=iterations,
                             quiet=True, targets=TARGETS)


def test_there_is_exactly_one_combat_axis():
    """MEASURED THE SAME DAY IT SHIPPED AS THREE.

    Across 13 decks at a uniform harness: power@6 vs damage@8 r=+0.97, vs
    kill@10 r=+0.98; damage@8 vs kill@8 r=+0.92. Three axes that rank
    identically are a trap rather than a redundancy — sweep on one, sweep on
    another, get the same order, read it as confirmation. The hoard is NOT part
    of that family (r=0.08 to 0.25), which is what makes it a real second axis.
    """
    combat = [a for a in candidates.AXES
              if candidates.AXIS_NEEDS.get(a) == "model_combat"]
    assert combat == ["damage_8"], combat
    treasure = [a for a in candidates.AXES
                if candidates.AXIS_NEEDS.get(a) == "model_treasures"]
    assert treasure, "the hoard axis went with them; it is independent"


def test_the_report_still_shows_every_series_and_names_the_correlation():
    """Dropping them as AXES must not drop them as READINGS — a pilot wants the
    kill clock even though it cannot be ranked on. Same shape as the mana note.
    """
    got = _read(_doc([_card("Doubler", DOUBLER, qty=8)]))
    assert "board_power_by_turn" in got["output"]
    assert "kill_by_turn" in got["output"]
    assert "one dimension" in diagnostic.COMBAT_READINGS_ARE_CORRELATED


def test_the_axis_registry_holds_both_kinds():
    assert set(candidates.MAGNITUDE_AXES) <= set(candidates.AXES)
    for axis in candidates.MAGNITUDE_AXES:
        assert candidates.AXES[axis][0] == "output", axis
        assert axis in candidates.AXIS_NEEDS, axis


def test_a_magnitude_axis_separates_two_cards_a_membership_axis_cannot():
    """THE CONTROL. Two cards, one component, wildly different magnitudes.

    `Doubler` doubles every Treasure event; `Blank` does nothing. Both satisfy
    the declared component identically, so a membership reading must call them
    equal — that is correct, and it is why it cannot answer "which one".
    The magnitude reading must not call them equal.
    """
    dbl = _read(_doc([_card("Doubler", DOUBLER, qty=8),
                      _card("Blank", "Draw a card.", qty=8)]))
    # Membership: the component is satisfied either way.
    assert dbl["engine"]["available"] is True

    with_doubler = _read(_doc([_card("Doubler", DOUBLER, qty=16)]))
    with_blank = _read(_doc([_card("Blank", "Draw a card.", qty=16)]))
    a = with_doubler["output"]["hoard_by_turn"]["8"]["rate"]
    b = with_blank["output"]["hoard_by_turn"]["8"]["rate"]
    assert a > b * 1.5, (
        f"the magnitude axis could not tell a doubler from a blank: {a} vs {b}. "
        f"If these agree, the axis is not doing what it claims and the eight "
        f"identical multiplier deltas will simply reappear wearing a new name.")


def test_absent_is_absent_and_names_the_flag_rather_than_reading_zero():
    """A deck that opts into neither model has no hoard and no clock. The key is
    missing, not 0.0 — a zero is a measurement nobody made — and the refusal
    names the flag, because a bare 'no reading' sends the pilot hunting a bug."""
    got = diagnostic.run_on(_doc([]), "heliod", iterations=200, quiet=True,
                            targets=[])
    # heliod declares neither flag, so the block refuses and says why.
    assert got["output"]["available"] is False
    assert "model_treasures" in got["output"]["why"]

    with pytest.raises(SystemExit) as e:
        candidates.sweep("heliod", ["Sol Ring"], axis="hoard_6")
    assert "model_treasures" in str(e.value)
    assert "not a zero" in str(e.value)


@requires_deck
def test_the_declared_multipliers_do_not_all_read_alike(tmp_path):
    """The real thing, on the real branch: remove each declared multiplier and
    the hoard must not come back identical eight times."""
    from manamap.pilot.common import load_deck_cards
    doc = load_deck_cards("ur-dragon", branch="treasure-v2")
    names = ["Anointed Procession", "Mondrak, Glory Dominus", "Jolene, the "
             "Plunder Queen", "Academy Manufactor"]
    got = []
    for name in names:
        d2 = copy.deepcopy(doc)
        d2["cards"] = [c for c in d2["cards"] if c["name"] != name]
        r = diagnostic.run_on(d2, "ur-dragon", branch="treasure-v2",
                              iterations=1500, quiet=True)
        got.append(r["output"]["hoard_by_turn"]["10"]["rate"])
    assert max(got) - min(got) > 0.1, f"all four read alike: {got}"


@requires_deck
def test_the_magnitude_means_agree_with_the_goldfish_that_owns_them():
    """Derived from the per-iteration rows so they can carry an interval — and
    a mean with no dispersion beside it cannot produce one. But the goldfish
    stays the owner of the figure, so the two must agree; if they ever drift,
    this document is publishing a second opinion under the first one's name."""
    from manamap.pilot import goldfish
    got = goldfish.run("ur-dragon", branch="treasure-v2", with_results=True,
                       quiet=True, iterations=400, seed=diagnostic.HARNESS["seed"],
                       max_turn=diagnostic.HARNESS["max_turn"])
    block = diagnostic.output(got)
    theirs = got["metrics"]["treasure"]["mean_treasures_in_hoard_by_turn"]
    for turn, cell in block["hoard_by_turn"].items():
        assert abs(cell["rate"] - theirs[turn]) < 0.001, (turn, cell, theirs[turn])


def test_every_magnitude_figure_carries_an_interval():
    """The whole document's contract. A magnitude axis must not become the one
    number here published bare, and the shape must match the other blocks or
    `candidates._read` and `compare` would each need a special case."""
    got = _read(_doc([_card("Doubler", DOUBLER, qty=8)]))
    for key, series in got["output"].items():
        if key in ("available", "basis", "why"):
            continue
        for turn, cell in series.items():
            assert set(cell) >= {"rate", "ci95", "n"}, (key, turn, cell)
            assert cell["ci95"][0] <= cell["rate"] <= cell["ci95"][1], (key, turn)


def test_an_ablation_without_a_placebo_measures_the_library_shrinking():
    """THE CONTROL THAT CAUGHT A WRONG FINDING.

    Removing N cards makes every remaining card come up sooner, so a group
    removal reads as an improvement on any speed axis whatever the cards did.
    Here the removed cards are inert filler — a correct ablation must call that
    inside the placebo band rather than an effect.
    """
    filler = [_card(f"Filler{i}", "Draw a card.", cmc=5) for i in range(8)]
    # The placebo needs comparable singles to draw FROM, or the control cannot
    # run — which the function reports rather than skipping silently.
    spare = [_card(f"Spare{i}", "Draw a card.", cmc=5) for i in range(10)]
    doc = _doc(filler + spare + [_card("Doubler", DOUBLER, qty=6)])
    got = diagnostic.ablate(doc, "ur-dragon",
                            [c["name"] for c in filler],
                            "output", "hoard_by_turn", "8",
                            iterations=400, targets=TARGETS, placebos=2)
    assert got["available"]
    assert got["placebo"] is not None, "the control did not run"
    assert set(got["placebo"]) >= {"draws", "band", "mean"}
    assert "placebo band" in got["verdict"]


def test_an_ablation_says_so_when_no_placebo_was_possible():
    """Silence here would be the worst outcome: a delta with no control reads
    exactly like a delta with one."""
    doc = _doc([_card("Doubler", DOUBLER, qty=4)])
    got = diagnostic.ablate(doc, "ur-dragon", ["Doubler", "Engine"],
                            "output", "hoard_by_turn", "6",
                            iterations=200, targets=TARGETS)
    assert got["placebo"] is None
    assert "cannot be separated" in got["verdict"]
