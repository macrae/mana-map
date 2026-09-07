"""`deck-branch new` prints what the CHAMPION reads on the objective's axis.

`--objective` takes a bare number and the number has to come from somewhere.
Twice in one session it came from the wrong place:

  board_power_6 >= 2.80    taken from a `goldfish` run, which uses the deck's own
                           seed. `net_change` runs 20260826 and reads the same
                           list at 2.4664. Verdict: NOT RESOLVABLE.
  interaction_6 >= 0.78    taken from a net-change table printed BEFORE the
                           X-spell draw model landed. That model made big X
                           spells eat the whole remaining pool — correct play,
                           correctly leaving less for interaction — and the
                           control fell to 0.6613 underneath a line already
                           written down. Verdict: NOT MET, on a branch that was
                           an IMPROVEMENT on the control it was graded against.

Both objectives stood as written, because a pre-registration that moves when it
is inconvenient is not one. The fix is not a rule in a document: it is printing
the control's current value at the moment the line is chosen, from the harness
that will do the grading.
"""

import pytest

from manamap.pilot import deck_branch

from conftest import requires_data, requires_deck


def _obj(axis, value, op=">="):
    return {"axis": axis, "op": op, "value": value}


def test_an_axis_with_no_diagnostic_cell_says_so_instead_of_raising():
    """A branch must still open. An unanchored line is worse than an anchored
    one and far better than a command that refuses to run."""
    lines = deck_branch.champion_reading("heliod", _obj("no_such_axis", 1.0))
    assert len(lines) == 1 and "cannot anchor" in lines[0]


def test_a_broken_read_degrades_to_a_note(monkeypatch):
    """Defensive on purpose: opening a branch must not depend on the goldfish
    being runnable. A fresh clone with no corpus still gets its branch."""
    from manamap.pilot import diagnostic

    def boom(*a, **k):
        raise RuntimeError("no corpus")
    monkeypatch.setattr(diagnostic, "run", boom)
    lines = deck_branch.champion_reading("heliod", _obj("board_power_6", 2.8))
    assert len(lines) == 1 and "unanchored" in lines[0]


def test_a_line_inside_the_noise_is_called_out(monkeypatch):
    """THE CHECK THAT EARNS ITS PLACE. A threshold closer to the control than
    the run can resolve cannot come back MET or NOT MET on evidence — it comes
    back as whichever side the noise landed on. `teeth-v1` missed its line by
    0.0291 against an MDE of 0.0844 and the harness reported NOT RESOLVABLE,
    hours after the number was chosen. This says it at the moment of choosing.
    """
    from manamap.pilot import diagnostic, net_change
    monkeypatch.setattr(diagnostic, "run", lambda *a, **k: {"_": 1})
    monkeypatch.setattr(net_change, "_cell", lambda *a, **k: {"rate": 2.4664})
    monkeypatch.setattr(diagnostic, "mde", lambda cell: 0.0844)

    tight = deck_branch.champion_reading("heliod", _obj("board_power_6", 2.50))
    assert any("INSIDE THE NOISE" in l for l in tight), tight

    wide = deck_branch.champion_reading("heliod", _obj("board_power_6", 2.80))
    assert not any("INSIDE THE NOISE" in l for l in wide), wide
    assert any("+0.3336" in l for l in wide), wide


@requires_data
@requires_deck
def test_the_anchor_uses_the_SAME_harness_that_will_grade_it(monkeypatch):
    """The whole point. `net_change.build` takes its iterations and seed from
    `diagnostic.HARNESS`; if the anchor read a different seed it would reproduce
    the exact defect it exists to prevent — a control that disagrees with the
    control the verdict is computed against.
    """
    from manamap.pilot import diagnostic
    seen = {}
    real = diagnostic.run

    def spy(slug, **kw):
        seen.update(kw)
        return real(slug, **kw)
    monkeypatch.setattr(diagnostic, "run", spy)
    deck_branch.champion_reading("heliod", _obj("board_power_6", 2.8))
    assert seen.get("iterations") == diagnostic.HARNESS["iterations"]
    assert seen.get("seed") == diagnostic.HARNESS["seed"]
    assert seen.get("branch") in (None, ...), "the ANCHOR is the champion, not a branch"
