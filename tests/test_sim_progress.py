"""`sim-progress` — a window into a run that is still going.

Written after an eight-hour job with no window into it, during which the same
question was answered four times by four slightly different ad-hoc greps. The
tests here are about the two things that make a progress view honest rather than
misleading: the ORDER the convergence trace claims, and the target it projects
against.
"""

import re

import pytest

from manamap.sim import progress


def test_the_target_comes_from_the_run_id():
    """The run id is the only place the launched N is recorded before the record
    exists — and the record does not exist until the run ends, which is exactly
    when this command is useless."""
    for rid, want in [
            ("giada-angels-vs-baylen-tokens-vs-abaddon-n120-996adb84-s573917060-podExperimental-c600", 120),
            ("v1.0.1-vs-@teeth-v1-x-giada-angels-baylen-tokens-abaddon-n100-e574597d-s1840775157-podExperimental", 100),
            ("edgar-vampires-vs-yawgmoth-swarm-vs-heliod-n8-dfd75e54", 8)]:
        m = progress._TARGET_RE.search(rid)
        assert m and int(m.group(1)) == want, rid


def test_the_convergence_trace_interleaves_jobs_rather_than_concatenating_them():
    """THE LOAD-BEARING APPROXIMATION. Jobs run in parallel, so job 0's third
    game finished at roughly the same time as job 1's third — not before job 1's
    first. Concatenating whole jobs would make the trace read as "job 0's entire
    sample, then job 1's", which for four jobs means the first quarter of the
    curve is one JVM's luck and the shape is an artefact of the file order.

    Four jobs where job 0 lost everything and the rest won everything: the
    interleaved trace must show that mix from the very first checkpoint.
    """
    per_job = [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    got = progress._order_round_robin(per_job)
    assert len(got) == 16
    assert got[:4] == [0, 1, 1, 1], (
        "the first four games are one from each job, not four from job 0 — "
        "concatenation would read [0, 0, 0, 0] and the early trace would be "
        "a single JVM's losing streak wearing the name of a running estimate")
    assert sum(got) == 12


def test_uneven_jobs_do_not_drop_games():
    """Jobs finish at different rates, so at any moment they have different
    counts — the shorter ones must simply run out rather than truncate the rest."""
    per_job = [[1, 1, 1], [0], [1, 0]]
    got = progress._order_round_robin(per_job)
    assert sorted(got) == sorted([1, 1, 1, 0, 1, 0])
    assert got[:3] == [1, 0, 1]


def test_the_interval_narrows_toward_the_target_and_the_projection_says_so():
    """The point of the trace is not the rate, which wanders; it is the HALF
    WIDTH, which only shrinks. A projection to the target N is what tells you
    whether finishing the run will actually settle anything."""
    flags = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0] * 5          # 50 games at 0.30
    rows, projected = progress._convergence(flags, 200)
    halves = [r[4] for r in rows]
    assert halves == sorted(halves, reverse=True), (
        f"the interval must narrow as n grows: {halves}")
    assert projected is not None
    n, p, lo, hi, half = projected
    assert n == 200 and abs(p - 0.30) < 1e-9
    assert half < halves[-1], "projecting to a larger N must give a tighter interval"


def test_no_projection_once_the_run_has_reached_its_target():
    """A finished run has nothing to project to, and inventing a row for it
    would read as "there is more to come" on a run that is over."""
    _rows, projected = progress._convergence([1, 0] * 25, 50)
    assert projected is None


def test_the_bar_is_honest_at_both_ends():
    assert progress._bar(0, 100, width=10) == "·" * 10
    assert progress._bar(100, 100, width=10) == "█" * 10
    assert progress._bar(50, 100, width=10) == "█" * 5 + "·" * 5
    # an unknown target renders as unknown rather than as complete
    assert set(progress._bar(3, 0, width=10)) == {"-"}


def test_an_experiment_arm_token_resolves_to_the_deck_slug():
    """`experiment` installs its arms as `mm-x-<slug>-a` / `-b` while a plain run
    uses `mm-<slug>`, and the record that would name them does not exist yet —
    so the mapping has to come out of the log text itself."""
    text = ("Turn: Turn 1 (Ai(1)-mm-x-heliod-a)\n"
            "Turn: Turn 2 (Ai(2)-mm-giada-angels)\n"
            "Turn: Turn 3 (Ai(3)-mm-abaddon)\n")
    lab = progress._seat_map([text])
    assert lab["Ai(1)-mm-x-heliod-a"] == "heliod"
    assert lab["Ai(2)-mm-giada-angels"] == "giada-angels"
    assert lab["Ai(3)-mm-abaddon"] == "abaddon"
