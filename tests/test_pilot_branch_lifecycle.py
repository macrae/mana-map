"""The branch as a decision: objective, commit trail, merge that closes the loop.

THE DEFECT THIS EXISTS AGAINST. The Ur-Dragon treasure branch stated "treasure is
the engine" and achieved it 4.4x over, then failed the purpose nobody wrote down.
A branch whose objective is unstated gets graded on whether it did what it does,
which is not the same question as whether it was worth doing — and it took a week
of measurement and a hand-built report to find that out.
"""

import json

import pytest

from conftest import requires_deck
from manamap.pilot import candidates, deck_branch

SLUG = "ur-dragon"


# ── the objective ────────────────────────────────────────────────────────

def test_an_objective_names_a_measure_the_bench_computes():
    got = deck_branch.parse_objective("kill_by_8 >= 0.30")
    assert got == {"axis": "kill_by_8", "op": ">=", "value": 0.30}


@pytest.mark.parametrize("bad", ["nonsense", "kill_by_8", ">= 0.3",
                                 "made_up_axis >= 1", "kill_by_8 ~ 0.3"])
def test_an_objective_that_cannot_be_measured_is_refused(bad):
    with pytest.raises(SystemExit):
        deck_branch.parse_objective(bad)


def test_the_objective_vocabulary_is_wider_than_the_ranking_one():
    """RANKING NEEDS INDEPENDENT AXES; STATING A GOAL DOES NOT.

    `AXES` is narrow on purpose — three combat measures correlate at r = 0.92-0.98
    and ranking on more than one is three confirmations of one fact. That argument
    is about sorting, not about aiming. A pilot's objective is "kill by turn
    eight", and forcing it onto `damage_8` makes the honest goal unsayable to
    protect a ranking nobody is doing.
    """
    assert set(candidates.AXES) < set(candidates.OBJECTIVE_AXES)
    assert "kill_by_8" in candidates.OBJECTIVE_AXES
    assert "kill_by_8" not in candidates.AXES


def test_grading_has_three_states_and_the_third_is_the_honest_one():
    o = {"axis": "kill_by_8", "op": ">=", "value": 0.30}
    assert deck_branch.grade_objective(o, 0.45)["state"] == "met"
    assert deck_branch.grade_objective(o, 0.011)["state"] == "not met"
    # A miss smaller than the run could see has NOT failed — the run could not
    # resolve it. Calling that "not met" is reporting a null as a finding.
    near = deck_branch.grade_objective(o, 0.298, mde=0.02)
    assert near["state"] == "not resolvable"
    assert "evidence of nothing" in near["why"]
    # ...and the same miss IS a failure once the run is powerful enough to see it.
    assert deck_branch.grade_objective(o, 0.298, mde=0.0005)["state"] == "not met"
    assert deck_branch.grade_objective(o, None)["state"] == "not measured"


# ── the round trip ───────────────────────────────────────────────────────

@pytest.fixture
def probe(tmp_path):
    """A branch identical to the deck, so a merge is a provable no-op."""
    from manamap.pilot.common import deck_dir
    src = deck_dir(SLUG) / "decklist.txt"
    name = "pytest-probe"
    root = deck_branch.branch_root(SLUG) / name
    if root.exists():
        deck_branch.delete(SLUG, name, force=True)
    deck_branch.new(SLUG, name, src.read_text(),
                    why="pytest", objective={"axis": "kill_by_8", "op": ">=",
                                             "value": 0.3, "why": "pytest"})
    yield name
    if root.exists():
        deck_branch.delete(SLUG, name, force=True)


@requires_deck
def test_a_commit_records_one_exact_list_and_refuses_a_repeat(probe):
    """A commit names one 99. Committing an unchanged list twice would make the
    trail say the pilot decided something when nothing moved."""
    first = deck_branch.commit(SLUG, probe, "the control")
    assert first["n"] == 1 and len(first["decklist_sha256"]) == 64
    with pytest.raises(SystemExit) as e:
        deck_branch.commit(SLUG, probe, "again")
    assert "has not changed" in str(e.value)
    with pytest.raises(SystemExit):
        deck_branch.commit(SLUG, probe, "   ")


@requires_deck
def test_commit_is_allowed_unsourced_and_merge_is_not(probe):
    """THE TWO-STEP IS THE PILOT'S OWN DISTINCTION. A commit says "this is the
    deck I am committed to running"; a merge says "this is the deck". The gap
    between them is CARDBOARD."""
    got = deck_branch.commit(SLUG, probe, "decided")
    assert "unsourced" in got and "mergeable" in got


@requires_deck
def test_merge_records_that_it_landed(probe):
    """Without this the branch survives untouched, `diff` reads +0 -0 forever,
    and nothing links the resulting version back to the work that produced it."""
    deck_branch.commit(SLUG, probe, "ready")
    got = deck_branch.merge(SLUG, probe, write=True, run_chain=False)
    assert got["written"] is True
    doc = deck_branch.meta(SLUG, probe)
    assert doc["merged"]["at"] and doc["merged"]["decklist_sha256"]
    assert doc["merged"]["into_version_before"] is not None


@requires_deck
def test_merging_an_identical_list_leaves_the_deck_byte_identical(probe):
    """THE CONTROL. The probe IS the deck, so a merge must be a no-op — if the
    canonical render or the backup path ever mangles the list, this is where it
    shows, and not in a deck the pilot then plays."""
    from manamap.pilot.common import deck_dir
    path = deck_dir(SLUG) / "decklist.txt"
    before = path.read_bytes()
    deck_branch.merge(SLUG, probe, write=True, run_chain=False)
    assert path.read_bytes() == before
    assert path.with_suffix(".txt.bak").exists(), (
        "merge overwrote the deck's tracked list with no backup — check-in has "
        "made one since it shipped and merge never did")
    path.with_suffix(".txt.bak").unlink()


@requires_deck
def test_deleting_an_unmerged_branch_is_refused(probe):
    """A branch holds measurements that cost real time — a 100-game Forge run is
    45 minutes."""
    with pytest.raises(SystemExit) as e:
        deck_branch.delete(SLUG, probe)
    assert "never merged" in str(e.value)
    deck_branch.merge(SLUG, probe, write=True, run_chain=False)
    from manamap.pilot.common import deck_dir
    (deck_dir(SLUG) / "decklist.txt.bak").unlink(missing_ok=True)
    assert deck_branch.delete(SLUG, probe)["deleted"]


@requires_deck
def test_a_v1_branch_still_loads_and_says_it_cannot_be_graded():
    """The upgrade direction that matters: our own older data must not be
    refused. A branch with no objective reports one absent, never a fabricated
    one."""
    for name in deck_branch.names(SLUG):
        got = deck_branch.log(SLUG, name)
        assert "commits" in got and isinstance(got["commits"], list)


def test_branch_sim_logs_are_not_tracked():
    """A PATH RULE WRITTEN BEFORE A DIRECTORY LEVEL EXISTED DOES NOT LEARN ABOUT IT.

    `.gitignore` carried `data/decks/*/sim/logs/`, which does not match
    `data/decks/<slug>/branches/<name>/sim/logs/` — so a branch's Forge logs were
    TRACKED while the deck's own were ignored. 12 MB of them, committed the day
    branches shipped. Nothing complains: the files are valid, just enormous.
    """
    import subprocess
    from conftest import ROOT
    out = subprocess.run(
        ["git", "ls-files", "data/decks/*/branches/*/sim/logs/*"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not out, f"{len(out)} branch sim log(s) are tracked: {out[:3]}"
    # ...and the run RECORDS beside them must stay tracked, or the evidence goes
    # with the noise.
    records = subprocess.run(
        ["git", "ls-files", "data/decks/*/branches/*/sim/*.json"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert records, "the branch sim records went untracked with the logs"
