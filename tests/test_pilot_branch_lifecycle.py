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
def sandbox(tmp_path, monkeypatch):
    """A COPY of the deck, in a temp data dir.

    THESE TESTS MERGE, AND A MERGE OVERWRITES `decklist.txt`. Run against the
    real deck they raced `test_a_branch_run_never_touches_the_decks_own_artifacts`
    under `-n auto` — one test writing the file another was asserting had not
    changed. Both were correct; the hazard was mine, and a marker would only
    have hidden it. `MANAMAP_DATA_DIR` is the override the repo already has for
    exactly this, and `deck_dir` resolves through `config.DECKS_DIR`, so
    repointing it isolates every writer at once.
    """
    import shutil

    from manamap import config
    from manamap.pilot import common, deck_versions
    real = config.DECKS_DIR / SLUG
    if not (real / "decklist.txt").exists():
        pytest.skip(f"no {SLUG} fixture")
    decks = tmp_path / "decks"
    shutil.copytree(real, decks / SLUG,
                    ignore=shutil.ignore_patterns("branches", "sim"))
    for mod in (config, common, deck_branch, deck_versions):
        if hasattr(mod, "DECKS_DIR"):
            monkeypatch.setattr(mod, "DECKS_DIR", decks)
    # Versions are a git walk over the REAL repo path; in a sandbox there is no
    # history, so the branch records `base_version: None` and that is honest.
    monkeypatch.setattr(deck_versions, "report", lambda slug: {"current_version": None})
    return decks


@pytest.fixture
def probe(sandbox):
    """A branch identical to the deck, so a merge is a provable no-op."""
    name = "pytest-probe"
    src = sandbox / SLUG / "decklist.txt"
    deck_branch.new(SLUG, name, src.read_text(),
                    why="pytest", objective={"axis": "kill_by_8", "op": ">=",
                                             "value": 0.3, "why": "pytest"})
    return name


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
    # The KEY, not a value: versions are a git walk, and a sandbox has no
    # history, so recording None here is the truthful answer rather than a
    # fabricated number. Asserting non-None would demand the code invent one.
    assert "into_version_before" in doc["merged"]


@requires_deck
def test_merging_an_identical_list_leaves_the_deck_byte_identical(probe, sandbox):
    """THE CONTROL. The probe IS the deck, so a merge must be a no-op — if the
    canonical render or the backup path ever mangles the list, this is where it
    shows, and not in a deck the pilot then plays."""
    path = sandbox / SLUG / "decklist.txt"
    before = path.read_bytes()
    deck_branch.merge(SLUG, probe, write=True, run_chain=False)
    assert path.read_bytes() == before
    assert path.with_suffix(".txt.bak").exists(), (
        "merge overwrote the deck's tracked list with no backup — check-in has "
        "made one since it shipped and merge never did")


@requires_deck
def test_deleting_an_unmerged_branch_is_refused(probe):
    """A branch holds measurements that cost real time — a 100-game Forge run is
    45 minutes."""
    with pytest.raises(SystemExit) as e:
        deck_branch.delete(SLUG, probe)
    assert "never merged" in str(e.value)
    deck_branch.merge(SLUG, probe, write=True, run_chain=False)
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


#: A real card ur-dragon does not run. Staging refuses a card already in the
#: list, so the fixture has to be one that is not — Sol Ring is in the 99.
ABSENT = "Llanowar Elves"


# ── staging: the swap is the unit ────────────────────────────────────────

@requires_deck
def test_a_swap_is_one_edit_that_says_what_it_displaced(probe):
    """A card added and a card cut are two edits a reader has to pair up by
    hand. A swap is one edit that already carries the pairing, which is what
    lets a report name WHICH swaps bought the delta rather than reporting that
    the list changed somehow."""
    before = deck_branch._parsed(SLUG, probe)
    out = next(e["name"] for e in before if not e.get("is_commander"))
    got = deck_branch.stage(SLUG, probe, out, ABSENT, strength=0.71,
                            why="pytest")
    assert got["out"] == out and got["in"] == ABSENT
    assert got["staged"] == 1
    after = deck_branch._parsed(SLUG, probe)
    names = {e["name"] for e in after}
    assert ABSENT in names and out not in names
    # ONE FOR ONE. The list is never briefly a 98 that some command measures,
    # and a substitution is exactly what `candidates` prices — which is why it
    # needs no placebo.
    assert (sum(int(e.get("quantity") or 1) for e in after)
            == sum(int(e.get("quantity") or 1) for e in before))
    row = deck_branch.meta(SLUG, probe)["staged"][0]
    assert row["out"] == out and row["in"] == ABSENT
    assert row["strength"] == 0.71, "the provenance of the swap, not a claim"


@requires_deck
def test_staging_writes_through_the_check_in_refusals(probe):
    """Editing decklist.txt directly skips singleton, size, commander and the
    corpus check. All four are exactly the refusals a paper list gets."""
    before = deck_branch._parsed(SLUG, probe)
    out = next(e["name"] for e in before if not e.get("is_commander"))
    held = next(e["name"] for e in before
                if not e.get("is_commander") and e["name"] != out)
    with pytest.raises(SystemExit) as e:
        deck_branch.stage(SLUG, probe, out, held)
    assert "already in" in str(e.value)
    with pytest.raises(SystemExit) as e:
        deck_branch.stage(SLUG, probe, "Not A Real Card", ABSENT)
    assert "nothing to swap out" in str(e.value)


@requires_deck
def test_the_commander_is_not_swappable(probe):
    """Changing it is a different deck, not a swap — the identity, the whole
    candidate pool and every declared component move with it."""
    entries = deck_branch._parsed(SLUG, probe)
    cmd = next(e["name"] for e in entries if e.get("is_commander"))
    with pytest.raises(SystemExit) as e:
        deck_branch.stage(SLUG, probe, cmd, ABSENT)
    assert "COMMANDER" in str(e.value)


@requires_deck
def test_a_staged_swap_can_be_put_back(probe):
    """A staging area you cannot back out of is a decision, not a draft."""
    before = deck_branch._list_text(SLUG, probe)
    out = next(e["name"] for e in deck_branch._parsed(SLUG, probe)
               if not e.get("is_commander"))
    deck_branch.stage(SLUG, probe, out, ABSENT)
    got = deck_branch.unstage(SLUG, probe, out, ABSENT)
    assert got["staged"] == 0
    assert deck_branch.meta(SLUG, probe)["staged"] == []
    names = {e["name"] for e in deck_branch._parsed(SLUG, probe)}
    assert out in names and ABSENT not in names
    with pytest.raises(SystemExit) as e:
        deck_branch.unstage(SLUG, probe, out, ABSENT)
    assert "Nothing staged" in str(e.value)


@requires_deck
def test_staging_never_touches_the_decks_own_list(probe):
    """The branched-write rule, on a new writer. Three instances of a branched
    write with an unbranched read have shipped; a write is the other half."""
    deck_path = deck_branch.deck_dir(SLUG) / "decklist.txt"
    before = deck_path.read_text()
    out = next(e["name"] for e in deck_branch._parsed(SLUG, probe)
               if not e.get("is_commander"))
    deck_branch.stage(SLUG, probe, out, ABSENT)
    assert deck_path.read_text() == before
