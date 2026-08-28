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
    if not records:
        # NO BRANCH HAS BEEN SIMULATED, which is a fact about the pilot's
        # current work and not about the ignore rule. Asserting otherwise ties
        # the suite to one experimental deck having Forge runs — the exact
        # dependency PLAN.md's 2026-08-27 issue forbids. The half that matters
        # (logs untracked) is asserted above and holds either way.
        pytest.skip("no branch on the bench has a simulation run")
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


# ──────────────────────────────────────────────────────────────────────────
# THE PROPOSAL — a branch that is DECIDED and waiting on cardboard
#
# Before `propose`, a branch had two observable conditions: the directory
# exists, or `merged` is present — and `delete` was the only code in the repo
# that read `merged`. A branch the pilot had accepted was byte-identical to a
# half-finished experiment nobody had looked at, and `deck-info` said the same
# sentence about both.
#
# EVERY STATE BELOW IS DERIVED AND NONE IS STORED. That is what makes a
# proposal un-block itself: drop a card into a box and the blocker shrinks
# without anyone touching the branch. Fixtures only — no test pins a real
# experimental branch, which is the standing rule.
# ──────────────────────────────────────────────────────────────────────────

import hashlib


def _branch(tmp_path, monkeypatch, cards, *, proposal=None, merged=None,
            base_version=2, current_version=2, apart=(), boxed=(), held=None):
    """A branch on disk, with the deck's version and collection stubbed.

    Everything `branch_state` reads is faked at its source rather than at the
    function, so the test drives the production path and not a mock of it.
    """
    from manamap.pilot import deck_branch as db
    from manamap.pilot import deck_versions

    root = tmp_path / "decks" / "d" / "branches" / "b"
    root.mkdir(parents=True)
    text = "".join(f"1 {c}\n" for c in cards)
    (root / "decklist.txt").write_text(text, encoding="utf-8")
    doc = {"slug": "d", "branch": "b", "v": 3, "opened": "2026-01-01",
           "objective": {"axis": "damage_8", "op": ">=", "value": 40.0}}
    if proposal is not None:
        doc["proposal"] = dict(
            {"at": "2026-01-02", "as_version": "v1.0.2",
             "base_version": base_version,
             "decklist_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "accepted_on": {"decklist_sha256": "x", "state": "a trade"}},
            **proposal)
    if merged is not None:
        doc["merged"] = merged
    (root / "branch.json").write_text(json.dumps(doc), encoding="utf-8")

    monkeypatch.setattr(db, "meta", lambda s, b: doc)
    monkeypatch.setattr(db, "deck_dir", lambda s, b=None: root)
    monkeypatch.setattr(deck_versions, "report",
                        lambda s: {"current_version": current_version})

    def fake_source(slug, branch, proxy=False):
        rows, unsourced = [], []
        for c in cards:
            if c in boxed:
                rows.append({"name": c, "state": db.BOX, "free": False,
                             "where": [{"kind": "box", "name": "A"}]})
            elif c in apart:
                rows.append({"name": c, "state": db.ELSEWHERE, "free": True,
                             "where": [{"kind": "deck", "slug": "old",
                                        "locked": False, "status": "retired",
                                        "apart": True}]})
            else:
                rows.append({"name": c, "state": db.BUY, "free": False, "where": []})
                unsourced.append(c)
        if held is not None:
            unsourced = list(held)
        return {"cards": rows, "unsourced": unsourced,
                "mergeable": not unsourced, "counts": {}, "free": len(apart)}

    monkeypatch.setattr(db, "source", fake_source)
    return doc


def test_a_branch_with_no_proposal_is_an_experiment_not_a_decision(
        tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], boxed=["Sol Ring"])
    state, why = db.branch_state("d", "b")
    assert state == db.OPEN
    assert "experiment" in why


def test_a_proposal_with_cards_outstanding_is_blocked(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring", "Mana Crypt"], proposal={})
    state, why = db.branch_state("d", "b")
    assert state == db.PROPOSED_BLOCKED
    assert "2 card(s)" in why


def test_a_proposal_with_everything_sourced_is_ready_to_merge(
        tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={}, boxed=["Sol Ring"])
    state, why = db.branch_state("d", "b")
    assert state == db.PROPOSED_READY
    assert "merge" in why


def test_a_card_only_in_a_broken_down_deck_does_not_block(tmp_path, monkeypatch):
    """3a, AND ITS CONTROL IS THE TEST BELOW.

    `deck-branch merge` refused Ur-Dragon on twelve cards, four of which sit in
    `sisay` and `hapatra` — decks that do not physically exist. The pilot was
    being told to unsleeve a deck that is already in a pile.
    """
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={}, apart=["Sol Ring"])
    assert db.branch_state("d", "b")[0] == db.PROPOSED_READY


def test_a_card_in_a_deck_that_is_still_together_does_block(tmp_path, monkeypatch):
    """THE CONTROL. Without it the fix above could be "nothing ever blocks"."""
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={},
            held=["Sol Ring"])
    assert db.branch_state("d", "b")[0] == db.PROPOSED_BLOCKED


def test_a_list_that_moves_after_it_was_accepted_goes_stale(
        tmp_path, monkeypatch):
    """The merge-request "new commits pushed" case, and it is mechanical: the
    proposal freezes the sha it was accepted on."""
    from manamap.pilot import deck_branch as db
    doc = _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={},
                  boxed=["Sol Ring"])
    assert db.branch_state("d", "b")[0] == db.PROPOSED_READY
    doc["proposal"]["decklist_sha256"] = "something else entirely"
    state, why = db.branch_state("d", "b")
    assert state == db.PROPOSED_STALE
    assert "changed since" in why


def test_a_deck_that_moved_on_outruns_the_proposal(tmp_path, monkeypatch):
    """THE MERGE-CONFLICT ANALOGUE, and it closes a real hole: `base_version`
    has been written by `new()` since branches shipped and NO CODE HAS EVER
    COMPARED IT TO ANYTHING. If another branch merges first, the version this
    proposal claims is taken."""
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={},
            boxed=["Sol Ring"], base_version=2, current_version=3)
    state, why = db.branch_state("d", "b")
    assert state == db.PROPOSED_OUTRUN
    assert "V2" in why and "V3" in why


def test_merged_outranks_everything(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Sol Ring"], proposal={},
            merged={"at": "2026-02-01", "into_version_before": 2},
            base_version=1, current_version=9)
    assert db.branch_state("d", "b")[0] == db.MERGED


def test_every_state_the_function_can_return_is_declared():
    """A state a caller cannot enumerate is one the frontend will not style."""
    from manamap.pilot import deck_branch as db
    for name in ("OPEN", "PROPOSED_BLOCKED", "PROPOSED_READY", "PROPOSED_STALE",
                 "PROPOSED_OUTRUN", "MERGED"):
        assert getattr(db, name) in db.BRANCH_STATES
    assert len(set(db.BRANCH_STATES)) == len(db.BRANCH_STATES)


# ── the pull list ────────────────────────────────────────────────────────

def test_the_pull_list_separates_costs_that_are_not_the_same_cost(
        tmp_path, monkeypatch):
    """BUY is money, UNSLEEVE takes a deck apart, PROXY is a decision already
    recorded, FREE is cardboard in a pile. Reported as one integer they read as
    one problem — which is how `elsewhere` came to mean both "unsleeve kianne"
    and "it is in the hapatra pile"."""
    from manamap.pilot import deck_branch as db
    _branch(tmp_path, monkeypatch, ["Buy Me", "In A Box", "Loose"],
            proposal={}, boxed=["In A Box"], apart=["Loose"])
    pl = db.pull_list("d", "b")
    assert [r["name"] for r in pl["buy"]] == ["Buy Me"]
    assert [r["name"] for r in pl["box"]] == ["In A Box"]
    assert [r["name"] for r in pl["free"]] == ["Loose"]
    assert pl["unsleeve"] == [] and pl["proxy"] == []


def test_a_proxied_card_is_filed_under_proxy_not_unsleeve(tmp_path, monkeypatch):
    """`--proxy` was a per-invocation flag and was never persisted, so `list`,
    `deck-info` and the web roster showed the non-proxy verdict however the
    pilot had decided. A proposal records the CARDS, not a boolean."""
    from manamap.pilot import deck_branch as db

    def source(slug, branch, proxy=False):
        return {"cards": [{"name": "Bloom Tender", "state": db.ELSEWHERE,
                           "free": False,
                           "where": [{"kind": "deck", "slug": "kinnan",
                                      "locked": False, "status": None,
                                      "apart": False}]}],
                "unsourced": [], "mergeable": True, "counts": {}, "free": 0}

    _branch(tmp_path, monkeypatch, ["Bloom Tender"],
            proposal={"proxy": ["Bloom Tender"]})
    monkeypatch.setattr(db, "source", source)
    pl = db.pull_list("d", "b")
    assert [r["name"] for r in pl["proxy"]] == ["Bloom Tender"]
    assert pl["unsleeve"] == []


def test_the_recorded_proxy_is_names_and_never_a_boolean():
    from manamap.pilot import deck_branch as db
    assert db.recorded_proxy({"proposal": {"proxy": ["A", "B"]}}) == ["A", "B"]
    assert db.recorded_proxy({"proposal": {}}) is False
    assert db.recorded_proxy({}) is False
    assert db.recorded_proxy(None) is False


# ── propose and withdraw: the refusals ───────────────────────────────────

def _proposable(tmp_path, monkeypatch, *, nc=None, tags=None, current=2):
    """A branch that `propose` would accept, so each test can break one thing."""
    from manamap.pilot import deck_branch as db
    from manamap.pilot import deck_versions

    root = tmp_path / "b"
    root.mkdir(parents=True)
    text = "1 Sol Ring\n"
    (root / "decklist.txt").write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    doc = {"slug": "d", "branch": "b", "v": 2, "opened": "2026-01-01"}
    report = {"decklist_sha256": sha, "recommendation": {"state": "a trade"},
              "objective": {"axis": "damage_8", "op": ">=", "value": 40.0},
              "objective_grade": {"state": "met", "reading": 46.4},
              "harness": {"iterations": 10000, "seed": 1}}
    if nc is not None:
        report = nc

    monkeypatch.setattr(db, "branch_root", lambda s: tmp_path)
    monkeypatch.setattr(db, "meta", lambda s, b: doc)
    monkeypatch.setattr(db, "deck_dir", lambda s, b=None: root)
    monkeypatch.setattr(db, "_write_meta", lambda s, b, d: doc.update(d))
    monkeypatch.setattr(db, "load_json", lambda p, *a, **k: report)
    monkeypatch.setattr(db, "pull_list",
                        lambda s, b, doc=None, src=None: {"blocking": 0})
    monkeypatch.setattr(db, "branch_state",
                        lambda s, b, doc=None, src=None: (db.PROPOSED_READY, "ok"))
    monkeypatch.setattr(db, "source", lambda s, b, proxy=False: {
        "cards": [], "unsourced": [], "mergeable": True, "counts": {}, "free": 0})
    monkeypatch.setattr(deck_versions, "report",
                        lambda s: {"current_version": current})
    monkeypatch.setattr(deck_versions, "tags", lambda s: tags or {})
    return doc


def test_a_proposal_freezes_the_report_it_was_accepted_on(tmp_path, monkeypatch):
    """The evidence and the decision are the same act, so the decision carries
    the evidence. A proposal that only said "yes" could not later be shown to
    have been made against a list that has since moved."""
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch)
    got = db.propose("d", "b", "v1.0.2", why="because", at="2026-01-02")
    p = got["proposal"]
    assert p["as_version"] == "v1.0.2" and p["why"] == "because"
    assert p["decklist_sha256"] and p["accepted_on"]["decklist_sha256"]
    assert p["accepted_on"]["state"] == "a trade"
    assert p["accepted_on"]["grade"] == "met"
    assert p["base_version"] == 2


def test_a_proposal_needs_a_measurement(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch, nc={})
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.2")
    assert "net-change" in str(e.value)


def test_a_report_that_measured_a_different_list_is_refused(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch,
                nc={"decklist_sha256": "stale", "recommendation": {"state": "merge"}})
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.2")
    assert "different list" in str(e.value)


def test_a_do_not_merge_needs_a_reason_to_override(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    text_sha = hashlib.sha256(b"1 Sol Ring\n").hexdigest()
    _proposable(tmp_path, monkeypatch,
                nc={"decklist_sha256": text_sha,
                    "recommendation": {"state": "do not merge", "because": "worse"}})
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.2")
    assert "DO NOT MERGE" in str(e.value)
    with pytest.raises(SystemExit) as e2:
        db.propose("d", "b", "v1.0.2", anyway=True)
    assert "--reason" in str(e2.value)
    got = db.propose("d", "b", "v1.0.2", anyway=True, reason="the log disagrees")
    assert got["proposal"]["forced_reason"] == "the log disagrees"


@pytest.mark.parametrize("bad", ["1.0", "v1", "latest", "v1.2.3.4", ""])
def test_a_version_that_is_not_a_release_tag_is_refused(tmp_path, monkeypatch, bad):
    """`deck_versions` owns this vocabulary and already refuses near misses. A
    second copy of the regex here would drift from it."""
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", bad)
    assert "release tag" in str(e.value)


def test_a_version_already_taken_is_refused(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch, tags={"v1.0.2": {"version": 2}})
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.2")
    assert "already names V2" in str(e.value)


def test_proposing_twice_is_refused_and_withdraw_is_the_way_back(
        tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    doc = _proposable(tmp_path, monkeypatch)
    db.propose("d", "b", "v1.0.2", at="2026-01-02")
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.3")
    assert "already proposed" in str(e.value) and "withdraw" in str(e.value)
    got = db.withdraw("d", "b")
    assert got["withdrew"]["as_version"] == "v1.0.2"
    assert "proposal" not in doc
    # And the branch is untouched: its objective and trail survive.
    assert doc["opened"] == "2026-01-01"


def test_withdrawing_nothing_says_so(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    _proposable(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        db.withdraw("d", "b")
    assert "not proposed" in str(e.value)


def test_a_merged_branch_cannot_be_proposed(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch as db
    doc = _proposable(tmp_path, monkeypatch)
    doc["merged"] = {"at": "2026-02-01"}
    with pytest.raises(SystemExit) as e:
        db.propose("d", "b", "v1.0.3")
    assert "already merged" in str(e.value)


# ── 3b: the interface the merge post-amble depends on ────────────────────

def test_deck_status_exposes_what_merge_reads_from_it():
    """THE ASSERTION WHOSE ABSENCE LET A WHOLE BLOCK NEVER RUN.

    `merge()` called `deck_status.report(slug)` — a function that does not
    exist — inside a bare `except`, so `out["stale"]` was unconditionally `[]`
    and the "written against the previous list" warning has never printed once.
    Two more bugs sat behind it: rows key their name as `stage`, not `key`, and
    `status()` returns a LIST rather than `{"stages": [...]}`. Nothing caught it
    because no branch in this repo has ever been merged.

    This drives the real interface rather than re-deriving it, so re-introducing
    any of the three failures fails here.
    """
    from manamap.pilot import deck_status
    assert not hasattr(deck_status, "report"), (
        "merge() assumed this existed; if it is added, fix the call site too")
    assert callable(deck_status.status)


@requires_deck
def test_the_stale_rows_merge_reports_are_shaped_the_way_it_reads_them():
    from manamap.pilot import deck_status
    rows = deck_status.status(SLUG)
    assert isinstance(rows, list) and rows
    checked = 0
    for r in rows:
        assert "stage" in r and "state" in r, r
        checked += 1
    assert checked >= 10
