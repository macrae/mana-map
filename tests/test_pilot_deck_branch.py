"""Deck branches — a candidate 99 the pilot cannot yet sleeve.

The load-bearing test here is `test_a_branch_run_never_touches_the_decks_own_artifacts`.
Every other property is visible when it breaks; that one fails SILENTLY and
plausibly, by writing a branch's numbers over the deck's tracked file under the
deck's own name. It is the same failure `resolve_out_path` exists to stop, one
level up, and the only control that catches a `deck_dir(slug)` call that never
got its `branch=`.
"""

import hashlib
import json

import pytest

from conftest import A_BRANCH, requires_branch, requires_deck

from manamap.pilot import deck_branch
from manamap.pilot.common import (
    DECKS_DIR, deck_dir, deck_file, load_deck_cards)

SLUG = "ur-dragon"
BRANCH = A_BRANCH

def _tree(path):
    """sha256 of every tracked-ish file in a deck dir, EXCLUDING its branches."""
    out = {}
    for p in sorted(path.rglob("*")):
        if not p.is_file() or deck_branch.BRANCHES_DIR in p.relative_to(path).parts:
            continue
        out[str(p.relative_to(path))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _has_branch():
    return (DECKS_DIR / SLUG).is_dir() and BRANCH in deck_branch.names(SLUG)


needs_branch = pytest.mark.skipif(not _has_branch(),
                                  reason=f"no {SLUG}/{BRANCH} branch on this machine")


@needs_branch
def test_a_branch_run_never_touches_the_decks_own_artifacts():
    """THE CONTROL. Run every branch-aware command THROUGH ITS CLI ENTRY POINT,
    then prove the deck's own directory is byte-identical.

    It must be `main(args)` and not `analyze(slug, branch)`: analyze RETURNS a
    document and main WRITES it, so a test that calls the analysis layer cannot
    see the only bug this exists to catch — a `deck_dir(slug)` on the write path
    that never got its `branch=`. The first version of this test did exactly
    that and passed against a deliberately broken write path.

    The failure it guards is silent and plausible: a branch's numbers land in
    the deck's tracked artifact, under the deck's name, and the wrong list's
    figures are the shape a reader expects.
    """
    import argparse

    from manamap.pilot import bracket, deck_map, goldfish, mana_analysis

    root = deck_dir(SLUG)
    before = _tree(root)
    assert before, "the deck directory is empty — the fixture is wrong, not the code"

    def args(**kw):
        ns = argparse.Namespace(slug=SLUG, branch=BRANCH, out=None, json=False,
                                as_json=False, force=False, archetype=None,
                                seed=None, iterations=None, max_turn=None)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    for mod in (bracket, mana_analysis, deck_map, goldfish):
        mod.main(args())

    after = _tree(root)
    changed = sorted(k for k in before if before[k] != after.get(k))
    added = sorted(set(after) - set(before))
    assert not changed, f"a branch run rewrote the deck's own artifact(s): {changed}"
    assert not added, f"a branch run created file(s) in the deck's directory: {added}"


@needs_branch
def test_a_branch_run_measured_the_branch_and_not_the_deck():
    """THE OTHER HALF OF THE CONTROL, AND IT WAS MISSING FOR AS LONG AS
    BRANCHES HAVE EXISTED.

    `test_a_branch_run_never_touches_the_decks_own_artifacts` proves the WRITE
    landed in the right directory. It cannot see a command that reads the
    CHAMPION and writes to the BRANCH — the file appears exactly where it
    should, holding the wrong deck's numbers.

    `goldfish.main` did precisely that: `run(args.slug)` with no `branch=`,
    beside a `deck_dir(args.slug, branch)` on the write path. On ur-dragon's
    treasure branch it understated the turn-10 hoard 5.29 -> 1.32 — a factor of
    four — and the artifact's own `meta.decklist_sha256` named the champion's
    list the whole time, because nothing compared it to anything.

    So compare it. Every measurement a branch writes must record the BRANCH's
    decklist sha, which is cheap, exact, and catches the entire class.
    """
    import argparse

    from manamap.pilot import goldfish, mana_analysis

    # A BRANCH THAT DIFFERS, BUILT HERE. This control needs a candidate list
    # that is not the deck's, and it used to borrow whichever experimental
    # branch happened to be on disk — so it failed the day the pilot deleted
    # one, and would have passed vacuously the day a branch was opened as an
    # exact copy. A branch is supposed to change and supposed to be thrown
    # away (PLAN.md, the 2026-08-27 issue); the fixture is ours to make.
    branch_sha = load_deck_cards(SLUG, BRANCH)["decklist_sha256"]
    deck_sha = load_deck_cards(SLUG)["decklist_sha256"]
    if branch_sha == deck_sha:
        pytest.skip(
            f"{SLUG}/{BRANCH} is currently an exact copy of the deck, so this "
            f"control cannot tell a branched read from an unbranched one. It "
            f"needs a branch with at least one staged swap.")

    def args(**kw):
        ns = argparse.Namespace(slug=SLUG, branch=BRANCH, out=None, json=False,
                                as_json=False, force=False, archetype=None,
                                seed=None, iterations=None, max_turn=None)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    for mod, name in ((goldfish, "goldfish_metrics.json"),
                      (mana_analysis, "mana_analysis.json")):
        mod.main(args())
        doc = json.loads((deck_dir(SLUG, BRANCH) / name).read_text())
        # The two artifacts stamp it at different depths — goldfish under
        # `meta`, mana-analysis at the top. Read both rather than normalising
        # the files: they are tracked, and tidying them to suit a test would
        # rewrite artifacts nobody asked to change.
        got = (doc.get("decklist_sha256")
               or (doc.get("meta") or {}).get("decklist_sha256"))
        assert got == branch_sha, (
            f"{name} was written into the branch but measured "
            f"{'the deck' if got == deck_sha else 'something else'}: {got}")


@needs_branch
def test_reads_fall_back_to_the_deck_but_writes_do_not():
    """A branch inherits AUTHORED inputs and owns its MEASUREMENTS.

    Nobody writes a second `goldfish_targets.json` to try a candidate list, so a
    branch that measured against no engine declaration would be reporting a
    different deck rather than a different list.
    """
    # A file the branch does NOT have falls back to the deck's. (Asserted with a
    # name no branch will ever carry: `goldfish_targets.json` used to serve here
    # and stopped the day the branch was given its own declaration, which is
    # exactly the behaviour the fallback exists to allow.)
    missing = deck_file(SLUG, "strategic_frame.json", BRANCH)
    assert missing == deck_dir(SLUG) / "strategic_frame.json"
    # measured, branch has its own -> stays in the branch
    mana = deck_file(SLUG, "mana_analysis.json", BRANCH)
    assert mana.parent.name == BRANCH, mana


@needs_branch
def test_the_sourcing_report_separates_a_box_from_another_deck():
    """`elsewhere` is the category nothing computed before, and it is the one
    that changes a decision: a card sleeved in a finished deck is not a card you
    can use, and it is not a purchase either."""
    s = deck_branch.source(SLUG, BRANCH)
    assert set(s["counts"]) == {"in_deck", "box", "elsewhere", "buy"}
    # Counts are DISTINCT NAMES, not copies: 36 basics are one thing to source.
    assert sum(s["counts"].values()) == s["diff"]["names"]
    assert s["diff"]["names"] <= s["diff"]["size"]
    for row in s["cards"]:
        if row["state"] == "elsewhere":
            assert row["where"], f"{row['name']} claims another deck and names none"
            for holder in row["where"]:
                assert holder["slug"] != SLUG, "a card cannot be 'elsewhere' in its own deck"
                assert "locked" in holder
    # Ownership still means A BOX. Anything in the box names the file it is in.
    for row in s["cards"]:
        if row["state"] == "box":
            assert row["where"], f"{row['name']} claims ownership and names no source file"


@needs_branch
def test_merge_refuses_what_it_cannot_honestly_apply():
    """Merging writes decklist.txt, which mints a version the captain's log
    stamps games against — so a version the pilot cannot physically play is a
    version that lies."""
    got = deck_branch.merge(SLUG, BRANCH, write=False)
    s = got["source"]
    if s["unsourced"]:
        assert got["blocking"], "unsourced cards did not block the merge"
        assert not got["written"]
        joined = " ".join(got["blocking"])
        assert str(len(s["unsourced"])) in joined, "the refusal does not say how many"
    # --force without a reason is refused whatever the sourcing says.
    forced = deck_branch.merge(SLUG, BRANCH, write=False, force=True, reason=None)
    assert any("--reason" in b for b in forced["blocking"]), forced["blocking"]


@needs_branch
def test_a_branch_is_a_legal_deck_or_it_is_not_a_branch():
    """`new` reuses check_in's refusals, so a branch that could never become a
    deck is refused at the moment it is opened rather than at merge time."""
    from manamap.pilot import check_in
    text = (deck_dir(SLUG, BRANCH) / "decklist.txt").read_text(encoding="utf-8")
    checked = check_in.analyze(SLUG, text)
    assert not checked["blocking"], checked["blocking"]


def test_deck_dir_refuses_an_unknown_branch():
    if not (DECKS_DIR / SLUG).is_dir():
        pytest.skip("no deck fixture")
    with pytest.raises(FileNotFoundError) as e:
        deck_dir(SLUG, branch="no-such-branch")
    assert "deck-branch" in str(e.value), "the error does not say how to list branches"


@needs_branch
def test_the_branch_reaches_info_json_and_the_next_line():
    """A branch nobody can see is a branch nobody acts on.

    `info.json` is the dossier's data model, so the composition is what puts the
    sourcing split on the page — and `next` is what tells the pilot which state
    the branch is in. That used to be two states and is now six: an experiment
    and a decision the pilot has ACCEPTED said the same sentence, which is the
    whole reason `propose` exists.
    """
    from manamap.pilot import deck_branch, deck_info
    info = deck_info.compose(SLUG)
    rows = info.get("branches") or []
    assert any(b["name"] == BRANCH for b in rows), "the branch is absent from info.json"
    b = next(x for x in rows if x["name"] == BRANCH)
    assert set(b["counts"]) == {"in_deck", "box", "elsewhere", "buy"}
    assert b["state"] in deck_branch.BRANCH_STATES
    joined = " ".join(info["next"])
    assert BRANCH in joined, f"`next` never mentions the branch: {info['next']}"

    # WHAT THE LINE SAYS FOLLOWS FROM THE STATE, and every state owes a line —
    # a branch that reached `next` with nothing to say about it would render as
    # a bare name.
    expect = {
        deck_branch.PROPOSED_BLOCKED: "waiting on cardboard",
        deck_branch.PROPOSED_READY: "every card is sourced",
        deck_branch.PROPOSED_STALE: "the list has changed",
        deck_branch.PROPOSED_OUTRUN: "moved on",
        deck_branch.MERGED: BRANCH,
    }.get(b["state"], "merge" if b["mergeable"] else "source")
    assert expect in joined, f"{b['state']} says nothing useful: {info['next']}"
    if b.get("proposal"):
        assert b["proposal"]["as_version"] in joined


def test_a_deck_with_no_branches_says_nothing_about_them():
    """Absence is silent here, deliberately. A branch is optional and is NOT a
    `deck_status` stage — adding one would change the denominator for every deck
    at once and mark twelve newly incomplete for something nobody asked for."""
    from manamap.pilot import deck_info, deck_status
    stages = {s[0] for s in deck_status.STAGES}
    assert "branch" not in stages and "branches" not in stages, \
        "a branch became a lifecycle stage — that changes /15 for every deck"
    for slug in ("goblin-storm", "heliod"):
        if not (DECKS_DIR / slug).is_dir():
            continue
        if deck_branch.names(slug):
            continue
        info = deck_info.compose(slug)
        assert info["branches"] == []
        assert not any("branch" in n for n in info["next"]), info["next"]


@needs_branch
def test_a_card_in_another_deck_is_owned_not_bought():
    """`elsewhere` IS A LOGISTICS PROBLEM, NOT AN OWNERSHIP ONE.

    The first cut counted a card sleeved in another deck as unsourced, which
    reads as "buy a second copy" — advice to spend money on something already in
    the house. It stays its own state, because the trade-off is real, but
    `--proxy` says the pilot will proxy across their own decks and that makes it
    sourced.

    `buy` is never proxiable here: that would be a claim about a card nobody
    owns, which is a different decision.
    """
    plain = deck_branch.source(SLUG, BRANCH)
    proxied = deck_branch.source(SLUG, BRANCH, proxy=True)
    n_else = plain["counts"]["elsewhere"]
    if not n_else:
        pytest.skip("nothing sleeved elsewhere on this branch")
    # A `free` card was NEVER unsourced — every deck holding it is broken down
    # or retired, so there is nothing to proxy and nothing to unsleeve. Counting
    # it here is what made this arithmetic wrong the moment that landed.
    contested = n_else - plain["free"]
    assert len(proxied["unsourced"]) == len(plain["unsourced"]) - contested, (
        "--proxy did not clear exactly the cards sleeved in decks that are "
        "still together")
    # A card nobody owns stays unsourced whatever the proxy policy is.
    buys = {r["name"] for r in plain["cards"] if r["state"] == "buy"}
    assert buys <= set(proxied["unsourced"]), (
        "--proxy cleared a card that nobody owns")
    assert proxied["owned_but_elsewhere"] == n_else


# --------------------------------------------------------------------------
# A BRANCH MAY NOT AIM AT A MEMBERSHIP AXIS
#
# The most expensive lesson the objective vocabulary carries. `engine_online_*`
# asks whether the parts named in `goldfish_targets.json` were drawn, and that
# file is authored — the same hand writes the declaration and the objective it
# is graded against.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("axis", deck_branch.MEMBERSHIP_AXES)
def test_an_authored_axis_may_not_be_a_branch_objective(axis):
    with pytest.raises(SystemExit) as e:
        deck_branch.parse_objective(f"{axis} >= 0.22")
    assert "authored" in str(e.value)
    assert "without touching a card" in str(e.value)


@pytest.mark.parametrize("expr,axis", [
    ("damage_8 >= 40", "damage_8"),
    ("kill_by_8 >= 0.75", "kill_by_8"),
    ("board_power_6 >= 12", "board_power_6"),
    ("stall <= 0.05", "stall"),
])
def test_an_output_axis_is_still_a_legal_objective(expr, axis):
    """The refusal is scoped. A figure the deck PRODUCES is exactly what a
    branch should aim at, and narrowing the vocabulary to nothing would be the
    worse error."""
    assert deck_branch.parse_objective(expr)["axis"] == axis


def test_the_refusal_names_what_to_aim_at_instead():
    """A refusal that does not say what to do next gets worked around."""
    with pytest.raises(SystemExit) as e:
        deck_branch.parse_objective("engine_online_5 >= 0.22")
    text = str(e.value)
    assert "damage_8" in text and "kill_by_8" in text
    for axis in deck_branch.MEMBERSHIP_AXES:
        assert f"\n  {axis}" not in text, f"{axis} offered as a replacement"


def test_every_membership_axis_is_a_real_axis():
    """A name in this tuple that the bench does not measure guards nothing, and
    would go on guarding nothing silently."""
    from manamap.pilot import candidates
    for axis in deck_branch.MEMBERSHIP_AXES:
        assert axis in candidates.OBJECTIVE_AXES, axis


def test_no_tracked_branch_is_still_graded_on_an_authored_axis():
    """The guard stops a NEW one; this catches an old one that predates it."""
    import glob
    import json
    checked = 0
    for path in glob.glob("data/decks/*/branches/*/branch.json"):
        axis = ((json.load(open(path)) or {}).get("objective") or {}).get("axis")
        if axis is None:
            continue
        assert axis not in deck_branch.MEMBERSHIP_AXES, path
        checked += 1
    assert checked >= 1, "no branch carries an objective to check"


# ──────────────────────────────────────────────────────────────────────────
# `deck_is_apart` — one predicate where four modules had grown their own
# ──────────────────────────────────────────────────────────────────────────

def test_a_broken_down_or_retired_deck_is_apart():
    """`merge` refused ur-dragon on twelve cards, four of which sit in decks
    that do not physically exist. The pilot was being told to unsleeve a deck
    already in a pile."""
    from manamap.pilot.common import deck_is_apart
    assert deck_is_apart("sisay") and deck_is_apart("hapatra")


def test_a_deck_that_is_still_together_is_not_apart():
    """The control. Without it the predicate could be "always true"."""
    from manamap.pilot.common import deck_is_apart
    assert not deck_is_apart("ur-dragon")
    assert not deck_is_apart("no-such-deck-anywhere")


def test_superseded_is_deliberately_not_apart():
    """A superseded list can still be sleeved and played, so its cards are
    spoken for. This is why the predicate reuses `UNPLAYABLE_STATUSES` rather
    than naming its own set — "cannot be played" and "its cards are free" are
    the same question asked twice."""
    from manamap.pilot.common import UNPLAYABLE_STATUSES
    assert "superseded" not in UNPLAYABLE_STATUSES
    assert UNPLAYABLE_STATUSES == frozenset({"broken-down", "retired"})


def test_the_cost_block_reads_the_branch_answer_rather_than_deriving_a_second():
    """FOUR PLACES ANSWERED THIS AND COULD DISAGREE. `net_change.FREE_TO_RAID`
    held its own copy of a set `common.UNPLAYABLE_STATUSES` already had. It is
    gone: `deck_branch.source` derives `free` and `apart` once, and `cost` reads
    them. Driven through the production function, so re-introducing a local list
    that disagreed with the row would fail here.
    """
    from manamap.pilot import net_change
    assert not hasattr(net_change, "FREE_TO_RAID")
    doc = {"bill": {"counts": {}, "cards": [
        {"name": "Loose", "state": "elsewhere", "free": True,
         "where": [{"kind": "deck", "slug": "sisay", "status": "retired",
                    "apart": True}]},
        {"name": "Spoken For", "state": "elsewhere", "free": False,
         "where": [{"kind": "deck", "slug": "kinnan", "status": None,
                    "apart": False}]}]}}
    got = net_change.cost(doc)
    assert [r["name"] for r in got["free_to_raid"]] == ["Loose"]
    assert [r["name"] for r in got["must_unsleeve"]] == ["Spoken For"]
    # And a card in BOTH kinds of home names only the deck still together.
    doc["bill"]["cards"] = [{"name": "Both", "state": "elsewhere", "free": False,
                             "where": [{"kind": "deck", "slug": "hapatra",
                                        "status": "broken-down", "apart": True},
                                       {"kind": "deck", "slug": "blar",
                                        "status": None, "apart": False}]}]
    assert net_change.cost(doc)["must_unsleeve"][0]["decks"] == ["blar"]


@requires_deck
def test_every_where_row_is_one_shape():
    """`where` was a STRING for a box row and a LIST OF DICTS for an elsewhere
    row, so every consumer had to know which state it was in before it could
    read the field. Five of them did."""
    from manamap.pilot import deck_branch
    checked = 0
    for slug in ("ur-dragon",):
        for name in deck_branch.names(slug):
            for r in deck_branch.source(slug, name)["cards"]:
                assert isinstance(r["where"], list), r
                assert isinstance(r["free"], bool)
                for w in r["where"]:
                    assert w["kind"] in ("box", "deck"), w
                    if w["kind"] == "deck":
                        assert set(w) >= {"slug", "locked", "status", "apart"}
                    else:
                        assert "name" in w
                checked += 1
    assert checked >= 20


def test_merge_runs_the_whole_build_not_three_commands():
    """A MERGE THAT LEAVES THE FIGURES BEHIND IS A MERGE THAT LIES.

    The chain was `fetch-deck, goldfish, mana-analysis` and stopped there, so a
    merge left `diagnostic.json`, `benchmark.json` and `info.json` describing the
    PREVIOUS 99 — and the way anyone found out was a failing test hours later.
    The rest now comes from `regen.STAGE_NAMES` rather than a second hand-written
    order, so there is one home for what depends on what.
    """
    import inspect

    from manamap.pilot import deck_branch, regen

    src = inspect.getsource(deck_branch)
    assert "regen.run(slug=slug" in src, "merge no longer runs the build"
    assert "regen.STAGE_NAMES" in src, (
        "the post-merge stage list is hand-written again — it will drift from "
        "regen's, which is the order that is actually correct")
    assert "diagnose" in regen.STAGE_NAMES and "benchmark" in regen.STAGE_NAMES


def test_merge_reports_what_a_rebuild_cannot_fix():
    """The other half of the build. An AUTHORED declaration naming cards the new
    list does not run cannot be regenerated — nobody but the pilot can say which
    components the new deck has — so the merge has to NAME it rather than leave
    it for a validator to find later.

    Asserted on the mechanism rather than on a deck's contents: which artifacts
    are invalid depends on what was merged, and pinning that would be a test
    about one experiment.
    """
    import inspect

    from manamap.pilot import deck_branch

    assert callable(getattr(deck_branch, "_validate_after_merge", None))
    src = inspect.getsource(deck_branch._validate_after_merge)
    assert "VALIDATED" in src, "it should ask deck_status which gates exist"
    # It must REPORT, never repair — prose is not hand-patched to green a gate.
    assert "write" not in src.lower().replace("written", ""), (
        "the post-merge validator appears to write something")
