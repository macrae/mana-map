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

from manamap.pilot import deck_branch
from manamap.pilot.common import DECKS_DIR, deck_dir, deck_file

SLUG = "ur-dragon"
BRANCH = "treasure-v2"

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

    from manamap.pilot import bracket, deck_map, mana_analysis

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

    for mod in (bracket, mana_analysis, deck_map):
        mod.main(args())

    after = _tree(root)
    changed = sorted(k for k in before if before[k] != after.get(k))
    added = sorted(set(after) - set(before))
    assert not changed, f"a branch run rewrote the deck's own artifact(s): {changed}"
    assert not added, f"a branch run created file(s) in the deck's directory: {added}"


@needs_branch
def test_reads_fall_back_to_the_deck_but_writes_do_not():
    """A branch inherits AUTHORED inputs and owns its MEASUREMENTS.

    Nobody writes a second `goldfish_targets.json` to try a candidate list, so a
    branch that measured against no engine declaration would be reporting a
    different deck rather than a different list.
    """
    # authored, branch has none -> falls back to the deck's
    targets = deck_file(SLUG, "goldfish_targets.json", BRANCH)
    assert targets == deck_dir(SLUG) / "goldfish_targets.json"
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
    sourcing split on the page — and `next` is what tells the pilot which of the
    two states the branch is in: a decision waiting to be taken, or a shopping
    list.
    """
    from manamap.pilot import deck_info
    info = deck_info.compose(SLUG)
    rows = info.get("branches") or []
    assert any(b["name"] == BRANCH for b in rows), "the branch is absent from info.json"
    b = next(x for x in rows if x["name"] == BRANCH)
    assert set(b["counts"]) == {"in_deck", "box", "elsewhere", "buy"}
    joined = " ".join(info["next"])
    assert BRANCH in joined, f"`next` never mentions the branch: {info['next']}"
    if b["mergeable"]:
        assert "merge" in joined
    else:
        assert "source" in joined


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
