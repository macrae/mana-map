"""`branch.json` — the last tracked pilot artifact that had no gate.

Every other artifact on a branch was validated and freshness-tested;
`branch.json`, which holds the objective the branch is graded against and now the
pilot's acceptance of it, was checked by nothing at all. It shipped that way and
survived two development cycles because branches were the newest thing here and
nothing had gone wrong yet.
"""

import pytest

from conftest import A_BRANCH, requires_branch, requires_deck
from manamap.pilot import deck_branch, validate_branch


def _doc(**over):
    doc = {"slug": "d", "branch": "b", "opened": "2026-01-01", "v": 2,
           "objective": {"axis": "damage_8", "op": ">=", "value": 40.0}}
    doc.update(over)
    return doc


def _proposal(**over):
    p = {"at": "2026-01-02", "as_version": "v1.0.2", "base_version": 2,
         "decklist_sha256": "a" * 64,
         "accepted_on": {"decklist_sha256": "b" * 64, "state": "a trade"}}
    p.update(over)
    return p


def test_a_well_formed_branch_passes():
    assert validate_branch.validate(_doc()) == []
    assert validate_branch.validate(_doc(proposal=_proposal())) == []


def test_a_v2_branch_owes_an_objective():
    doc = _doc()
    doc.pop("objective")
    assert any("no objective" in e for e in validate_branch.validate(doc))


def test_a_v1_branch_predates_the_requirement_and_still_loads():
    """A file that loaded yesterday must load today — the same allowance
    `deck_branch.new` documents for the objective it added."""
    doc = _doc(v=1)
    doc.pop("objective")
    assert not any("objective" in e for e in validate_branch.validate(doc))


@pytest.mark.parametrize("axis", deck_branch.MEMBERSHIP_AXES)
def test_an_authored_axis_is_refused_at_rest_and_not_only_at_the_prompt(axis):
    """`parse_objective` refuses it on the way in; this catches a hand-edited
    file, which is how the Ur-Dragon objective got there in the first place."""
    doc = _doc(objective={"axis": axis, "op": ">=", "value": 0.22})
    errors = validate_branch.validate(doc)
    assert any("AUTHORED" in e for e in errors)


def test_an_axis_the_bench_does_not_measure_is_refused():
    doc = _doc(objective={"axis": "vibes", "op": ">=", "value": 1})
    assert any("not something the bench measures" in e
               for e in validate_branch.validate(doc))


# ── the proposal ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing", validate_branch.PROPOSAL_REQUIRED)
def test_a_proposal_owes_every_key_the_state_machine_reads(missing):
    p = _proposal()
    p.pop(missing)
    assert any(f"no {missing!r}" in e
               for e in validate_branch.validate(_doc(proposal=p)))


def test_a_proposal_must_say_what_it_intends_to_become():
    """Without `as_version` a proposal cannot be found to have been OUTRUN by
    another merge, which is the whole reason `base_version` is recorded."""
    p = _proposal(as_version="next")
    errors = validate_branch.validate(_doc(proposal=p))
    assert any("release tag" in e for e in errors)


def test_a_proposal_freezes_the_sha_the_report_measured():
    """One sha without the other cannot answer "has this gone stale", and
    staleness is the only thing between a decision and a list that moved under
    it."""
    p = _proposal(accepted_on={"state": "a trade"})
    errors = validate_branch.validate(_doc(proposal=p))
    assert any("has moved since" in e for e in errors)


def test_a_do_not_merge_accepted_anyway_must_say_why():
    p = _proposal(accepted_on={"decklist_sha256": "b" * 64,
                               "state": "do not merge"})
    assert any("forced_reason" in e for e in validate_branch.validate(_doc(proposal=p)))
    p["forced_reason"] = "the log disagrees with the goldfish"
    assert validate_branch.validate(_doc(proposal=p)) == []


def test_a_state_that_is_not_a_net_change_state_is_refused():
    p = _proposal(accepted_on={"decklist_sha256": "b" * 64, "state": "lgtm"})
    assert any("not a net-change state" in e
               for e in validate_branch.validate(_doc(proposal=p)))


def test_proxy_is_named_cards_and_never_a_boolean():
    """A decision about specific cardboard is recorded as specific cardboard.
    `--proxy` was a bare flag for exactly as long as it was never persisted."""
    assert any("never a boolean" in e for e in
               validate_branch.validate(_doc(proposal=_proposal(proxy=True))))
    assert validate_branch.validate(
        _doc(proposal=_proposal(proxy=["Bloom Tender"]))) == []


def test_a_procurement_note_that_says_nothing_is_refused():
    assert any("procurement" in e for e in
               validate_branch.validate(_doc(proposal=_proposal(procurement={}))))


# ── objective_history: hand-written, so it owes a shape ──────────────────

def test_an_objective_history_entry_owes_its_three_keys():
    """Nothing in the repo writes or reads this key — it was hand-added when
    ur-dragon's objective moved off an authored axis. An ungated key that only
    ever arrives by hand is the shape a typo lives in forever."""
    doc = _doc(objective_history=[{"at": "2026-08-28"}])
    errors = validate_branch.validate(doc)
    assert any("'was'" in e for e in errors)
    assert any("'why_changed'" in e for e in errors)


def test_an_objective_history_that_is_not_a_list_is_refused():
    assert any("not a list" in e
               for e in validate_branch.validate(_doc(objective_history={})))


# ── against the real fleet ───────────────────────────────────────────────

def test_the_slug_and_the_directory_must_agree():
    errors = validate_branch.validate(_doc(), slug="other", branch="b")
    assert any("lives under" in e for e in errors)
    errors = validate_branch.validate(_doc(), slug="d", branch="elsewhere")
    assert any("the directory is" in e for e in errors)


@requires_branch
@requires_deck
def test_every_tracked_branch_passes_its_own_gate():
    import glob
    import json
    checked = 0
    for path in glob.glob("data/decks/*/branches/*/branch.json"):
        parts = path.split("/")
        doc = json.loads(open(path).read())
        assert validate_branch.validate(
            doc, slug=parts[2], branch=parts[4]) == [], path
        checked += 1
    assert checked >= 1, "no tracked branch to check"
