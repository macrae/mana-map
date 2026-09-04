"""The environment ladder (pilot/promote.py).

PRD §3 calls the three environments the spine. Before this the ladder had two
rungs and one of them was `living.filter(e => !e.locked)` in `workbench.js` —
computed in the browser, stored nowhere, gated on by nothing.

These pin the two design decisions that keep it from becoming a second source of
truth: SLEEVED is the paper lock and is never stored, and BENCH is the default so
its absence keeps meaning what it already meant on ten existing files.
"""

import json
from types import SimpleNamespace

import pytest

from conftest import requires_deck

from manamap.pilot import deck_versions, promote
from manamap.pilot.promote import BENCH, DEV, GATES, LADDER, SLEEVED


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """A deck directory with nothing in it but a `deck_versions.json`."""
    root = tmp_path / "decks"
    (root / "x").mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", root)
    monkeypatch.setattr("manamap.pilot.deck_versions.DECKS_DIR", root,
                        raising=False)
    return root / "x"


def _versions(deck, **blocks):
    doc = {"slug": "x", "tags": {}, **blocks}
    (deck / "deck_versions.json").write_text(json.dumps(doc), encoding="utf-8")


# ── the ladder itself ───────────────────────────────────────────────────────

def test_the_ladder_is_ordered_and_a_promotion_cannot_skip():
    """Index is the rung, so skipping one skips its gate entirely."""
    assert LADDER == (DEV, BENCH, SLEEVED)
    assert promote._next_rung(DEV, +1) == BENCH
    assert promote._next_rung(BENCH, +1) == SLEEVED
    assert promote._next_rung(SLEEVED, +1) is None
    assert promote._next_rung(DEV, -1) is None


def test_sleeved_is_the_paper_lock_and_is_never_stored(deck):
    """A stored `sleeved` would be a second claim about cardboard.

    The workbench has already been burned by that shape once — it filtered on
    `locked` before `status` and rendered a broken-down deck under SLEEVED — so
    the lock stays the single source and this refuses to compete with it.
    """
    _versions(deck, paper={"version": 1, "decklist_sha256": "a" * 64,
                           "built_at": "2026-01-01"})
    assert promote.stage("x") == SLEEVED

    with pytest.raises(SystemExit) as exc:
        promote.set_stage("x", SLEEVED)
    assert "paper" in str(exc.value)


def test_bench_is_the_default_and_writing_it_stores_nothing(deck):
    """Storing the default would make its ABSENCE mean something new.

    Ten decks on disk have no stage block. If `bench` were written, every one of
    them would become "unknown" rather than "on the bench", which is a migration
    disguised as a feature.
    """
    _versions(deck)
    assert promote.stage("x") == BENCH
    assert promote.stored("x") is None

    promote.set_stage("x", DEV)
    assert promote.stage("x") == DEV and promote.stored("x") == DEV

    promote.set_stage("x", BENCH)
    assert promote.stage("x") == BENCH
    assert promote.stored("x") is None, "bench must leave no trace"
    # And with nothing else asserted, no trace means no FILE — see
    # `test_a_round_trip_through_dev_leaves_no_artifact_behind`.
    assert not (deck / "deck_versions.json").exists()


def test_the_lock_wins_over_a_stored_stage(deck):
    """They cannot be allowed to disagree, and the lock is the claim."""
    _versions(deck, stage={"name": DEV, "at": "2026-01-01"},
              paper={"version": 1, "decklist_sha256": "a" * 64,
                     "built_at": "2026-01-01"})
    assert promote.stage("x") == SLEEVED


def test_an_unknown_stage_is_refused(deck):
    _versions(deck)
    with pytest.raises(SystemExit) as exc:
        promote.set_stage("x", "prod")
    assert "not a stage" in str(exc.value)


def test_the_stage_is_written_by_the_one_writer_of_that_file(deck):
    """`_write_tags` orders the keys, so `stage` cannot land out of place."""
    _versions(deck, tags={"v1.0.0": {"decklist_sha256": "b" * 64}})
    promote.set_stage("x", DEV)
    doc = json.loads((deck / "deck_versions.json").read_text(encoding="utf-8"))
    assert list(doc) == ["slug", "stage", "tags"], list(doc)


# ── the gate reports per requirement ────────────────────────────────────────

def test_every_gate_row_names_an_artifact_a_reason_and_a_way_forward():
    for rung, rows in GATES.items():
        assert rows, rung
        for label, artifact, why in rows:
            assert label and artifact and why, (rung, label)


def test_the_gates_tighten_going_up():
    """PRD §3's whole claim: requirements TIGHTEN at each promotion.

    A rung that asked for less than the one below it would let a deck improve
    its way out of a requirement.
    """
    assert set(GATES) == {BENCH, SLEEVED}
    assert len(GATES[SLEEVED]) >= 5
    bench_artifacts = {a for _, a, _ in GATES[BENCH]}
    sleeved_artifacts = {a for _, a, _ in GATES[SLEEVED]}
    assert not (bench_artifacts & sleeved_artifacts), \
        "a requirement is checked at the rung that first needs it, once"


@requires_deck
def test_a_real_deck_reports_a_gate_row_for_every_requirement():
    rows = promote.gate("zur-enchantress", SLEEVED)
    assert len(rows) == len(GATES[SLEEVED])
    for row in rows:
        assert row["state"] in ("present", "missing", "STALE", "INVALID",
                                "unknown"), row
        assert row["label"] and row["why"]


@requires_deck
def test_blockers_are_the_rows_that_are_not_present():
    """`present` is the only passing state — STALE and INVALID are not."""
    rows = [{"state": s, "label": s} for s in
            ("present", "missing", "STALE", "INVALID", "unknown")]
    assert [r["label"] for r in promote.blockers(rows)] == \
        ["missing", "STALE", "INVALID", "unknown"]


# ── ownership stays three buckets ───────────────────────────────────────────

@requires_deck
def test_ownership_keeps_deck_membership_beside_the_boxes_never_inside():
    """C-3 asks for three groups, and the repo forbids folding two of them.

    `collection` is deliberately the only reader of the boxes and deliberately
    does not count deck membership: a card sleeved in another deck is not one
    you can put in this deck without taking that one apart. So "in another deck"
    is REPORTED beside ownership, with the deck named — which is also C-3's
    conflict warning.
    """
    state, detail, missing = promote._ownership("zur-enchantress")
    assert state in ("present", "missing", "unknown")
    if state == "missing":
        assert "in a box" in detail and "sleeved elsewhere" in detail \
            and "to buy" in detail
        assert any("(in " in m for m in missing), \
            "a card held by another deck must name that deck"


@requires_deck
def test_basics_are_not_a_shopping_list():
    """Every deck would otherwise fail its ownership gate on Swamps."""
    _, _, missing = promote._ownership("zur-enchantress")
    for basic in ("Swamp", "Island", "Plains", "Mountain", "Forest"):
        assert not any(m.startswith(basic) for m in missing), basic


# ── a dead deck is not on the ladder ────────────────────────────────────────

@requires_deck
def test_an_archived_deck_does_not_move_between_environments():
    """A deck in a pile is not in an environment; it is history."""
    args = SimpleNamespace(slug="hapatra", to=None, show=True, force=False,
                           reason=None, pilot_command="promote")
    with pytest.raises(SystemExit) as exc:
        promote.main(args)
    assert "revive" in str(exc.value)


@requires_deck
def test_a_sleeved_deck_reports_sleeved():
    assert promote.stage("edgar-vampires") == SLEEVED
    assert deck_versions.paper("edgar-vampires")


@requires_deck
def test_regen_already_sweeps_exactly_the_sleeved_decks():
    """The ladder needed no change here, and that is worth asserting.

    `regen.is_pinned` reads the paper lock, and SLEEVED *is* the paper lock — so
    "sleeved decks regenerate automatically, everything else is asked for" was
    already the rule the ladder describes. A second predicate would have been
    the bug this module is shaped to avoid.
    """
    from manamap.pilot import regen

    for slug in ("edgar-vampires", "zur-enchantress"):
        assert regen.is_pinned(slug) is (promote.stage(slug) == SLEEVED), slug


def test_a_round_trip_through_dev_leaves_no_artifact_behind(deck):
    """`deck_versions.json` holds ASSERTIONS, and the default is not one.

    Found by doing it to a real deck: `demote` wrote the file to store `dev`,
    `promote --to bench` popped the key, and what was left was a tracked
    artifact containing nothing, on a deck that had never had one.
    """
    path = deck / "deck_versions.json"
    assert not path.exists()

    promote.set_stage("x", BENCH)
    assert not path.exists(), "the default must not create a file"

    promote.set_stage("x", DEV)
    assert path.exists() and promote.stage("x") == DEV

    promote.set_stage("x", BENCH)
    assert not path.exists(), "an empty assertion file must not survive"
    assert promote.stage("x") == BENCH


def test_a_file_holding_other_claims_survives_the_round_trip(deck):
    """Only an EMPTY one goes. A tag or a lock is a claim and must not be lost."""
    path = deck / "deck_versions.json"
    _versions(deck, tags={"v1.0.0": {"decklist_sha256": "c" * 64}})

    promote.set_stage("x", DEV)
    promote.set_stage("x", BENCH)

    assert path.exists()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["tags"] == {"v1.0.0": {"decklist_sha256": "c" * 64}}
    assert promote.STAGE_KEY not in doc
