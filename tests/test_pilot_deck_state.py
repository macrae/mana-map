"""The deck lifecycle: where it lives, who writes it, and what it contradicts.

The failure this whole module exists for: `yawgmoth-swarm` was broken down for
parts and rendered under **SLEEVED — you can play these tonight** for weeks,
because the status was a hand edit nobody made and the front door filtered on the
paper lock first. Two facts about one piece of cardboard, in two files, with no
writer and no gate.
"""

import json

import pytest

from manamap.config import DATA_DIR
from manamap.pilot import deck_delete, deck_state, deck_versions
from manamap.pilot.common import (
    DECK_STATUSES, UNPLAYABLE_STATUSES, deck_is_apart, deck_lifecycle)
from manamap.pilot.validate_deck_versions import validate

DECKS = DATA_DIR / "decks"


# ── the home ────────────────────────────────────────────────────────────

def test_the_lifecycle_is_read_from_deck_versions_and_nowhere_else():
    """ONE HOME, NO FALLBACK. Reading `deck_versions.json` and then falling back
    to `issue.json` would be two homes for one fact — the divergence this repo
    keeps paying for. `validate-issue` reports a leftover key instead."""
    checked = 0
    for deck in sorted(DECKS.iterdir()):
        if not deck.is_dir():
            continue
        checked += 1
        doc = json.loads((deck / "deck_versions.json").read_text()) \
            if (deck / "deck_versions.json").exists() else {}
        expected = (doc.get("lifecycle") or {}).get("status")
        life = deck_lifecycle(deck.name)
        assert (life[0] if life else None) == (
            expected if expected in DECK_STATUSES else None), deck.name
    assert checked >= 8


def test_no_issue_json_still_carries_a_status():
    """The migration moved three and `validate-issue` now rejects the key. A
    leftover would be OBEYED BY NOBODY while looking exactly like it worked,
    which is quieter than the typo the old check was written for."""
    checked = 0
    for path in sorted(DECKS.glob("*/issue.json")):
        checked += 1
        assert "status" not in json.loads(path.read_text()), (
            f"{path} still carries a status — it moved to deck_versions.json")
    assert checked >= 5


def test_the_three_migrated_decks_kept_their_verdicts():
    """The golden. A migration that changes what `deck_is_apart` answers for any
    deck has moved a fact, not a file."""
    assert deck_lifecycle("hapatra")[0] == "broken-down"
    assert deck_lifecycle("radagast")[0] == "broken-down"
    assert deck_lifecycle("sisay")[0] == "retired"
    for slug in ("hapatra", "radagast", "sisay"):
        assert deck_is_apart(slug), slug
    for slug in ("ur-dragon", "edgar-vampires", "goblin-storm"):
        assert not deck_is_apart(slug), slug


# ── the writer ──────────────────────────────────────────────────────────

def _scratch(tmp_path, monkeypatch, paper=None, lifecycle=None):
    """A deck directory the writer can be pointed at without touching the fleet."""
    base = tmp_path / "decks" / "scratch"
    base.mkdir(parents=True)
    doc = {"slug": "scratch", "tags": {}}
    if paper:
        doc["paper"] = paper
    if lifecycle:
        doc["lifecycle"] = lifecycle
    (base / "deck_versions.json").write_text(json.dumps(doc))
    monkeypatch.setattr(deck_versions, "deck_dir", lambda slug, branch=None: base)
    return base


PAPER = {"version": 5, "sha": "a2a16459cafe",
         "decklist_sha256": "a" * 64, "built_at": "2026-08-29", "note": ""}


def test_archiving_withdraws_the_paper_lock(tmp_path, monkeypatch):
    """The contradiction resolved at the source. "These cards are in a pile" and
    "this exact 99 is in sleeves" cannot both be true, and leaving both set is
    what let the workbench guess — wrongly."""
    base = _scratch(tmp_path, monkeypatch, paper=PAPER)
    block, withdrew = deck_versions.set_lifecycle(
        "scratch", status="broken-down", reason="parts for zur")
    assert block["status"] == "broken-down"
    assert block["reason"] == "parts for zur"
    assert withdrew["version"] == 5, "the withdrawn lock must be RETURNED, not dropped"
    doc = json.loads((base / "deck_versions.json").read_text())
    assert "paper" not in doc
    assert doc["lifecycle"]["status"] == "broken-down"


def test_superseding_a_sleeved_deck_leaves_the_lock_alone(tmp_path, monkeypatch):
    """THE TEST THAT PINS WHY `UNPLAYABLE_STATUSES` IS REUSED rather than a
    fourth list of dead statuses being born in the writer. A superseded list is
    still sleeved and still playable — it is just no longer the best version of
    itself — so `superseded` + a live lock is a legal, meaningful state."""
    assert "superseded" not in UNPLAYABLE_STATUSES
    base = _scratch(tmp_path, monkeypatch, paper=PAPER)
    _block, withdrew = deck_versions.set_lifecycle("scratch", status="superseded")
    assert withdrew is None
    doc = json.loads((base / "deck_versions.json").read_text())
    assert doc["paper"]["version"] == 5


def test_reviving_clears_the_mark_and_refuses_twice(tmp_path, monkeypatch):
    base = _scratch(tmp_path, monkeypatch,
                    lifecycle={"status": "retired", "at": "2026-08-18", "reason": ""})
    deck_versions.set_lifecycle("scratch", clear=True)
    assert "lifecycle" not in json.loads((base / "deck_versions.json").read_text())
    with pytest.raises(SystemExit, match="already live"):
        deck_versions.set_lifecycle("scratch", clear=True)


def test_an_unknown_status_is_refused_by_the_writer(tmp_path, monkeypatch):
    _scratch(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="not a lifecycle status"):
        deck_versions.set_lifecycle("scratch", status="mothballed")


def test_the_pilots_words_map_onto_the_vocabulary():
    """`archive` is the pilot's word for the rack; `broken-down` is what the rest
    of the repo reads. Every action must land on a real status."""
    for action, status in deck_state.ACTIONS.items():
        assert status in DECK_STATUSES, action
    assert deck_state.ACTIONS["archive"] == "broken-down"


# ── the gate ────────────────────────────────────────────────────────────

def test_the_validator_is_silent_on_every_tracked_file():
    """A VALIDATOR THAT FIRES ON CORRECT DATA IS WORSE THAN NO VALIDATOR, and the
    only way to know is to measure it against the whole fleet."""
    checked = 0
    for path in sorted(DECKS.glob("*/deck_versions.json")):
        checked += 1
        errs = validate(json.loads(path.read_text()), slug=path.parent.name)
        assert errs == [], f"{path.parent.name}: {errs}"
    assert checked >= 8


def test_the_validator_catches_a_dead_deck_that_is_still_sleeved():
    """THE BUG THIS GATE WAS WRITTEN FOR, re-introduced by hand.

    On disk this state is now unreachable — `set_lifecycle` withdraws the lock —
    but the file is authored and tracked, so a hand edit can still create it, and
    the workbench renders it under SLEEVED.
    """
    doc = {"slug": "zombie", "lifecycle": {"status": "broken-down", "at": "2026-09-01"},
           "paper": dict(PAPER), "tags": {}}
    errs = validate(doc, slug="zombie")
    assert any("cannot also have an exact 99 in sleeves" in e for e in errs), errs


def test_the_validator_catches_a_misspelled_status():
    """`deck_status_of` returns None for an unknown value on purpose — a typo
    must not take the workbench offline — so the deck reads as LIVE. Tolerated
    there, reported here."""
    doc = {"slug": "z", "lifecycle": {"status": "brokendown", "at": "2026-09-01"},
           "tags": {}}
    assert any("reads as LIVE" in e for e in validate(doc, slug="z"))


def test_the_validator_catches_a_nearly_release_tag():
    doc = {"slug": "z", "tags": {"v1.2": {"version": 1, "decklist_sha256": "b" * 64}}}
    assert any("nearly a release" in e for e in validate(doc, slug="z"))


def test_the_validator_catches_a_truncated_sha():
    doc = {"slug": "z", "tags": {}, "paper": {"version": 1, "built_at": "2026-01-01",
                                              "decklist_sha256": "abc123"}}
    assert any("64 hex" in e for e in validate(doc, slug="z"))


# ── the destructive verb ────────────────────────────────────────────────

def test_delete_refuses_every_deck_that_is_a_record():
    """A destructive command that never refuses has not been tested. Every deck
    that was sleeved, played or published must come back with a reason."""
    checked = 0
    for slug in ("goblin-storm", "ur-dragon", "edgar-vampires", "heliod",
                 "yawgmoth-swarm", "hapatra", "sisay", "radagast"):
        if not (DECKS / slug).is_dir():
            continue
        checked += 1
        why = deck_delete.blockers(slug)
        assert why, f"{slug} would be deleted with no objection"
    assert checked >= 6


def test_the_refusal_is_not_keyed_on_the_magazine_renderer_alone():
    """`published` means "the FROZEN renderer ran", and manual-v5 retires it —
    a destructive gate keyed only on that inverts silently. Two of the three
    questions must survive the unfreeze."""
    why = deck_delete.blockers("gishath")
    assert any("logged game" in w or "SLEEVED" in w for w in why), why


def test_locking_an_archived_deck_is_refused(tmp_path, monkeypatch):
    """THE OTHER HALF OF THE INVARIANT, and it was found by tripping it.

    `set_lifecycle` withdraws the lock when a deck is archived. Without the
    matching refusal in `set_paper`, the identical contradiction can be built
    from the other side — and it is easy to do by accident, because
    `deck-version <slug> paper` with no ref is a WRITE that reads like a report.
    Minutes after archiving yawgmoth-swarm, running it to CHECK the lock had
    gone silently re-locked a deck whose cards are in a pile. The validator
    caught it, which is the gate working; this stops it happening.
    """
    base = tmp_path / "decks" / "scratch"
    base.mkdir(parents=True)
    (base / "deck_versions.json").write_text(json.dumps({
        "slug": "scratch", "tags": {},
        "lifecycle": {"status": "broken-down", "at": "2026-09-01", "reason": ""}}))
    monkeypatch.setattr(deck_versions, "deck_dir", lambda slug, branch=None: base)
    monkeypatch.setattr(deck_versions.common, "DECKS_DIR", tmp_path / "decks")
    with pytest.raises(SystemExit, match="cards are in a pile"):
        deck_versions.set_paper("scratch")


def test_no_tracked_deck_is_both_dead_and_sleeved():
    """The fleet-wide assertion, so the invariant is not only checked one deck at
    a time by a validator somebody has to remember to run."""
    checked = 0
    for path in sorted(DECKS.glob("*/deck_versions.json")):
        doc = json.loads(path.read_text())
        status = (doc.get("lifecycle") or {}).get("status")
        checked += 1
        assert not (status in UNPLAYABLE_STATUSES and doc.get("paper")), (
            f"{path.parent.name} is {status} and still carries a paper lock")
    assert checked >= 8


def test_the_holder_report_reads_the_path_the_artifact_actually_uses():
    """A REPORT THAT IS SILENT LOOKS EXACTLY LIKE A REPORT WITH NOTHING TO SAY.

    `holders` first read a top-level `cost` key on `net_change.json`. There is
    none — the block lives at `recommendation.cost` — so it returned [] for every
    deck and printed nothing, while Edgar's `bloodline-v4` was `mergeable: false`
    on "unsleeve The Ozolith from kianne", a deck that has never existed in paper.
    Caught by deleting kianne and noticing the report say nothing about a claim
    that had been read out of the same file by hand ten minutes earlier.

    Driven against the real committed artifact rather than a fixture: the point
    is the SHAPE on disk, and a fixture would have re-encoded the wrong guess.
    """
    branch = DECKS / "edgar-vampires" / "branches" / "bloodline-v4" / "net_change.json"
    if not branch.exists():
        pytest.skip("bloodline-v4 has been merged or deleted")
    doc = json.loads(branch.read_text())
    rows = ((doc.get("recommendation") or {}).get("cost") or {}).get("must_unsleeve")
    assert rows is not None, (
        "net_change.json no longer carries recommendation.cost.must_unsleeve — "
        "deck_delete.holders reads that path and would go silent again")
