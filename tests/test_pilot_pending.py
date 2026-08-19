"""The pending queue: a decision that outlives the session that made it.

Nine decks, decisions made in conversation, and before this there was no home for
a change that had been decided but not applied. Applied swaps are derived from
git; proposed ones live in `considering.json`, which is fixed at exactly ten
entries, forbids a pick already in the deck, and is regenerated wholesale on any
decklist edit. A three-land swap decided on a Tuesday fell straight through the
gap — which is what happened to yawgmoth's mono-black land fix.

Two properties carry the whole design, and both are tested here:

1. **Closure is DERIVED, never declared.** There is no `applied: true` field,
   because a hand-set flag is exactly how `HISTORY.md` became "append-only and
   append-forgotten". The deck itself says whether a change landed.
2. **The cuts decide it, not the additions.** The first implementation counted an
   `in` card as arrived whenever the deck contained it — and reported "+3 Swamp"
   as half-done before anything happened, because the deck already ran 21. That
   bug is pinned below so it cannot come back.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot import validate_pending as vp

DECK = {"cards": [{"name": "Swamp", "quantity": 21}, {"name": "City of Brass"},
                  {"name": "Exotic Orchard"}, {"name": "Sol Ring"}]}
NAMES = {c["name"] for c in DECK["cards"]}


def _entry(**kw):
    base = {"id": "x-1", "decided": "2026-08-17", "why": "because"}
    base.update(kw)
    return base


# ── the derived-closure rule ──────────────────────────────────────────────

def test_cuts_still_in_the_deck_means_open():
    e = _entry(**{"in": ["Swamp"], "out": ["City of Brass"]})
    assert vp.state_of(e, NAMES) == vp.OPEN


def test_all_cuts_gone_means_applied():
    e = _entry(**{"in": ["Swamp"], "out": ["Gone Card"]})
    assert vp.state_of(e, NAMES) == vp.APPLIED


def test_some_cuts_gone_means_partial():
    e = _entry(**{"in": ["Swamp"], "out": ["City of Brass", "Gone Card"]})
    assert vp.state_of(e, NAMES) == vp.PARTIAL


def test_an_addition_the_deck_already_runs_does_not_prove_arrival():
    """The bug this test exists for: adding three Swamps to a deck that already
    runs twenty-one reported PARTIAL before anything was applied, because
    presence was read as arrival. Cuts are unambiguous; presence is not."""
    e = _entry(**{"in": ["Swamp", "Swamp", "Swamp"],
                  "out": ["City of Brass", "Exotic Orchard"]})
    assert vp.state_of(e, NAMES) == vp.OPEN


def test_a_pure_addition_falls_back_to_its_in_cards():
    """The one case where presence IS the signal — there are no cuts to read."""
    assert vp.state_of(_entry(**{"in": ["Sol Ring"]}), NAMES) == vp.APPLIED
    assert vp.state_of(_entry(**{"in": ["Nothing Here"]}), NAMES) == vp.OPEN


def test_there_is_no_applied_flag_in_the_schema():
    """A hand-set flag is the failure `HISTORY.md` had. If someone adds one, the
    deck must still be the authority — this asserts the field is not consulted."""
    e = _entry(**{"in": ["Swamp"], "out": ["City of Brass"], "applied": True})
    assert vp.state_of(e, NAMES) == vp.OPEN


# ── the shape considering.json cannot express ─────────────────────────────

def test_in_and_out_are_lists_so_one_swap_is_one_decision():
    e = _entry(**{"in": ["Swamp", "Swamp", "Swamp"],
                  "out": ["City of Brass", "Exotic Orchard", "Survivors' Encampment"]})
    assert vp._names(e, "in") == ["Swamp"] * 3
    assert len(vp._names(e, "out")) == 3


def test_a_bare_string_is_accepted_as_a_one_card_list():
    assert vp._names(_entry(**{"in": "Sol Ring"}), "in") == ["Sol Ring"]


# ── the validator ─────────────────────────────────────────────────────────

def test_an_empty_queue_is_valid():
    assert vp.validate({"slug": "x", "pending": []}, DECK) == []


@pytest.mark.parametrize("bad,expect", [
    ({"id": "Not Kebab", "decided": "2026-08-17", "why": "w"}, "kebab"),
    ({"id": "ok", "decided": "17-08-2026", "why": "w"}, "YYYY-MM-DD"),
    ({"id": "ok", "decided": "2026-08-17", "why": "  "}, "why is empty"),
])
def test_form_errors_are_reported(bad, expect):
    errs = vp.validate({"slug": "x", "pending": [dict(bad, **{"out": ["City of Brass"]})]}, DECK)
    assert any(expect in e for e in errs), errs


def test_an_entry_with_nothing_to_apply_is_an_error():
    errs = vp.validate({"slug": "x", "pending": [_entry()]}, DECK)
    assert any("neither an `in` nor an `out`" in e for e in errs), errs


def test_a_half_satisfied_entry_reports_partial_rather_than_erroring():
    """The ambiguity this design refuses to guess at.

    A cut naming a card no longer in the deck means EITHER the entry was applied
    OR that card left for an unrelated reason, and nothing in the data separates
    them. So there is no stranded-cut error — PARTIAL is the signal, and a human
    reads the entry. (The obvious check would also be unreachable: any entry with
    a cut already gone is PARTIAL or APPLIED, never OPEN.)
    """
    half = _entry(**{"out": ["City of Brass", "Never Here"]})
    assert vp.state_of(half, NAMES) == vp.PARTIAL
    assert vp.validate({"slug": "x", "pending": [half]}, DECK) == []


def test_settled_by_must_name_a_real_routine():
    e = _entry(**{"out": ["City of Brass"], "settled_by": "make-it-good"})
    errs = vp.validate({"slug": "x", "pending": [e]}, DECK,
                       known_settlers={"fetch-deck", "goldfish"})
    assert any("settled_by" in x for x in errs), errs
    ok = _entry(**{"out": ["City of Brass"], "settled_by": "fetch-deck"})
    assert vp.validate({"slug": "x", "pending": [ok]}, DECK,
                       known_settlers={"fetch-deck", "goldfish"}) == []


def test_duplicate_ids_are_rejected():
    a = _entry(**{"out": ["City of Brass"]})
    b = _entry(**{"out": ["Exotic Orchard"]})
    errs = vp.validate({"slug": "x", "pending": [a, b]}, DECK)
    assert any("duplicate id" in e for e in errs), errs


# ── the tracked artifact ──────────────────────────────────────────────────

@pytest.mark.skipif(not (DECKS_DIR / "yawgmoth-swarm" / "pending.json").exists(),
                    reason="requires the tracked decks")
def test_yawgmoths_queued_land_swap_is_open_and_valid():
    doc = json.loads((DECKS_DIR / "yawgmoth-swarm" / "pending.json").read_text())
    deck = json.loads((DECKS_DIR / "yawgmoth-swarm" / "cards.json").read_text())
    assert vp.validate(doc, deck) == []
    summary = vp.summarise("yawgmoth-swarm")
    assert summary["open"] == 1 and summary["applied"] == 0
    entry = summary["entries"][0]
    assert entry["state"] == vp.OPEN
    assert "City of Brass" in entry["out"]


@pytest.mark.skipif(not DECKS_DIR.exists(), reason="requires the tracked decks")
def test_every_tracked_pending_file_validates():
    """A queue that fails its own validator is worse than no queue."""
    for path in sorted(DECKS_DIR.glob("*/pending.json")):
        slug = path.parent.name
        doc = json.loads(path.read_text())
        deck = json.loads((path.parent / "cards.json").read_text())
        assert vp.validate(doc, deck) == [], f"{slug}: {vp.validate(doc, deck)}"


# ── deck-status reports VALIDITY, not just bookkeeping ────────────────────

def test_deck_status_runs_the_gates_and_reports_a_failure(tmp_path, monkeypatch):
    """The gap this closes, pinned.

    `deck-status` compared shas and counted files and called that health. The
    gates existed and nothing in the command ran them, so PLAN.md recorded it
    reading nine decks green while two failed their own validators — and it did
    it again live on ur-dragon mid-swap: `deck-status` FAIL=0 while
    `validate-issue` (the legacy plan gate) FAIL=1 on the same deck in the same second.

    A dashboard that is green while the gate is red is worse than no dashboard,
    because people stop checking the gate.
    """
    from manamap.pilot import deck_status as ds

    monkeypatch.setitem(ds.VALIDATED, "cards.json", "tests._always_fails")
    import sys
    import types
    mod = types.ModuleType("tests._always_fails")

    def _main(args):
        print("  - deliberately broken")
        raise SystemExit(1)

    mod.main = _main
    sys.modules["tests._always_fails"] = mod
    try:
        ok, why = ds._validity("yawgmoth-swarm", "cards.json")
        assert ok is False and "error(s)" in why, (ok, why)
    finally:
        del sys.modules["tests._always_fails"]


def test_a_validator_that_raises_is_not_a_green_deck(monkeypatch):
    """A broken gate must never read as a passing artifact — silence is the
    failure mode this whole file exists to stop."""
    from manamap.pilot import deck_status as ds
    import sys
    import types
    mod = types.ModuleType("tests._explodes")

    def _main(args):
        raise ValueError("gate itself is broken")

    mod.main = _main
    sys.modules["tests._explodes"] = mod
    monkeypatch.setitem(ds.VALIDATED, "cards.json", "tests._explodes")
    try:
        ok, why = ds._validity("yawgmoth-swarm", "cards.json")
        assert ok is False and "ValueError" in why, (ok, why)
    finally:
        del sys.modules["tests._explodes"]


def test_the_validator_map_is_the_one_the_test_suite_gates_on():
    """Two maps that can disagree about what is gated is the same defect as two
    records of what is applied. The artifact test imports this one."""
    from manamap.pilot.deck_status import VALIDATED
    from tests.test_pilot_tracked_artifacts_validate import GATED  # noqa: F401
    assert set(VALIDATED) <= set(GATED)
