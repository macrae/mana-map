"""The dossier prints what to physically move.

`deck-info` showed `SLEEVED · V6` on one line and `V8 of 8` on the next and
NOTHING about the difference, so a pilot could read both and have no idea nine
cards had to change hands. The lock is not a status badge — it is the thing
drift is measured FROM — and until this the only place the difference appeared
was `deck-version <slug> paper`, which WRITES the lock as a side effect of being
run, so reading the drift could silently claim a deck was sleeved when it was
not. (It did, once, on heliod.)
"""

import pytest

from manamap.pilot import deck_info

from conftest import requires_deck


def test_the_drift_block_prints_copies_and_names(monkeypatch):
    """Counts AND names. A count with no names is not an instruction, and a
    name with no count sends the pilot after one basic when five moved."""
    monkeypatch.setattr(deck_info.versions_mod, "paper_state", lambda slug: {
        "version": 6, "in_sync": False,
        "drift": {"pull": ["Iron Maiden", "5x Island"], "add": ["5x Plains"],
                  "pull_copies": 6, "add_copies": 5}})
    out = "\n".join(deck_info.drift_lines("x", 8))
    assert "TO SLEEVE" in out and "V6 is in your hands" in out
    assert "pull 6, add 5" in out
    assert "- 5x Island" in out and "+ 5x Plains" in out and "- Iron Maiden" in out


def test_a_deck_in_sync_prints_no_drift_block(monkeypatch):
    """A line that always appears stops being read. Nothing to do, nothing said."""
    monkeypatch.setattr(deck_info.versions_mod, "paper_state",
                        lambda slug: {"version": 8, "in_sync": True})
    assert deck_info.drift_lines("x", 8) == []


def test_a_deck_with_no_lock_or_no_history_gets_no_block(monkeypatch):
    """A dossier must render for a deck nobody has claimed exists in paper."""
    monkeypatch.setattr(deck_info.versions_mod, "paper_state", lambda slug: None)
    assert deck_info.drift_lines("x", 8) == []

    def boom(slug):
        raise RuntimeError("no git history")
    monkeypatch.setattr(deck_info.versions_mod, "paper_state", boom)
    assert deck_info.drift_lines("x", 8) == []


def test_reading_the_drift_never_writes_the_lock(monkeypatch):
    """THE BUG THIS BLOCK EXISTS TO REPLACE. `deck-version <slug> paper` with no
    ref MOVES the lock; it was run to READ the drift and claimed heliod was
    sleeved at V8 while two of its cards were unbought. The dossier must reach
    for `paper_state`, which only reads."""
    import inspect
    src = inspect.getsource(deck_info.drift_lines)
    assert "paper_state" in src
    assert "set_paper" not in src


@requires_deck
def test_the_real_dossier_computes_a_drift_without_touching_the_file():
    """End to end on a deck that HAS drift, and the lock is unchanged after."""
    import json
    from manamap.config import DECKS_DIR
    p = DECKS_DIR / "heliod" / "deck_versions.json"
    if not p.exists():
        pytest.skip("heliod is not in this checkout")
    before = p.read_bytes()
    st = deck_info.versions_mod.paper_state("heliod")
    assert p.read_bytes() == before, "reading the drift wrote to deck_versions.json"
    if st and not st.get("in_sync"):
        d = st["drift"]
        assert d["pull_copies"] >= len(d["pull"]), "copies can never be under names"
