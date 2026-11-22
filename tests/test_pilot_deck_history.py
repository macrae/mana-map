"""deck-history: applied swaps derived from git, plus the pending ones.

Three places recorded swaps before this and none could be trusted — decklist
comment blocks, HISTORY.md on half the decks, and considering.json, which is
replaced wholesale so an applied ten leaves no trace. This derives from
`decklist.txt`'s own git history, which cannot drift because it IS the change.
"""

import pytest

from manamap.pilot import deck_history as dh

from conftest import requires_deck


def test_entries_reads_the_mainboard_only():
    """History is about the 99. A sideboard edit is not a swap, so the sideboard
    section is parsed (since 2026-10-09) and left out of the history."""
    text = ("Commander:\n1 Edgar Markov\n\nDeck:\n1 Sol Ring\n2 Swamp\n"
            "Sideboard:\n1 Bitterblossom\n")
    assert dh._entries(text) == {"Edgar Markov": 1, "Sol Ring": 1, "Swamp": 2}


def test_entries_counts_copies():
    assert dh._entries("Deck:\n22 Swamp\n")["Swamp"] == 22


@requires_deck
def test_history_reports_ins_and_outs_per_revision():
    changes = dh.history("yawgmoth-swarm")
    if not changes:
        pytest.skip("no git history available in this checkout")
    assert changes[0].get("note", "").startswith("first tracked revision")
    for rev in changes:
        assert set(rev) >= {"sha", "date", "reason", "in", "out", "size"}
        assert rev["date"].count("-") == 2
        # An entry that changed nothing should not have been recorded at all.
        if "note" not in rev:
            assert rev["in"] or rev["out"] or rev["quantity_changes"]


@requires_deck
def test_history_is_oldest_first():
    changes = dh.history("yawgmoth-swarm")
    if len(changes) < 2:
        pytest.skip("needs at least two revisions")
    assert changes[0]["date"] <= changes[-1]["date"]


@requires_deck
def test_pending_is_a_swap_and_carries_no_ownership_claim():
    """Until 2026-11-22 a pending swap derived an `acquisition` (owned or
    purchase) from the pilot's boxes. The bench has no notion of a box any more,
    so a pending entry is a card in, a card out, what it closes and where the
    entry came from — and an `acquisition` block an older `considering.json`
    still carries never reaches the report."""
    prop = dh.pending("yawgmoth-swarm")
    if not prop:
        pytest.skip("no considering.json for this deck")
    for p in prop:
        assert set(p) == {"in", "out", "closes", "source", "id", "state"}, p
        assert p["source"] in ("considering.json", "pending.json")
    text = dh.format_report(dh.analyze("yawgmoth-swarm"))
    assert "[owned" not in text and "[buy" not in text


@requires_deck
def test_pending_marks_swaps_with_no_cut():
    prop = dh.pending("yawgmoth-swarm")
    if not prop:
        pytest.skip("no considering.json for this deck")
    assert all("out" in p for p in prop)


@requires_deck
def test_analyze_always_explains_its_reason_field():
    """`reason` is a commit subject, not an authored rationale. Say so."""
    doc = dh.analyze("yawgmoth-swarm")
    assert any("commit subject" in n for n in doc["notes"])


def test_analyze_rejects_an_unknown_slug():
    with pytest.raises(FileNotFoundError):
        dh.analyze("no-such-deck-here")
