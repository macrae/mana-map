"""The drift report counts COPIES, not entries.

`diff_vs_working` compared NAME MEMBERSHIP. heliod's v1.0.1 moved five Islands
to five Plains — the deck's largest measured fix, and the whole of it — and the
drift line read:

    the sleeved list is V6; the repo is at V7 — pull 0, add 0 to bring the
    cardboard level

on the one report whose entire job is telling the pilot what cardboard to move.
Both names are in both lists; only the counts changed. `dh._entries` has always
returned name -> copies and nothing used the second half of it.

Same defect this repo already documents from the magazine era, where counting
entries once published "18 lands" for a 33-land deck.
"""

import pytest

from manamap.pilot import deck_versions


def _diff(then_text, now_text, monkeypatch, tmp_path):
    (tmp_path / "decklist.txt").write_text(now_text)
    monkeypatch.setattr(deck_versions, "blob_at", lambda slug, v: then_text)
    monkeypatch.setattr(deck_versions, "deck_dir", lambda slug: tmp_path)
    return deck_versions.diff_vs_working("x", {"version": 1})


def test_a_pure_quantity_change_is_not_invisible(monkeypatch, tmp_path):
    """THE BUG, exactly as it shipped. Nothing enters or leaves by name."""
    then = "Commander:\n1 Heliod, the Radiant Dawn\n\nDeck:\n14 Island\n3 Plains\n"
    now = "Commander:\n1 Heliod, the Radiant Dawn\n\nDeck:\n9 Island\n8 Plains\n"
    d = _diff(then, now, monkeypatch, tmp_path)
    assert d["pull_copies"] == 5 and d["add_copies"] == 5, (
        "five basics moved and the report said nothing")
    assert d["in_then_not_now"] == ["5x Island"]
    assert d["in_now_not_then"] == ["5x Plains"]


def test_a_single_copy_is_named_without_a_count(monkeypatch, tmp_path):
    """The common case must not become noisier to fix the rare one."""
    then = "Deck:\n1 Sol Ring\n1 Wall of Glare\n"
    now = "Deck:\n1 Sol Ring\n1 Wrath of God\n"
    d = _diff(then, now, monkeypatch, tmp_path)
    assert d["in_then_not_now"] == ["Wall of Glare"]
    assert d["in_now_not_then"] == ["Wrath of God"]
    assert d["pull_copies"] == 1 and d["add_copies"] == 1


def test_a_partial_quantity_change_reports_only_the_delta(monkeypatch, tmp_path):
    """Going 10 Island -> 8 Island is a two-card swap, not a ten-card one. The
    naive fix — treat any changed count as the whole entry — would send the
    pilot to their sleeves for eight cards that never move."""
    then = "Deck:\n10 Island\n"
    now = "Deck:\n8 Island\n1 Plains\n1 Sol Ring\n"
    d = _diff(then, now, monkeypatch, tmp_path)
    assert d["in_then_not_now"] == ["2x Island"]
    assert sorted(d["in_now_not_then"]) == ["Plains", "Sol Ring"]
    assert d["pull_copies"] == 2 and d["add_copies"] == 2


def test_an_identical_list_has_no_drift(monkeypatch, tmp_path):
    same = "Deck:\n9 Island\n8 Plains\n1 Sol Ring\n"
    d = _diff(same, same, monkeypatch, tmp_path)
    assert d == {"in_then_not_now": [], "in_now_not_then": [],
                 "pull_copies": 0, "add_copies": 0}
