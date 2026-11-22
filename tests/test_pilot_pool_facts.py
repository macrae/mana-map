"""Tests for the pool-file readers: the names a brief constrains a build to.

`pool_facts` was a command — "what deck can I build from a box of cards" — until
2026-11-22, when the bench stopped having any notion of what a person holds. What
is left is the readers `build_deck` and `validate_build` share, and a pool is a
constraint on a build, never an inventory. The pure functions are tested against a
hand-built card table so they hold without any generated artifact; the memo tests
need cards.csv and are gated behind `requires_data`.
"""

import pytest

from conftest import requires_data
from manamap.pilot import pool_facts
from manamap.pilot.common import commander_rejection

# A miniature cards.csv: enough shape for identity, land and commander logic.
CARDS = {
    "Hapatra, Vizier of Poisons": {
        "name": "Hapatra, Vizier of Poisons",
        "type_line": "Legendary Creature — Zombie Cleric",
        "oracle_text": "Deathtouch",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 4033.0,
        "mechanical_tags": "counters_minus, tokens",
    },
    "Grist, Voracious Larva // Grist, the Plague Swarm": {
        "name": "Grist, Voracious Larva // Grist, the Plague Swarm",
        "type_line": "Legendary Creature — Insect // Legendary Planeswalker — Grist",
        "oracle_text": "",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 4455.0,
        "mechanical_tags": "tokens",
    },
    "Overgrown Tomb": {
        "name": "Overgrown Tomb",
        "type_line": "Land — Swamp Forest",
        "oracle_text": "({T}: Add {B} or {G}.)",
        "color_identity": "B, G",
        "legal_commander": "legal",
        "edhrec_rank": 100.0,
        "mechanical_tags": "",
    },
    "Command Tower": {
        "name": "Command Tower",
        "type_line": "Land",
        "oracle_text": "{T}: Add one mana of any color that a commander in the command zone could produce.",
        "color_identity": "",
        "legal_commander": "legal",
        "edhrec_rank": 1.0,
        "mechanical_tags": "",
    },
    "Island": {
        "name": "Island",
        "type_line": "Basic Land — Island",
        "oracle_text": "",
        "color_identity": "U",
        "legal_commander": "legal",
        "edhrec_rank": None,
        "mechanical_tags": "",
    },
    "Sol Ring": {
        "name": "Sol Ring",
        "type_line": "Artifact",
        "oracle_text": "{T}: Add {C}{C}.",
        "color_identity": "",
        "legal_commander": "legal",
        "edhrec_rank": 1.0,
        "mechanical_tags": "ramp",
    },
    "Black Lotus": {  # legal_commander != legal — must never be a commander
        "name": "Black Lotus",
        "type_line": "Artifact",
        "oracle_text": "",
        "color_identity": "",
        "legal_commander": "banned",
        "edhrec_rank": None,
        "mechanical_tags": "",
    },
}


# ── The shared commander predicate ──────────────────────────────────────


def test_legendary_creature_is_a_commander():
    assert commander_rejection(CARDS["Hapatra, Vizier of Poisons"]) is None


def test_front_face_decides_for_a_dfc():
    """Grist's back face is a planeswalker; the front face is what matters."""
    assert commander_rejection(CARDS["Grist, Voracious Larva // Grist, the Plague Swarm"]) is None


def test_non_creature_without_the_text_is_rejected():
    reason = commander_rejection(CARDS["Sol Ring"])
    assert reason and "not a legal commander" in reason


def test_can_be_your_commander_text_qualifies():
    row = dict(CARDS["Sol Ring"], oracle_text="Sol Ring can be your commander.")
    assert commander_rejection(row) is None


def test_banned_card_is_rejected_even_if_legendary():
    row = dict(CARDS["Hapatra, Vizier of Poisons"], legal_commander="banned")
    assert commander_rejection(row) == "not legal in Commander"


def test_build_deck_still_raises_with_the_reason():
    """The refactor kept build_deck's error messages, which name the card."""
    from manamap.pilot.build_deck import BriefError, commander_identity

    with pytest.raises(BriefError, match="not a legal commander"):
        commander_identity(CARDS["Sol Ring"])
    assert commander_identity(CARDS["Hapatra, Vizier of Poisons"]) == {"B", "G"}


# ── Name resolution: the front-face translation ─────────────────────────


def test_front_face_map_keys_on_the_front_only():
    mapping = pool_facts.front_face_map(CARDS)
    assert mapping == {"Grist, Voracious Larva": "Grist, Voracious Larva // Grist, the Plague Swarm"}


def test_decklist_front_face_resolves_to_the_joined_name(tmp_path):
    """A decklist names the front face; every global artifact keys on the join.

    Without this translation the card is silently absent from every count
    rather than reported unresolved — the expensive failure mode.
    """
    lst = tmp_path / "deck.txt"
    lst.write_text("1 Grist, Voracious Larva (MH3) 251\n2 Sol Ring\n")
    per_file, unresolved = pool_facts.read_sources([lst], CARDS)
    assert unresolved == {}
    assert per_file["deck.txt"] == {
        "Grist, Voracious Larva // Grist, the Plague Swarm": 1,
        "Sol Ring": 2,
    }


def test_unresolved_names_are_reported_not_dropped(tmp_path):
    lst = tmp_path / "deck.txt"
    lst.write_text("1 Sol Ring\n1 Not A Real Card\n")
    per_file, unresolved = pool_facts.read_sources([lst], CARDS)
    assert per_file["deck.txt"] == {"Sol Ring": 1}
    assert unresolved == {"Not A Real Card": ["deck.txt"]}


def test_quantities_accumulate_across_files(tmp_path):
    (tmp_path / "a.txt").write_text("2 Sol Ring\n")
    (tmp_path / "b.txt").write_text("1 Sol Ring\n")
    per_file, _ = pool_facts.read_sources(
        [tmp_path / "a.txt", tmp_path / "b.txt"], CARDS
    )
    assert per_file["a.txt"]["Sol Ring"] == 2
    assert per_file["b.txt"]["Sol Ring"] == 1


def test_collect_paths_expands_a_directory_and_honours_exclude(tmp_path):
    (tmp_path / "a.txt").write_text("1 Sol Ring\n")
    (tmp_path / "b.txt").write_text("1 Sol Ring\n")
    (tmp_path / "notes.md").write_text("ignored")
    assert len(pool_facts.collect_paths([tmp_path], [])) == 2
    kept = pool_facts.collect_paths([tmp_path], [tmp_path / "b.txt"])
    assert [p.name for p in kept] == ["a.txt"]


def test_collect_paths_rejects_a_missing_target(tmp_path):
    with pytest.raises(SystemExit):
        pool_facts.collect_paths([tmp_path / "nope"], [])


# ── The printing a pool file pins ───────────────────────────────────────


def test_pool_printings_keeps_the_pinned_printing_and_the_first_one_wins(tmp_path):
    """`read_sources` drops the `(SET) CN *F*` suffix because counting copies does
    not care; a build that pins printings does. A bare line pins nothing, a
    front-face line resolves to the joined name, and a card listed twice keeps
    the printing it was first seen with — which line it came from is not a
    decision the reader should invent."""
    (tmp_path / "a.txt").write_text(
        "1 Sol Ring (ECC) 58 *F*\n1 Grist, Voracious Larva (MH3) 251\n1 Command Tower\n")
    (tmp_path / "b.txt").write_text("1 Sol Ring (C21) 263\n")
    got = pool_facts.pool_printings([tmp_path / "a.txt", tmp_path / "b.txt"], CARDS)
    assert got["Sol Ring"] == {"set": "ecc", "collector_number": "58", "foil": True}
    assert got["Grist, Voracious Larva // Grist, the Plague Swarm"] == {
        "set": "mh3", "collector_number": "251", "foil": False}
    assert "Command Tower" not in got, "a bare line pins no printing"


# ── One parse per process ────────────────────────────────────────────────


@requires_data
def test_load_cards_is_memoized(monkeypatch):
    """A build that named a pool scanned the 24 MB CSV twice — once here and once
    inside `bracket.load_reference`. Same key discipline as `bracket._card_flags`."""
    import pandas as pd
    from manamap.pilot import pool_facts as pf
    from manamap.pilot.common import clear_memo

    clear_memo()
    calls = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    first = pf.load_cards()
    second = pf.load_cards()
    assert len(calls) == 1, "second call re-read the CSV"
    assert first is second, "callers must share one parse"


@requires_data
def test_load_cards_reparses_when_the_file_changes():
    """Keyed on (mtime_ns, size), so a regenerated cards.csv is picked up.

    The storage moved to `common._MTIME_MEMO` when five hand-rolled copies of
    this cache were consolidated; the property under test is unchanged.
    """
    from manamap.pilot import pool_facts as pf
    from manamap.pilot.common import _MTIME_MEMO, clear_memo

    clear_memo()
    pf.load_cards()
    sig, _ = _MTIME_MEMO["pool_facts:cards"]
    _MTIME_MEMO["pool_facts:cards"] = ((sig[0] - 1, sig[1]), {"stale": True})
    assert "stale" not in pf.load_cards()
