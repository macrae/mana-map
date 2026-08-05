"""validate-goldfish-targets: the engine declaration is itself an artifact.

`goldfish_targets.json` drives the assembly rates deck-audit quotes and a
diagnosis prescribes against, and nothing checked it until a seven-deck run found
it wrong on six of eight decks. These tests pin the two checks that survived
being measured against the whole fleet — and the third that did not.
"""

import json

import pytest

from manamap.pilot import validate_goldfish_targets as vgt

from conftest import requires_deck


def _doc(*groups):
    """One target per group, each group a plain any_of list."""
    return {"targets": [{"label": f"target {i}", "need": [{"any_of": list(g)}]}
                        for i, g in enumerate(groups)]}


# ── Shape ────────────────────────────────────────────────────────────────

def test_empty_targets_is_an_error():
    errors = vgt._validate_shape({"targets": []})
    assert errors and "non-empty list" in errors[0]


def test_a_duplicate_member_overstates_redundancy():
    """A group's SIZE is its redundancy claim, so a repeat inflates it."""
    errors = vgt._validate_shape(_doc(["Sol Ring", "Sol Ring", "Mana Crypt"]))
    assert any("listed twice" in e for e in errors)


def test_duplicate_labels_are_flagged():
    doc = {"targets": [{"label": "same", "need": [{"any_of": ["A"]}]},
                       {"label": "same", "need": [{"any_of": ["B"]}]}]}
    assert any("duplicate label" in e for e in vgt._validate_shape(doc))


def test_an_empty_group_is_an_error():
    doc = {"targets": [{"label": "x", "need": [{"any_of": []}]}]}
    assert any("non-empty list of card names" in e for e in vgt._validate_shape(doc))


def test_a_well_formed_declaration_passes_shape():
    assert vgt._validate_shape(_doc(["Sol Ring", "Arcane Signet"])) == []


# ── Membership: the staleness guard ──────────────────────────────────────

def test_a_declared_card_no_longer_in_the_deck_is_reported():
    """A swap strands the name it removed, and the group keeps its old size."""
    doc = _doc(["Sol Ring", "Cut Long Ago"])
    errors = vgt._validate_membership(doc, {"Sol Ring"}, set())
    assert len(errors) == 1
    assert "Cut Long Ago" in errors[0]
    assert "overstates its redundancy" in errors[0]


def test_the_commander_counts_as_in_the_deck():
    """The commander is not in the 99 but is legitimately declarable."""
    doc = _doc(["Yawgmoth, Thran Physician"])
    assert vgt._validate_membership(doc, set(), {"Yawgmoth, Thran Physician"}) == []


# ── Win-line coverage ────────────────────────────────────────────────────

def test_quorum_is_two_stacks():
    """One passing stack is a line; two is a pattern.

    Both real omissions the fleet survey found clear two, so the threshold buys
    the finding without reporting every card that ever appeared on a board.
    """
    assert vgt.WIN_LINE_QUORUM == 2


@requires_deck
def test_heliod_primary_win_line_is_undeclared():
    """The regression this module exists for.

    Hullbreaker Horror + Sol Ring + Arcane Signet + Aetherflux Reservoir is
    heliod's primary win line, verified by a checker-passed stack and named in
    four other artifacts — and no goldfish target mentions it, so the simulator
    has never measured how the deck actually wins.
    """
    from manamap.pilot.common import deck_dir
    base = deck_dir("heliod")
    path = base / "goldfish_targets.json"
    if not path.exists():
        pytest.skip("heliod goldfish_targets.json not present")
    with open(path) as f:
        doc = json.load(f)
    errors = vgt.validate(doc, "heliod", base)
    assert any("Hullbreaker Horror" in e for e in errors), (
        "the primary win line must be reported as undeclared")


@requires_deck
def test_a_commander_is_never_reported_as_an_omission():
    """Commanders sit on every board and say nothing about the engine.

    Without the exclusion this fires on Edgar Markov, Gishath, Zada and The
    Ur-Dragon — four false positives on four decks.
    """
    from manamap.pilot.common import deck_dir, load_deck_cards
    for slug in ("gishath", "goblin-storm", "ur-dragon"):
        base = deck_dir(slug)
        path = base / "goldfish_targets.json"
        if not path.exists():
            continue
        with open(path) as f:
            doc = json.load(f)
        commanders = {c["name"] for c in load_deck_cards(slug).get("cards", [])
                      if c.get("is_commander")}
        errors = vgt.validate(doc, slug, base)
        for name in commanders:
            assert not any(f"'{name}'" in e for e in errors), (
                f"{slug}: commander {name} reported as an undeclared component")
