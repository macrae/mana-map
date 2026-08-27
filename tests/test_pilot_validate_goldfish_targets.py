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
    """The regression this module exists for, now the other way round.

    Hullbreaker Horror + a cheap rock + Aetherflux Reservoir is heliod's primary
    win line, verified by checker-passed stacks 001 and 006 and named in four
    other artifacts. No goldfish target mentioned it, so the simulator had never
    measured how the deck actually wins — and when one was finally declared the
    answer was 0.7% assembled, 0.3% by turn six.

    This asserts the declaration STAYS. Deleting the target would restore the
    blind spot silently: every rate the engine block prints would still be
    correct, and the one that matters would simply be absent again.
    """
    from manamap.pilot.common import deck_dir
    base = deck_dir("heliod")
    path = base / "goldfish_targets.json"
    if not path.exists():
        pytest.skip("heliod goldfish_targets.json not present")
    with open(path) as f:
        doc = json.load(f)
    declared = vgt._declared_names(doc)
    assert "Hullbreaker Horror" in declared, (
        "the primary win line must stay declared — an undeclared win line is "
        "measured by nothing")
    assert not any("Hullbreaker Horror" in e for e in vgt.validate(doc, "heliod", base))


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


def test_an_unedited_scaffold_is_reported_on_every_run(tmp_path, capsys):
    """A scaffold that nobody rewrites is the `DECK_ROLE_BUDGET` failure exactly
    — provisional, labelled provisional, and left in place for months. It is not
    an ERROR (a gate that reddens a legitimate intermediate state teaches its
    reader to ignore the gate), so it is SAID, every run, until the key goes."""
    import json as _json

    from manamap.pilot import validate_goldfish_targets as v

    base = tmp_path / "d"
    base.mkdir()
    (base / "cards.json").write_text(_json.dumps(
        {"cards": [{"name": "Sol Ring"}, {"name": "Cmd", "is_commander": True}]}))
    (base / "goldfish_targets.json").write_text(_json.dumps({
        "scaffolded": True,
        "targets": [{"label": "RAMP drawn", "_from": "role:ramp",
                     "need": [{"any_of": ["Sol Ring"]}]}],
    }))

    class A:
        slug = "d"

    import manamap.pilot.validate_goldfish_targets as mod

    original = mod.deck_dir
    # takes (slug, branch) now — a one-arg stub is the same shape of
    # half-plumbing this pass exists to remove.
    mod.deck_dir = lambda slug, branch=None: base
    try:
        mod.main(A())
    except SystemExit:
        pass
    finally:
        mod.deck_dir = original

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "SCAFFOLD" in out, "an unedited draft went unreported"
    assert "never edited" in out


def test_a_declaration_with_no_required_marking_says_so(capsys, monkeypatch, tmp_path):
    """NO `required` SILENTLY DISABLES THE FLAGSHIP METRIC.

    `diagnostic.engine` needs to know which components the deck cannot do
    without; absent that it withholds the figure — correctly, and out of sight
    of anyone running the validator, which printed a clean OK over it. Measured
    2026-08-26: 1 of 13 decks carried the marking, so `engine_online` and every
    axis built on it were validated on a sample of one.

    Reported, never failed: a declaration without it is a legitimate older file,
    and a gate that reddens twelve correct artifacts teaches its reader to
    ignore the gate. Driven through `main` — a test that re-derives the rule
    is testing itself.
    """
    import argparse

    unmarked = _doc(["Sol Ring", "Mana Crypt"])
    marked = {"targets": [dict(unmarked["targets"][0], required=True)]}

    def run(payload):
        path = tmp_path / "goldfish_targets.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(vgt, "deck_dir", lambda *a, **k: tmp_path)
        try:
            vgt.main(argparse.Namespace(slug="x", branch=None))
        except SystemExit:
            pass
        return capsys.readouterr().out

    assert "NO `required` MARKING" in run(unmarked)
    out = run(marked)
    assert "NO `required` MARKING" not in out
    assert out.startswith("OK"), out
