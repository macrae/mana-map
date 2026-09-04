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
        # REAL CARDS, because without them the deck-dependent checks do not run
        # and the headline is PARTIAL, not OK. This test asserted `OK` for
        # months over a run that had checked nothing but the file's shape —
        # it was passing THROUGH the blind spot fixed on 2026-09-04, which is
        # how that blind spot stayed invisible.
        monkeypatch.setattr(vgt, "load_deck_cards", lambda *a, **k: {"cards": [
            {"name": "Sol Ring"}, {"name": "Mana Crypt"},
            {"name": "Cmd", "is_commander": True}]})
        try:
            vgt.main(argparse.Namespace(slug="x", branch=None))
        except SystemExit:
            pass
        return capsys.readouterr().out

    assert "NO `required` MARKING" in run(unmarked)
    out = run(marked)
    assert "NO `required` MARKING" not in out
    assert out.startswith("OK"), out


# ── a kill that has to connect ──────────────────────────────────────────────

def _route_doc(label, route="commander", **flags):
    """A one-route declaration. NOT `_doc` — that name is taken at the top of
    this file, and appending a second definition silently overrode it and broke
    four passing tests that were nothing to do with this change."""
    return {"targets": [{"label": label, "route": route,
                         "any_of": [["Sol Ring"]]}], **flags}


def test_a_combat_kill_route_is_noted_because_the_goldfish_has_no_blockers():
    """The check that would have saved 39 games.

    Zur's V6 engine is "Zur attacks, fetches an aura, connects with lifelink,
    Vito drains" — every step gated on a 1/4 commander connecting. The goldfish
    reported the route assembled by turn six in 23.6% of games, and Forge on the
    pilot's own table returned **0.35 commander damage a game, best single game
    2, and 0 of 39 games reaching 21**. The route was never wrong about the
    draw and was never evidence about the kill.
    """
    from manamap.pilot.validate_goldfish_targets import _combat_route_notes

    off = _combat_route_notes(_route_doc("KILL — commander damage: a buff aura on Zur"))
    assert len(off) == 1
    assert "model_combat` is OFF" in off[0]
    assert "0 of 39" in off[0], "the note carries the measurement that earned it"

    on = _combat_route_notes(
        _route_doc("THE COMBAT KILL: a Dragon-damage MULTIPLIER", model_combat=True))
    assert len(on) == 1
    assert "NO BLOCKERS" in on[0]
    assert "simulate" in on[0], "the note must name the engine that CAN judge it"


def test_a_non_combat_route_is_not_noted():
    """The entry criterion. A drain or storm route is graded fine by the model."""
    from manamap.pilot.validate_goldfish_targets import _combat_route_notes

    for label in ("KILL — drain: a payoff plus something to feed it",
                  "RESOURCE ENGINE drawn — a draw engine that keeps drawing",
                  "A TUTOR drawn", "RAMP drawn — an accelerant of any kind"):
        assert _combat_route_notes(_route_doc(label)) == [], label


def test_a_component_is_not_a_route_and_is_not_noted():
    """`required` components are not kills, so they are not judged as one."""
    from manamap.pilot.validate_goldfish_targets import _combat_route_notes

    doc = {"targets": [{"label": "A Vampire lord or anthem drawn (the combat plan)",
                        "required": True, "any_of": [["Sol Ring"]]}]}
    assert _combat_route_notes(doc) == []


def test_it_fires_on_exactly_the_fleet_it_was_measured_against():
    """Two of ten decks, and both are genuine combat routes.

    A check that fires on correct data is worse than no check, and six proposals
    have been rejected in this repo on that ground. If this count moves, either
    a deck declared a new combat route — fine, and the note is right — or the
    pattern widened past what it was measured on.
    """
    import glob
    import json
    import pathlib as _pl

    from manamap.pilot.validate_goldfish_targets import _combat_route_notes

    root = _pl.Path(__file__).resolve().parent.parent
    files = sorted(glob.glob(str(root / "data/decks/*/goldfish_targets.json")))
    assert len(files) >= 8, "the guard iterated almost nothing"
    fired = {_pl.Path(f).parent.name for f in files
             if _combat_route_notes(json.loads(_pl.Path(f).read_text(encoding="utf-8")))}
    assert fired == {"ur-dragon", "zur-enchantress"}, fired


def test_the_note_survives_a_rename_of_the_kill():
    """THE STRUCTURED FIELD, NOT THE LABEL PROSE.

    zur-enchantress abandoned commander damage and renamed its kill from
    "KILL — commander damage: a buff aura on Zur" to "KILL — a BOARD: a real
    body plus a way through". The rename was correct — the deck really did
    change win-con — and `_COMBAT_ROUTE`, which greps the label, silently
    stopped firing on a route that still has to CONNECT.

    That is the note's whole subject. A deck can rename its way out of the one
    warning that says its assembly rate is about the draw and not the kill.
    """
    from manamap.pilot.validate_goldfish_targets import _combat_route_notes

    renamed = {"model_combat": True, "targets": [
        {"label": "KILL — a BOARD: a real body plus a way through",
         "route": "board", "any_of": [["Sol Ring"]]}]}
    notes = _combat_route_notes(renamed)
    assert len(notes) == 1 and "NO BLOCKERS" in notes[0]

    # The control: a route the model grades honestly must stay silent, or this
    # widening would have fired on every deck with a declared kill.
    for route in ("drain", "entry"):
        quiet = {"targets": [{"label": "KILL — drain", "route": route,
                              "any_of": [["Sol Ring"]]}]}
        assert _combat_route_notes(quiet) == [], route


# ── the checks that did not run ─────────────────────────────────────────────

def test_a_missing_cards_json_is_said_and_never_printed_as_OK(tmp_path, capsys,
                                                              monkeypatch):
    """A GUARD THAT CANNOT FAIL IS A CLAIM, NOT A GUARD.

    Every check below the shape pass needs the deck's cards. Without them
    `validate` caught a bare `Exception` and returned an empty error list, which
    `main` printed as a clean `OK` — the validator asserting, in its own voice,
    that a declaration held when it had looked at nothing but the file's shape.

    Caught live on 2026-09-04: `deck-branch new` writes `decklist.txt` and no
    `cards.json`, so a branch validated between `new` and `fetch-deck` reported
    OK on a declaration naming two cards the swap had just removed. The goldfish
    found them a minute later. Nothing in the gate would ever have.

    Reported, not failed — a fresh clone legitimately has no card data, and a
    gate that reddens one teaches its reader to ignore the gate.
    """
    import argparse

    doc = _doc(["Cut Long Ago", "Also Gone"])
    (tmp_path / "goldfish_targets.json").write_text(json.dumps(doc))
    monkeypatch.setattr(vgt, "deck_dir", lambda *a, **k: tmp_path)

    def no_cards(*a, **k):
        raise FileNotFoundError("cards.json not found")

    monkeypatch.setattr(vgt, "load_deck_cards", no_cards)
    try:
        vgt.main(argparse.Namespace(slug="x", branch=None))
    except SystemExit:
        pass
    out = capsys.readouterr().out

    assert not out.startswith("OK"), (
        "the validator reported OK over checks it never ran:\n" + out)
    assert "PARTIAL" in out
    assert "DID NOT RUN" in out
    assert "fetch-deck x" in out, "the note must name the command that fixes it"


def test_the_branch_form_of_the_fix_is_named(tmp_path, capsys, monkeypatch):
    """The note has to be runnable on the deck it was printed for.

    A branch needs `--branch <name>` and the deck-level command would not help,
    which is exactly the state this was found in.
    """
    import argparse

    (tmp_path / "goldfish_targets.json").write_text(json.dumps(_doc(["A"])))
    monkeypatch.setattr(vgt, "deck_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(vgt, "load_deck_cards",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no")))
    try:
        vgt.main(argparse.Namespace(slug="zur-enchantress", branch="toolbox-v1"))
    except SystemExit:
        pass
    assert "fetch-deck zur-enchantress --branch toolbox-v1" in capsys.readouterr().out


def test_a_deck_that_loads_still_reports_OK():
    """The control. Without it this change could have made every run PARTIAL."""
    import argparse

    from manamap.pilot.common import deck_dir

    base = deck_dir("zur-enchantress")
    if not (base / "goldfish_targets.json").exists():
        pytest.skip("zur-enchantress targets not present")
    notes = []
    with open(base / "goldfish_targets.json") as f:
        vgt.validate(json.load(f), "zur-enchantress", base, None, notes=notes)
    assert notes == [], f"a deck with real cards should skip nothing: {notes}"
