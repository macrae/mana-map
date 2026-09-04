"""Tests for the build brief's gate (pilot/validate_brief.py).

`brief.json` was the input to every build and the one tracked pilot artifact
with no validator. Every check here was measured against all four briefs on disk
before it was written and fires on none of them — that measurement is the entry
criterion, and these tests are what stops a later edit quietly widening a check
past it.
"""

import glob
import json
import pathlib

import pytest

from conftest import requires_data, requires_deck

from manamap.pilot import validate_brief
from manamap.pilot.validate_brief import CONSUMED, INERT, validate

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Two corpus rows, enough to drive every name and identity check without
#: loading 36,563 of them. The shape is `load_frame().to_dict("records")`.
_ROWS = {
    "Zur the Enchanter": {
        "name": "Zur the Enchanter", "color_identity": "B, U, W",
        "type_line": "Legendary Creature — Human Wizard", "legal_commander": "legal"},
    "Rhystic Study": {
        "name": "Rhystic Study", "color_identity": "U",
        "type_line": "Enchantment", "legal_commander": "not_legal"},
    "Sterling Grove": {
        "name": "Sterling Grove", "color_identity": "G, W",
        "type_line": "Enchantment", "legal_commander": "not_legal"},
    "Sol Ring": {
        "name": "Sol Ring", "color_identity": "",
        "type_line": "Artifact", "legal_commander": "not_legal"},
}
_NAMES = set(_ROWS)


def _check(doc, slug="zur", **kw):
    return validate(doc, slug, rows=_ROWS, names=_NAMES, **kw)


def _brief(**over):
    doc = {"slug": "zur", "commander": "Zur the Enchanter", "bracket": 3,
           "must_include": [], "must_exclude": []}
    doc.update(over)
    return doc


# ── the entry criterion: it fires on none of the real briefs ────────────────

@requires_data
@requires_deck
@pytest.mark.parametrize("path", sorted(glob.glob(str(ROOT / "data/decks/*/brief.json"))))
def test_every_brief_on_disk_passes(path):
    """A validator that fires on correct data is worse than none.

    Six proposed checks have been rejected in this repo on exactly this ground,
    one of them firing on 27% of correct authored data. `--themes` is
    deliberately off: it is a network call, and this asserts the offline gate.
    """
    slug = pathlib.Path(path).parent.name
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    errors, _ = validate(doc, slug)
    assert errors == [], f"{slug}: {errors}"


@requires_data
@requires_deck
def test_the_fleet_is_not_empty():
    """`assert checked >= N`. Fourteen tests here once passed on zero rows."""
    assert len(glob.glob(str(ROOT / "data/decks/*/brief.json"))) >= 4


# ── and it can actually fail ────────────────────────────────────────────────

def test_a_slug_that_disagrees_with_its_directory_fails():
    errors, _ = _check(_brief(slug="not-zur"))
    assert any("sits in zur/" in e for e in errors)


def test_a_commander_that_is_not_in_the_corpus_fails():
    errors, _ = _check(_brief(commander="Zur the Enchantre"))
    assert any("not in the corpus" in e for e in errors)


def test_a_card_that_cannot_be_a_commander_fails():
    """`commander_rejection` owns the rule; this only asserts it is consulted."""
    errors, _ = _check(_brief(commander="Rhystic Study"))
    assert any("cannot be a commander" in e for e in errors)


def test_a_must_include_outside_the_colour_identity_fails():
    """`legal_must_includes` DROPS these at build time, which is not enough.

    A promise the builder silently cannot keep is a defect in the brief, and the
    plan's report of the drop is read after the build rather than before it.
    """
    errors, _ = _check(_brief(must_include=["Sterling Grove"]))
    assert any("Sterling Grove" in e and "outside" in e for e in errors)


def test_a_colourless_must_include_is_inside_every_identity():
    """The `str(nan) == 'nan'` class, which shipped once in `build_deck`.

    A colourless card's identity cell is empty, and an ad-hoc split reads it as
    a colour outside every identity — so EVERY colourless card was reported
    illegal. `parse_color_identity` is the shared reader that gets this right.
    """
    errors, _ = _check(_brief(must_include=["Sol Ring"]))
    assert errors == []


def test_an_unresolvable_name_fails_on_either_list():
    inc, _ = _check(_brief(must_include=["Not A Real Card"]))
    exc, _ = _check(_brief(must_exclude=["Not A Real Card"]))
    assert any("not in the corpus" in e for e in inc)
    assert any("not in the corpus" in e for e in exc)


@pytest.mark.parametrize("bracket", [0, 6, 9, -1])
def test_a_bracket_outside_the_range_fails(bracket):
    errors, _ = _check(_brief(bracket=bracket))
    assert any("bracket must be 1-5" in e for e in errors)


def test_a_pool_file_that_is_not_on_disk_fails():
    errors, _ = _check(_brief(pool_files=["no-such-pool.txt"]))
    assert any("not on disk" in e for e in errors)


def test_a_missing_commander_fails_and_stops():
    """It returns early rather than reporting every name against no identity."""
    errors, _ = _check(_brief(commander=None, must_include=["Sterling Grove"]))
    assert errors == ["no commander — the builder cannot start without one"]


# ── the two things reported and never failed ────────────────────────────────

def test_inert_keys_are_reported_and_never_failed():
    """Three of the four real briefs carry one; failing would redden them all."""
    errors, notes = _check(_brief(playstyle="grindy", notes="a note", partner=None))
    assert errors == []
    # `partner: None` is falsy but PRESENT, and presence is what is reported.
    assert any("read by nothing" in n and "playstyle" in n and "notes" in n
               for n in notes)


def test_an_unrecognised_key_is_reported_and_never_failed():
    errors, notes = _check(_brief(wingspan=7))
    assert errors == []
    assert any("unrecognised" in n and "wingspan" in n for n in notes)


def test_a_theme_is_shape_checked_without_the_network():
    """A gate that fails when EDHREC is down is a gate that gets switched off."""
    errors, notes = _check(_brief(theme="enchantress"))
    assert errors == []
    assert any("shape only" in n and "--themes" in n for n in notes)


def test_a_theme_lookup_that_raises_becomes_a_note_not_an_error(monkeypatch):
    monkeypatch.setattr("manamap.pilot.archetypes.list_themes",
                        lambda c: (_ for _ in ()).throw(OSError("no network")))
    errors, notes = _check(_brief(theme="enchantress"), check_themes=True)
    assert errors == []
    assert any("not resolved" in n and "no network" in n for n in notes)


def test_a_theme_the_commander_does_not_have_is_named(monkeypatch):
    """The defect this whole validator exists for.

    `role_budget_for` falls back to the flat provisional budget and says so only
    in `role_budget_grounding`, which nothing reads — so a typo'd style produces
    a legal 99 built to the wrong shape and leaves no visible trace.
    """
    monkeypatch.setattr(
        "manamap.pilot.archetypes.list_themes",
        lambda c: [{"slug": "enchantress", "name": "Enchantress", "decks": 1201}])
    errors, notes = _check(_brief(theme="enchantres"), check_themes=True)
    assert errors == []
    assert any("NOT one of" in n for n in notes)
    assert any("Did you mean enchantress" in n for n in notes)


def test_a_thin_theme_is_named_with_its_deck_count(monkeypatch):
    from manamap.pilot.archetypes import MIN_DECKS_FOR_TEMPLATE

    monkeypatch.setattr(
        "manamap.pilot.archetypes.list_themes",
        lambda c: [{"slug": "tempo", "name": "Tempo", "decks": 29},
                   {"slug": "stax", "name": "Stax", "decks": 542}])
    _, thin = _check(_brief(theme="tempo"), check_themes=True)
    assert any("29 decks behind it" in n for n in thin)

    _, fat = _check(_brief(theme="stax"), check_themes=True)
    assert any("542 decks" in n for n in fat)
    assert not any("behind it" in n for n in fat)
    assert MIN_DECKS_FOR_TEMPLATE > 29


# ── the consumed/inert split must not drift from the code it describes ──────

def test_consumed_names_exactly_what_the_builder_reads():
    """`CONSUMED` is a claim about `build_deck`, so it is checked against it.

    A list like this rots silently: the builder grows a key, the docstring keeps
    saying it is inert, and the note under every build becomes a lie.
    """
    import inspect

    from manamap.pilot import build_deck

    source = "".join(inspect.getsource(f) for f in
                     (build_deck.load_brief, build_deck.resolve_pool,
                      build_deck.role_budget_for, build_deck.build,
                      build_deck.deck_printings))
    for key in CONSUMED - {"format"}:            # `format` is read by serve.py
        assert f'"{key}"' in source or f"'{key}'" in source, key

    for key in INERT:
        assert f'brief["{key}"]' not in source, f"{key} is no longer inert"
        assert f'brief.get("{key}")' not in source, f"{key} is no longer inert"


def test_consumed_and_inert_do_not_overlap():
    assert not (CONSUMED & INERT)
