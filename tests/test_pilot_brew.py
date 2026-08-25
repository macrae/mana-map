"""The build-out: cards you kept become a deck on the bench. PRD §7.4."""

import json

import pytest

from conftest import requires_data
from manamap.config import DECK_ROLE_BUDGET
from manamap.pilot import brew, build_deck


def test_the_library_becomes_must_include(tmp_path, monkeypatch):
    """`must_include` is the promise that these cards are in the 99, and the
    library is exactly the set the pilot deliberately kept."""
    monkeypatch.setattr(build_deck, "DECKS_DIR", tmp_path)
    path, doc = build_deck.scaffold_brief(
        "zz", "Zur the Enchanter", library=["Ethereal Armor"], theme="voltron")
    assert doc["must_include"] == ["Ethereal Armor"]
    assert doc["theme"] == "voltron"
    assert json.loads(path.read_text())["commander"] == "Zur the Enchanter"


def test_a_new_deck_is_not_sleeved(tmp_path, monkeypatch):
    """§7.4 lands a deck at v0.1.0. 0.x is the version of a list that exists
    only digitally; reaching 1.0.0 is the act of sleeving it, which only the
    pilot can do. Writing a `paper` block here would claim cardboard that does
    not exist — the exact defect the rehearsal locks were withdrawn for."""
    monkeypatch.setattr(build_deck, "DECKS_DIR", tmp_path)
    path, doc = build_deck.scaffold_brief("zz", "Zur the Enchanter")
    assert "paper" not in doc
    assert not (path.parent / "deck_versions.json").exists()


def test_it_refuses_to_overwrite_an_existing_brief(tmp_path, monkeypatch):
    monkeypatch.setattr(build_deck, "DECKS_DIR", tmp_path)
    build_deck.scaffold_brief("zz", "Zur the Enchanter")
    with pytest.raises(SystemExit):
        build_deck.scaffold_brief("zz", "Someone Else")


def test_an_exported_brief_is_read_as_one(tmp_path):
    """The Atlas exports `brief.json` with `must_include` already resolved, so a
    walk in the browser continues at the CLI without a translation step."""
    p = tmp_path / "brief.json"
    p.write_text(json.dumps({"commander": "Zur the Enchanter",
                             "must_include": ["Ethereal Armor", "Sol Ring"]}))
    cards, commander = brew._library_from_file(str(p))
    assert cards == ["Ethereal Armor", "Sol Ring"]
    assert commander == "Zur the Enchanter"


def test_a_decklist_file_goes_through_the_one_parser(tmp_path):
    """Quantities and `*CMDR*` for free, and no second name reader."""
    p = tmp_path / "library.txt"
    p.write_text("1 Zur the Enchanter *CMDR*\n1 Ethereal Armor\n1 Sol Ring\n")
    cards, commander = brew._library_from_file(str(p))
    assert commander == "Zur the Enchanter"
    assert "Zur the Enchanter" not in cards, "the commander is not a kept card"
    assert cards == ["Ethereal Armor", "Sol Ring"]


# ── The role budget the style asks for ─────────────────────────────────────


@requires_data
def test_a_style_shapes_the_budget_and_says_so():
    """`DECK_ROLE_BUDGET` is one flat budget for every deck and its own comment
    calls it PROVISIONAL. A style's budget is MEASURED from that archetype's
    average deck, and the grounding string says which."""
    from manamap.pilot.archetypes import _roles

    roles = _roles()
    flat, flat_why = build_deck.role_budget_for({}, roles)
    assert flat == DECK_ROLE_BUDGET and "provisional" in flat_why

    themed, why = build_deck.role_budget_for(
        {"commander": "Zur the Enchanter", "theme": "voltron"}, roles)
    assert "measured" in why and "voltron" in why
    assert themed != flat, "the style did not change the budget at all"
    assert sum(themed.values()) == sum(flat.values()), (
        "the style changed the deck's SIZE — it may only change its shape, "
        "because land counts, the curve quota and the bracket pass are all "
        "sized against that total")


@requires_data
def test_lands_do_not_get_counted_into_the_spell_budget():
    """A bug in the first version. `role_group` has no land line, so every
    `land:basic` / `land:tapped` / `land:utility` fell into `flex` — 32 of
    voltron's 135 role-copies — inflating flex and deflating every real line at
    once. Lands are budgeted separately by `land_counts`; counting them here was
    double-counting the mana base into the spells.
    """
    from manamap.pilot.archetypes import _roles

    themed, _ = build_deck.role_budget_for(
        {"commander": "Zur the Enchanter", "theme": "voltron"}, _roles())
    assert themed["lands"] == DECK_ROLE_BUDGET["lands"], "the land line moved"
    spells = sum(v for k, v in themed.items() if k != "lands")
    assert themed["flex"] < spells * 0.55, (
        f"flex is {themed['flex']} of {spells} — lands are probably being "
        f"counted into the spell pool again")


@requires_data
def test_an_unfetchable_style_falls_back_with_a_reason():
    """A theme that cannot be read must not stop a build: the flat budget is
    what every deck here was built on until now."""
    from manamap.pilot.archetypes import _roles

    budget, why = build_deck.role_budget_for(
        {"commander": "Nobody At All", "theme": "not-a-theme"}, _roles())
    assert budget == DECK_ROLE_BUDGET
    assert "could not read" in why
