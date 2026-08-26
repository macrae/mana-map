"""A starting `goldfish_targets.json` — and the line it must not cross.

The file is the deck's engine DECLARATION, and it stays authored. What is
scaffolded is the blank page: a new deck's dossier named a path, a schema nobody
had seen, and two panels blocked behind it.

The danger is the one `DECK_ROLE_BUDGET` fell into — provisional, labelled
provisional, and left in place for months — so these tests are mostly about the
draft staying VISIBLE as a draft.
"""

import json

import pytest

from manamap.pilot import scaffold_targets as st

from conftest import requires_deck

pytestmark = requires_deck


def _deck(tmp_path, monkeypatch, slug="t", cards=None, commander="Cmd"):
    base = tmp_path / slug
    base.mkdir(parents=True)
    doc = {"cards": ([{"name": commander, "is_commander": True}]
                     + [{"name": n} for n in (cards or [])])}
    (base / "cards.json").write_text(json.dumps(doc))
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path, raising=False)
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path)
    return base


def test_the_broad_threshold_is_above_every_group_anyone_has_authored():
    """MEASURED, not chosen — and the check it replaced was measured away.

    Across the tracked fleet's authored declarations the largest group is 20, so
    "wider than 20" means "wider than anything a person has ever declared". The
    first version of this constant was the OPPOSITE check, a THIN warning at 3,
    and the same sweep killed it: 31 of 113 authored groups are under 3 and 22
    are exactly 1, because a size-1 group is a deliberate declaration that a
    component has no backup. It would have fired on a quarter of the correct
    data.

    This test is the guard on that reasoning: if a deck is ever authored with a
    group of 21, the constant is no longer describing the fleet.
    """
    from manamap.config import DECKS_DIR

    sizes = []
    for path in sorted(DECKS_DIR.glob("*/goldfish_targets.json")):
        doc = json.loads(path.read_text())
        if doc.get("scaffolded"):
            continue                      # a draft is not evidence about authoring
        for target in doc.get("targets", []):
            for leg in target.get("need", []):
                sizes.append((len(leg.get("any_of", [])), path.parent.name))
    if len(sizes) < 20:
        pytest.skip("not enough authored declarations to characterise")

    biggest, where = max(sizes)
    assert biggest < st.BROAD_GROUP, (
        f"{where} authored a group of {biggest}; BROAD_GROUP={st.BROAD_GROUP} no "
        f"longer means 'wider than anything anyone has declared'")
    # And the deleted check, kept as a statement about the data rather than a
    # comment that can drift away from it.
    tiny = [n for n, _ in sizes if n < 3]
    assert len(tiny) > len(sizes) * 0.15, (
        "small groups have stopped being normal — the THIN check this module "
        "deleted may be worth revisiting, with numbers")


def test_a_scaffold_never_overwrites_an_authored_file(tmp_path, monkeypatch):
    """An authored declaration is the one thing here no command can rebuild."""
    base = _deck(tmp_path, monkeypatch)
    (base / "goldfish_targets.json").write_text(json.dumps(
        {"targets": [{"label": "REAL", "need": [{"any_of": ["x"]}]}]}))

    with pytest.raises(SystemExit) as exc:
        st.scaffold("t")
    assert "AUTHORED" in str(exc.value)
    # Even --force says what it is about to destroy is not a draft.
    assert "no command can rebuild" in str(exc.value)


def test_a_scaffold_may_be_redrawn_with_force(tmp_path, monkeypatch):
    """`--force` exists for a draft nobody has touched, and says so."""
    base = _deck(tmp_path, monkeypatch)
    (base / "goldfish_targets.json").write_text(json.dumps(
        {"scaffolded": True, "targets": []}))

    with pytest.raises(SystemExit) as exc:
        st.scaffold("t")
    assert "still a scaffold" in str(exc.value)


def test_the_draft_marks_itself_and_names_each_group_s_source(tmp_path, monkeypatch):
    """`scaffolded` is what keeps an unedited draft visible, and `_from` is what
    lets a reader tell a real combo line from a role bucket at a glance."""
    from manamap.config import DECKS_DIR as REAL

    src = REAL / "zur-enchantress" / "cards.json"
    if not src.exists():
        pytest.skip("needs a fetched deck")
    base = tmp_path / "t"
    base.mkdir(parents=True)
    (base / "cards.json").write_text(src.read_text())
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path, raising=False)
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path)

    st.scaffold("t")
    doc = json.loads((base / "goldfish_targets.json").read_text())

    assert doc["scaffolded"] is True
    assert "SCAFFOLD" in doc["_note"] and "PLACEHOLDER" in doc["_note"]
    assert doc["targets"], "a draft with no targets helps nobody"
    for target in doc["targets"]:
        assert target["_from"], "a group with no provenance is indistinguishable from a finding"
        assert target["_from"] == "combo_details" or target["_from"].startswith("role:")


def test_a_combo_line_becomes_one_leg_per_card(tmp_path, monkeypatch):
    """The schema represents a combo exactly: `need` is an AND of ORs, so a line
    wanting A and B is two legs of one card — not one group of two, which would
    say either card alone assembles it."""
    from manamap.config import DECKS_DIR as REAL

    src = REAL / "zur-enchantress" / "cards.json"
    if not src.exists():
        pytest.skip("needs a fetched deck")
    base = tmp_path / "t"
    base.mkdir(parents=True)
    (base / "cards.json").write_text(src.read_text())
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path, raising=False)
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path)

    got = st.derive("t")
    lines = [t for t in got["targets"] if t["_from"] == "combo_details"]
    if not lines:
        pytest.skip("this deck contains no combo lines")
    for line in lines:
        assert len(line["need"]) >= 2, "a combo needs every card, not any of them"
        assert all(len(leg["any_of"]) == 1 for leg in line["need"])


def test_the_commander_is_never_a_target(tmp_path, monkeypatch):
    """It is on every board from turn one, so it says nothing about what a draw
    assembles — the same exclusion `validate_goldfish_targets` makes."""
    from manamap.config import DECKS_DIR as REAL

    src = REAL / "zur-enchantress" / "cards.json"
    if not src.exists():
        pytest.skip("needs a fetched deck")
    base = tmp_path / "t"
    base.mkdir(parents=True)
    (base / "cards.json").write_text(src.read_text())
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path, raising=False)
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path)

    commander = next(c["name"] for c in json.loads(src.read_text())["cards"]
                     if c.get("is_commander"))
    got = st.derive("t")
    for target in got["targets"]:
        for leg in target["need"]:
            assert commander not in leg["any_of"], target["label"]


def test_a_scaffold_declares_no_win_line(tmp_path, monkeypatch):
    """THE defect this whole area exists to catch: heliod's Hullbreaker Horror
    and ur-dragon's Aggravated Assault are each named in two passing stacks and
    in no component, so the simulator never measured how those decks win. A
    machine guessing at one manufactures exactly the claim the validator demands
    — so the draft states finishers as a GROUP and never a LINE."""
    from manamap.config import DECKS_DIR as REAL

    src = REAL / "zur-enchantress" / "cards.json"
    if not src.exists():
        pytest.skip("needs a fetched deck")
    base = tmp_path / "t"
    base.mkdir(parents=True)
    (base / "cards.json").write_text(src.read_text())
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", tmp_path, raising=False)
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path)

    got = st.derive("t")
    role_targets = [t for t in got["targets"] if t["_from"].startswith("role:")]
    for target in role_targets:
        assert len(target["need"]) == 1, (
            f"{target['label']!r} declares a multi-leg LINE from a role axis — "
            f"that is a claim about how the deck wins, and it is the pilot's")
