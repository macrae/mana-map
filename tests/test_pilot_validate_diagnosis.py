"""validate-diagnosis: the contract on a deck diagnosis.

The check worth the file is `orphans_stack`. A cut list that quietly proposes the
one card a checker-passed line rests on is the exact failure this artifact exists
to prevent, and nothing else in the repo detects it — `validate_considering`
checks that a `natural_cut` is a real card and stops there.
"""

import json

import pytest

from manamap.pilot import validate_diagnosis as vd
from manamap.pilot.common import deck_dir

from conftest import requires_deck, requires_roles, requires_strategy


def _deck(main=("Alpha", "Beta"), commander="Cmdr", bench=()):
    cards = [{"name": commander, "is_commander": True, "type_line": "Legendary Creature"}]
    cards += [{"name": n, "type_line": "Creature"} for n in main]
    cards += [{"name": n, "type_line": "Creature", "is_sideboard": True} for n in bench]
    return {"cards": cards}


def _doc(**overrides):
    base = {
        "slug": "test",
        "verdict": "It grinds; it does not close.",
        "axes": [],
        "engine": {},
        "lean_into": [],
        "cut_candidates": [],
        "add_candidates": [],
        "open_questions": [],
        "gaps": [],
    }
    base.update(overrides)
    return base


def _errors(doc, deck=None, **kw):
    return vd.validate(doc, deck or _deck(), **kw)


# ── Shape ────────────────────────────────────────────────────────────────

def test_missing_top_keys_short_circuits():
    errors = _errors({"slug": "test"})
    assert len(errors) == 1 and "Missing top-level keys" in errors[0]


def test_empty_verdict_fails():
    assert any("verdict is empty" in e for e in _errors(_doc(verdict="   ")))


def test_a_minimal_diagnosis_passes():
    assert _errors(_doc()) == []


# ── Axes are re-derived, not trusted ─────────────────────────────────────

MEASURED = {"card-advantage": {"axis": "card-advantage",
                               "measured": {"value": 5, "unit": "copies"}}}


def test_axis_value_must_match_the_audit():
    doc = _doc(axes=[{"axis": "card-advantage", "verdict": "weakness",
                      "measured": {"value": 9}, "reading": "thin"}])
    errors = _errors(doc, measured_axes=MEASURED)
    assert any("deck-audit computes 5" in e for e in errors)


def test_matching_axis_value_passes():
    doc = _doc(axes=[{"axis": "card-advantage", "verdict": "weakness",
                      "measured": {"value": 5}, "reading": "thin"}])
    assert _errors(doc, measured_axes=MEASURED) == []


def test_unknown_axis_name_fails():
    doc = _doc(axes=[{"axis": "vibes", "verdict": "strength",
                      "measured": {"value": 1}, "reading": "good"}])
    assert any("measures no axis by that name" in e
               for e in _errors(doc, measured_axes=MEASURED))


def test_axis_verdict_vocabulary_is_closed():
    doc = _doc(axes=[{"axis": "card-advantage", "verdict": "fine",
                      "measured": {"value": 5}, "reading": "ok"}])
    assert any("verdict must be one of" in e for e in _errors(doc))


def test_axis_without_a_reading_fails():
    doc = _doc(axes=[{"axis": "card-advantage", "verdict": "weakness",
                      "measured": {"value": 5}, "reading": ""}])
    assert any("`reading` is empty" in e for e in _errors(doc))


def test_duplicate_axis_fails():
    axis = {"axis": "card-advantage", "verdict": "weakness",
            "measured": {"value": 5}, "reading": "thin"}
    assert any("duplicate" in e for e in _errors(_doc(axes=[axis, dict(axis)])))


# ── Cuts ─────────────────────────────────────────────────────────────────

def _cut(**kw):
    base = {"card": "Alpha", "why": "redundant", "cost_of_cutting": "a body",
            "difficulty": "easy"}
    base.update(kw)
    return base


def test_cut_must_be_in_the_maindeck():
    errors = _errors(_doc(cut_candidates=[_cut(card="Not In Deck")]))
    assert any("not in the maindeck" in e for e in errors)


def test_cut_may_not_be_the_commander():
    assert any("commander cannot be cut" in e
               for e in _errors(_doc(cut_candidates=[_cut(card="Cmdr")])))


def test_duplicate_cut_fails():
    assert any("duplicate" in e
               for e in _errors(_doc(cut_candidates=[_cut(), _cut()])))


def test_cut_difficulty_vocabulary_is_closed():
    assert any("difficulty must be one of" in e
               for e in _errors(_doc(cut_candidates=[_cut(difficulty="hard")])))


def test_cut_without_a_stated_cost_fails():
    """Every cut costs something, and naming it is the job."""
    assert any("cost_of_cutting" in e
               for e in _errors(_doc(cut_candidates=[_cut(cost_of_cutting="  ")])))


# ── orphans_stack is computed, not read ──────────────────────────────────

@requires_deck
def test_cutting_a_verified_lines_card_must_be_priced(tmp_path):
    """South Wind Avatar carries stack 005; a cut list that omits that lies."""
    base = deck_dir("yawgmoth-swarm")
    doc = _doc(cut_candidates=[_cut(card="South Wind Avatar", orphans_stack=None)])
    deck = _deck(main=("South Wind Avatar",))
    errors = vd.validate(doc, deck, deck_path=base)
    assert any("checker-passed stack(s) 005" in e for e in errors)


@requires_deck
def test_declaring_the_right_stack_passes():
    base = deck_dir("yawgmoth-swarm")
    doc = _doc(cut_candidates=[_cut(card="South Wind Avatar", orphans_stack=["005"])])
    errors = vd.validate(doc, _deck(main=("South Wind Avatar",)), deck_path=base)
    assert errors == []


@requires_deck
def test_claiming_a_stack_that_does_not_name_the_card_fails():
    base = deck_dir("yawgmoth-swarm")
    doc = _doc(cut_candidates=[_cut(card="Toxic Deluge", orphans_stack=["001"])])
    errors = vd.validate(doc, _deck(main=("Toxic Deluge",)), deck_path=base)
    assert any("no checker-passed stack" in e for e in errors)


@requires_deck
def test_stacks_naming_reads_the_scenario_only():
    """A checker note may discuss a card the board never held.

    A discussion is not a dependency, so the probe reads the scenario block and
    nothing else.
    """
    base = deck_dir("yawgmoth-swarm")
    assert vd.stacks_naming(base, "South Wind Avatar") == ["005"]
    assert vd.stacks_naming(base, "Skullclamp") == []


# ── Adds ─────────────────────────────────────────────────────────────────

def _add(**kw):
    base = {"card": "Gamma", "closes": "card-advantage", "source": "pool",
            "why": "draws two"}
    base.update(kw)
    return base


def test_add_already_in_the_deck_fails():
    assert any("already in the maindeck" in e
               for e in _errors(_doc(add_candidates=[_add(card="Alpha")])))


def test_add_source_vocabulary_is_closed():
    assert any("source must be one of" in e
               for e in _errors(_doc(add_candidates=[_add(source="wishlist")])))


def test_bench_source_must_really_be_on_the_bench():
    doc = _doc(add_candidates=[_add(card="Gamma", source="sideboard")])
    assert any("sideboard holds no such card" in e for e in _errors(doc))
    deck = _deck(bench=("Gamma",))
    assert vd.validate(doc, deck) == []


def test_add_that_closes_nothing_fails():
    assert any("`closes` is empty" in e
               for e in _errors(_doc(add_candidates=[_add(closes="")])))


def test_natural_cut_must_be_real_and_claimed_once():
    doc = _doc(add_candidates=[_add(card="Gamma", natural_cut="Nope")])
    assert any("natural_cut 'Nope' is not in the maindeck" in e for e in _errors(doc))
    doc = _doc(add_candidates=[_add(card="Gamma", natural_cut="Alpha"),
                               _add(card="Delta", natural_cut="Alpha")])
    assert any("already claimed" in e for e in _errors(doc))


def test_natural_cut_may_not_be_the_commander():
    doc = _doc(add_candidates=[_add(natural_cut="Cmdr")])
    assert any("may not be the commander" in e for e in _errors(doc))


def test_combo_line_status_vocabulary_is_closed():
    doc = _doc(add_candidates=[_add(combo_lines_opened=[
        {"cards": ["A", "B"], "status": "confirmed"}])])
    assert any("status must be" in e for e in _errors(doc))


# ── Open questions ───────────────────────────────────────────────────────

def test_settled_by_must_route_to_a_real_skill():
    doc = _doc(open_questions=[{"question": "does it loop?",
                                "settled_by": "thinking about it",
                                "why_it_matters": "the bracket"}])
    assert any("settled_by must be one of" in e for e in _errors(doc))


def test_a_routable_question_passes():
    doc = _doc(open_questions=[{"question": "does it loop?",
                                "settled_by": "resolve-stack",
                                "why_it_matters": "the bracket floor rests on it"}])
    assert _errors(doc) == []


# ── The skeptic ──────────────────────────────────────────────────────────

def test_pass_with_an_open_finding_is_an_inconsistency():
    doc = _doc(skeptic={"verdict": "pass", "findings": [
        {"status": "miscounted", "where": "axes[2]", "note": "off by three"}]})
    assert any("pass alongside an open finding" in e for e in _errors(doc))


def test_fail_with_an_open_finding_is_fine():
    doc = _doc(skeptic={"verdict": "fail", "findings": [
        {"status": "miscounted", "where": "axes[2]", "note": "off by three"}]})
    assert _errors(doc) == []


def test_pass_with_only_supported_findings_is_fine():
    doc = _doc(skeptic={"verdict": "pass", "findings": [
        {"status": "supported", "where": "axes[0]", "note": ""}]})
    assert _errors(doc) == []


def test_skeptic_status_vocabulary_is_closed():
    doc = _doc(skeptic={"verdict": "fail", "findings": [
        {"status": "wrong", "where": "axes[0]"}]})
    assert any("status must be one of" in e for e in _errors(doc))


def test_absent_skeptic_block_is_allowed():
    """A diagnosis exists before the skeptic reads it; the loop merges it later."""
    assert _errors(_doc()) == []


# ── Citations ────────────────────────────────────────────────────────────

@requires_strategy
def test_a_non_verbatim_strategy_quote_fails():
    from manamap.pilot.validate_stack import load_strategy_sections
    sections = load_strategy_sections()
    doc = _doc(lean_into=[{"what": "the engine", "why": "it grinds", "citations": [
        {"rule": "strategy:deckbuilding.ratios",
         "quote": "run exactly eleven card draw spells"}]}])
    errors = vd.validate(doc, _deck(), strategy_sections=sections)
    assert any("not verbatim" in e for e in errors)


@requires_strategy
def test_a_verbatim_strategy_quote_passes():
    from manamap.pilot.validate_stack import load_strategy_sections
    sections = load_strategy_sections()
    doc = _doc(lean_into=[{"what": "the engine", "why": "it grinds", "citations": [
        {"rule": "strategy:deckbuilding.ratios",
         "quote": "Take the categories, derive the counts from the deck's actual "
                  "failure modes."}]}])
    assert vd.validate(doc, _deck(), strategy_sections=sections) == []


@requires_strategy
def test_a_nonexistent_strategy_section_fails():
    from manamap.pilot.validate_stack import load_strategy_sections
    sections = load_strategy_sections()
    doc = _doc(cut_candidates=[_cut(citations=[
        {"rule": "strategy:deckbuilding.vibes", "quote": "trust me"}])])
    errors = vd.validate(doc, _deck(), strategy_sections=sections)
    assert any("nonexistent strategy section" in e for e in errors)


# ── Integration ──────────────────────────────────────────────────────────

@requires_deck
@requires_roles
def test_a_real_diagnosis_against_a_real_audit(tmp_path):
    """Build a diagnosis from the audit's own figures; it must validate clean."""
    from manamap.pilot import deck_audit
    from manamap.pilot.common import load_deck_cards

    audit = deck_audit.analyze("yawgmoth-swarm")
    measured = {a["axis"]: a for a in audit["axes"]}
    doc = _doc(
        slug="yawgmoth-swarm",
        axes=[{"axis": name, "verdict": "adequate",
               "measured": {"value": a["measured"]["value"]},
               "reading": "carried from the audit"}
              for name, a in list(measured.items())[:4]],
    )
    errors = vd.validate(doc, load_deck_cards("yawgmoth-swarm"),
                         deck_path=deck_dir("yawgmoth-swarm"),
                         measured_axes=measured)
    assert errors == []


@requires_deck
@requires_roles
def test_a_diagnosis_that_retypes_a_figure_is_caught():
    from manamap.pilot import deck_audit
    from manamap.pilot.common import load_deck_cards

    audit = deck_audit.analyze("yawgmoth-swarm")
    measured = {a["axis"]: a for a in audit["axes"]}
    real = measured["card-advantage"]["measured"]["value"]
    doc = _doc(slug="yawgmoth-swarm",
               axes=[{"axis": "card-advantage", "verdict": "weakness",
                      "measured": {"value": real + 3}, "reading": "thin"}])
    errors = vd.validate(doc, load_deck_cards("yawgmoth-swarm"),
                         measured_axes=measured)
    assert any(f"deck-audit computes {real!r}" in e for e in errors)
