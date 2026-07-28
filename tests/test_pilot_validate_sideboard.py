"""Tests for the sideboard-analysis contract.

Synthetic fixtures throughout: the repo contains exactly one analysable sideboard
card, and coverage that depends on one Instant is not coverage.
"""

import json

import pytest

from manamap.pilot import validate_sideboard as vs


def card(name, **overrides):
    base = {"name": name, "is_commander": False, "is_sideboard": False,
            "type_line": "Instant", "mana_cost": "{1}{R}", "cmc": 2.0}
    base.update(overrides)
    return base


DECK = {"cards": [
    card("Zada, Hedron Grinder", is_commander=True, type_line="Legendary Creature"),
    card("Witch's Mark"),
    card("Chaos Warp"),
    card("Mountain", type_line="Basic Land — Mountain"),
    card("Sazacap's Brew", is_sideboard=True),
    card("Storm Counter", is_sideboard=True, type_line="Card", mana_cost="", cmc=0.0),
]}


def analysis(**overrides):
    doc = {
        "slug": "toy",
        "assessment": "One flex slot for graveyard-light metas.",
        "swaps": [{
            "in": "Sazacap's Brew", "out": "Witch's Mark", "role": "draw:engine",
            "when": "against graveyard-light tables",
            "why": "Instant speed matters when the storm turn is the same turn.",
        }],
        "opens_lines": [],
        "long_term_defaults": [
            {"card": "Sazacap's Brew", "verdict": "keep-in-sideboard",
             "why": "Only better when the meta is graveyard-light."},
        ],
    }
    doc.update(overrides)
    return doc


def errors(**overrides):
    return vs.validate(analysis(**overrides), DECK)


# ── The pool constraint — the whole point of the feature ─────────────────


def test_a_clean_analysis_validates():
    assert errors() == []


def test_incoming_card_must_be_in_the_sideboard():
    """The constraint that makes this agent safe: it cannot conjure a card."""
    bad = errors(swaps=[dict(analysis()["swaps"][0], **{"in": "Lightning Bolt"})])
    assert any("not in this deck's sideboard" in e for e in bad)


def test_incoming_card_may_not_be_a_table_accessory():
    bad = errors(swaps=[dict(analysis()["swaps"][0], **{"in": "Storm Counter"})])
    assert any("table accessory" in e for e in bad)


def test_outgoing_card_must_be_in_the_maindeck():
    bad = errors(swaps=[dict(analysis()["swaps"][0], out="Nonexistent Card")])
    assert any("not a maindeck card" in e for e in bad)


def test_you_cannot_cut_your_commander():
    bad = errors(swaps=[dict(analysis()["swaps"][0], out="Zada, Hedron Grinder")])
    assert any("cannot cut your commander" in e for e in bad)


def test_two_swaps_cannot_claim_the_same_slot():
    base = analysis()["swaps"][0]
    bad = errors(swaps=[base, dict(base, role="removal:spot")])
    assert any("already cut by swap 0" in e for e in bad)


# ── The empty-`why` check that 56 hapatra swaps would have failed ────────


def test_an_empty_why_is_rejected():
    bad = errors(swaps=[dict(analysis()["swaps"][0], why="")])
    assert any("`why` is empty" in e for e in bad)


def test_a_whitespace_why_is_also_rejected():
    bad = errors(swaps=[dict(analysis()["swaps"][0], why="   ")])
    assert any("`why` is empty" in e for e in bad)


def test_an_empty_when_is_rejected():
    bad = errors(swaps=[dict(analysis()["swaps"][0], when="")])
    assert any("`when` is empty" in e for e in bad)


def test_missing_swap_keys_are_named():
    bad = errors(swaps=[{"in": "Sazacap's Brew", "out": "Witch's Mark"}])
    assert any("missing keys" in e and "when" in e for e in bad)


# ── Candidate lines stay candidates ──────────────────────────────────────


def test_an_opened_line_with_an_unknown_status_is_rejected():
    bad = errors(opens_lines=[
        {"cards": ["A", "B"], "why_plausible": "Both are now present.", "status": "probably fine"},
    ])
    assert any("needs a stack scenario" in e for e in bad)


def test_a_verified_line_without_an_artifact_is_rejected():
    """The claim needs a checker-passed stack behind it, not just the word."""
    bad = errors(opens_lines=[
        {"cards": ["A", "B"], "why_plausible": "Both are now present.", "status": "verified"},
    ])
    assert any("requires a `stack_artifact`" in e for e in bad)


def _stack_artifact(tmp_path, verdict="pass", cards=("Sazacap's Brew", "Chaos Warp")):
    stacks = tmp_path / "stacks"
    stacks.mkdir(exist_ok=True)
    path = stacks / "001-toy-line.json"
    path.write_text(json.dumps({
        "scenario": {"question": " and ".join(cards)},
        "checker": {"verdict": verdict},
    }))
    return "stacks/001-toy-line.json"


def _verified_line(rel, cards=("Sazacap's Brew", "Chaos Warp")):
    return {"cards": list(cards), "why_plausible": "Checker-passed.",
            "status": vs.VERIFIED_STATUS, "stack_artifact": rel}


def test_a_verified_line_with_a_passing_artifact_passes(tmp_path):
    rel = _stack_artifact(tmp_path)
    doc = analysis(opens_lines=[_verified_line(rel)])
    assert vs.validate(doc, DECK, deck_path=tmp_path) == []


def test_a_verified_line_with_a_failing_artifact_is_rejected(tmp_path):
    rel = _stack_artifact(tmp_path, verdict="fail")
    doc = analysis(opens_lines=[_verified_line(rel)])
    bad = vs.validate(doc, DECK, deck_path=tmp_path)
    assert any("not 'pass'" in e for e in bad)


def test_a_verified_line_with_a_missing_artifact_is_rejected(tmp_path):
    doc = analysis(opens_lines=[_verified_line("stacks/999-nope.json")])
    bad = vs.validate(doc, DECK, deck_path=tmp_path)
    assert any("does not exist" in e for e in bad)


def test_a_verified_line_whose_artifact_never_names_the_cards_is_rejected(tmp_path):
    rel = _stack_artifact(tmp_path, cards=("Lightning Bolt",))
    doc = analysis(opens_lines=[_verified_line(rel)])
    bad = vs.validate(doc, DECK, deck_path=tmp_path)
    assert any("never mentions" in e for e in bad)


def test_a_verified_line_without_a_deck_path_cannot_be_confirmed():
    doc = analysis(opens_lines=[_verified_line("stacks/001-toy-line.json")])
    bad = vs.validate(doc, DECK)
    assert any("cannot be confirmed" in e for e in bad)


def test_a_correctly_flagged_line_passes():
    assert vs.validate(analysis(opens_lines=[
        {"cards": ["Sazacap's Brew", "Chaos Warp"], "why_plausible": "Both present.",
         "status": vs.UNVERIFIED_STATUS},
    ]), DECK) == []


def test_a_line_with_no_cards_is_rejected():
    bad = errors(opens_lines=[
        {"cards": [], "why_plausible": "x", "status": vs.UNVERIFIED_STATUS}])
    assert any("no cards named" in e for e in bad)


# ── Long-term defaults ───────────────────────────────────────────────────


def test_verdict_must_be_in_the_closed_set():
    bad = errors(long_term_defaults=[
        {"card": "Sazacap's Brew", "verdict": "maybe", "why": "unsure"}])
    assert any("verdict must be one of" in e for e in bad)


def test_a_default_verdict_about_a_card_not_in_the_sideboard_is_rejected():
    bad = errors(long_term_defaults=[
        {"card": "Chaos Warp", "verdict": "promote", "why": "good card"}])
    assert any("not in the sideboard" in e for e in bad)


def test_a_default_verdict_about_an_accessory_says_so():
    bad = errors(long_term_defaults=[
        {"card": "Storm Counter", "verdict": "promote", "why": "counts things"}])
    assert any("table accessory" in e for e in bad)


# ── Structure ────────────────────────────────────────────────────────────


def test_missing_top_level_keys_short_circuit():
    bad = vs.validate({"slug": "toy"}, DECK)
    assert len(bad) == 1 and "Missing top-level keys" in bad[0]


def test_an_empty_assessment_is_rejected():
    bad = errors(assessment="")
    assert any("assessment is empty" in e for e in bad)


def test_no_swaps_is_a_legitimate_answer():
    """"Nothing here is worth a slot" is a real conclusion, not a failure."""
    assert vs.validate(analysis(swaps=[], long_term_defaults=[]), DECK) == []


# ── Bracket deltas are recomputed, never trusted ─────────────────────────


def test_a_wrong_bracket_delta_is_caught(monkeypatch):
    """An agent asserting a power-level change is not evidence of one."""
    monkeypatch.setattr(vs.bracket_mod, "load_reference", lambda: ({}, {}, {"combos": [], "by_card": {}}))
    monkeypatch.setattr(vs.bracket_mod, "assess",
                        lambda names, *a, **k: {"floor": 2, "drivers": []})
    bad = vs.validate(analysis(swaps=[dict(analysis()["swaps"][0],
                                           bracket_delta={"before": 2, "after": 4})]), DECK)
    assert any("recomputed" in e for e in bad)


def test_a_correct_bracket_delta_passes(monkeypatch):
    monkeypatch.setattr(vs.bracket_mod, "load_reference", lambda: ({}, {}, {"combos": [], "by_card": {}}))
    monkeypatch.setattr(vs.bracket_mod, "assess",
                        lambda names, *a, **k: {"floor": 3, "drivers": []})
    assert vs.validate(analysis(swaps=[dict(analysis()["swaps"][0],
                                            bracket_delta={"before": 3, "after": 3})]), DECK) == []


def test_absent_reference_artifacts_skip_the_check_rather_than_fail(monkeypatch):
    def boom():
        raise SystemExit("no cards.csv")
    monkeypatch.setattr(vs.bracket_mod, "load_reference", boom)
    assert vs.validate(analysis(swaps=[dict(analysis()["swaps"][0],
                                            bracket_delta={"before": 1, "after": 9})]), DECK) == []
