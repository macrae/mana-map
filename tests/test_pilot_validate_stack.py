"""Tests for the citation contract enforcer (pilot validate_stack)."""

import copy
import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot import validate_stack as vs
from manamap.pilot.validate_stack import validate_any, validate_decision, validate_scenario

from conftest import requires_deck, requires_rules

RULES = {
    "702.40a": {
        "text": (
            "Storm is a triggered ability that functions on the stack. “Storm” means "
            "“When you cast this spell, copy it for each other spell that was cast "
            "before it this turn.”"
        ),
        "section": "702. Keyword Abilities",
        "parent": "702.40",
    },
    "601.2a": {
        "text": "To propose the casting of a spell, a player first moves that card to the stack.",
        "section": "601. Casting Spells",
        "parent": "601.2",
    },
}


def valid_doc():
    return {
        "id": "001",
        "slug": "storm-count",
        "deck": "test-deck",
        "title": "Storm count check",
        "rules_version": "June 19, 2026",
        "scenario": {
            "stack": [{"pos": 0, "object": "Empty the Warrens", "controller": "you"}],
            "question": "How many copies?",
        },
        "resolution": {
            "steps": [
                {
                    "n": 1,
                    "action": "Storm trigger goes on the stack",
                    "effect": "Creates one copy per prior spell",
                    "citations": [
                        {
                            "rule": "702.40a",
                            "quote": "copy it for each other spell that was cast before it this turn",
                        }
                    ],
                }
            ],
            "final_state": {"summary": "4 copies"},
        },
        "checker": {
            "verdict": "pass",
            "iterations": 1,
            "findings": [{"step": 1, "rule": "702.40a", "status": "supported", "note": ""}],
        },
    }


def test_valid_doc_passes():
    assert validate_scenario(valid_doc(), RULES) == []


def test_step_without_citation_fails():
    doc = valid_doc()
    doc["resolution"]["steps"][0]["citations"] = []
    errors = validate_scenario(doc, RULES)
    assert any("NO CITATIONS" in e for e in errors)


def test_bogus_rule_id_fails():
    doc = valid_doc()
    doc["resolution"]["steps"][0]["citations"][0]["rule"] = "999.99z"
    errors = validate_scenario(doc, RULES)
    assert any("nonexistent rule" in e for e in errors)


def test_malformed_rule_id_fails():
    doc = valid_doc()
    doc["resolution"]["steps"][0]["citations"][0]["rule"] = "70240a"
    errors = validate_scenario(doc, RULES)
    assert any("malformed rule id" in e for e in errors)


def test_non_substring_quote_fails():
    doc = valid_doc()
    doc["resolution"]["steps"][0]["citations"][0]["quote"] = "storm makes infinite copies"
    errors = validate_scenario(doc, RULES)
    assert any("not verbatim" in e for e in errors)


def test_whitespace_normalized_quote_passes():
    doc = valid_doc()
    # Same words, different whitespace/line breaks — still verbatim after normalization.
    doc["resolution"]["steps"][0]["citations"][0]["quote"] = (
        "copy it  for each other\nspell that was cast before it this turn"
    )
    assert validate_scenario(doc, RULES) == []


def test_empty_steps_fail():
    doc = valid_doc()
    doc["resolution"]["steps"] = []
    errors = validate_scenario(doc, RULES)
    assert any("steps is empty" in e for e in errors)


def test_pass_verdict_with_unsupported_finding_fails():
    doc = valid_doc()
    doc["checker"]["findings"].append(
        {"step": 1, "rule": "601.2a", "status": "irrelevant", "note": "wrong rule"}
    )
    errors = validate_scenario(doc, RULES)
    assert any("not 'supported'" in e for e in errors)


def test_invalid_checker_status_fails():
    doc = valid_doc()
    doc["checker"]["findings"][0]["status"] = "looks-fine"
    errors = validate_scenario(doc, RULES)
    assert any("invalid status" in e for e in errors)


def test_missing_top_keys_fail():
    errors = validate_scenario({"id": "001"}, RULES)
    assert any("Missing top-level keys" in e for e in errors)


# ── Decision scenarios (kind: "decision", coaching tier) ──


def valid_decision():
    return {
        "id": "001",
        "slug": "open-mana-signal",
        "deck": "test-deck",
        "kind": "decision",
        "title": "Cast now or hold?",
        "scenario": {
            "board": {"you": "Zada + 4 tokens", "table": "sweeper mana open"},
            "question": "Cast the cantrip now or hold?",
        },
        "branches": [
            {"choice": "Cast now", "line": "Fire it.", "signals": "Engine live.",
             "coalition_risk": "High.", "coaching": "Only if lethal."},
            {"choice": "Hold", "line": "Pass.", "signals": "Looks like interaction.",
             "coalition_risk": "Low.", "coaching": "Default.",
             "citations": [{"rule": "702.40a", "quote": "copy it for each other spell that was cast before it this turn"}]},
        ],
        "recommendation": {"choice": "Hold", "rationale": "Sweeper undoes everything."},
    }


def test_valid_decision_passes():
    assert validate_decision(valid_decision(), RULES) == []


def test_validate_any_dispatches_on_kind():
    assert validate_any(valid_decision(), RULES) == []
    assert validate_any(valid_doc(), RULES) == []  # missing kind -> stack


def test_decision_needs_two_branches():
    doc = valid_decision()
    doc["branches"] = doc["branches"][:1]
    errors = validate_decision(doc, RULES)
    assert any(">=2 branches" in e for e in errors)


def test_decision_branch_missing_fields():
    doc = valid_decision()
    del doc["branches"][0]["coalition_risk"]
    errors = validate_decision(doc, RULES)
    assert any("missing keys" in e and "coalition_risk" in e for e in errors)


def test_decision_recommendation_must_match_branch():
    doc = valid_decision()
    doc["recommendation"]["choice"] = "Concede"
    errors = validate_decision(doc, RULES)
    assert any("does not match any branch" in e for e in errors)


def test_decision_branch_citations_use_same_contract():
    doc = valid_decision()
    doc["branches"][1]["citations"][0]["quote"] = "made-up rule text"
    errors = validate_decision(doc, RULES)
    assert any("not verbatim" in e for e in errors)


def test_decision_requires_board_and_question():
    doc = valid_decision()
    doc["scenario"]["board"] = {}
    errors = validate_decision(doc, RULES)
    assert any("scenario.board" in e for e in errors)


# ── Golden artifacts: every committed scenario must hold the contract and pass ──


@requires_deck
@requires_rules
def test_all_committed_stacks_validate_and_pass():
    from manamap.pilot.common import load_rules_db

    rules, _, _ = load_rules_db()
    stacks_dir = DECKS_DIR / "goblin-storm" / "stacks"
    paths = sorted(stacks_dir.glob("*.json"))
    if not paths:
        pytest.skip("no stack scenarios committed yet (P4 pending)")
    for path in paths:
        with open(path) as f:
            doc = json.load(f)
        assert validate_scenario(doc, rules) == [], f"{path.name} violates the contract"
        assert doc["checker"]["verdict"] == "pass", f"{path.name} is not checker-verified"


# ── Preflight: format conventions, warn-for-published ────────────────────


def _scenario(**over):
    doc = {
        "id": "900", "slug": "x", "deck": "X", "title": "t",
        "scenario": {"board": {"you": []}, "hand": [], "mana_available": "{0}",
                     "stack": [{"pos": 0, "object": "o"}], "question": "What?"},
    }
    doc["scenario"].update(over)
    return doc


def test_preflight_errors_on_prose_in_hand_for_an_unresolved_scenario():
    """The exact bug that shipped: a placeholder sentence written into `hand`.

    `build_index.line_cards` read it as a card name and put it in the deck
    manifest. Caught here for free, before a ~35k-token resolver spawn.
    """
    errors, _ = vs.validate_preflight(
        _scenario(hand=["no cards relevant to this question."]))
    assert any("looks like prose" in e for e in errors)


def test_preflight_errors_on_empty_mana_available():
    errors, _ = vs.validate_preflight(_scenario(mana_available=""))
    assert any('use "{0}"' in e for e in errors)


def test_preflight_only_WARNS_once_a_resolution_exists():
    """Published work is warned about, never blocked.

    `scenario:self` is a fingerprint input, so normalising a committed scenario
    costs a respawn. The gate exists to stop a wasted spawn; against work already
    done it has no spawn left to save.
    """
    doc = _scenario(mana_available="", hand=["a whole sentence that is not a card."])
    doc["resolution"] = {"steps": [], "final_state": {}}
    errors, warnings = vs.validate_preflight(doc)
    assert errors == []
    assert any('use "{0}"' in w for w in warnings)
    assert any("looks like prose" in w for w in warnings)


def test_preflight_warns_on_the_outlier_board_shape():
    _, warnings = vs.validate_preflight(
        _scenario(board={"you": [], "opponent_a": ["x"]}))
    assert any("opponent_a..d" in w for w in warnings)


@requires_deck
def test_every_committed_scenario_preflights_without_errors():
    """The corpus predates these rules and must not be blocked by them."""
    from manamap.config import DECKS_DIR
    checked = 0
    for deck in sorted(DECKS_DIR.iterdir()):
        stacks = deck / "stacks"
        if not stacks.is_dir():
            continue
        for path in sorted(stacks.glob("*.json")):
            doc = json.loads(path.read_text())
            if doc.get("kind") == "decision":
                continue
            errors, _ = vs.validate_preflight(doc)
            card_errors, _ = vs.unknown_cards(doc, deck.name)
            if "resolution" not in doc:
                errors = errors + card_errors
            assert errors == [], f"{path.name}: {errors}"
            checked += 1
    assert checked >= 40


@requires_deck
def test_unknown_cards_separates_a_real_card_from_a_typo():
    """Three outcomes, graded against the CORPUS rather than a zone.

    With no sideboard the zone cannot grade anything, so the question becomes
    whether the name is a Magic card at all. A scenario may legitimately name a
    card outside the 99 — an opponent's permanent, or one under consideration —
    while a name the corpus has never heard of is a typo, and the line as written
    cannot be played.

    Builds from live data rather than hardcoding a decklist: the earlier version
    named Exquisite Blood and broke the day that card left edgar-vampires.
    """
    from manamap.pilot.common import load_deck_cards

    try:
        cards = load_deck_cards("edgar-vampires")["cards"]
    except FileNotFoundError:
        pytest.skip("edgar-vampires not fetched")
    if vs._corpus_names() is None:
        pytest.skip("cards.csv absent — the corpus check degrades to warnings")

    in_deck = cards[0]["name"]
    doc = {"scenario": {"board": {"you": [in_deck]}, "hand": [], "stack": []}}
    errors, warnings = vs.unknown_cards(doc, "edgar-vampires")
    assert errors == [] and warnings == [], (
        f"a card the deck runs must be silent: {errors} {warnings}")

    # A real card this deck does not run — warn, never error.
    outside = {"scenario": {"board": {"you": ["Black Lotus"]},
                            "hand": [], "stack": []}}
    errors, warnings = vs.unknown_cards(outside, "edgar-vampires")
    assert errors == [], f"a real card outside the 99 must warn, not error: {errors}"
    assert any("does not run" in w for w in warnings), warnings

    # Not a Magic card at all — that is a typo and it errors.
    typo = {"scenario": {"board": {"you": ["Nicol Bolas, Planewalker"]},
                         "hand": [], "stack": []}}
    errors, _ = vs.unknown_cards(typo, "edgar-vampires")
    assert errors, "a name no Magic card bears must be an error"


@requires_deck
def test_a_dfc_front_face_counts_as_the_card_the_deck_runs():
    """`cards.json` stores "A // B"; a scenario names the front face.

    Matching full names only reported sisay/003 as exploring a card outside the
    99 when the deck runs the card that face belongs to.
    """
    from manamap.pilot.common import load_deck_cards

    try:
        cards = load_deck_cards("sisay")["cards"]
    except FileNotFoundError:
        pytest.skip("sisay not fetched")
    dfc = next((c["name"] for c in cards if " // " in c["name"]), None)
    if dfc is None:
        pytest.skip("sisay runs no double-faced card")

    front = dfc.split(" // ")[0].strip()
    doc = {"scenario": {"board": {"you": [front]}, "hand": [], "stack": []}}
    errors, warnings = vs.unknown_cards(doc, "sisay")
    assert errors == [] and warnings == [], (
        f"the front face of a card the deck runs must be silent: {errors} {warnings}")
