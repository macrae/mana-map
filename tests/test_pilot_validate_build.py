"""Tests for the build-plan form gate (pilot/validate_build.py)."""

import pytest

from manamap.pilot.validate_build import deck_card_names, validate


def _plan(**overrides):
    plan = {
        "slug": "test-deck",
        "commander": "Hapatra",
        "color_identity": ["B", "G"],
        "bracket": {"target": 3, "computed_floor": 1},
        "role_budget": {"lands": 2, "flex": 2},
        "slots": [
            {"name": "Sol Ring", "role": "flex"},
            {"name": "Deadly Dispute", "role": "flex"},
        ],
        "land_counts": {"Swamp": 1, "Forest": 1},
    }
    plan.update(overrides)
    return plan


def _cards(extra=None):
    cards = {
        "Hapatra": {"color_identity": "B, G", "legal_commander": "legal",
                    "type_line": "Legendary Creature — Snake"},
        "Sol Ring": {"color_identity": "", "legal_commander": "legal",
                     "type_line": "Artifact"},
        "Deadly Dispute": {"color_identity": "B", "legal_commander": "legal",
                           "type_line": "Instant"},
        "Swamp": {"color_identity": "B", "legal_commander": "legal",
                  "type_line": "Basic Land — Swamp"},
        "Forest": {"color_identity": "G", "legal_commander": "legal",
                   "type_line": "Basic Land — Forest"},
    }
    cards.update(extra or {})
    return cards


def _sized(plan, cards):
    """Pad to 100 cards so size doesn't mask the check under test.

    1 commander + N slots + lands == 100.
    """
    lands = 99 - len(plan["slots"])
    plan["land_counts"] = {"Swamp": lands - 1, "Forest": 1}
    plan["role_budget"] = {"lands": lands, "flex": len(plan["slots"])}
    return plan, cards


# ── card counting ──


def test_deck_card_names_counts_land_quantities():
    names = deck_card_names(_plan())
    assert len(names) == 5
    assert names.count("Swamp") == 1


def test_wrong_card_count_is_an_error():
    errors = validate(_plan(), _cards())
    assert any("expected exactly 100" in e for e in errors)


def test_correct_card_count_passes():
    plan, cards = _sized(_plan(), _cards())
    assert not any("expected exactly" in e for e in validate(plan, cards))


# ── required structure ──


def test_missing_required_keys_short_circuits():
    errors = validate({"slug": "x"})
    assert len(errors) == 1
    assert "missing required key" in errors[0]


def test_slot_without_a_name_is_an_error():
    plan, cards = _sized(_plan(slots=[{"role": "flex"}, {"name": "Sol Ring", "role": "flex"}]),
                         _cards())
    assert any("has no name" in e for e in validate(plan, cards))


def test_slot_without_a_role_is_an_error():
    plan, cards = _sized(_plan(slots=[{"name": "Sol Ring"}, {"name": "Deadly Dispute"}]),
                         _cards())
    assert any("has no role" in e for e in validate(plan, cards))


# ── legality ──


def test_unknown_card_is_an_error():
    plan, cards = _sized(_plan(slots=[{"name": "Not A Real Card", "role": "flex"}]), _cards())
    assert any("unknown card" in e for e in validate(plan, cards))


def test_color_identity_violation_is_caught():
    extra = {"Counterspell": {"color_identity": "U", "legal_commander": "legal",
                             "type_line": "Instant"}}
    plan, cards = _sized(_plan(slots=[{"name": "Counterspell", "role": "flex"}]),
                         _cards(extra))
    assert any("color identity violation" in e for e in validate(plan, cards))


def test_illegal_card_is_caught():
    extra = {"Shahrazad": {"color_identity": "", "legal_commander": "banned",
                           "type_line": "Sorcery"}}
    plan, cards = _sized(_plan(slots=[{"name": "Shahrazad", "role": "flex"}]),
                         _cards(extra))
    assert any("not legal in Commander" in e for e in validate(plan, cards))


def test_singleton_violation_is_caught():
    plan, cards = _sized(
        _plan(slots=[{"name": "Sol Ring", "role": "flex"},
                     {"name": "Sol Ring", "role": "flex"}]), _cards())
    assert any("singleton violation" in e for e in validate(plan, cards))


def test_basic_lands_are_exempt_from_singleton():
    plan, cards = _sized(_plan(), _cards())
    assert not any("singleton" in e for e in validate(plan, cards))


def test_none_cards_skips_reality_checks_without_claiming_the_deck_is_empty():
    plan, _ = _sized(_plan(slots=[{"name": "Anything At All", "role": "flex"}]), _cards())
    assert not any("unknown card" in e for e in validate(plan, cards=None))


# ── bracket ──


def test_floor_above_target_is_out_of_tier():
    plan, cards = _sized(_plan(bracket={"target": 2, "computed_floor": 4}), _cards())
    errors = validate(plan, cards)
    assert any("out of tier" in e for e in errors)


def test_floor_at_target_is_fine():
    plan, cards = _sized(_plan(bracket={"target": 3, "computed_floor": 3}), _cards())
    assert not any("out of tier" in e for e in validate(plan, cards))


def test_missing_computed_floor_is_an_error():
    plan, cards = _sized(_plan(bracket={"target": 3}), _cards())
    assert any("computed_floor is missing" in e for e in validate(plan, cards))


def test_invalid_bracket_target():
    plan, cards = _sized(_plan(bracket={"target": 9, "computed_floor": 1}), _cards())
    assert any("must be 1-5" in e for e in validate(plan, cards))


def test_plan_disagreeing_with_the_bracket_artifact_is_caught():
    """A self-reported floor is a claim; bracket_report.json is the evidence."""
    plan, cards = _sized(_plan(bracket={"target": 3, "computed_floor": 3}), _cards())
    errors = validate(plan, cards, bracket_report={"floor": 4})
    assert any("bracket_report.json measured 4" in e for e in errors)


def test_plan_agreeing_with_the_bracket_artifact_passes():
    plan, cards = _sized(_plan(bracket={"target": 3, "computed_floor": 3}), _cards())
    assert not any("bracket_report" in e
                   for e in validate(plan, cards, bracket_report={"floor": 3}))


def test_absent_bracket_artifact_only_skips_the_cross_check():
    plan, cards = _sized(_plan(), _cards())
    assert validate(plan, cards, bracket_report=None) == []


# ── budget arithmetic ──


def test_land_count_mismatch_is_caught():
    plan = _plan(role_budget={"lands": 40, "flex": 2})
    assert any("role_budget says 40" in e for e in validate(plan, _cards()))


def test_nonland_budget_mismatch_is_caught():
    plan = _plan(role_budget={"lands": 2, "flex": 10})
    assert any("declares 10 nonland slots" in e for e in validate(plan, _cards()))


def test_per_role_mismatch_is_caught_even_when_the_total_is_right():
    """The deck-critic found a real build where the total summed but every line
    was wrong — 9 declared ramp against 10 slots, 9 draw against 6, net zero."""
    plan = _plan(
        slots=[{"name": "Sol Ring", "role": "ramp"},
               {"name": "Deadly Dispute", "role": "ramp"}],
        role_budget={"lands": 2, "ramp": 1, "draw": 1},
    )
    errors = validate(plan, _cards())
    assert any("says 1 ramp, plan labels 2" in e for e in errors)
    assert any("says 1 draw, plan labels 0" in e for e in errors)
    # ...and the total check alone would have passed it
    assert not any("nonland slots" in e for e in errors)


def test_undeclared_role_is_caught():
    plan = _plan(
        slots=[{"name": "Sol Ring", "role": "ramp"}, {"name": "Deadly Dispute", "role": "wincon"}],
        role_budget={"lands": 2, "ramp": 1, "flex": 1},
    )
    assert any("labelled 'wincon', which the budget never declares" in e
               for e in validate(plan, _cards()))


# ── derived blocks must describe the deck they claim to ──


def test_lands_array_disagreeing_with_land_counts_is_caught():
    """A swap that edits land_counts but not `lands` leaves two 36-card lists."""
    plan = _plan(lands=["Swamp", "Haven of the Spirit Dragon"])
    assert any("lands and land_counts disagree" in e for e in validate(plan, _cards()))


def test_matching_lands_array_passes():
    plan = _plan(lands=["Forest", "Swamp"])
    assert not any("disagree" in e for e in validate(plan, _cards()))


def test_stale_manabase_diagnostics_are_caught():
    """Diagnostics computed against a spell list the deck no longer runs."""
    plan = _plan(manabase={"spell_slots": 63, "sources": {"B": 29}})
    assert any("re-run the mana base" in e for e in validate(plan, _cards()))


def test_fresh_manabase_diagnostics_pass():
    plan = _plan(manabase={"spell_slots": 2, "sources": {"B": 29}})
    assert not any("re-run the mana base" in e for e in validate(plan, _cards()))


def test_manabase_without_a_stamp_is_not_flagged():
    """Older plans predate the stamp — absence is not staleness."""
    plan = _plan(manabase={"sources": {"B": 29}})
    assert not any("re-run the mana base" in e for e in validate(plan, _cards()))


# ── critic verdict consistency (mirrors the stack contract) ──


def test_critic_pass_with_unsupported_findings_is_rejected():
    plan, cards = _sized(_plan(critic={
        "verdict": "pass",
        "findings": [{"claim": "ratios", "status": "unjustified"}],
    }), _cards())
    assert any("not 'supported'" in e for e in validate(plan, cards))


def test_critic_pass_with_all_supported_is_accepted():
    plan, cards = _sized(_plan(critic={
        "verdict": "pass",
        "findings": [{"claim": "ratios", "status": "supported"}],
    }), _cards())
    assert not any("critic" in e for e in validate(plan, cards))


def test_invalid_critic_status_is_rejected():
    plan, cards = _sized(_plan(critic={
        "verdict": "fail",
        "findings": [{"claim": "x", "status": "made-up-status"}],
    }), _cards())
    assert any("invalid status" in e for e in validate(plan, cards))


def test_invalid_critic_verdict_is_rejected():
    plan, cards = _sized(_plan(critic={"verdict": "maybe", "findings": []}), _cards())
    assert any("must be pass|fail" in e for e in validate(plan, cards))


def test_no_critic_block_is_fine():
    """The deterministic builder makes no claims, so it carries no critic."""
    plan, cards = _sized(_plan(), _cards())
    assert not any("critic" in e for e in validate(plan, cards))


# ── citations go through the existing contract ──


def test_bad_strategy_citation_is_rejected():
    plan, cards = _sized(_plan(role_budget_citations=[
        {"rule": "strategy:nonexistent", "quote": "made up"},
    ]), _cards())
    errors = validate(plan, cards, rules={}, strategy_sections={})
    assert any("nonexistent strategy section" in e for e in errors)


def test_non_verbatim_quote_is_rejected():
    sections = {"strategy:ratios": {"text": "Run about thirty-six lands."}}
    plan, cards = _sized(_plan(role_budget_citations=[
        {"rule": "strategy:ratios", "quote": "run forty lands"},
    ]), _cards())
    errors = validate(plan, cards, rules={}, strategy_sections=sections)
    assert any("not verbatim" in e for e in errors)


def test_verbatim_quote_is_accepted():
    sections = {"strategy:ratios": {"text": "Run about thirty-six lands."}}
    plan, cards = _sized(_plan(role_budget_citations=[
        {"rule": "strategy:ratios", "quote": "about thirty-six lands"},
    ]), _cards())
    errors = validate(plan, cards, rules={}, strategy_sections=sections)
    assert not any("verbatim" in e or "nonexistent" in e for e in errors)


def test_a_clean_plan_validates():
    plan, cards = _sized(_plan(), _cards())
    assert validate(plan, cards) == []
