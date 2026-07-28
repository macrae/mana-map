"""Tests for the upgrade-watch (pool scout) contract.

Synthetic fixtures throughout, mirroring test_pilot_validate_sideboard: the
scout's honesty comes from claim-tracing, so the tests monkeypatch the tracked
indexes and check that untraceable claims fail.
"""

import json

import pytest

from manamap.pilot import validate_upgrade_watch as vu


def card(name, **overrides):
    base = {"name": name, "is_commander": False, "is_sideboard": False,
            "type_line": "Creature", "mana_cost": "{1}{R}", "cmc": 2.0}
    base.update(overrides)
    return base


DECK = {"cards": [
    card("Zada, Hedron Grinder", is_commander=True, type_line="Legendary Creature"),
    card("Witch's Mark"),
    card("Chaos Warp"),
    card("Mountain", type_line="Basic Land — Mountain"),
]}


def entry(**overrides):
    base = {
        "card": "Lightning Bolt",
        "role": "removal:damage",
        "evidence": {},
        "why": "Cheapest possible answer in a deck whose curve tops at two.",
    }
    base.update(overrides)
    return base


def report(**overrides):
    doc = {
        "slug": "toy",
        "assessment": "The pool offers cheap interaction this list is short on.",
        "lookout": [entry()],
        "gaps": [],
    }
    doc.update(overrides)
    return doc


def errors(**overrides):
    return vu.validate(report(**overrides), DECK)


def _no_indexes(monkeypatch):
    """Reference indexes absent — cross-checks skip rather than fail."""
    monkeypatch.setattr(vu, "load_json", lambda path, default=None: default)


# ── Structure ────────────────────────────────────────────────────────────


def test_a_clean_report_validates(monkeypatch):
    _no_indexes(monkeypatch)
    assert errors() == []


def test_missing_top_level_keys_short_circuit():
    bad = vu.validate({"slug": "toy"}, DECK)
    assert len(bad) == 1 and "Missing top-level keys" in bad[0]


def test_an_empty_assessment_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    assert any("assessment is empty" in e for e in errors(assessment=""))


def test_an_empty_lookout_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    assert any("non-empty list" in e for e in errors(lookout=[]))


def test_more_than_ten_entries_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    eleven = [entry(card=f"Card {i}") for i in range(11)]
    assert any("top 10" in e for e in errors(lookout=eleven))


def test_duplicate_picks_are_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    assert any("duplicate" in e for e in errors(lookout=[entry(), entry()]))


# ── The pool constraint, inverted: a pick must NOT be in the deck ────────


def test_a_pick_already_in_the_deck_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    bad = errors(lookout=[entry(card="Chaos Warp")])
    assert any("already in the deck" in e for e in bad)


def test_an_empty_why_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    assert any("`why` is empty" in e for e in errors(lookout=[entry(why="  ")]))


# ── Combo-line claims stay candidates ────────────────────────────────────


def test_an_opened_line_with_a_bare_verified_status_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    bad = errors(lookout=[entry(evidence={"combo_lines_opened": [
        {"cards": ["A", "B"], "status": "verified"}]})])
    assert any("stack_artifact" in e for e in bad)


def test_an_unknown_line_status_is_rejected(monkeypatch):
    _no_indexes(monkeypatch)
    bad = errors(lookout=[entry(evidence={"combo_lines_opened": [
        {"cards": ["A", "B"], "status": "probably fine"}]})])
    assert any("needs a stack scenario" in e for e in bad)


def test_a_verified_line_with_a_passing_artifact_passes(monkeypatch, tmp_path):
    _no_indexes(monkeypatch)
    stacks = tmp_path / "stacks"
    stacks.mkdir()
    (stacks / "001-toy.json").write_text(json.dumps({
        "scenario": {"question": "A and B"},
        "checker": {"verdict": "pass"},
    }))
    doc = report(lookout=[entry(evidence={"combo_lines_opened": [
        {"cards": ["A", "B"], "status": "verified",
         "stack_artifact": "stacks/001-toy.json"}]})])
    assert vu.validate(doc, DECK, deck_path=tmp_path) == []


# ── Obsolescence claims are re-checked against the index ─────────────────


def _index(monkeypatch, index=None, graph=None):
    def fake_load(path, default=None):
        if "obsolescence" in str(path):
            return index
        if "synergy" in str(path):
            return graph
        return default
    monkeypatch.setattr(vu, "load_json", fake_load)


def test_a_traceable_obsolescence_claim_passes(monkeypatch):
    _index(monkeypatch, index={
        "Witch's Mark": {"obsoleted_by": [{"name": "Lightning Bolt"}]}})
    assert errors(lookout=[entry(evidence={"obsoletes": ["Witch's Mark"]})]) == []


def test_an_untraceable_obsolescence_claim_is_rejected(monkeypatch):
    _index(monkeypatch, index={"Witch's Mark": {"obsoleted_by": [{"name": "Other"}]}})
    bad = errors(lookout=[entry(evidence={"obsoletes": ["Witch's Mark"]})])
    assert any("obsolescence_index.json lists no such replacement" in e for e in bad)


def test_an_absent_index_skips_the_check(monkeypatch):
    _index(monkeypatch, index=None)
    assert errors(lookout=[entry(evidence={"obsoletes": ["Witch's Mark"]})]) == []


# ── Synergy claims are re-checked against the graph ──────────────────────


def test_a_traceable_synergy_claim_passes(monkeypatch):
    _index(monkeypatch, graph={
        "Lightning Bolt": [{"partner": "Chaos Warp", "score": 3}]})
    good = errors(lookout=[entry(evidence={
        "synergy_partners_in_deck": [{"partner": "Chaos Warp"}]})])
    assert good == []


def test_a_partner_not_on_the_shortlist_is_rejected(monkeypatch):
    _index(monkeypatch, graph={"Lightning Bolt": [{"partner": "Other Card"}]})
    bad = errors(lookout=[entry(evidence={
        "synergy_partners_in_deck": [{"partner": "Chaos Warp"}]})])
    assert any("does not include it" in e for e in bad)


def test_a_partner_not_in_the_deck_is_rejected(monkeypatch):
    _index(monkeypatch, graph={
        "Lightning Bolt": [{"partner": "Uncast Card"}]})
    bad = errors(lookout=[entry(evidence={
        "synergy_partners_in_deck": [{"partner": "Uncast Card"}]})])
    assert any("not in this deck" in e for e in bad)
