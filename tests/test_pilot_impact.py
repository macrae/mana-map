"""Tests for the impact report — the mechanical version of the regeneration
sweep judgment. Report-only by contract: nothing here may edit an artifact."""

import json

import pytest

from manamap.pilot import agent_cache as ac
from manamap.pilot import common
from manamap.pilot import impact

SLUG = "toy-impact"


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    base = decks / SLUG
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    write_json(base / "cards.json", {"deck": SLUG, "decklist_sha256": "abc", "cards": [
        {"name": "Loop Piece", "oracle_text": "Untap everything."},
        {"name": "Plain Land", "oracle_text": "T: Add W.", "is_sideboard": False},
        {"name": "Bench Card", "oracle_text": "Flash.", "is_sideboard": True},
    ]})
    write_json(base / "stacks" / "001-loop.json", {
        "id": "001", "title": "The Loop Piece line",
        "scenario": {"question": "Does Loop Piece go infinite?", "stack": [{"pos": 0, "object": "X"}]},
        "resolution": {"steps": [], "final_state": {}},
        "checker": {"verdict": "pass", "iterations": 1},
    })
    write_json(base / "manual_prose.json", {
        "how_it_wins": "Loop Piece untaps everything and the table dies.",
        "mulligan": "Keep lands.",
    })
    write_json(base / "goldfish_targets.json", {"targets": [
        {"label": "Engine drawn", "need": [{"any_of": ["Loop Piece", "Bench Card"]}]},
    ]})
    ac._SHA_MEMO.clear()
    common.clear_memo()
    return base


def _seed_baseline(base):
    cache = ac.load_cache(SLUG)
    cache["cards_map"] = {
        "digest": ac.cards_semantic_digest(base / "cards.json"),
        "cards": ac.cards_semantic_card_map(base / "cards.json"),
    }
    ac.save_cache(SLUG, cache)


# ── deck diff ────────────────────────────────────────────────────────────


def test_no_baseline_is_reported_not_guessed(deck):
    report = impact.analyze(SLUG)
    assert report["deck_diff"]["available"] is False
    assert "cache-rebless" in report["deck_diff"]["reason"]


def test_changed_and_zone_moved_cards_are_named(deck):
    _seed_baseline(deck)
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][0]["oracle_text"] = "Untap everything twice."
    cards["cards"][2]["is_sideboard"] = False  # bench -> maindeck
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    diff = impact.analyze(SLUG)["deck_diff"]
    assert "Loop Piece" in diff["changed"]
    assert "Bench Card" in diff["zone_moved"]


# ── reference impact ─────────────────────────────────────────────────────


def test_reference_impact_names_artifact_and_key(deck):
    _seed_baseline(deck)
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][0]["oracle_text"] = "Changed."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    hits = impact.analyze(SLUG)["reference_impact"]
    artifacts = {(h["artifact"], h.get("key")) for h in hits}
    assert any("001-loop.json" in a for a, _ in artifacts)
    assert any(k == "how_it_wins" for _, k in artifacts)
    assert not any(k == "mulligan" for _, k in artifacts)


def test_unreferenced_change_has_no_impact(deck):
    _seed_baseline(deck)
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"][1]["oracle_text"] = "T: Add one white mana."
    write_json(deck / "cards.json", cards)
    ac._SHA_MEMO.clear(); common.clear_memo()
    report = impact.analyze(SLUG)
    assert report["deck_diff"]["changed"] == ["Plain Land"]
    assert report["reference_impact"] == []


# ── target audit (the benched-Goreclaw trap) ─────────────────────────────


def test_target_audit_flags_non_maindeck_members(deck):
    problems = impact.target_audit(deck)
    assert problems == [{"target": "Engine drawn", "not_in_maindeck": ["Bench Card"]}]


# ── zone framing ─────────────────────────────────────────────────────────


def test_zone_framing_flags_referencing_stacks(deck):
    flags = impact.zone_framing(deck, ["Loop Piece"])
    assert flags and "001-loop.json" in flags[0]["artifact"]
    assert impact.zone_framing(deck, ["Plain Land"]) == []


# ── figure audit ─────────────────────────────────────────────────────────


def test_figure_audit_flags_stale_literal(deck, monkeypatch):
    write_json(deck / "goldfish_metrics.json", {
        "meta": {"seed": 42},
        "metrics": {"commander": {"mean_cast_turn": 7.982}},
    })
    write_json(deck / "manual_prose.json", {
        "how_it_wins": "the mean cast is 7.932 which ages poorly",
        "mulligan": "mean cast 7.982 stays current",
    })
    monkeypatch.setattr(impact, "_previous_goldfish", lambda base: {
        "meta": {"seed": 42},
        "metrics": {"commander": {"mean_cast_turn": 7.932}},
    })
    common.clear_memo()
    findings = impact.figure_audit(deck)
    keys = {f["key"] for f in findings}
    assert "how_it_wins" in keys
    assert "mulligan" not in keys


def test_figure_audit_silent_when_goldfish_unchanged(deck, monkeypatch):
    write_json(deck / "goldfish_metrics.json", {"meta": {}, "metrics": {"x": 1.5}})
    monkeypatch.setattr(impact, "_previous_goldfish",
                        lambda base: {"meta": {}, "metrics": {"x": 1.5}})
    common.clear_memo()
    assert impact.figure_audit(deck) == []


def test_canonical_figures_rounding_variants(deck):
    write_json(deck / "goldfish_metrics.json", {
        "meta": {}, "metrics": {"rate": 0.721, "mean": 7.982},
    })
    common.clear_memo()
    figures = impact.canonical_figures(deck)
    assert 0.721 in figures        # raw
    assert 72.1 in figures         # percent, 1dp
    assert 72 in figures           # percent, 0dp
    assert 7.982 in figures
    assert 8 in figures            # "turn 8"
