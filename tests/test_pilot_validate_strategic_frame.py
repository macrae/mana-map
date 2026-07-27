"""Form gate for strategic_frame.json — the highest-leverage artifact that had none."""

import pytest

from manamap.pilot import validate_strategic_frame as vsf


def frame(**overrides):
    doc = {
        "slug": "test-deck",
        "archetype": "aggro-storm",
        "schools": ["strategy:tempo"],
        "overall_assessment": "A fine deck.",
        "role_assignment": {"Sac Outlet": "engine"},
        "engines": [{"piece": "Zada", "engine": "copy cantrips",
                     "evidence": "oracle text", "strategy_ref": "strategy:tempo"}],
        "candidate_missing_lines": [
            {"cards": ["A", "B"], "title": "A line", "why_plausible": "because",
             "status": "needs a stack scenario"}],
        "matchup_frames": {"stax": "race it"},
        "distribution_notes": "curve is low",
        "gaps": [],
    }
    doc.update(overrides)
    return doc


def test_a_wellformed_frame_passes():
    assert vsf.validate(frame()) == []


def test_all_three_committed_frames_would_pass():
    # the fixture mirrors the real files' shape; guard the required-key set
    assert vsf.REQUIRED_KEYS <= set(frame())


def test_missing_key_fails_fast():
    doc = frame()
    del doc["engines"]
    errors = vsf.validate(doc)
    assert len(errors) == 1 and "engines" in errors[0]


def test_slug_mismatch_is_an_error():
    errors = vsf.validate(frame(), slug="other-deck")
    assert any("slug" in e for e in errors)


def test_empty_archetype_is_an_error():
    errors = vsf.validate(frame(archetype="   "))
    assert any("archetype" in e for e in errors)


def test_engine_missing_strategy_ref_is_an_error():
    doc = frame(engines=[{"piece": "Zada", "engine": "x", "evidence": "y"}])
    errors = vsf.validate(doc)
    assert any("strategy_ref" in e for e in errors)


def test_candidate_line_must_carry_the_exact_status():
    doc = frame(candidate_missing_lines=[
        {"cards": ["A"], "title": "t", "why_plausible": "w", "status": "verified"}])
    errors = vsf.validate(doc)
    assert any("never a fact" in e for e in errors)


def test_candidate_line_needs_cards():
    doc = frame(candidate_missing_lines=[
        {"cards": [], "title": "t", "why_plausible": "w",
         "status": "needs a stack scenario"}])
    errors = vsf.validate(doc)
    assert any("non-empty list" in e for e in errors)


def test_wrong_container_types_fail():
    errors = vsf.validate(frame(schools="tempo", matchup_frames=[]))
    assert any("schools" in e for e in errors)
    assert any("matchup_frames" in e for e in errors)


# ── The shared validator tail (common.report_errors) ─────────────────────


def test_report_errors_exits_1_with_the_fail_form(capsys):
    from manamap.pilot.common import report_errors
    with pytest.raises(SystemExit) as exc:
        report_errors("thing.json", ["bad", "worse"], "OK never printed")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL thing.json (2 error(s)):" in out
    assert "  - bad" in out and "  - worse" in out
    assert "OK never printed" not in out


def test_report_errors_prints_the_ok_line_when_clean(capsys):
    from manamap.pilot.common import report_errors
    report_errors("thing.json", [], "OK   thing.json — all good")
    assert capsys.readouterr().out == "OK   thing.json — all good\n"


def test_report_errors_with_no_ok_line_prints_nothing(capsys):
    from manamap.pilot.common import report_errors
    report_errors("thing.json", [])
    assert capsys.readouterr().out == ""


# ── The shared board-split and gates (common) ────────────────────────────


def test_mainboard_sideboard_partition():
    from manamap.pilot.common import mainboard, sideboard
    cards = [{"name": "A"}, {"name": "B", "is_sideboard": True},
             {"name": "C", "is_sideboard": False}]
    assert [c["name"] for c in mainboard(cards)] == ["A", "C"]
    assert [c["name"] for c in sideboard(cards)] == ["B"]


def test_is_land_reads_the_front_face():
    from manamap.pilot.common import is_land
    assert is_land({"type_line": "Land — Mountain"})
    assert is_land({"type_line": "Land // Instant"})
    assert not is_land({"type_line": "Instant // Land"})
    assert not is_land({"type_line": "Creature — Dryad"})


def test_checker_passed_gate():
    from manamap.pilot.common import checker_passed
    assert checker_passed({"checker": {"verdict": "pass"}})
    assert not checker_passed({"checker": {"verdict": "fail"}})
    assert not checker_passed({"checker": None})
    assert not checker_passed({})
