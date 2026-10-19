"""`validate-log`: the captain's log gated on FORM, now that the survey is a
field a figure is computed from. Every negative re-introduces one check."""

import json

import pytest

from manamap.pilot import validate_log

GOOD = [
    {"id": "001", "at": "2026-08-22T21:00:00", "decklist_sha256": "a" * 64, "result": "loss",
     "opponents": 2, "tags": ["pod3"], "text": "Lost it."},
    {"id": "002", "at": "2026-08-25T22:00", "decklist_sha256": None, "result": None,
     "opponents": None, "tags": [], "text": "Just notes.", "feel": {"power": 4, "fun": 5}},
]


def _lines(entries):
    return [json.dumps(e) for e in entries]


def _errors(mutate):
    doc = json.loads(json.dumps(GOOD))
    mutate(doc)
    errors, _ = validate_log.validate_lines(_lines(doc))
    return errors


def test_a_well_formed_log_passes():
    errors, entries = validate_log.validate_lines(_lines(GOOD))
    assert errors == [] and len(entries) == 2


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d[1].__setitem__("id", "001"), "ids increase"),
    (lambda d: d[0].__setitem__("id", "1"), "zero-padded"),
    (lambda d: d[0].pop("text"), "missing 'text'"),
    (lambda d: d[0].__setitem__("text", "  "), "non-empty"),
    (lambda d: d[0].__setitem__("result", "won"), "result must be"),
    (lambda d: d[0].__setitem__("opponents", 9), "opponents must be 1-7"),
    (lambda d: d[0].__setitem__("opponents", True), "opponents must be 1-7"),
    (lambda d: d[0].__setitem__("tags", "pod3"), "tags must be a list"),
    (lambda d: d[0].__setitem__("decklist_sha256", "abc"), "64-hex"),
    (lambda d: d[1].__setitem__("feel", {"mood": 4}), "feel key 'mood'"),
    (lambda d: d[1].__setitem__("feel", {"power": 6}), "whole number 1-5"),
    (lambda d: d[1].__setitem__("feel", {"power": 4.0}), "whole number 1-5"),
    (lambda d: d[1].__setitem__("feel", {}), "non-empty object"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_validator_negatives(mutate, needle):
    errors = _errors(mutate)
    assert errors and any(needle in e for e in errors), errors


def test_a_non_json_line_and_a_blank_line_are_named_by_number():
    errors, entries = validate_log.validate_lines(_lines(GOOD[:1]) + ["", "not json"])
    assert any(e.startswith("line 2: blank") for e in errors)
    assert any(e.startswith("line 3: not JSON") for e in errors)
    assert len(entries) == 1


def test_main_absent_is_absent_and_a_good_log_prints_the_survey(tmp_path, monkeypatch, capsys):
    decks = tmp_path / "decks"
    (decks / "d").mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    validate_log.main(type("A", (), {"slug": "d"})())
    assert "absent means absent" in capsys.readouterr().out
    (decks / "d" / "log.jsonl").write_text("\n".join(_lines(GOOD)) + "\n")
    validate_log.main(type("A", (), {"slug": "d"})())
    out = capsys.readouterr().out
    assert out.startswith("OK") and "2 entries, 1 with a survey" in out and "power 4.0·1" in out
    (decks / "d" / "log.jsonl").write_text("\n".join(_lines(GOOD)) + "\n{\"id\": \"001\"}\n")
    with pytest.raises(SystemExit):
        validate_log.main(type("A", (), {"slug": "d"})())


def test_the_registry_gates_the_log():
    """The `log` stage row existed before the gate; the VALIDATED row is what
    makes `deck-status` and the tracked-artifact test run the form check."""
    from manamap.pilot import deck_status
    assert deck_status.VALIDATED["log.jsonl"] == "manamap.pilot.validate_log"
