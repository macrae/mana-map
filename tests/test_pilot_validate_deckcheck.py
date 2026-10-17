"""`validate-deckcheck`: form only; time is a WARN and never a failure."""

import copy
import json
import re

import pytest

from manamap import config
from manamap.pilot import dti, validate_deckcheck

BM = {"R1": "A", "R2": "B", "A1": "S", "A2": "A", "P1": "B", "P2": "C", "P3": "B",
      "I1": "A", "I2": "B", "S1": "C", "S2": "B", "S3": "F"}
SHA = "a" * 64
READING = {"as_of": "2026-10-17", "source_updated_at": "2026-10-17T10:05:00Z",
           "decklist_sha256": SHA, "deckcheck_id": "AbC_12-xyz",
           "url": "https://deckcheck.co/deck/AbC_12-xyz", "score": 61, "max_score": 96,
           "bracket": 4, "benchmarks": dict(BM)}


def _good():
    rs = [copy.deepcopy(READING)]          # deep: a mutating case must not leak
    return {"schema_version": 1, "slug": "ddeck", "branch": None, "source": "deckcheck.co",
            "last_pulled": "2026-10-17", "readings": rs, "omni": dti.omni(rs, SHA),
            "floor_vs_dti": {"floor": 4, "floor_name": "Optimized",
                             "floor_decklist_sha256": SHA, "their_bracket": 4,
                             "relation": "equal", "reading_as_of": "2026-10-17"}}


def _errors(mutate):
    doc = _good()
    mutate(doc)
    errors, _ = validate_deckcheck.validate("ddeck", None, doc)
    return errors


def test_a_well_formed_document_passes_with_no_warnings():
    assert validate_deckcheck.validate("ddeck", None, _good(), measured=SHA,
                                       bracket_report={"floor": 4}) == ([], [])


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.pop("last_pulled"), "lacks 'last_pulled'"),
    (lambda d: d.update(schema_version=2), "schema_version 2 is not 1"),
    (lambda d: d.update(slug="other"), "slug 'other'"),
    (lambda d: d.update(branch="b"), "branch 'b'"),
    (lambda d: d.update(source="moxfield"), "source 'moxfield'"),
    (lambda d: d.update(readings=[]), "non-empty list"),
    (lambda d: d["readings"][0].update(benchmarks={**BM, "R1": "D"}), "outside"),
    (lambda d: d["readings"][0]["benchmarks"].pop("S1"), "exactly the twelve"),
    (lambda d: d["readings"][0]["benchmarks"].update(Z9="A"), "exactly the twelve"),
    (lambda d: d["readings"][0].update(score=97), r"not an int in \[0, 96\]"),
    (lambda d: d["readings"][0].update(score=-1), r"not an int in \[0, 96\]"),
    (lambda d: d["readings"][0].update(score=True), r"not an int in \[0, 96\]"),
    (lambda d: d["readings"][0].update(bracket=6), "not one of"),
    (lambda d: d["readings"][0].update(max_score=95), "max_score 95 is not 96"),
    (lambda d: d["readings"][0].update(url="https://evil.com/deck/AbC"), "is not https on"),
    (lambda d: d["readings"][0].update(url="https://deckcheck.co/deck/AbC?x=1"), "query string"),
    (lambda d: d["readings"][0].update(as_of="17 Oct 2026"), "not an ISO date"),
    (lambda d: d["readings"][0].update(source_updated_at="yesterday"), "not an ISO timestamp"),
    (lambda d: d["readings"][0].update(decklist_sha256="abc"), "not a sha256"),
    (lambda d: d["readings"][0].update(deckcheck_id="a b"), "does not match"),
    (lambda d: d["readings"][0].update(extra=1), "unknown key 'extra'"),
    (lambda d: d["readings"][0].update(fullAnalysis="<p>"), "never stored"),
    (lambda d: d.update(fullAnalysis="<p>"), "never stored"),
    (lambda d: d["readings"].insert(0, {**READING, "as_of": "2026-10-18", "score": 50}),
     "dated before"),
    (lambda d: d["readings"].append(dict(READING, as_of="2026-10-18")), "repeats"),
    (lambda d: d["omni"].update(median_score=99), "does not re-add"),
    (lambda d: d["floor_vs_dti"].update(relation="same"), "relation 'same'"),
])
def test_validator_negatives(mutate, needle):
    errors = _errors(mutate)
    assert any(re.search(needle, e) for e in errors), errors


def test_time_is_a_warning_never_an_error():
    doc = _good()
    errors, warnings = validate_deckcheck.validate("ddeck", None, doc, measured="b" * 64,
                                                   bracket_report={"floor": 3})
    assert errors == []
    assert any("older list" in w for w in warnings)
    assert any("not the current floor 3" in w for w in warnings)
    doc["floor_vs_dti"].update(relation="dti_lower", their_bracket=3)
    _, warnings = validate_deckcheck.validate("ddeck", None, doc)
    assert any("below our floor" in w for w in warnings)


def test_main_absent_is_absent_and_warns_without_failing(tmp_path, monkeypatch, capsys):
    root = tmp_path / "decks"
    d = root / "ddeck"
    d.mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", root)
    validate_deckcheck.main(type("A", (), {"slug": "ddeck", "branch": None})())
    assert "absent means absent" in capsys.readouterr().out
    (d / "cards.json").write_text(json.dumps({"decklist_sha256": "b" * 64}))
    (d / "deckcheck.json").write_text(json.dumps(_good()))
    validate_deckcheck.main(type("A", (), {"slug": "ddeck", "branch": None})())
    out = capsys.readouterr().out
    assert out.startswith("WARN") and "OK   ddeck" in out


def test_the_registry_gates_the_artifact_as_form_only_and_never_as_a_stage():
    from manamap.pilot import deck_status, regen

    assert deck_status.VALIDATED["deckcheck.json"] == "manamap.pilot.validate_deckcheck"
    assert "deckcheck.json" not in {row[1] for row in deck_status.STAGES}
    assert "deckcheck.json" not in {s[1] for s in regen.STAGES}
