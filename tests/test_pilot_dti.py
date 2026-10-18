"""`dti.py`: DeckCheck's Threat Index as dated external evidence.

Readings are never edited or deleted; `omni` and `floor_vs_dti` are derived and
recomputed; nothing here reads the clock or the network. The floor gates, the
DTI never does.
"""

import copy
import json
from datetime import date

import pytest

from manamap import config
from manamap.pilot import deckcheck, dti

BM = {"R1": "A", "R2": "B", "A1": "S", "A2": "A", "P1": "B", "P2": "C", "P3": "B",
      "I1": "A", "I2": "B", "S1": "C", "S2": "B", "S3": "F"}
PAYLOAD = {"dti": {"score": 61, "maxScore": 96, "bracket": 4, "benchmarks": dict(BM)},
           "threatIndex": 61, "bracket": 4, "fullAnalysis": "<p>prose</p>",
           "updatedAtUtc": "2026-10-17T10:05:00Z", "name": "x"}
SHA = "a" * 64
URL = "https://deckcheck.co/app/decklist/AbC_12-xyz"


def _reading(as_of="2026-10-17", sha=SHA, score=61, bracket=4, updated="2026-10-17T10:05:00Z",
             **grades):
    r = dti.reading_from_payload(
        {"dti": {"score": score, "maxScore": 96, "bracket": bracket,
                 "benchmarks": {**BM, **grades}}, "updatedAtUtc": updated},
        deck_id="AbC_12-xyz", url=URL, as_of=as_of, decklist_sha256=sha)
    return r


@pytest.fixture
def deck(tmp_path, monkeypatch):
    root = tmp_path / "decks"
    d = root / "ddeck"
    d.mkdir(parents=True)
    monkeypatch.setattr(config, "DECKS_DIR", root)
    (d / "cards.json").write_text(json.dumps({"decklist_sha256": SHA, "cards": []}))
    (d / "bracket_report.json").write_text(json.dumps(
        {"floor": 4, "floor_name": "Optimized", "decklist_sha256": SHA, "drivers": []}))
    return d


# ── one reading ────────────────────────────────────────────────────────────

def test_a_reading_keeps_the_twelve_and_the_numbers_and_drops_the_html():
    r = dti.reading_from_payload(PAYLOAD, deck_id="AbC_12-xyz", url=URL,
                                 as_of="2026-10-17", decklist_sha256=SHA)
    assert set(r) == set(dti.READING_KEYS)
    assert r["score"] == 61 and r["max_score"] == 96 and r["bracket"] == 4
    assert r["benchmarks"] == BM and "fullAnalysis" not in json.dumps(r)
    assert r["source_updated_at"] == "2026-10-17T10:05:00Z"


@pytest.mark.parametrize("mutate,phrase", [
    (lambda p: p.pop("dti"), "carries no `dti`"),
    (lambda p: p["dti"]["benchmarks"].pop("S3"), "missing \\['S3'\\]"),
    (lambda p: p["dti"]["benchmarks"].update(X9="A"), "extra \\['X9'\\]"),
    (lambda p: p["dti"]["benchmarks"].update(R1="D"), "outside"),
    (lambda p: p["dti"].pop("score"), "unreadable"),
])
def test_a_document_off_the_documented_shape_is_refused(mutate, phrase):
    p = copy.deepcopy(PAYLOAD)
    mutate(p)
    with pytest.raises(ValueError, match=phrase):
        dti.reading_from_payload(p, deck_id="x", url=URL, as_of="2026-10-17", decklist_sha256=SHA)


# ── omni ───────────────────────────────────────────────────────────────────

def test_omni_is_the_lower_median_over_readings_of_one_list():
    rs = [_reading(score=60, R1="A"), _reading(as_of="2026-10-18", score=64, R1="S")]
    o = dti.omni(rs, SHA)
    assert o["n"] == 2 and o["median_score"] == 60 and o["benchmarks"]["R1"] == "A"
    rs.append(_reading(as_of="2026-10-19", score=62, R1="S"))
    o = dti.omni(rs, SHA)
    assert o["n"] == 3 and o["median_score"] == 62 and o["benchmarks"]["R1"] == "S"
    assert "not their figure" in o["rule"]


def test_omni_counts_only_the_asked_list_and_zero_is_an_absence_not_a_number():
    rs = [_reading(sha="b" * 64, score=90), _reading(as_of="2026-10-18", score=50)]
    o = dti.omni(rs, SHA)
    assert o["n"] == 1 and o["median_score"] == 50
    o = dti.omni(rs, "c" * 64)
    assert o["n"] == 0 and "absent_because" in o and "median_score" not in o


# ── the floor against theirs ───────────────────────────────────────────────

def test_compare_floor_names_the_relation_and_never_lowers_the_floor():
    rep = {"floor": 4, "floor_name": "Optimized", "decklist_sha256": SHA}
    assert dti.compare_floor(_reading(bracket=4), rep)["relation"] == "equal"
    assert dti.compare_floor(_reading(bracket=5), rep)["relation"] == "dti_higher"
    low = dti.compare_floor(_reading(bracket=3), rep)
    assert low["relation"] == "dti_lower" and low["floor"] == 4 and "older list" in low["note"]
    assert "absent_because" in dti.compare_floor(_reading(), None)


# ── append, dedupe, reconcile ──────────────────────────────────────────────

def test_append_dedupes_against_the_last_reading_only_and_always_dates_the_pull(deck):
    got = dti.append("ddeck", _reading(), today=date(2026, 10, 17))
    assert got["added"] and got["readings_n"] == 1
    got = dti.append("ddeck", _reading(as_of="2026-10-18"), today=date(2026, 10, 18))
    assert not got["added"] and got["deduped_against"] == "2026-10-17"
    doc = dti.load("ddeck")
    assert len(doc["readings"]) == 1 and doc["last_pulled"] == "2026-10-18"
    # a changed score is new; a return to the first score after it is new too
    assert dti.append("ddeck", _reading(as_of="2026-10-19", score=64), today=date(2026, 10, 19))["added"]
    assert dti.append("ddeck", _reading(as_of="2026-10-20", score=61), today=date(2026, 10, 20))["added"]
    # their own edit time changing is a new reading even at the same score
    assert dti.append("ddeck", _reading(as_of="2026-10-21", updated="2026-10-21T00:00:00Z"),
                      today=date(2026, 10, 21))["added"]
    doc = dti.load("ddeck")
    assert doc["omni"]["n"] == 4 and doc["floor_vs_dti"]["relation"] == "equal"
    assert doc["schema_version"] == 1 and doc["source"] == "deckcheck.co"


def test_reconcile_after_the_list_moves_leaves_readings_alone(deck):
    dti.append("ddeck", _reading(), today=date(2026, 10, 17))
    before = dti.load("ddeck")["readings"]
    (deck / "cards.json").write_text(json.dumps({"decklist_sha256": "b" * 64, "cards": []}))
    got = dti.reconcile("ddeck")
    assert got["changed"] and got["readings_on_current_list"] == 0
    doc = dti.load("ddeck")
    assert doc["readings"] == before and doc["omni"]["n"] == 0
    assert dti.reconcile("ddeck")["changed"] is False
    assert dti.reconcile("nodeck") is None


# ── reading it back ────────────────────────────────────────────────────────

def test_info_block_and_next_lines_never_need_a_clock(deck):
    dti.append("ddeck", _reading(), today=date(2026, 10, 17))
    block = dti.info_block(dti.load("ddeck"), SHA)
    assert block["score"] == 61 and block["stale_list"] is False and block["readings_n"] == 1
    assert block["max_age_days"] == config.DECKCHECK_MAX_AGE_DAYS
    assert dti.next_lines(block, "ddeck") == []
    assert dti.next_lines(block, "ddeck", today=date(2026, 10, 20)) == []
    (age,) = dti.next_lines(block, "ddeck", today=date(2026, 12, 1))
    assert age.startswith("DTI is 45 days old") and "deckcheck ddeck --write" in age
    stale = dti.info_block(dti.load("ddeck"), "b" * 64)
    (line,) = dti.next_lines(stale, "ddeck", today=date(2026, 12, 1))
    assert "describes an older list" in line, "the sha line, and no second line for age"
    assert dti.info_block(None, SHA) is None


def test_near_names_the_decks_without_a_file_and_never_scores_them(deck):
    root = deck.parent
    for slug, score in (("near1", 58), ("far1", 20)):
        (root / slug).mkdir()
        (root / slug / "cards.json").write_text(json.dumps({"decklist_sha256": SHA}))
        dti.append(slug, _reading(score=score), today=date(2026, 10, 17))
    (root / "nofile").mkdir()
    dti.append("ddeck", _reading(), today=date(2026, 10, 17))
    doc = dti.near("ddeck", 5)
    assert [r["slug"] for r in doc["rows"]] == ["near1"] and doc["rows"][0]["delta"] == -3
    assert doc["without_dti"] == ["nofile"]
    assert "near1" in dti.format_near(doc) and "nofile" in dti.format_near(doc)
    assert dti.near("nofile", 5)["anchor_score"] is None


# ── the writer on the CLI ──────────────────────────────────────────────────

def test_write_appends_and_the_file_only_flags_never_fetch(deck, monkeypatch, capsys):
    (deck / "decklist.txt").write_text("1 Forest\n")
    (deck / "links.json").write_text(json.dumps(
        {"deckcheck": {"url": URL, "id": "AbC_12-xyz", "as_of": "2026-10-17"}}))
    doc = {**PAYLOAD, "publicUrl": URL, "boards": {"mainboard": {"cards": [
        {"quantity": 1, "card": {"name": "Forest"}}]}}}
    monkeypatch.setattr(deckcheck, "fetch", lambda **kw: doc)
    monkeypatch.setattr(deckcheck, "today_iso", lambda: "2026-10-17")

    args = type("A", (), {"slug": "ddeck", "url": None, "text": False, "as_json": False,
                          "write": True, "near": None, "history": False})()
    deckcheck.main(args)
    out = capsys.readouterr().out
    assert "DTI 61/96 → B4" in out and "reading 1 added" in out
    assert dti.load("ddeck")["readings"][0]["decklist_sha256"] == SHA
    deckcheck.main(args)
    assert "nothing added" in capsys.readouterr().out

    monkeypatch.setattr(deckcheck, "fetch", lambda **kw: (_ for _ in ()).throw(AssertionError("fetched")))
    deckcheck.main(type("A", (), {"slug": "ddeck", "near": 5, "history": False, "as_json": False})())
    assert "DECKS WITHIN" in capsys.readouterr().out
    deckcheck.main(type("A", (), {"slug": "ddeck", "near": None, "history": True, "as_json": False})())
    assert "DTI READINGS" in capsys.readouterr().out
