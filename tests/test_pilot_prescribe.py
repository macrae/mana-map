"""Prescriptions: one question to the doctor, answered under the diagnosis contract.

What these pin:

- the question is AUTHORED and its id is a hash of it — asking twice finds the same
  file, whitespace and case notwithstanding, and nothing overwrites it;
- the merge writes answer keys only (the prompt and the stamp are never touched) and
  folds the skeptic in when present;
- the cache routine `prescription:<id>` digests only the prompt slice, so merging the
  answer never self-invalidates, and `record` refuses a file without a passing
  skeptic — a prescription reaches a decklist;
- the validator runs the diagnosis contract when the stamp is the deck, and FORM ONLY
  when it is not: prescriptions accumulate, and a gate that reddens history teaches
  its reader to ignore it.
"""

import hashlib
import json

import pytest

from manamap.pilot import agent_cache as ac
from manamap.pilot import deck_status, prescribe, validate_prescription as vp

SLUG = "rxdeck"
DECKLIST = "1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n1 Llanowar Elves\n30 Forest\n"


@pytest.fixture
def deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    base = decks / SLUG
    (base / ".agent-out").mkdir(parents=True)
    (base / "stacks").mkdir()
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "decklist.txt").write_text(DECKLIST)
    sha = hashlib.sha256(DECKLIST.encode()).hexdigest()
    (base / "cards.json").write_text(json.dumps({
        "deck": SLUG, "decklist_sha256": sha,
        "cards": [{"name": "Radagast of Rhosgobel", "is_commander": True},
                  {"name": "Craterhoof Behemoth"}, {"name": "Llanowar Elves"},
                  {"name": "Forest", "quantity": 30}]}))
    (base / "goldfish_metrics.json").write_text(json.dumps({"meta": {"seed": 1}, "metrics": {}}))
    (base / "log.jsonl").write_text(json.dumps(
        {"id": "001", "at": "2026-08-19T09:00:00-07:00", "decklist_sha256": sha,
         "result": "loss", "opponents": 3, "tags": [], "text": "Wrathed on five."}) + "\n")
    ac._SHA_MEMO.clear()
    return base


ANSWER = {
    "reading": "You asked for draw; the audit says draw is fine and the curve is the problem.",
    "log_entries_read": ["001"],
    "cut_candidates": [{"card": "Llanowar Elves", "why": "dies to the wrath you keep eating",
                        "cost_of_cutting": "a turn-two ramp piece", "difficulty": "contested"}],
    "add_candidates": [{"card": "Heroic Intervention", "closes": "the thinnest engine component",
                        "source": "pool", "why": "answers the wrath the log names twice",
                        "natural_cut": "Llanowar Elves"}],
    "open_questions": [],
    "gaps": [],
}


# ── Create ───────────────────────────────────────────────────────────────


def test_the_id_is_a_hash_of_the_question_and_the_file_is_never_overwritten(deck):
    p1, created = prescribe.create(SLUG, "  I keep getting WRATHED  on five ")
    assert created and p1.parent.name == "prescriptions"
    doc = json.loads(p1.read_text())
    assert doc["id"] == prescribe.prescription_id("i keep getting wrathed on five")
    assert doc["prompt"] == "I keep getting WRATHED on five"
    assert doc["as_of_decklist_sha256"] == hashlib.sha256(DECKLIST.encode()).hexdigest()
    p2, created2 = prescribe.create(SLUG, "i keep getting wrathed on five")
    assert p2 == p1 and not created2
    assert json.loads(p1.read_text()) == doc, "an existing question is never rewritten"


def test_create_refuses_an_empty_question(deck):
    with pytest.raises(SystemExit):
        prescribe.create(SLUG, "   ")


# ── Merge ────────────────────────────────────────────────────────────────


def _handoff(deck, pid, answer=ANSWER, skeptic=None):
    (deck / ".agent-out" / f"deck-doctor-prescribe-{pid}.json").write_text(json.dumps(answer))
    if skeptic is not None:
        (deck / ".agent-out" / f"deck-skeptic-prescribe-{pid}.json").write_text(json.dumps(skeptic))


def test_merge_writes_answer_keys_only_and_folds_the_skeptic(deck):
    path, _ = prescribe.create(SLUG, "more draw?")
    pid = json.loads(path.read_text())["id"]
    _handoff(deck, pid, dict(ANSWER, prompt="A DIFFERENT QUESTION", id="deadbeef0000"))
    _, merged = prescribe.merge(SLUG, pid)
    doc = json.loads(path.read_text())
    assert doc["prompt"] == "more draw?" and doc["id"] == pid, "the authored half is never touched"
    assert doc["reading"] == ANSWER["reading"] and "skeptic" not in doc
    _handoff(deck, pid, skeptic={"verdict": "pass", "findings": []})
    _, merged = prescribe.merge(SLUG, pid)
    assert "skeptic" in merged and json.loads(path.read_text())["skeptic"]["verdict"] == "pass"


def test_merge_refuses_a_missing_or_empty_handoff(deck):
    path, _ = prescribe.create(SLUG, "q")
    pid = json.loads(path.read_text())["id"]
    with pytest.raises(SystemExit):
        prescribe.merge(SLUG, pid)
    _handoff(deck, pid, {"unrelated": 1})
    with pytest.raises(SystemExit):
        prescribe.merge(SLUG, pid)
    with pytest.raises(SystemExit):
        prescribe.merge(SLUG, "000000000000")


# ── The cache routine ────────────────────────────────────────────────────


def test_the_routine_digests_only_the_prompt_and_survives_the_merge(deck):
    path, _ = prescribe.create(SLUG, "faster?")
    pid = json.loads(path.read_text())["id"]
    routine = f"prescription:{pid}"
    assert routine in ac.discover_routines(SLUG)
    spec = ac.routine_spec(SLUG, routine)
    assert spec["artifact_subdir"] == "prescriptions" and "prompt:self" in spec["inputs"]
    entries, extra = ac.resolve_inputs(SLUG, spec)
    before = ac.fingerprint(routine, spec, entries, extra)
    _handoff(deck, pid, skeptic={"verdict": "pass", "findings": []})
    prescribe.merge(SLUG, pid)
    ac._SHA_MEMO.clear()
    entries, extra = ac.resolve_inputs(SLUG, spec)
    assert ac.fingerprint(routine, spec, entries, extra) == before, (
        "merging the answer must not invalidate the question — scenario:self's rule")
    (deck / "decklist.txt").write_text(DECKLIST + "1 Sol Ring\n")
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"].append({"name": "Sol Ring"}); cards["decklist_sha256"] = "moved"
    (deck / "cards.json").write_text(json.dumps(cards))
    ac._SHA_MEMO.clear()
    entries, extra = ac.resolve_inputs(SLUG, spec)
    assert ac.fingerprint(routine, spec, entries, extra) != before, "a deck change is a MISS"


def test_record_refuses_a_prescription_without_a_passing_skeptic(deck):
    path, _ = prescribe.create(SLUG, "q")
    pid = json.loads(path.read_text())["id"]
    _handoff(deck, pid)
    prescribe.merge(SLUG, pid)
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, f"prescription:{pid}")
    _handoff(deck, pid, skeptic={"verdict": "fail", "findings": []})
    prescribe.merge(SLUG, pid)
    with pytest.raises(ac.MissingInput):
        ac.record(SLUG, f"prescription:{pid}")
    _handoff(deck, pid, skeptic={"verdict": "pass", "findings": []})
    prescribe.merge(SLUG, pid)
    entry, _ = ac.record(SLUG, f"prescription:{pid}")
    assert entry["verdict"] == "pass"


# ── The validator ────────────────────────────────────────────────────────


def _doc(deck, prompt="q", answer=ANSWER, **over):
    path, _ = prescribe.create(SLUG, prompt)
    doc = json.loads(path.read_text())
    if answer is not None:
        doc.update(answer)
    doc.update(over)
    return doc


def _validate(deck, doc):
    cards = json.loads((deck / "cards.json").read_text())
    return vp.validate(doc, cards, deck_path=deck, measured_axes=None, rules={},
                       strategy_sections=None, log_ids={"001"})


def test_an_open_question_is_a_valid_file_and_a_sound_answer_passes(deck):
    assert _validate(deck, _doc(deck, answer=None)) == []
    assert _validate(deck, _doc(deck)) == []


def test_an_edited_prompt_under_an_old_id_fails(deck):
    doc = _doc(deck, prompt="original question")
    doc["prompt"] = "a different question"
    assert any("not the hash of the prompt" in e for e in _validate(deck, doc))


def test_the_add_list_is_capped_at_ten_and_the_log_ids_must_exist(deck):
    adds = [dict(ANSWER["add_candidates"][0], card=f"Card {i}", natural_cut=None)
            for i in range(11)]
    errs = _validate(deck, _doc(deck, add_candidates=adds))
    assert any("the cap is 10" in e for e in errs)
    errs = _validate(deck, _doc(deck, log_entries_read=["001", "042"]))
    assert any("no log entry '042'" in e for e in errs)


def test_a_current_prescription_runs_the_diagnosis_contract(deck):
    """The diagnosis validator's checks, reused: an add already in the deck, a cut
    of a card the deck does not run, an unpriced cut."""
    doc = _doc(deck, add_candidates=[dict(ANSWER["add_candidates"][0], card="Craterhoof Behemoth")])
    assert any("already in the maindeck" in e for e in _validate(deck, doc))
    doc = _doc(deck, cut_candidates=[dict(ANSWER["cut_candidates"][0], card="Sol Ring")])
    assert any("not in the maindeck" in e for e in _validate(deck, doc))
    doc = _doc(deck, cut_candidates=[dict(ANSWER["cut_candidates"][0], cost_of_cutting="")])
    assert any("cost_of_cutting" in e for e in _validate(deck, doc))
    doc = _doc(deck, reading="")
    assert any("reading is empty" in e for e in _validate(deck, doc))


def test_a_stale_prescription_is_form_checked_only(deck):
    """Written against an older list whose cut has since been APPLIED: the cut is no
    longer in the deck and the add now is. Neither is an error — it is history."""
    doc = _doc(deck)
    doc["as_of_decklist_sha256"] = "an-older-decklist"
    cards = json.loads((deck / "cards.json").read_text())
    cards["cards"] = [c for c in cards["cards"] if c["name"] != "Llanowar Elves"]
    cards["cards"].append({"name": "Heroic Intervention"})
    (deck / "cards.json").write_text(json.dumps(cards))
    assert _validate(deck, doc) == []
    # ...but form still holds: a closed set is a closed set
    bad = dict(doc, add_candidates=[dict(ANSWER["add_candidates"][0], source="vibes")])
    assert any("source 'vibes'" in e for e in _validate(deck, bad))


def test_the_short_list_stages_are_retired_from_deck_status():
    stages = {s[0] for s in deck_status.STAGES}
    assert "shortlist" not in stages and "shortlist-art" not in stages
    assert "considering.json" in deck_status.VALIDATED, (
        "the frozen legacy files are still gated, just not a lifecycle stage")
