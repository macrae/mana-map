"""The captain's log and its debrief: the log is authored, the annotation is derived,
and the validator holds the second to the first.

Three things these tests pin, each the reason the feature is shaped as it is:

- `deck-notes add` APPENDS and stamps the decklist as it stood — a note written
  before a swap is distinguishable from one written after without a date.
- `merge-debrief` writes by id, rejects ids the log lacks, carries earlier
  annotations, and never touches `log.jsonl`.
- `validate-debrief` fails an annotation that names anything the note and the 99
  do not: an opponent reading without a verbatim phrase, a card from nowhere, a
  line asserted without a passing stack, a route to a loop that does not exist.
"""

import json

import pytest

from manamap.pilot import deck_notes, deck_status, merge_debrief, validate_debrief
from manamap.pilot.agent_cache import _debrief_applicable

SLUG = "logdeck"


def _args(**kw):
    base = {"slug": SLUG, "action": "list", "text": None, "file": None, "result": None,
            "opponents": None, "tag": [], "at": None, "since": None, "as_json": False}
    base.update(kw)
    return type("Args", (), base)()


@pytest.fixture
def deck(tmp_path, monkeypatch):
    decks = tmp_path / "decks"
    base = decks / SLUG
    (base / ".agent-out").mkdir(parents=True)
    (base / "stacks").mkdir()
    monkeypatch.setattr("manamap.pilot.common.DECKS_DIR", decks)
    (base / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n1 Forest\n")
    (base / "cards.json").write_text(json.dumps({
        "deck": SLUG, "decklist_sha256": "stale-stamp",
        "cards": [{"name": "Radagast of Rhosgobel", "is_commander": True},
                  {"name": "Craterhoof Behemoth"},
                  {"name": "Fable of the Mirror-Breaker // Reflection of Kiki-Jiki"},
                  {"name": "Forest", "quantity": 30}]}))
    (base / "engine.json").write_text(json.dumps({
        "stages": [{"stage": "ignition"}, {"stage": "wincon"}]}))
    (base / "stacks" / "003-hoof.json").write_text(json.dumps({
        "id": "003", "title": "hoof", "scenario": {"board": {}},
        "resolution": {"steps": [{"action": "Craterhoof Behemoth resolves"}]},
        "checker": {"verdict": "pass"}}))
    return base


def _add(text, **kw):
    return deck_notes.append_entry(SLUG, text, **kw)


# ── The log ──────────────────────────────────────────────────────────────


def test_add_appends_and_numbers_sequentially(deck):
    a = _add("Game one. Lost to a wrath on five.")
    b = _add("Game two. Won on six.", result="win", opponents=3, tags=["orinda", "weekly"])
    assert (a["id"], b["id"]) == ("001", "002")
    lines = (deck / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1]) == b
    assert b["tags"] == ["orinda", "weekly"] and b["result"] == "win" and b["opponents"] == 3


def test_the_stamp_is_the_decklist_as_it_stood(deck):
    """decklist.txt, not cards.json's stamp: the note is about the deck the pilot
    was HOLDING, and the text file is what moves when a card is swapped."""
    before = _add("before the swap")["decklist_sha256"]
    assert before != "stale-stamp"
    (deck / "decklist.txt").write_text("1 Radagast of Rhosgobel *CMDR*\n1 Forest\n")
    after = _add("after the swap")["decklist_sha256"]
    assert before != after


def test_add_refuses_an_empty_note_and_a_bad_result(deck):
    with pytest.raises(SystemExit):
        _add("   ")
    with pytest.raises(SystemExit):
        _add("fine", result="won")
    assert not (deck / "log.jsonl").exists(), "a refusal must not have written"


def test_list_and_show_report_debrief_coverage(deck, capsys):
    _add("one"); _add("two")
    (deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": {"summary": "s", "takeaways": []}}}))
    deck_notes.main(_args(action="list"))
    out = capsys.readouterr().out
    assert "2 entries, 1 debriefed" in out and "✓ debriefed" in out
    deck_notes.main(_args(action="show", text="001"))
    assert "— debrief —" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        deck_notes.main(_args(action="show", text="009"))


def test_a_malformed_log_line_is_an_error_not_a_skip(deck):
    _add("fine")
    with open(deck / "log.jsonl", "a") as f:
        f.write("not json\n")
    with pytest.raises(SystemExit):
        deck_notes.read_log(SLUG)


# ── The merge ────────────────────────────────────────────────────────────


def _handoff(deck, entries):
    (deck / ".agent-out" / "debrief.json").write_text(json.dumps(
        {"slug": SLUG, "entries": entries}))


def test_merge_writes_by_id_and_rejects_ids_the_log_lacks(deck):
    _add("one"); _add("two")
    _handoff(deck, {"002": {"summary": "read two", "takeaways": []},
                    "007": {"summary": "a game nobody logged", "takeaways": []}})
    merged, rejected, total = merge_debrief.merge(SLUG)
    assert merged == ["002"] and rejected == ["007"] and total == 1
    doc = json.loads((deck / "log_annotations.json").read_text())
    assert "007" not in doc["entries"], "the annotation cannot add games to the log"


def test_merge_carries_earlier_annotations_and_never_touches_the_log(deck):
    _add("one"); _add("two")
    log_before = (deck / "log.jsonl").read_text()
    _handoff(deck, {"001": {"summary": "first read", "takeaways": []}})
    merge_debrief.merge(SLUG)
    _handoff(deck, {"002": {"summary": "second read", "takeaways": []}})
    merge_debrief.merge(SLUG)
    doc = json.loads((deck / "log_annotations.json").read_text())
    assert doc["entries"]["001"]["summary"] == "first read"
    assert doc["entries"]["002"]["summary"] == "second read"
    assert (deck / "log.jsonl").read_text() == log_before


def test_merge_refuses_an_empty_handoff(deck):
    _add("one")
    _handoff(deck, {})
    with pytest.raises(SystemExit):
        merge_debrief.merge(SLUG)
    assert not (deck / "log_annotations.json").exists()


# ── The validator ────────────────────────────────────────────────────────


def _validate(deck, entries):
    doc = {"slug": SLUG, "entries": entries}
    names = set()
    for c in json.loads((deck / "cards.json").read_text())["cards"]:
        names |= validate_debrief.expand_faces(c["name"])
    return validate_debrief.validate(SLUG, doc, deck_notes.read_log(SLUG), names,
                                     {"ignition", "wincon"}, deck)


GOOD = {
    "summary": "Lost to a wrath; the Hoof was the only finisher drawn.",
    "opponents": [{"seat": "the Dimir player", "archetype": "control",
                   "evidence": "held up two every turn"}],
    "cards": [{"card": "Craterhoof Behemoth", "read": "under", "why": "countered"},
              {"card": "Reflection of Kiki-Jiki", "read": "as-expected", "why": "a face name"},
              {"card": "Wrath of God", "read": "missed", "why": "named in the note"}],
    "decisions": [{"spot": "swing with four or wait", "worth_a_spread": True}],
    "takeaways": ["Hold the Hoof against open blue."],
    "engine_stages": ["wincon"],
    "lines": [{"cards": ["Craterhoof Behemoth"], "status": "verified",
               "stack_artifact": "stacks/003-hoof.json"},
              {"cards": ["Radagast of Rhosgobel", "Craterhoof Behemoth"],
               "status": "needs a stack scenario"}],
    "open_questions": [{"question": "second finisher?", "settled_by": "diagnose",
                        "why_it_matters": "two games lost the same way"}],
}
NOTE = ("The Dimir player held up two every turn and countered the Hoof. "
        "Wrath of God on five. Should have waited for the fifth body.")


def test_a_sound_annotation_passes(deck):
    _add(NOTE)
    assert _validate(deck, {"001": GOOD}) == []


def test_an_annotation_for_a_game_nobody_logged_fails(deck):
    _add(NOTE)
    errs = _validate(deck, {"001": GOOD, "002": GOOD})
    assert any("no such log entry" in e for e in errs)


def test_an_opponent_reading_needs_a_verbatim_phrase(deck):
    _add(NOTE)
    bad = dict(GOOD, opponents=[{"seat": "the Boros player", "archetype": "aggro",
                                 "evidence": "curved out with hasty creatures"}])
    errs = _validate(deck, {"001": bad})
    assert any("not a phrase of the note" in e for e in errs)
    # whitespace and case do not count against a real phrase
    ok = dict(GOOD, opponents=[{"seat": "x", "archetype": "y",
                                "evidence": "Held  up TWO every turn"}])
    assert _validate(deck, {"001": ok}) == []


def test_a_card_from_nowhere_fails_but_a_note_named_card_passes(deck):
    _add(NOTE)
    bad = dict(GOOD, cards=[{"card": "Sol Ring", "read": "over", "why": "?"}])
    errs = _validate(deck, {"001": bad})
    assert any("neither in the 99 nor named in the note" in e for e in errs)
    bad_read = dict(GOOD, cards=[{"card": "Craterhoof Behemoth", "read": "great", "why": "?"}])
    assert any(".read:" in e for e in _validate(deck, {"001": bad_read}))


def test_a_line_is_not_verified_on_the_pilots_word(deck):
    _add(NOTE)
    no_stack = dict(GOOD, lines=[{"cards": ["Craterhoof Behemoth"], "status": "verified"}])
    assert any("stack_artifact" in e for e in _validate(deck, {"001": no_stack}))
    wrong_cards = dict(GOOD, lines=[{"cards": ["Sol Ring"], "status": "verified",
                                     "stack_artifact": "stacks/003-hoof.json"}])
    assert any("never mentions" in e for e in _validate(deck, {"001": wrong_cards}))
    asserted = dict(GOOD, lines=[{"cards": ["Craterhoof Behemoth"], "status": "works"}])
    assert any(".status:" in e for e in _validate(deck, {"001": asserted}))


def test_routes_and_stages_are_closed_sets(deck):
    _add(NOTE)
    bad_route = dict(GOOD, open_questions=[{"question": "?", "settled_by": "the-pilot",
                                            "why_it_matters": "?"}])
    assert any("settled_by" in e for e in _validate(deck, {"001": bad_route}))
    bad_stage = dict(GOOD, engine_stages=["mana", "wincon"])
    assert any("engine_stages" in e for e in _validate(deck, {"001": bad_stage}))
    assert "diagnose" in validate_debrief.DEBRIEF_SETTLED_BY, (
        "a logged game is exactly what should send a deck back to the doctor")


def test_summary_and_takeaways_are_required(deck):
    _add(NOTE)
    errs = _validate(deck, {"001": {"summary": ""}})
    assert any("missing 'takeaways'" in e for e in errs)
    assert any("summary: empty" in e for e in errs)


# ── deck-status and the cache gate ───────────────────────────────────────


def test_deck_status_reports_the_log_and_its_coverage(deck):
    rows = {r["stage"]: r for r in deck_status.status(SLUG, validate=False)}
    assert rows["log"]["state"] == "missing"
    _add("one"); _add("two")
    (deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": {"summary": "s", "takeaways": []}}}))
    rows = {r["stage"]: r for r in deck_status.status(SLUG, validate=False)}
    assert rows["log"]["state"] == "present"
    assert rows["log"]["detail"] == "2 logged, 1 debriefed"
    assert rows["log"]["new"], "a stage added this cycle reports MISSING on old decks"


def test_deck_status_runs_the_debrief_gate_on_the_log_row(deck):
    """The log has no validator; the annotation beside it does, and the row must
    run it — a green row over a broken debrief is a dashboard hiding a gate."""
    _add(NOTE)
    (deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"009": {"summary": "a game nobody logged",
                                            "takeaways": []}}}))
    rows = {r["stage"]: r for r in deck_status.status(SLUG, validate=True)}
    assert rows["log"]["state"] == "INVALID"
    (deck / "log_annotations.json").write_text(json.dumps(
        {"slug": SLUG, "entries": {"001": GOOD}}))
    rows = {r["stage"]: r for r in deck_status.status(SLUG, validate=True)}
    assert rows["log"]["state"] == "present"


def test_debrief_is_not_applicable_until_something_is_logged(deck):
    assert not _debrief_applicable(SLUG)
    _add("one")
    assert _debrief_applicable(SLUG)
