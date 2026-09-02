"""How a game ended — the pilot's own claim, filed apart from the log it annotates.

`--result` says whether you won. The cause says WHY, and the difference is the
whole reason the dossier's game table can show counts instead of paragraphs:
three losses to `removal` and three to `mana-drought` are two different decks
with the same record, and prose cannot be counted.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot import deck_notes
from manamap.pilot.deck_notes import CAUSES, CAUSES_FILE, causes, read_log
from manamap.pilot.validate_log_causes import validate


def test_the_vocabulary_is_closed_and_small():
    """CLOSED, not free text, and not a `--tag`. The moment "comboed" and
    "combo'd" both exist the count silently splits in two and the table
    understates by half while still rendering. Small because a vocabulary
    nobody can hold in their head gets used wrong."""
    assert 5 <= len(CAUSES) <= 12, CAUSES
    for key, gloss in CAUSES.items():
        assert key == key.lower() and " " not in key, key
        assert len(gloss) > 20, f"{key} has no usable gloss"
    assert "won" in CAUSES, "a win has a cause too"


def test_the_cli_choices_are_derived_from_the_vocabulary():
    """`registry.py` carried a hand-copied literal of the embedding-space slugs
    once, and a rename left the flag accepting a slug nothing could resolve.
    The same mistake here would accept a cause the writer then refuses."""
    import argparse

    from manamap.pilot.registry import add_pilot_parser

    parser = argparse.ArgumentParser()
    add_pilot_parser(parser.add_subparsers(dest="command"))
    ns = parser.parse_args(["pilot", "deck-notes", "x", "list"])
    action = [a for a in parser._subparsers._group_actions][0]
    notes = action.choices["pilot"]._subparsers._group_actions[0].choices["deck-notes"]
    flag = [a for a in notes._actions if a.dest == "cause"][0]
    assert sorted(flag.choices) == sorted(CAUSES), flag.choices
    assert ns is not None


# ── the writer ───────────────────────────────────────────────────────────

def _scratch(tmp_path, monkeypatch, entries):
    base = tmp_path / "decks" / "scratch"
    base.mkdir(parents=True)
    with open(base / "log.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    monkeypatch.setattr(deck_notes, "deck_dir", lambda slug, branch=None: base)
    return base


ENTRY = {"id": "001", "at": "2026-09-01T19:00:00-07:00", "result": "loss",
         "opponents": 3, "tags": [], "text": "x", "decklist_sha256": "a" * 64}


def test_a_cause_is_written_to_a_sidecar_and_the_log_is_untouched(tmp_path, monkeypatch):
    """THE LOG IS APPEND-ONLY AND NEVER REWRITTEN — its own module docstring.
    Nine games were logged before this field existed, so putting the cause on
    the entry would mean rewriting those lines or leaving the field permanently
    absent on most of the evidence there is."""
    base = _scratch(tmp_path, monkeypatch, [ENTRY])
    before = (base / "log.jsonl").read_bytes()
    deck_notes.set_cause("scratch", "001", "wipe", note="third wipe")
    assert (base / "log.jsonl").read_bytes() == before, "the log was rewritten"
    doc = json.loads((base / CAUSES_FILE).read_text())
    assert doc["entries"]["001"]["cause"] == "wipe"
    assert doc["entries"]["001"]["note"] == "third wipe"
    assert doc["entries"]["001"]["at"], "a claim with no date cannot be read later"


def test_the_writer_refuses_a_cause_outside_the_vocabulary(tmp_path, monkeypatch):
    _scratch(tmp_path, monkeypatch, [ENTRY])
    with pytest.raises(SystemExit, match="is not a cause"):
        deck_notes.set_cause("scratch", "001", "comboed")


def test_the_writer_refuses_an_id_the_log_does_not_have(tmp_path, monkeypatch):
    """A cause filed against a missing id counts toward nothing and appears in
    no table. It errors nowhere; the roll-up is simply short and looks fine."""
    _scratch(tmp_path, monkeypatch, [ENTRY])
    with pytest.raises(SystemExit, match="no log entry"):
        deck_notes.set_cause("scratch", "007", "wipe")


def test_add_with_a_cause_goes_through_the_same_writer(tmp_path, monkeypatch):
    """ONE WRITER, so `add --cause` and the standalone verb cannot disagree
    about the shape of the sidecar or skip its vocabulary check."""
    import argparse

    base = _scratch(tmp_path, monkeypatch, [])
    monkeypatch.setattr(deck_notes, "decklist_sha256", lambda slug: "b" * 64)
    deck_notes.main(argparse.Namespace(
        slug="scratch", action="add", text="a game", file=None, result="loss",
        opponents=3, tag=[], at=None, cause="combo", note=None))
    doc = json.loads((base / CAUSES_FILE).read_text())
    assert doc["entries"]["001"]["cause"] == "combo"


# ── the gate ─────────────────────────────────────────────────────────────

def test_the_validator_is_silent_on_every_tracked_file():
    """A VALIDATOR THAT FIRES ON CORRECT DATA IS WORSE THAN NO VALIDATOR, and
    the only way to know is to measure it against the whole fleet."""
    checked = 0
    for path in sorted(DECKS_DIR.glob("*/" + CAUSES_FILE)):
        checked += 1
        slug = path.parent.name
        errs = validate(json.loads(path.read_text()), entries=read_log(slug))
        assert errs == [], f"{slug}: {errs}"
    assert checked >= 3


def test_the_validator_catches_a_cause_that_contradicts_its_result():
    """Two authored claims about one game that disagree, and only a gate that
    reads both files can see it."""
    won = [{"id": "001", "result": "win"}]
    errs = validate({"entries": {"001": {"cause": "wipe"}}}, entries=won)
    assert any("WIN" in e for e in errs), errs
    lost = [{"id": "001", "result": "loss"}]
    errs = validate({"entries": {"001": {"cause": "won"}}}, entries=lost)
    assert any("other direction" in e for e in errs), errs


def test_every_logged_game_on_a_played_deck_has_a_stated_cause():
    """The backfill, held at rest. A cause is the pilot's claim about their own
    game — nothing derives it — so a game without one is a row the dossier's
    priors table cannot count. Not a gate on new games (a cause can be filed
    later); a check that the eleven already played did not lose theirs."""
    checked = 0
    for path in sorted(DECKS_DIR.glob("*/log.jsonl")):
        slug = path.parent.name
        entries = read_log(slug)
        if not entries:
            continue
        filed = causes(slug)
        for e in entries:
            checked += 1
            assert e["id"] in filed, (
                f"{slug} entry {e['id']} has no stated cause — "
                f"`manamap pilot deck-notes {slug} cause {e['id']} --cause <code>`")
    assert checked >= 10
