"""`merge-prose`: one agent, one file, and the keys nobody owns must survive.

`pilot-notes` owns five keys of `manual_prose.json`. The published decks still
carry `card_roles`, `mana_base`, `upgrades`, `editors_letter` and `pilots_log` —
frozen legacy copy from the retired magazine agents that no routine owns. A
whole-file copy from the agent's `.agent-out/` artifact deletes them silently,
and the loss surfaces much later as a manual section rendering short.

The ownership map is not a convention to remember: it is
`AGENT_ROUTINES[routine]["artifact_keys"]`, the same declaration the cache
fingerprints per key. (Until 2026-08-19 the same merge kept two agents from
clobbering each other; one agent and a frozen remainder is the same problem.)
"""

import json

import pytest

from manamap.config import AGENT_ROUTINES
from manamap.pilot import merge_prose

from conftest import requires_deck

OWNED = AGENT_ROUTINES["pilot-notes"]["artifact_keys"]
LEGACY = ("card_roles", "mana_base", "upgrades", "editors_letter", "pilots_log")


@pytest.fixture
def deck(tmp_path, monkeypatch):
    base = tmp_path / "decks" / "testdeck"
    (base / ".agent-out").mkdir(parents=True)
    monkeypatch.setattr(merge_prose, "deck_dir", lambda slug: base)
    return base


def _write(base, doc, agent_file="pilot-notes.json"):
    (base / ".agent-out" / agent_file).write_text(json.dumps(doc))


def test_a_merge_touches_only_the_keys_the_routine_owns(deck):
    """The property the whole command exists to hold."""
    (deck / "manual_prose.json").write_text(json.dumps(
        {"how_it_wins": "old", "mana_base": "frozen legacy"}))
    _write(deck, {"how_it_wins": "new", "matchups": "new",
                  "mana_base": "THE AGENT MUST NOT OWN THIS"})

    merge_prose.merge("testdeck", "pilot-notes")

    doc = json.loads((deck / "manual_prose.json").read_text())
    assert doc["how_it_wins"] == "new" and doc["matchups"] == "new"
    assert doc["mana_base"] == "frozen legacy", (
        "the agent wrote a legacy key it does not own and the merge took it — "
        "exactly the clobber the key-scoping prevents")


def test_every_legacy_key_survives_a_full_merge(deck):
    """A whole-file copy would drop all five of these and nothing would say so."""
    (deck / "manual_prose.json").write_text(json.dumps({k: "legacy" for k in LEGACY}))
    _write(deck, {k: "written" for k in OWNED})

    merge_prose.merge("testdeck", "pilot-notes")

    doc = json.loads((deck / "manual_prose.json").read_text())
    assert all(doc[k] == "legacy" for k in LEGACY)
    assert len(doc) == len(OWNED) + len(LEGACY)


def test_the_routine_owns_no_legacy_key():
    assert not set(OWNED) & set(LEGACY), (
        "a retired key re-entered the owned set; it is frozen copy, not a routine's")


def test_a_wrapped_payload_is_unwrapped(deck):
    """Charters ask for flat keys; a wrapper is a common harmless variation.
    Guessing wrong means merging zero keys, so the test is whether an OWNED key
    is present — not what the wrapper happens to be called."""
    _write(deck, {"prose": {"threat_assessment": "t", "matchups": "m"}})
    merged, missing, _ = merge_prose.merge("testdeck", "pilot-notes")
    assert sorted(merged) == ["matchups", "threat_assessment"]
    assert sorted(missing) == ["combo_lines", "how_it_wins", "mulligan"]


def test_a_payload_with_none_of_its_keys_refuses_to_write(deck):
    """Merging nothing and reporting success is how a section renders empty
    with every check still green."""
    (deck / "manual_prose.json").write_text(json.dumps({"matchups": "keep me"}))
    _write(deck, {"something_else": "?"})

    with pytest.raises(SystemExit) as e:
        merge_prose.merge("testdeck", "pilot-notes")
    assert "refusing to write" in str(e.value)
    doc = json.loads((deck / "manual_prose.json").read_text())
    assert doc == {"matchups": "keep me"}, "a refusal must not have written"


def test_a_partial_payload_reports_what_is_missing(deck):
    """Partial revision is a supported mode, so this reports rather than fails."""
    _write(deck, {"how_it_wins": "only this one"})
    merged, missing, _ = merge_prose.merge("testdeck", "pilot-notes")
    assert merged == ["how_it_wins"]
    assert "mulligan" in missing


def test_a_missing_handoff_names_the_agent_to_spawn(deck):
    with pytest.raises(SystemExit) as e:
        merge_prose.merge("testdeck", "pilot-notes")
    assert "pilot-notes" in str(e.value)


@requires_deck
def test_merging_a_real_decks_handoff_reproduces_the_tracked_artifact(
        tmp_path, monkeypatch):
    """Re-running the merge that produced the tracked file must change nothing.
    Run against a COPY — a test that can corrupt committed data to prove data
    is not corrupted is not a test worth having."""
    import shutil

    from manamap.pilot.common import deck_dir
    real = deck_dir("heliod")
    if not (real / ".agent-out" / "pilot-notes.json").exists():
        pytest.skip("heliod .agent-out handoff not present (gitignored; none spawned yet)")

    base = tmp_path / "heliod"
    (base / ".agent-out").mkdir(parents=True)
    shutil.copy(real / "manual_prose.json", base / "manual_prose.json")
    shutil.copy(real / ".agent-out" / "pilot-notes.json", base / ".agent-out" / "pilot-notes.json")
    monkeypatch.setattr(merge_prose, "deck_dir", lambda slug: base)

    before = (base / "manual_prose.json").read_text()
    merge_prose.merge("heliod", "pilot-notes")
    assert (base / "manual_prose.json").read_text() == before
