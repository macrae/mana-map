"""`merge-prose`: two agents, one file, and neither may clobber the other.

`manual-writer` owns six keys of `manual_prose.json` and `pilot-coach` owns two.
A whole-file copy from either agent's `.agent-out/` artifact deletes the other's
work silently — nothing errors, and the loss surfaces much later as a manual
section rendering short.

This was a ~40-line script re-derived by hand at the end of every prose refresh.
The ownership map is not a convention to remember: it is
`AGENT_ROUTINES[routine]["artifact_keys"]`, the same declaration the cache
fingerprints per key.
"""

import json

import pytest

from manamap.config import AGENT_ROUTINES
from manamap.pilot import merge_prose

from conftest import requires_deck


@pytest.fixture
def deck(tmp_path, monkeypatch):
    """A deck directory with both agents' handoffs staged."""
    base = tmp_path / "decks" / "testdeck"
    (base / ".agent-out").mkdir(parents=True)
    monkeypatch.setattr(merge_prose, "deck_dir", lambda slug: base)
    return base


def _write(base, agent_file, doc):
    (base / ".agent-out" / agent_file).write_text(json.dumps(doc))


def test_a_merge_touches_only_the_keys_the_routine_owns(deck):
    """The property the whole command exists to hold."""
    (deck / "manual_prose.json").write_text(json.dumps(
        {"how_it_wins": "writer's work", "matchups": "old coach"}))
    _write(deck, "pilot-coach.json",
           {"threat_assessment": "new", "matchups": "new",
            "how_it_wins": "COACH SHOULD NOT OWN THIS"})

    merge_prose.merge("testdeck", "coach-prose")

    doc = json.loads((deck / "manual_prose.json").read_text())
    assert doc["matchups"] == "new"
    assert doc["threat_assessment"] == "new"
    assert doc["how_it_wins"] == "writer's work", (
        "the coach wrote a key it does not own and the merge took it — this is "
        "exactly the clobber the key-scoping prevents")


def test_the_other_agents_keys_survive(deck):
    """A whole-file copy would drop `matchups` here and nothing would say so."""
    (deck / "manual_prose.json").write_text(json.dumps(
        {"matchups": "coach's work", "threat_assessment": "coach's work"}))
    _write(deck, "manual-writer.json",
           {k: "written" for k in AGENT_ROUTINES["writer-prose"]["artifact_keys"]})

    merge_prose.merge("testdeck", "writer-prose")

    doc = json.loads((deck / "manual_prose.json").read_text())
    assert doc["matchups"] == "coach's work"
    assert len(doc) == 8, "six writer keys plus the coach's two"


def test_a_wrapped_payload_is_unwrapped(deck):
    """Charters ask for flat keys; a wrapper is a common harmless variation.

    Guessing wrong means merging zero keys, so the test is whether an OWNED key
    is present — not what the wrapper happens to be called.
    """
    _write(deck, "pilot-coach.json",
           {"prose": {"threat_assessment": "t", "matchups": "m"}})
    merged, missing, _ = merge_prose.merge("testdeck", "coach-prose")
    assert sorted(merged) == ["matchups", "threat_assessment"]
    assert not missing


def test_a_payload_with_none_of_its_keys_refuses_to_write(deck):
    """Merging nothing and reporting success is how a section renders empty
    with every check still green."""
    (deck / "manual_prose.json").write_text(json.dumps({"matchups": "keep me"}))
    _write(deck, "pilot-coach.json", {"something_else": "?"})

    with pytest.raises(SystemExit) as e:
        merge_prose.merge("testdeck", "coach-prose")
    assert "refusing to write" in str(e.value)
    doc = json.loads((deck / "manual_prose.json").read_text())
    assert doc == {"matchups": "keep me"}, "a refusal must not have written"


def test_a_partial_payload_reports_what_is_missing(deck):
    """Partial revision is a supported mode, so this reports rather than fails."""
    _write(deck, "manual-writer.json", {"how_it_wins": "only this one"})
    merged, missing, _ = merge_prose.merge("testdeck", "writer-prose")
    assert merged == ["how_it_wins"]
    assert "mana_base" in missing


def test_a_missing_handoff_names_the_agent_to_spawn(deck):
    with pytest.raises(SystemExit) as e:
        merge_prose.merge("testdeck", "coach-prose")
    assert "pilot-coach" in str(e.value)


def test_the_two_routines_partition_the_artifact():
    """Overlapping ownership would make the merge order significant, and the
    last writer would win a race nobody declared."""
    coach = set(AGENT_ROUTINES["coach-prose"]["artifact_keys"])
    writer = set(AGENT_ROUTINES["writer-prose"]["artifact_keys"])
    assert not (coach & writer), f"both routines claim {coach & writer}"


@requires_deck
def test_merging_a_real_decks_handoffs_reproduces_the_tracked_artifact(
        tmp_path, monkeypatch):
    """The strongest available check: the tracked file was produced by these
    two merges, so re-running them must change nothing — same keys, same
    ownership split, same JSON formatting.

    Run against a COPY. Writing to the tracked artifact would be idempotent in
    the happy path and a partial-file race under `-n auto` in every other, and
    a test that can corrupt committed data to prove data is not corrupted is
    not a test worth having.
    """
    import shutil

    from manamap.pilot.common import deck_dir
    real = deck_dir("heliod")
    if not all((real / ".agent-out" / f).exists()
               for f in ("pilot-coach.json", "manual-writer.json")):
        pytest.skip("heliod .agent-out handoffs not present (they are gitignored)")

    base = tmp_path / "heliod"
    (base / ".agent-out").mkdir(parents=True)
    shutil.copy(real / "manual_prose.json", base / "manual_prose.json")
    for f in ("pilot-coach.json", "manual-writer.json"):
        shutil.copy(real / ".agent-out" / f, base / ".agent-out" / f)
    monkeypatch.setattr(merge_prose, "deck_dir", lambda slug: base)

    before = (base / "manual_prose.json").read_text()
    merge_prose.merge("heliod", "coach-prose")
    merge_prose.merge("heliod", "writer-prose")
    assert (base / "manual_prose.json").read_text() == before, (
        "merging the handoffs did not reproduce the tracked artifact")
