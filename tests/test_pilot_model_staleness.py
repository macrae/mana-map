"""The model stamp, and the staleness nothing could see.

Regenerating the fleet after the mana-rock and colour fixes left 39 figures
quoted in authored prose describing a model that no longer runs — and
`validate-diagnosis`, `validate-strategic-frame` and `validate-tutor-guide` all
passed, because the decklist sha had not moved and no artifact recorded which
MODEL produced a number.
"""

import json

import pytest

from conftest import requires_deck
from manamap.pilot import goldfish, model_staleness
from manamap.pilot.common import DECKS_DIR


def test_the_stamp_is_derived_not_hand_kept():
    """A version somebody has to remember to bump is one that will not be."""
    a = goldfish.model_version()
    assert a == goldfish.model_version(), "not deterministic"
    assert len(a) == 12 and all(c in "0123456789abcdef" for c in a)


def test_the_stamp_moves_when_the_simulator_moves(tmp_path, monkeypatch):
    """Coarse ON PURPOSE — a sha over the file, not a curated list of
    'model-facing' lines. The curated version is the judgement call that goes
    wrong silently; the cost of the coarse one is a regeneration nobody needed
    after a comment edit, which is the cheaper mistake."""
    import hashlib
    import pathlib
    original = pathlib.Path(goldfish.__file__).read_bytes()
    before = goldfish.model_version()
    fake = tmp_path / "goldfish.py"
    fake.write_bytes(original + b"\n# a comment\n")
    monkeypatch.setattr(goldfish, "__file__", str(fake))
    assert goldfish.model_version() != before, (
        "editing the simulator did not move its version — the stamp is not "
        "derived from what it claims to describe")
    assert goldfish.model_version() == hashlib.sha256(
        fake.read_bytes()).hexdigest()[:12]


def test_an_unknown_deck_is_not_a_staleness_verdict():
    """`stamp_of` runs inside three validators. `deck_dir` raises on a missing
    directory, and letting that through turns an absent deck into a crash in a
    gate about something else."""
    assert model_staleness.stamp_of("no-such-deck-here") is None
    assert model_staleness.note("no-such-deck-here", {}) == ''


@requires_deck
def test_every_tracked_goldfish_carries_the_stamp():
    stamped = 0
    for path in sorted(DECKS_DIR.glob("*/goldfish_metrics.json")):
        meta = json.loads(path.read_text())["meta"]
        assert "model_version" in meta, path
        stamped += 1
    assert stamped >= 8, f"only {stamped} artifacts checked"


@requires_deck
def test_the_fleet_is_stamped_with_the_model_that_is_running():
    """If this fails the fleet needs regenerating — which is the whole point of
    having the stamp, and was undecidable before it."""
    current = goldfish.model_version()
    stale = [p.parent.name for p in sorted(DECKS_DIR.glob("*/goldfish_metrics.json"))
             if json.loads(p.read_text())["meta"].get("model_version") != current]
    assert not stale, (
        f"{len(stale)} deck(s) were measured by an older simulator: {stale}. "
        f"Re-run `manamap pilot goldfish <slug>` for each.")


@requires_deck
def test_prose_written_against_an_older_model_is_REPORTED_not_failed():
    """The three states, and only one of them is evidence.

    A document with no stamp predates stamping — unknown, not stale — and
    saying otherwise would redden every agent artifact in the repo on day one.
    """
    current = goldfish.model_version()
    assert model_staleness.note("heliod", {"model_version": current}) == ""
    assert model_staleness.note("heliod", {}) == "", "an unstamped doc is not stale"
    note = model_staleness.note("heliod", {"model_version": "0" * 12})
    assert "OLDER MODEL" in note
    assert "Do NOT hand-edit" in note, (
        "the note must say what NOT to do — hand-patching prose to green a "
        "gate puts a fresh claim under an old byline")
