"""deck-status: the dashboard must not be green while a gate is red.

`VALIDATED` (which artifacts have a validator) and `STAGES` (which artifacts are
steps in building a deck) are different lists, and the status loop only ever walked
`STAGES`. So three gated artifacts — `diagnosis.json`, `build_plan.json` and
`deck_recon.json` — had validators `deck-status` could not run.

Measured 2026-08-22, while fixing the DFC-pip defect: the fleet view reported "0
failing a gate" across all 11 decks in the same second that `validate-diagnosis
heliod` failed. That is the precise divergence the `VALIDATED` map was extracted
from the test suite to end, reappearing through the other door.
"""

import json

import pytest

from manamap.config import DECKS_DIR
from manamap.pilot.deck_status import STAGES, VALIDATED, status

from conftest import requires_deck


@requires_deck
def test_every_gated_artifact_is_reported_even_without_a_lifecycle_stage():
    staged = {row[1] for row in STAGES}
    orphans = set(VALIDATED) - staged
    assert orphans, "if every gate gained a stage row, delete this test"

    for slug in ("heliod", "radagast"):
        reported = {r["artifact"] for r in status(slug, validate=False)}
        for artifact in orphans:
            if (DECKS_DIR / slug / artifact).exists():
                assert artifact in reported, (
                    f"{slug}: {artifact} has a gate but deck-status never runs it")


@requires_deck
def test_a_gate_row_is_not_counted_as_a_lifecycle_stage():
    """A deck with MORE evidence must not read as less finished. Before this, the
    count jumped from 13/15 to 13/17 purely because two gated artifacts existed."""
    rows = status("radagast", validate=False)
    stages = [r for r in rows if r["stage"] != "—"]
    gates = [r for r in rows if r["stage"] == "—"]
    assert gates, "radagast has diagnosis.json, which is gated and not a stage"
    assert len(stages) == len(STAGES), "the lifecycle count must not move"
    assert all(r["state"] in ("gate", "INVALID") for r in gates)


@requires_deck
def test_a_failing_gate_names_the_artifact_not_a_dash():
    """A gate row has no stage, so the fleet view reported "FAILS ITS GATE: —",
    which tells a reader nothing about what to fix.

    THE LOOP USED TO PASS PRECISELY WHEN NOTHING WAS WRONG: `invalid` and
    `stale` are empty on a healthy fleet, so it could only fail by accident and
    proved nothing on every green run. A synthetic failing row exercises the
    naming, and the fleet loop keeps its coverage with a count that proves it
    ran at all.
    """
    from manamap.pilot.deck_status import fleet
    rows = fleet()
    assert rows, "the fleet is empty — this test cannot see the bug it guards"
    for row in rows:
        for name in row["invalid"] + row["stale"]:
            assert name and name != "—", row["slug"]

    # THE PROPERTY, DRIVEN THROUGH THE PRINTER, on a row that IS failing —
    # which the fleet loop above cannot supply on a healthy checkout. A
    # `hasattr` fallback here would be the same vacuous shape one level down.
    import argparse
    import io
    from contextlib import redirect_stdout

    from manamap.pilot import deck_status
    row = {"slug": "synthetic", "done": 3, "total": 15, "stale": [],
           "invalid": ["engine.json"], "pending_open": 0,
           "pending_partial": 0, "pending_applied": 0}
    buf = io.StringIO()
    with redirect_stdout(buf):
        original, deck_status.fleet = deck_status.fleet, lambda: [row]
        try:
            deck_status._fleet_main(argparse.Namespace(slug=None, as_json=False))
        except SystemExit:
            pass
        finally:
            deck_status.fleet = original
    out = buf.getvalue()
    assert "FAILS ITS GATE: engine.json" in out, out
    assert "FAILS ITS GATE: —" not in out


def test_the_engines_staleness_check_can_actually_fire(tmp_path, monkeypatch):
    """THE WIRING WAS THERE AND THE CURRENT FLOWED NOWHERE.

    `STAGES` declares `engine.json`'s stamp path as `decklist_sha256`, but the
    row was computed by an `elif key == "engine"` branch sitting ABOVE the
    staleness check in the same chain — so it short-circuited, and the check
    could never run on any deck, ever.

    What it cost: edgar-vampires' engine model named twelve cards the deck
    stopped running the day the bloodline branch merged, and `deck-status` read
    `OK  engine  critic: pass` for a week. A model that describes a different
    list is stale whatever its critic thought of it — the critic signed off on
    the OLD deck.

    Prove it by reverting the fix: move the critic branch back above the sha
    check and this test goes red while everything else stays green.
    """
    from manamap.pilot import deck_status

    base = tmp_path / "decks" / "scratch"
    base.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Sol Ring\n")
    (base / "cards.json").write_text(json.dumps(
        {"decklist_sha256": "b" * 64, "cards": [{"name": "Sol Ring"}]}))
    (base / "engine.json").write_text(json.dumps(
        {"decklist_sha256": "a" * 64,          # a DIFFERENT list
         "thesis": "x", "stages": [], "lines": [],
         "critic": {"verdict": "pass"}}))
    monkeypatch.setattr(deck_status, "deck_dir", lambda slug, branch=None: base)

    rows = {r["stage"]: r for r in deck_status.status("scratch", validate=False)}
    engine = rows.get("engine")
    assert engine, rows
    assert engine["state"] == "STALE", (
        f"engine row is {engine['state']} with a stamp naming another list — "
        f"the critic branch is short-circuiting the staleness check again")
    assert "critic: pass" in engine["detail"], (
        "the critic verdict must be ADDED to the staleness read, not replaced "
        "by it — both facts matter and they answer different questions")


def test_the_tutor_guide_has_a_staleness_path_at_all():
    """Its stamp path was `None`, so there was no check — on the one artifact
    whose entire content is card names, which is what rots first when a list
    moves. Both of the fleet's tutor guides currently fail their validator for
    exactly that reason."""
    from manamap.pilot.deck_status import STAGES

    paths = {row[0]: row[2] for row in STAGES}
    assert paths["tutors"], "tutor_guide.json has no staleness path"
    assert "decklist_sha256" in paths["tutors"]
