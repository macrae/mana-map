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
    which tells a reader nothing about what to fix."""
    from manamap.pilot.deck_status import fleet
    for row in fleet():
        for name in row["invalid"] + row["stale"]:
            assert name and name != "—", row["slug"]
