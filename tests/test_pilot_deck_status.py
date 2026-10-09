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


@pytest.mark.slow
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


def test_the_sim_row_counts_runs_on_the_current_list(tmp_path, monkeypatch):
    """A RUN DESCRIBES THE LIST IT PLAYED, and the status row now says how many
    describe the list on disk.

    Measured 2026-09-21 (known-issues §13): five of six sleeved decks had ZERO
    Forge runs on the sleeved list, and this row read "4 run(s)" for each — a
    dossier printed a win rate beside a version it did not belong to and nothing
    said so. Re-introduce the bug by putting `detail = f"{len(files)} run(s)"`
    back and this fails on the count; the state must stay `present`, because a
    run on an older list is older evidence and `promote.GATES` reads this row.
    """
    from manamap.pilot import deck_status

    base = tmp_path / "decks" / "scratch"
    (base / "sim").mkdir(parents=True)
    (base / "cards.json").write_text(json.dumps(
        {"decklist_sha256": "b" * 64, "cards": []}))
    (base / "sim" / "old.json").write_text(json.dumps(
        {"run_id": "old", "seats": [{"slug": "scratch", "decklist_sha256": "a" * 64}]}))
    (base / "sim" / "new.json").write_text(json.dumps(
        {"run_id": "new", "seats": [{"slug": "scratch", "decklist_sha256": "b" * 64}]}))
    (base / "sim" / "unstamped.json").write_text(json.dumps(
        {"run_id": "unstamped", "seats": [{"slug": "scratch"}]}))
    monkeypatch.setattr(deck_status, "deck_dir", lambda slug, branch=None: base)

    rows = {r["stage"]: r for r in deck_status.status("scratch", validate=False)}
    sim = rows["sim"]
    assert sim["state"] == "present"
    assert (sim["runs"], sim["runs_on_current_list"]) == (3, 1), sim
    assert sim["detail"] == "3 run(s), 1 on the current list", sim["detail"]
    # The subject is found by NAME, not by position, and an unstamped record
    # answers False rather than matching an absent sha.
    assert deck_status.sim_run_describes(
        {"seats": [{"slug": "other", "decklist_sha256": "b" * 64},
                   {"slug": "scratch", "decklist_sha256": "a" * 64}]}, "scratch", "b" * 64) is False
    assert deck_status.sim_run_describes({"seats": [{"slug": "scratch"}]}, "scratch", None) is False


# ── 60-card formats (2026-10-09): the stages a Modern deck cannot have read n/a ──

def _format_deck(tmp_path, monkeypatch, fmt=None):
    decks = tmp_path / "decks"
    base = decks / "elvesish"
    base.mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", decks)
    # Sixty basics: legal in every format (basics are exempt from the copy
    # limit, and 60 is both Commander-short and constructed-minimum), so the
    # gate on cards.json has nothing to say and the printed OK line appears.
    doc = {"decklist_sha256": "b" * 64,
           "cards": [{"name": "Forest", "quantity": 60, "type_line": "Basic Land — Forest",
                      "color_identity": [], "colors": [], "mana_cost": "", "cmc": 0,
                      "oracle_text": "({T}: Add {G}.)", "layout": "normal"}]}
    if fmt:
        doc["format"] = fmt
    (base / "decklist.txt").write_text("60 Forest\n")
    (base / "cards.json").write_text(json.dumps(doc))
    return base


def test_applies_is_the_one_predicate():
    from manamap.pilot import deck_status, formats

    for stage in deck_status.COMMANDER_ONLY_STAGES:
        assert deck_status.applies(stage, formats.COMMANDER)
        assert not deck_status.applies(stage, formats.MODERN), stage
    assert deck_status.applies("mana", formats.MODERN) and deck_status.applies("combos", formats.MODERN)
    # Every name in the set is a real stage, so a typo cannot exempt nothing.
    assert deck_status.COMMANDER_ONLY_STAGES <= {row[0] for row in STAGES}


def test_a_60_card_deck_reads_n_a_for_the_commander_only_stages_and_they_leave_the_denominator(
        tmp_path, monkeypatch):
    """`missing` would hold elves to a denominator it can never reach and name
    commands that refuse (`goldfish`, `bracket-check`). Re-introduce the bug by
    making `applies` return True and the rows come back as `missing`."""
    from manamap.pilot import deck_status

    _format_deck(tmp_path, monkeypatch, fmt="modern")
    rows = deck_status.status("elvesish", validate=False)
    by = {r["stage"]: r for r in rows}
    for stage in deck_status.COMMANDER_ONLY_STAGES:
        assert by[stage]["state"] == "n/a", (stage, by[stage])
        assert by[stage]["detail"] == "not measured for Modern"
        assert by[stage]["required"] is False
    assert by["mana"]["state"] == "missing" and by["combos"]["state"] == "missing"
    counted = [r for r in rows if r["stage"] != "—" and r["state"] != "n/a"]
    assert len(counted) == len(STAGES) - len(deck_status.COMMANDER_ONLY_STAGES)

    # The Commander control: the same deck with no `format` is MISSING those
    # stages, not exempt from them.
    (tmp_path / "decks" / "elvesish" / "cards.json").write_text(json.dumps(
        {"decklist_sha256": "b" * 64, "cards": [{"name": "Forest", "quantity": 60}]}))
    rows = deck_status.status("elvesish", validate=False)
    assert all(r["state"] != "n/a" for r in rows)
    assert {r["stage"]: r["state"] for r in rows}["goldfish"] == "missing"


def test_the_printed_count_and_the_fleet_count_leave_the_n_a_rows_out(tmp_path, monkeypatch, capsys):
    from manamap.pilot import deck_status

    _format_deck(tmp_path, monkeypatch, fmt="modern")
    deck_status.main(type("A", (), {"slug": "elvesish", "as_json": False, "all_decks": False})())
    out = capsys.readouterr().out
    expected = len(STAGES) - len(deck_status.COMMANDER_ONLY_STAGES)
    assert f"/{expected} stages complete · {len(deck_status.COMMANDER_ONLY_STAGES)} not measured for this format" in out
    assert "  n/a    goldfish" in out and "not measured for Modern" in out
    assert f"/{expected} lifecycle stages present" in out

    monkeypatch.setattr("manamap.pilot.validate_pending.summarise",
                        lambda slug: {"open": 0, "applied": 0, "partial": 0})
    fleet = {r["slug"]: r for r in deck_status.fleet()}
    assert fleet["elvesish"]["total"] == expected
