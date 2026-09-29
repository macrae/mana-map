"""`poh_procedures.json` — the authored half of the handbook, which had no gate.

Tracked on six decks, stamped with a `decklist_sha256_prefix` by `install_agent`, and
never form-checked: no `deck_status.VALIDATED` entry, no `STAGES` row, no freshness test.
`validate-poh` checks the RENDERED HTML, which cannot see a condition outside the closed
vocabulary or a `grounded_in` citing a game nobody played.
"""

import glob
import json
import pathlib

import pytest

from manamap.pilot import poh_spec
from manamap.pilot import validate_poh_procedures as v


def _good():
    """A minimal file shaped like the six real ones."""
    return {
        "emergency": [{"condition": c, "condition_text": "x", "indications": ["i"],
                       "immediate": ["a"], "subsequent": ["b"], "notes": "n"}
                      for c in sorted(poh_spec.EMERGENCY_CONDITIONS)],
        "normal": {p: {"steps": ["s"]} for p in v.PHASES},
        "handling": {k: ["x"] for k in v.HANDLING},
    }


def test_it_does_not_fire_on_any_tracked_file():
    """THE BAR THIS REPO SETS BEFORE A VALIDATOR SHIPS. Six proposed checks have been
    rejected here for firing on correct data, and one written earlier today would have.
    Swept against all six real files; heliod's two extra keys are a NOTE, not an error."""
    from manamap.pilot import deck_notes

    checked = 0
    for p in sorted(glob.glob("data/decks/*/poh_procedures.json")):
        slug = pathlib.Path(p).parts[2]
        try:
            ids = {e["id"] for e in deck_notes.read_log(slug)}
        except Exception:                                    # noqa: BLE001
            ids = None
        errors, _ = v.validate(json.load(open(p)), log_ids=ids)
        assert not errors, f"{slug}: {errors}"
        checked += 1
    assert checked >= 6, f"expected the tracked corpus, swept {checked}"


def test_a_minimal_well_formed_file_passes():
    errors, notes = v.validate(_good(), log_ids=set())
    assert errors == [], errors
    assert notes == []


def test_a_condition_outside_the_closed_vocabulary_is_refused():
    """The conditions mirror `deck_notes.CAUSES` so the dossier can COUNT how games end.
    An invented one silently splits the count — the reason that vocabulary is closed."""
    doc = _good()
    doc["emergency"][0]["condition"] = "comboed"
    errors, _ = v.validate(doc, log_ids=set())
    assert any("not one the log can record" in e for e in errors), errors


def test_an_ordered_field_given_as_a_string_is_refused():
    """The renderer emits `<ol>` from `immediate`; a string draws one character per step."""
    doc = _good()
    doc["emergency"][0]["immediate"] = "Cast Teferi's Protection."
    errors, _ = v.validate(doc, log_ids=set())
    assert any("not a \n" not in e and "immediate is str" in e for e in errors), errors


def test_a_grounded_in_citing_a_game_nobody_logged_is_refused():
    """The log is the authority. This is the same admissibility rule `merge_debrief`
    enforces by id — an annotation may not add games the pilot did not log."""
    doc = _good()
    doc["emergency"][0]["grounded_in"] = ["log:007"]
    errors, _ = v.validate(doc, log_ids={"001", "002"})
    assert any("not a log entry" in e for e in errors), errors
    # And the same page passes once that game exists.
    errors, _ = v.validate(doc, log_ids={"001", "002", "007"})
    assert errors == [], errors


def test_an_empty_grounded_in_is_honest_and_never_an_error():
    """zur-enchantress has zero across all seven pages, and the agent charter says so
    explicitly: "a page with an empty `grounded_in` is honest". Erroring would demand the
    pilot invent games."""
    doc = _good()
    for page in doc["emergency"]:
        page["grounded_in"] = []
    errors, _ = v.validate(doc, log_ids=set())
    assert errors == [], errors


def test_no_log_at_all_does_not_check_references():
    """A deck can carry a handbook before it has games; a gate that reddened on that would
    demand the pilot play before writing."""
    doc = _good()
    doc["emergency"][0]["grounded_in"] = ["log:001"]
    errors, _ = v.validate(doc, log_ids=None)
    assert errors == [], errors


def test_a_missing_phase_and_an_unknown_phase_are_both_refused():
    doc = _good()
    del doc["normal"]["cruise"]
    errors, _ = v.validate(doc, log_ids=set())
    assert any("no 'cruise' phase" in e for e in errors), errors

    doc = _good()
    doc["normal"]["taxiing"] = {"steps": ["x"]}
    errors, _ = v.validate(doc, log_ids=set())
    assert any("cannot draw" in e for e in errors), errors


def test_two_pages_for_one_condition_are_refused():
    """One condition per page is the form; two means the dossier counts one twice."""
    doc = _good()
    doc["emergency"].append(dict(doc["emergency"][0]))
    errors, _ = v.validate(doc, log_ids=set())
    assert any("two pages for the same condition" in e for e in errors), errors


def test_it_is_registered_where_the_status_command_looks():
    """A validator is only a gate if something runs it — `deck_status.VALIDATED` is how
    `deck-status` and the fleet sweep in `test_pilot_tracked_artifacts_validate` see it."""
    from manamap.pilot.deck_status import VALIDATED

    assert VALIDATED.get("poh_procedures.json") == \
        "manamap.pilot.validate_poh_procedures"
    assert VALIDATED.get("pilot_policy.json") == "manamap.pilot.validate_pilot_policy"
