"""`validate-engine` v2 (2026-11-22): the computed half is held to a fresh build, the
authored `reading` to its form. Driven through `validate_engine.validate(slug, doc=…)`
on the synthetic deck `test_pilot_engine_graph` builds, so every check exercises the
production gate; each test reintroduces one bug and watches the gate catch it.
"""
import copy
import json

import pytest

from manamap.pilot import engine_graph as eg
from manamap.pilot import validate_engine as ve
from test_pilot_engine_graph import SLUG, synthetic_deck  # noqa: F401 - the fixture


@pytest.fixture
def written(synthetic_deck):  # noqa: F811
    doc, path = eg.write(SLUG)
    return doc, path


def _errors(doc):
    return ve.validate(SLUG, doc=doc)[0]


def test_a_fresh_write_passes(written):
    doc, _ = written
    assert _errors(doc) == []


def test_a_graph_differing_from_a_fresh_compute_is_refused(written):
    doc, _ = written
    bad = copy.deepcopy(doc)
    # A hand edit: promote a card into ignition.
    for st in bad["stages"]:
        if st["stage"] == "ignition":
            st["cards"].append("Sol Ring")
    errs = _errors(bad)
    assert errs and "differs from a fresh build" in errs[0] and "stages" in errs[0]
    assert "hand edit" in errs[0]
    # The list moved: the stamp names the why.
    moved = copy.deepcopy(doc)
    moved["decklist_sha256"] = "f" * 64
    errs = _errors(moved)
    assert errs and "the list moved" in errs[0]


def test_the_reading_is_copied_forward_untouched(written):
    doc, path = written
    doc["reading"]["thesis"] = "Edgar makes bodies; the outlets bill the table."
    doc["reading"]["lines"] = [{"line": doc["lines"][0]["id"], "name": "THE LOOP", "note": "n"}]
    path.write_text(json.dumps(doc))
    again, _ = eg.write(SLUG)
    assert again["reading"]["thesis"] == doc["reading"]["thesis"]
    assert again["reading"]["lines"][0]["name"] == "THE LOOP"
    assert _errors(again) == []


def test_a_reading_for_another_list_is_marked_stale_not_dropped(written):
    doc, _ = written
    other = copy.deepcopy(doc)
    other["reading"]["written_for"] = "b" * 64
    other["reading"]["thesis"] = "written against last month's list"
    errs = _errors(other)
    assert any("reading.stale is not true" in e for e in errs), errs
    other["reading"]["stale"] = True
    assert not any("stale" in e for e in _errors(other))
    # And the reverse lie: stale on the current list.
    lie = copy.deepcopy(doc)
    lie["reading"]["stale"] = True
    assert any("stale is true for the current list" in e for e in _errors(lie))


def test_a_reading_line_must_name_a_graph_line(written):
    doc, _ = written
    bad = copy.deepcopy(doc)
    bad["reading"]["lines"] = [{"line": "L99", "name": "ghost", "note": "n"}]
    errs = _errors(bad)
    assert any("L99" in e and "neither a computed line nor a component" in e for e in errs), errs
    ok = copy.deepcopy(doc)
    ok["reading"]["lines"] = [{"line": "ev:DIES", "name": "the death component", "note": "n"}]
    assert not any("neither" in e for e in _errors(ok))


def test_a_stage_override_names_a_card_a_stage_and_a_why(written):
    doc, _ = written
    bad = copy.deepcopy(doc)
    bad["reading"]["stage_overrides"] = [{"card": "Nothing Here", "stage": "wincon", "why": ""},
                                         {"card": "Blood Artist", "stage": "sideboard", "why": "x"}]
    errs = _errors(bad)
    assert any("Nothing Here" in e for e in errs)
    assert any("no why" in e for e in errs)
    assert any("'sideboard' is not a stage" in e for e in errs)


def test_a_critic_verdict_is_pass_or_fail(written):
    doc, _ = written
    bad = copy.deepcopy(doc)
    bad["reading"]["critic"] = {"verdict": "meh", "findings": [{"status": "vibes"}]}
    errs = _errors(bad)
    assert any("must be pass or fail" in e for e in errs)
    assert any("status 'vibes'" in e for e in errs)


def test_a_v1_file_is_refused_with_the_migration_command(synthetic_deck):  # noqa: F811
    v1 = {"slug": SLUG, "thesis": "t", "stages": [{"stage": "mana", "cards": ["Sol Ring"]}], "lines": []}
    errs = _errors(v1)
    assert len(errs) == 1 and "v1" in errs[0] and f"engine-graph {SLUG} --write" in errs[0]


@pytest.mark.skipif(not __import__("manamap.sim.forge_cards", fromlist=["x"]).installed(),
                    reason="the record is read from Forge's card scripts")
def test_every_tracked_engine_json_matches_a_fresh_run():
    """The freshness gate the tracked fleet is held to (regression tier): every
    committed `engine.json` is v2 and its computed half equals a fresh build —
    the same claim `combos.json` makes. A record that differs was hand-edited,
    or the parser moved without `engine-graph --write`."""
    from manamap.config import DECKS_DIR
    checked, bad = 0, []
    for path in sorted(DECKS_DIR.glob("*/engine.json")):
        slug = path.parent.name
        doc = json.loads(path.read_text())
        errs = ve.validate(slug, doc=doc)[0]
        checked += 1
        if errs:
            bad.append(f"{slug}: {errs[0][:160]}")
    assert checked >= 1, "no tracked engine.json found"
    assert not bad, "\n".join(bad)
