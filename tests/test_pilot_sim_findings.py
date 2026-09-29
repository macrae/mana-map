"""`sim_findings.json`: the sim debrief's skeleton, computed from the run
records; prose that may cite only finding ids; a merge that recomputes the
skeleton and takes the prose; a gate that holds every number to the record.
"""

import json

import pytest

from manamap.pilot import merge_sim_findings as msf
from manamap.pilot import pilot_findings as pf
from manamap.pilot import validate_sim_findings as vsf
from conftest import ROOT


def _record(run_id, wins=9, decided=82, games=100, pod="standard-v3", sha="a" * 64):
    return {"run_id": run_id, "at": "2026-09-28", "slug": "x", "games_completed": games,
            "pod": {"name": pod}, "profiles": ["Default", "Experimental"],
            "seats": [{"slug": "x", "decklist_sha256": sha}],
            "summary": {"wins": {"x": wins}, "decided": decided},
            "analysis": {"games": games, "decided": decided,
                         "seats": {"x": {"wins": wins, "win_rate": round(wins / decided, 3),
                                         "win_rate_ci95": [0.05, 0.19],
                                         "eliminated_by": {"jarad": 40, "sythis": 30},
                                         "eliminated_how": {"damage": 60, "life loss": 10},
                                         "first_attack_turn": {"mean": 14.3, "median": 13, "n": 71,
                                                               "ci95": [13.1, 15.5]}}},
                         "wipe_recovery": {"available": False, "why": "none seen"}},
            "engine_casts": {"seat": "mm-x", "games": games, "turns": 900, "kept_hand_mean": 7.0,
                             "by_card": {"Zada, Hedron Grinder": {"cast": 100, "activated": 0, "discarded": 0},
                                         "Haze of Rage": {"cast": 0, "activated": 0, "discarded": 4}}}}


@pytest.fixture
def deck(tmp_path, monkeypatch):
    base = tmp_path / "decks" / "x"
    (base / "sim").mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Zada, Hedron Grinder *CMDR*\n1 Haze of Rage\n")
    (base / "sim" / "r1.json").write_text(json.dumps(_record("run-one-aaaaaaaa")))
    monkeypatch.setattr("manamap.config.DECKS_DIR", tmp_path / "decks")
    monkeypatch.setattr(pf, "deck_dir", lambda slug, branch=None: base)
    monkeypatch.setattr(vsf, "deck_dir", lambda slug, branch=None: base)
    monkeypatch.setattr("manamap.sim.forge._out_dir", lambda slug: base / "sim")
    monkeypatch.setattr("manamap.sim.power.null_rate", lambda pod: (0.233, 412))
    monkeypatch.setattr("manamap.sim.engine_casts.engine_set", lambda slug, branch=None: None)
    return base


def test_the_skeleton_is_findings_with_ids_figures_and_intervals(deck):
    doc = pf.skeleton("x")
    run = doc["runs"]["run-one-aaaaaaaa"]
    kinds = [f["kind"] for f in run["findings"]]
    assert kinds[:4] == ["rate_vs_null", "pilot_quality", "loss_decomposition", "held"] or "rate_vs_null" in kinds
    rate = next(f for f in run["findings"] if f["kind"] == "rate_vs_null")
    assert rate["id"] == pf._fid("run-one-aaaaaaaa", 1) and rate["figure"] == 0.11 and rate["n"] == 82
    assert pf._fid("a-podExperimental-c600", 1) != pf._fid("b-podExperimental-c600", 1), (
        "two runs sharing a harness suffix must not share finding ids")
    assert rate["null"]["rate"] == 0.233 and "47%" in rate["text"]
    held = next(f for f in run["findings"] if f["kind"] == "held")
    assert "Haze of Rage" in held["cards"] and held["measured"] == ["Haze of Rage"]
    assert run["prose"] == {} and run["run"]["current"] is False
    assert vsf.validate(doc, "x") == ([], [])


def test_prose_may_cite_only_this_runs_findings_and_quote_only_their_numbers(deck):
    doc = pf.skeleton("x")
    run = doc["runs"]["run-one-aaaaaaaa"]
    rate = next(f["id"] for f in run["findings"] if f["kind"] == "rate_vs_null")
    held = next(f["id"] for f in run["findings"] if f["kind"] == "held")
    run["prose"] = {"reading": f"9 of 82 decided games, 47% of par ({rate}).",
                    "so_what": [{"text": "Haze of Rage was discarded 4 times and never cast.",
                                 "cites": [held]}],
                    "open_questions": [{"question": "does the copy engine ever fire?",
                                        "settled_by": "resolve-stack", "cites": [held]}]}
    errs, notes = vsf.validate(doc, "x")
    assert errs == [], errs
    # a number the record does not carry
    run["prose"]["reading"] = f"the deck wins 0.11 of games and 57 of them were blowouts ({rate})"
    errs, _ = vsf.validate(doc, "x")
    assert any("57" in e and "not in any cited finding" in e for e in errs), errs
    # a citation to another run
    run["prose"]["reading"] = "see F-bbbbbbbb-01"
    errs, _ = vsf.validate(doc, "x")
    assert any("not a finding of this run" in e for e in errs)
    # a claim with no citation, and an unknown route
    run["prose"] = {"reading": "", "so_what": [{"text": "fine", "cites": []}],
                    "open_questions": [{"question": "q", "settled_by": "vibes", "cites": []}]}
    errs, _ = vsf.validate(doc, "x")
    assert any("cites nothing" in e for e in errs) and any("vibes" in e for e in errs)


def test_merge_recomputes_the_skeleton_and_takes_prose_only(deck, monkeypatch):
    base = deck
    (base / ".agent-out").mkdir()
    rate = pf._fid("run-one-aaaaaaaa", 1)
    handoff = {"runs": {"run-one-aaaaaaaa": {"prose": {"reading": f"47% of par ({rate}).",
                                                        "findings": [{"id": "F-aaaaaaaa-99", "kind": "invented"}]},
                        "findings": "an agent's own skeleton, ignored"},
               "run-nine": {"prose": {"reading": "a run that does not exist"}}}}
    (base / ".agent-out" / msf.AGENT_FILE).write_text(json.dumps(handoff))
    monkeypatch.setattr(msf.config, "DECKS_DIR", base.parent)
    merged, rejected, notes, path = msf.merge("x")
    assert merged == ["run-one-aaaaaaaa"] and rejected == ["run-nine"]
    on_disk = json.loads(path.read_text())
    run = on_disk["runs"]["run-one-aaaaaaaa"]
    assert run["prose"] == {"reading": f"47% of par ({rate})."}, "whitelisted to PROSE_KEYS"
    assert all(f["kind"] != "invented" for f in run["findings"]), "the skeleton is the records'"
    # a merge that would break the record refuses before writing
    handoff["runs"]["run-one-aaaaaaaa"]["prose"]["reading"] = f"wins 0.11 and 999 blowouts ({rate})"
    (base / ".agent-out" / msf.AGENT_FILE).write_text(json.dumps(handoff))
    with pytest.raises(SystemExit) as e:
        msf.merge("x")
    assert "does not hold to the record" in str(e.value)
    assert json.loads(path.read_text())["runs"]["run-one-aaaaaaaa"]["prose"]["reading"].startswith("47%")


def test_every_tracked_findings_file_passes_its_gate_and_matches_a_fresh_skeleton():
    files = sorted((ROOT / "data" / "decks").glob("*/sim_findings.json"))
    assert files, "no tracked sim_findings.json — the registry would name an artifact nothing writes"
    for f in files:
        slug = f.parent.name
        doc = json.loads(f.read_text(encoding="utf-8"))
        errs, _ = vsf.validate(doc, slug)
        assert errs == [], (slug, errs)
        fresh = pf.skeleton(slug)
        for rid, run in doc["runs"].items():
            assert rid in fresh["runs"], (slug, rid, "a run in the file that the records no longer hold")
            got = [{k: v for k, v in x.items()} for x in fresh["runs"][rid]["findings"]
                   if x["kind"] not in ("board_shape",)]
            want = [x for x in run["findings"] if x["kind"] not in ("board_shape",)]
            assert got == want, (slug, rid, "the skeleton on disk is stale — `sim-findings <slug> --write`")
