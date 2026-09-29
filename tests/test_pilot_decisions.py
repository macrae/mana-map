"""The decision ledger: append-only, a reason for every withdrawal and rejection,
a prediction frozen on every propose and merge, and an outcome that joins the
merged list's own runs back to the merge it closes.
"""

import json

import pytest

from manamap.pilot import decisions, validate_decisions


@pytest.fixture
def deck(tmp_path, monkeypatch):
    base = tmp_path / "decks" / "x"
    base.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Sol Ring\n")
    monkeypatch.setattr(decisions, "deck_dir", lambda slug, branch=None: base / "branches" / branch if branch else base)
    monkeypatch.setattr(decisions, "_deck_sha", lambda slug, base=None: "d" * 64)
    return base


def test_the_ledger_is_append_only_and_refuses_a_malformed_line(deck):
    a = decisions.append("x", "propose", branch="b", as_version="v1.1.0", prediction={"endpoint": "kill_by_8"})
    b = decisions.append("x", "withdraw", branch="b", reason="the goldfish cannot see it")
    assert [e["id"] for e in decisions.read("x")] == ["001", "002"]
    assert a["deck_decklist_sha256"] == "d" * 64 and b["reason"]
    with pytest.raises(SystemExit) as e:
        decisions.append("x", "withdraw", branch="b")
    assert "needs a reason" in str(e.value)
    with pytest.raises(SystemExit):
        decisions.append("x", "sell", branch="b")
    with open(decisions.path("x"), "a") as f:
        f.write("{not json\n")
    with pytest.raises(SystemExit) as e:
        decisions.read("x")
    assert "append-only" in str(e.value)


def test_withdraw_records_a_reason_and_leaves_branch_json_clean(tmp_path, monkeypatch):
    """THE BUG: a withdrawal vanished. `withdraw` pops the proposal and wrote
    nothing anywhere; now it refuses without a reason, records it in the
    ledger, and STILL adds no key to branch.json — the graveyard stays out."""
    from manamap.pilot import deck_branch
    base = tmp_path / "decks" / "x"
    bdir = base / "branches" / "b"
    bdir.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Sol Ring\n")
    (bdir / "branch.json").write_text(json.dumps(
        {"slug": "x", "branch": "b", "v": 3, "objective": {"axis": "kill_by_8", "op": ">=", "value": 0.3},
         "commits": [], "proposal": {"as_version": "v1.1.0", "at": "2026-09-01", "accepted_on": {}}}))
    monkeypatch.setattr(deck_branch, "deck_dir", lambda slug, branch=None: bdir if branch else base)
    monkeypatch.setattr(deck_branch, "branch_root", lambda slug: base / "branches")
    monkeypatch.setattr(decisions, "deck_dir", lambda slug, branch=None: bdir if branch else base)
    monkeypatch.setattr(decisions, "_deck_sha", lambda slug, base=None: "d" * 64)
    with pytest.raises(SystemExit) as e:
        deck_branch.withdraw("x", "b")
    assert "--reason" in str(e.value)
    got = deck_branch.withdraw("x", "b", reason="cards never arrived")
    assert got["withdrew"]["as_version"] == "v1.1.0"
    on_disk = json.loads((bdir / "branch.json").read_text())
    assert "proposal" not in on_disk and "withdrawn" not in on_disk, on_disk
    lines = decisions.read("x")
    assert lines[-1]["kind"] == "withdraw" and lines[-1]["reason"] == "cards never arrived"
    assert lines[-1]["as_version"] == "v1.1.0"


def test_reject_is_derived_into_branch_state_from_the_ledger(tmp_path, monkeypatch):
    from manamap.pilot import deck_branch
    base = tmp_path / "decks" / "x"
    bdir = base / "branches" / "b"
    bdir.mkdir(parents=True)
    (base / "decklist.txt").write_text("1 Sol Ring\n")
    (bdir / "branch.json").write_text(json.dumps(
        {"slug": "x", "branch": "b", "v": 2, "objective": {"axis": "kill_by_8", "op": ">=", "value": 0.3},
         "commits": []}))
    monkeypatch.setattr(deck_branch, "deck_dir", lambda slug, branch=None: bdir if branch else base)
    monkeypatch.setattr(decisions, "deck_dir", lambda slug, branch=None: bdir if branch else base)
    monkeypatch.setattr(decisions, "_deck_sha", lambda slug, base=None: "d" * 64)
    assert deck_branch.branch_state("x", "b")[0] == deck_branch.OPEN
    with pytest.raises(SystemExit):
        deck_branch.reject("x", "b")
    got = deck_branch.reject("x", "b", reason="Forge read it 0.061 worse at an interval excluding zero")
    assert got["state"][0] == deck_branch.REJECTED
    assert "rejected" in got["state"][1]
    assert "reject" not in json.loads((bdir / "branch.json").read_text()), "stored nowhere in the branch file"
    # a later propose line reopens it: the state follows the ledger's last word
    decisions.append("x", "propose", branch="b", prediction={"endpoint": "kill_by_8"})
    assert deck_branch.branch_state("x", "b")[0] == deck_branch.OPEN
    assert deck_branch.REJECTED in deck_branch.BRANCH_STATES


def _sim_record(path, slug, sha, pod, wins, decided, games, overrides=None, profile="Default"):
    doc = {"run_id": path.stem, "pod": {"name": pod},
           "seats": [{"slug": slug, "decklist_sha256": sha}],
           "profiles": [profile, "Experimental"],
           "analysis": {"games": games, "decided": decided, "seats": {slug: {"wins": wins}}}}
    if overrides:
        doc["card_overrides"] = {"sha": overrides}
    path.write_text(json.dumps(doc))


def test_outcome_joins_only_runs_of_the_merged_list_at_the_same_pod_and_harness(deck, monkeypatch):
    """A merge predicted +0.05 [-0.02, +0.12] at standard-v3 under the plain
    harness against a champion of 20/100. Runs of the merged list at another
    pod, under overrides, or of another list are not its outcome; the two that
    qualify pool to 30/100 — a realised difference inside the prediction. The
    paper record joins by sha and is shown, never pooled."""
    merged_sha = "m" * 64
    (deck / "sim").mkdir()
    _sim_record(deck / "sim" / "a.json", "x", merged_sha, "standard-v3", 12, 40, 40)
    _sim_record(deck / "sim" / "b.json", "x", merged_sha, "standard-v3", 18, 60, 60)
    _sim_record(deck / "sim" / "other-pod.json", "x", merged_sha, "vito-era", 30, 60, 60)
    _sim_record(deck / "sim" / "overridden.json", "x", merged_sha, "standard-v3", 30, 60, 60, overrides="60636e9e5565")
    _sim_record(deck / "sim" / "older-list.json", "x", "o" * 64, "standard-v3", 30, 60, 60)
    (deck / "log.jsonl").write_text(json.dumps({"id": "001", "decklist_sha256": merged_sha, "result": "win"}) + "\n"
                                    + json.dumps({"id": "002", "decklist_sha256": "o" * 64, "result": "loss"}) + "\n")
    monkeypatch.setattr("manamap.pilot.deck_notes.log_path", lambda slug: deck / "log.jsonl")
    decisions.append("x", "merge", branch="b", decklist_sha256=merged_sha,
                     prediction={"endpoint": "forge.win_rate",
                                 "forge": {"pod": "standard-v3", "card_overrides": None, "ai_profile": None,
                                           "champion": {"wins": 20, "games": 100, "rate": 0.2},
                                           "delta": 0.05, "ci95": [-0.02, 0.12], "mde": 0.15, "null": 0.233}})
    assert decisions.awaiting("x")[0]["runs_of_merged_list"] == 2
    added = decisions.outcome("x")
    assert len(added) == 1
    r = added[0]["realised"]
    assert (r["wins"], r["decided"]) == (30, 100) and sorted(r["run_ids"]) == ["a", "b"]
    assert r["difference"]["delta"] == 0.1 and added[0]["inside_prediction"] is True
    assert r["paper"] == {"games": 1, "win": 1, "loss": 0, "draw": 0}
    # idempotent: a second pass appends nothing, and the validator is happy
    assert decisions.outcome("x") == []
    assert decisions.awaiting("x") == []
    assert validate_decisions.validate(decisions.read("x")) == []


def test_a_proposal_freezes_the_forge_prediction_beside_the_goldfish_one():
    nc = {"objective": {"axis": "damage_10", "op": ">=", "value": 24}, "decklist_sha256": "c" * 64,
          "objective_grade": {"state": "met", "reading": 24.3},
          "recommendation": {"state": "merge"}, "harness": {"iterations": 10000, "seed": 1, "model_version": "v"},
          "forge": {"available": True, "pod": "standard-v3", "card_overrides": None,
                    "champion": {"wins": 7, "games": 94, "rate": 0.0745}, "branch": {"wins": 1, "games": 73, "rate": 0.0137},
                    "delta": -0.061, "ci95": [-0.133, 0.010], "excludes_zero": False, "mde": 0.15,
                    "null": {"rate": 0.233}, "run_ids": {"champion": ["r1", "r2"], "branch": ["r3"]},
                    "endpoints": {"forge.win_rate": {"delta": -0.061, "ci95": [-0.133, 0.010], "mde": 0.15}}}}
    p = decisions.prediction_from(nc)
    assert p["endpoint"] == "damage_10" and p["grade"] == "met" and p["recommendation"] == "merge"
    assert p["forge"]["delta"] == -0.061 and p["forge"]["null"] == 0.233 and p["forge"]["run_ids"]["champion"] == ["r1", "r2"]
    absent = decisions.prediction_from(dict(nc, forge={"available": False, "why": "no run"}))
    assert absent["forge"] is None and absent["forge_why"] == "no run"
    assert decisions.prediction_from(None) is None


def test_the_validator_holds_the_ledger_to_its_form():
    ok = [{"id": "001", "at": "2026-09-29", "kind": "propose", "branch": "b", "prediction": {}},
          {"id": "002", "at": "2026-09-29", "kind": "merge", "branch": "b", "decklist_sha256": "m" * 64, "prediction": {}},
          {"id": "003", "at": "2026-09-30", "kind": "outcome", "of": "002", "realised": {"rate": 0.3, "run_ids": ["r"]}}]
    assert validate_decisions.validate(ok) == []
    gap = [ok[0], dict(ok[1], id="003")]
    assert any("sequential" in e for e in validate_decisions.validate(gap))
    assert any("no reason" in e for e in validate_decisions.validate(
        [{"id": "001", "at": "x", "kind": "reject", "branch": "b"}]))
    assert any("not a merge" in e for e in validate_decisions.validate(
        [{"id": "001", "at": "x", "kind": "outcome", "of": "009", "realised": {"rate": 0.1, "run_ids": ["r"]}}]))
    assert any("no prediction" in e for e in validate_decisions.validate(
        [{"id": "001", "at": "x", "kind": "propose", "branch": "b"}]))
    assert validate_decisions.validate(
        [{"id": "001", "at": "x", "kind": "propose", "branch": "b", "backfilled": True}]) == []
    twice = ok + [dict(ok[2], id="004")]
    assert any("already has an outcome" in e for e in validate_decisions.validate(twice))


def test_every_tracked_ledger_passes_its_gate():
    from conftest import ROOT
    files = sorted((ROOT / "data" / "decks").glob("*/decisions.jsonl"))
    assert files, "backfill seeded no ledger — the registry would name an artifact nothing writes"
    for f in files:
        lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert validate_decisions.validate(lines) == [], (f, validate_decisions.validate(lines))
