"""The pre-registered campaign: an overnight queue that measures and never merges.

Pinned: a tracked campaign passes its own gate; `plan` pins a `working` ref to
a sha and the entry becomes a run id; a list that moved is STALE and is not
run; `run` skips DONE and resumes RUNNING; state is derived from the records,
never stored; an A/A is prepended per harness; and the module never imports
the merge.
"""

import ast
import json
import pathlib
import subprocess

import pytest

from manamap.sim import campaign as cp
from manamap.sim import experiment as ex
from manamap.pilot import deck_history as dh
from conftest import ROOT

SLUG = "xdeck"
V1 = "1 Radagast of Rhosgobel *CMDR*\n1 Craterhoof Behemoth\n30 Forest\n"
V2 = V1.replace("Craterhoof Behemoth", "Hornet Queen")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path
    deck = root / "data" / "decks" / SLUG
    deck.mkdir(parents=True)
    monkeypatch.setattr("manamap.config.DECKS_DIR", root / "data" / "decks")
    monkeypatch.setattr(dh, "_REPO_ROOT", root)
    monkeypatch.setattr(cp, "CAMPAIGNS_DIR", root / "data" / "campaigns")
    monkeypatch.setattr(ex, "card_overrides", lambda: None)
    monkeypatch.setattr(ex, "seat_sha", lambda o: "0" * 64)      # the pod's seats are not in this repo
    monkeypatch.setattr(cp, "harness_fingerprint",
                        lambda: {"forge": "test", "card_overrides": None, "clock_seconds": 600})
    monkeypatch.setattr("manamap.sim.power.null_rate", lambda pod: (0.233, 412))
    _git(root, "init", "-q")
    (deck / "decklist.txt").write_text(V1)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "v1")
    (deck / "decklist.txt").write_text(V2)
    _git(root, "add", "."); _git(root, "commit", "-q", "-m", "v2")
    (root / "data" / "campaigns").mkdir()
    doc = {"name": "t", "authored": "2026-09-29", "hypothesis": "does V2 beat V1",
           "entries": [{"id": "ab", "slug": SLUG, "a": "V1", "b": "working", "pod": "standard-v3",
                        "games": 8, "looks": 2, "primary": "win_rate",
                        "hypothesis": "V2 wins more at standard-v3"}]}
    (root / "data" / "campaigns" / "t.json").write_text(json.dumps(doc))
    return root


def test_every_tracked_campaign_passes_its_validator():
    files = sorted((ROOT / "data" / "campaigns").glob("*.json"))
    files = [f for f in files if not f.name.endswith(".state.json")]
    assert files, "no tracked campaign — the registry would name an artifact nothing writes"
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        assert cp.validate(doc) == [], (f.name, cp.validate(doc))
        assert doc["name"] == f.stem


def test_the_validator_refuses_what_cannot_be_run():
    base = {"name": "x", "hypothesis": "h", "entries": [
        {"id": "e", "slug": "goblin-storm", "a": "V1", "b": "working", "pod": "standard-v3",
         "games": 40, "looks": 4, "hypothesis": "h"}]}
    assert cp.validate(base) == []
    bad = json.loads(json.dumps(base)); bad["entries"][0]["games"] = 42
    assert any("divide" in e for e in cp.validate(bad))
    bad = json.loads(json.dumps(base)); bad["entries"][0]["b"] = "V1"
    assert any("same ref" in e for e in cp.validate(bad))
    bad = json.loads(json.dumps(base)); bad["entries"][0]["pod"] = "nope"
    assert any("not a table" in e for e in cp.validate(bad))
    bad = json.loads(json.dumps(base)); bad["entries"][0]["primary"] = "damage"
    assert any("registered endpoint" in e for e in cp.validate(bad))
    bad = json.loads(json.dumps(base)); bad["entries"].append(dict(base["entries"][0]))
    assert any("duplicate" in e for e in cp.validate(bad))
    assert any("no hypothesis" in e for e in cp.validate({"name": "x", "entries": base["entries"]}))


def test_plan_pins_working_to_a_sha_and_the_entry_becomes_a_run_id(repo):
    doc, lines = cp.plan("t")
    ab = next(e for e in doc["entries"] if e["id"] == "ab")
    r = ab["resolved"]
    assert r["b_sha"] == ex.resolve_arm(SLUG, "working")["decklist_sha256"]
    assert r["experiment_id"].endswith("-k2") and "-podExperimental" in r["experiment_id"]
    assert r["harness"]["forge"] == "test" and r["planned"]
    # an A/A for this harness was prepended, once
    aa = [e for e in doc["entries"] if e.get("aa")]
    assert len(aa) == 1 and doc["entries"][0]["aa"] and aa[0]["a"] == aa[0]["b"] == "working"
    again, _ = cp.plan("t")
    assert sum(1 for e in again["entries"] if e.get("aa")) == 1, "a second plan adds no second A/A"
    assert any("night" in l for l in lines)
    on_disk = json.loads((repo / "data" / "campaigns" / "t.json").read_text())
    assert on_disk["entries"][1]["resolved"]["experiment_id"] == r["experiment_id"]
    assert "state" not in on_disk["entries"][1], "state is derived, never stored"


def test_an_entry_whose_list_moved_is_stale_and_is_not_run(repo, monkeypatch):
    cp.plan("t")
    (repo / "data" / "decks" / SLUG / "decklist.txt").write_text(V2 + "1 Sol Ring\n")
    rows = {r["id"]: r for r in cp.status("t")}
    assert rows["ab"]["state"] == "STALE" and "pinned" in rows["ab"]["why"]
    started = []
    monkeypatch.setattr(ex, "run", lambda *a, **kw: started.append(kw) or (None, {"status": "complete", "experiment_id": "x", "delta": {}}))
    ran = cp.run("t", only=["ab"])
    assert ran == [] and started == [], "a stale entry measures a list nobody holds"


def test_run_skips_done_and_resumes_running(repo, monkeypatch):
    doc, _ = cp.plan("t")
    ab = next(e for e in doc["entries"] if e["id"] == "ab")
    exp_dir = repo / "data" / "decks" / SLUG / "experiments"
    exp_dir.mkdir(parents=True)
    # a half-finished sequential record for `ab`, a finished one for the A/A
    (exp_dir / f"{ab['resolved']['experiment_id']}.json").write_text(json.dumps(
        {"experiment_id": ab["resolved"]["experiment_id"], "status": "running",
         "design": {"looks": 2}, "looks": [{"k": 1}], "delta": {}}))
    aa = doc["entries"][0]
    (exp_dir / f"{aa['resolved']['experiment_id']}.json").write_text(json.dumps(
        {"experiment_id": aa["resolved"]["experiment_id"], "status": "complete",
         "delta": {"reading": "noise floor"}}))
    rows = {r["id"]: r for r in cp.status("t")}
    assert rows["ab"]["state"] == "RUNNING" and rows[aa["id"]]["state"] == "DONE"
    calls = []
    def fake_run(slug, a, b, opponents, **kw):
        calls.append(kw)
        return None, {"status": "complete", "experiment_id": "x", "delta": {"win_rate": {}}}
    monkeypatch.setattr(ex, "run", fake_run)
    ran = cp.run("t")
    assert [c["resume"] for c in calls] == [True], calls
    assert calls[0]["looks"] == 2 and calls[0]["pod_name"] == "standard-v3" and calls[0]["seed"] == ab["resolved"]["seed"]
    assert ran == [("ab", "complete")]
    assert not cp.live_for(SLUG), "the state file is idle after the run"


def test_a_harness_that_changed_since_planning_is_refused(repo, monkeypatch):
    cp.plan("t")
    monkeypatch.setattr(cp, "harness_fingerprint",
                        lambda: {"forge": "test", "card_overrides": "deadbeef0000", "clock_seconds": 600})
    with pytest.raises(SystemExit) as e:
        cp.run("t", only=["ab"])
    assert "another harness is another measurement" in str(e.value)


def test_nothing_in_a_campaign_merges():
    """Read the module, not its behaviour: the queue may never reach the one
    function that rewrites a decklist."""
    src = (ROOT / "src" / "manamap" / "sim" / "campaign.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert "deck_branch" not in names and "merge" not in names and "propose" not in names, names
