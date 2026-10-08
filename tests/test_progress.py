"""`manamap.progress`: the live file the job band draws. What it has to get right
is the one thing a terminal cannot say — that a job STOPPED — so the heartbeat
is tested apart from the count, and the log counter on the boundary case that
would make a running simulation read short.
"""
import json
import re
import time

from manamap import progress


def _read(p):
    return json.loads(p.path.read_text())


def test_the_heartbeat_moves_while_the_count_does_not(tmp_path):
    """A 5-minute test, a 20-minute Forge job: no progress for a long time is
    normal. Only a heartbeat that STOPS means the job died or the machine slept."""
    p = progress.Progress("pytest unit", total=10, unit="tests",
                          heartbeat=0.05, directory=tmp_path).start()
    first = _read(p)["updated_at"]
    time.sleep(0.3)
    second = _read(p)
    p.finish()
    assert second["done"] == 0 and second["state"] == "running"
    assert second["updated_at"] > first


def test_a_finish_says_whether_it_passed(tmp_path):
    with progress.Progress("regen", total=3, unit="targets", directory=tmp_path) as p:
        p.advance()
        p.advance(failed=1)
    doc = _read(p)
    assert (doc["done"], doc["failed"], doc["state"]) == (2, 1, "failed")
    ok = progress.Progress("regen", total=1, directory=tmp_path, name="other").start()
    ok.advance()
    ok.finish()
    assert _read(ok)["state"] == "passed"


def test_a_counter_is_polled_on_every_beat_and_at_the_end(tmp_path):
    seen = iter(range(100))
    p = progress.Progress("simulate x", total=100, unit="games", heartbeat=0.05,
                          counter=lambda: next(seen), directory=tmp_path).start()
    time.sleep(0.3)
    p.finish()
    assert _read(p)["done"] >= 3


def test_the_log_counter_reads_only_what_is_new_and_never_splits_a_line(tmp_path):
    """Forge writes a game's result line in one go, but a read can land in the
    middle of it; counting the half line twice, or not at all, would make a
    running simulation read wrong."""
    done = re.compile(r"^Game Result: Game (\d+) ended in ", re.M)
    log = tmp_path / "part-00.log"
    count = progress.LogCounter(done, str(tmp_path / "part-*.log"))
    log.write_text("Turn: Turn 1\nGame Result: Game 1 ended in a win\nGame Res")
    assert count() == 1
    with open(log, "a") as f:
        f.write("ult: Game 2 ended in a draw\n")
    assert count() == 2
    (tmp_path / "part-01.log").write_text("Game Result: Game 1 ended in a loss\n")
    assert count() == 3
    assert count() == 3, "a second read of unchanged files counted again"


def test_the_off_switch_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MANAMAP_NO_PROGRESS", "1")
    with progress.Progress("regen", total=1, directory=tmp_path) as p:
        p.advance()
    assert not list(tmp_path.iterdir())


def test_a_dead_runs_file_is_pruned_and_a_live_ones_kept(tmp_path):
    """A killed run left NO HEARTBEAT on the band for half an hour; a live
    process with a stale heartbeat (a machine that slept) must still show."""
    import json as _json
    import os
    dead = tmp_path / "simulate-999999.json"            # no such pid
    dead.write_text(_json.dumps({"state": "running", "started_at": 0, "updated_at": 0}))
    alive = tmp_path / f"pytest-{os.getpid()}9.json"    # unlikely pid, but finished
    alive.write_text(_json.dumps({"state": "passed", "started_at": 0, "updated_at": 0}))
    me = tmp_path / f"regen-{os.getppid()}.json"         # running, process alive
    me.write_text(_json.dumps({"state": "running", "started_at": 0, "updated_at": 0}))
    progress.Progress("regen", directory=tmp_path, name="new").start().finish()
    assert not dead.exists()
    assert me.exists()
    assert alive.exists(), "a FRESH finished file stays until it is ten minutes old"


def test_a_running_job_is_the_parent_of_whatever_starts_under_it(tmp_path, monkeypatch):
    """THE JOB GRAPH: a job exports its id while it runs, so a child — in this process
    or a subprocess it launches — names it as parent with no wiring at the call site.
    Siblings started one after another are siblings, not a chain."""
    import os
    import subprocess
    import sys
    monkeypatch.delenv(progress.PARENT_ENV, raising=False)
    with progress.Progress("prepush full", directory=tmp_path, name="prepush") as top:
        with progress.Progress("pytest unit", directory=tmp_path, name="unit") as a:
            pass
        with progress.Progress("pytest regression", directory=tmp_path, name="regr",
                               feeds=["make manuals"]) as b:
            seen = subprocess.run([sys.executable, "-c",
                                   f"import os; print(os.environ['{progress.PARENT_ENV}'])"],
                                  capture_output=True, text=True).stdout.strip()
    assert "parent" not in _read(top) and _read(top)["id"] == top.id == f"prepush-{os.getpid()}"
    assert _read(a)["parent"] == top.id and _read(b)["parent"] == top.id
    assert _read(b)["feeds"] == ["make manuals"] and seen == b.id
    assert progress.PARENT_ENV not in os.environ


def test_a_stated_parent_wins_over_the_inherited_one(tmp_path, monkeypatch):
    monkeypatch.setenv(progress.PARENT_ENV, "regen-1")
    p = progress.Progress("simulate x", directory=tmp_path, parent="experiment-9").start()
    p.finish()
    assert _read(p)["parent"] == "experiment-9"
    assert progress.os.environ[progress.PARENT_ENV] == "regen-1"   # restored, not cleared
